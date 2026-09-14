
from typing import Optional, Union
import torch
from ..lightning import AbstractLightningWrapper
from torchdiffeq import odeint
from ...utils.fft import fourier_transform, inverse_fourier_transform
from .samples import Samples
import torch.nn.functional as F
import wandb

"""
    This class implements flow matching in time domain
"""

class GaussianPriorFlowMatching(AbstractLightningWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = self.config.unet_model.domain if self.config.unet_model is not None else "time"

    @property
    def other_domain(self):
        if self.domain == "time":
            return "frequency"
        return "time"

    def forward(self, x, t, condition=None):
        if self.config.unet_model is not None and self.config.unet_model.include_other_domain:
            other_domain_x = self.tensor_to_domain(x, self.domain, self.other_domain)
            x = torch.cat([x, other_domain_x], dim=1)
        return self.model(x, t, condition)
    
    def get_initial_states(self, batch_size: int):
        """
            This function returns sample from the prior distribution
        """
        original_state = torch.randn(batch_size, self.data_channels, self.model_config.sequence_size, device=self.device) if self.config.unet_model is not None else torch.randn(batch_size, 2, self.model_config.sequence_size, device=self.device)
        return Samples(
            original_state,
            self.domain,
            0
        )
    
    def interpolate(self, source: torch.Tensor, target: torch.Tensor, t: torch.Tensor):
        """
            source: samples from the prior distribution in the models domain
            target: samples from the data distribution in the models domain
            t: interpolation time
        """
        return t * target + (1-t) * source
    
    def get_reference_flow(self, source: torch.Tensor, target: torch.Tensor, t: torch.Tensor):
        """
            returns the GT flow given a source and a target in the models domain
        """
        return target - source
    
    @property
    def inference_domains(self):
        return ["time", "frequency"]

    def tensor_to_domain(self, curves: torch.Tensor, curves_domain: str, target_domain: str):
        assert curves_domain in self.inference_domains and target_domain in self.inference_domains
        if curves_domain == target_domain:
            return curves
        if target_domain == "time":
            return inverse_fourier_transform(curves)
        elif target_domain == "frequency":
            return fourier_transform(curves)
        else:
            raise NotImplementedError(f"not implemented for domain: {target_domain}")

    def to_domain(self, samples: Samples, domain: Optional[str] = None):
        """
            this function accepts samples and moved from one domain to another
        """
        if domain is None:
            domain = self.domain
        assert domain in self.inference_domains
        assert samples.domain in self.inference_domains
        return Samples(
            self.tensor_to_domain(samples.curves, samples.domain, domain),
            domain,
            samples.t
        )
    
    def call_model_in_domain(self, t: torch.Tensor, x: torch.Tensor, domain:str = "time", condition=None):
        """
            this function will use the model on the correct domain, if 
        """
        assert domain in self.inference_domains
        assert self.domain is not None
        if domain == self.domain:
            return self.forward(x, t, condition)
        if domain == "frequency":
            return fourier_transform(self.forward(inverse_fourier_transform(x), t, condition))
        elif domain == "time":
            return inverse_fourier_transform(self.forward(fourier_transform(x), t, condition))
        else:
            raise NotImplementedError(f"not implemented for domain: {domain}")

    def sample(self, num_samples):
        samples = self.domain_sample(num_samples, domain=self.domain)
        return self.to_domain(samples, "time").curves

    def domain_sample(self, samples: Union[int, torch.Tensor], masks=None, references=None, condition=None, domain="time", num_steps=100, inverse=False, solver="euler"):
        """
            if mask and references are given, the flow matching will be done everywhere that mask = False
        """
        assert (masks is None and references is None) or (masks is not None and references is not None)
        if isinstance(samples, int):
            x_0 = self.to_domain(self.get_initial_states(samples), domain).curves
        else:
            x_0 = samples
        x_0_original = x_0.clone()
        t = torch.linspace(0, 1, num_steps, device=self.device) if not inverse else torch.linspace(1, 0, num_steps, device=self.device)
        def model_call(t,x):
            model_output = self.call_model_in_domain(t, x, domain, condition=condition)
            if masks is not None:
                model_output[~masks] = self.get_reference_flow(x_0_original, references, t)[~masks]
            return model_output

        with torch.no_grad():
            output_curves = odeint(model_call, x_0, t, method=solver)[-1]
        return Samples(
            output_curves, 
            domain=domain,
            t = 0 if inverse else 1
        )
    
    def auto_regressive_generation(self, num_samples, desired_length, autoregressive_step=None, domain="time", solver="euler"):
        current = self.domain_sample(num_samples, domain=domain, solver=solver).curves
        encoding = self.encode(current)
        current_length = current.shape[-1]
        if autoregressive_step is None:
            autoregressive_step = current.shape[-1] // 2
        curve_slices = [current]
        masks = torch.zeros_like(current, dtype=torch.bool)
        masks[..., :autoregressive_step] = 1
        while current_length < desired_length:
            new_slice = self.domain_sample(num_samples, domain="time", masks=~masks, references=current, condition=encoding).curves
            curve_slices.append(new_slice[:, -autoregressive_step:])
            current = new_slice
            current_length += autoregressive_step
        return Samples(
            torch.cat(curve_slices, dim=-1)[..., :desired_length], 
            domain=domain,
            t = 1
        )
    
    def encode(self, x):
        if (self.config.unet_model is not None) and (self.config.unet_model.include_other_domain):
            other_condition_curves = self.tensor_to_domain(x, "time", "frequency")
            x = torch.cat([x, other_condition_curves], dim=-2)
            return F.normalize(self.encoder(x), dim=-1)
        return F.normalize(self.encoder(x.reshape(x.shape[0], -1, x.shape[-1])), dim=-1)

    def get_t(self, batch):
        t = torch.rand(batch.shape[0], 1, 1, device=batch.device)
        return t

    def training_step(self, batch, batch_idx):
        loss_dict = {}

        un_conditional_targets = Samples(
            self._get_batch_trajectory_slices(batch),
            "time",
            t=1
        )
        un_conditional_targets_domain = self.to_domain(un_conditional_targets, self.domain).curves

        # conditional loss
        if self.has_conditional:
            condition_curves = Samples(
                self._get_batch_trajectory_slices(batch),
                "time",
                t=1
            )
            condition_curves_domain = self.to_domain(condition_curves, self.domain).curves
            condition_encoding = self.encode(condition_curves_domain)
            un_conditional_encoding = self.encode(un_conditional_targets_domain)

            conditional_targets = Samples(
                self._get_batch_trajectory_slices(batch),
                "time",
                t=1
            )
            conditional_targets_domain = self.to_domain(conditional_targets, self.domain).curves
            x_start = self.get_initial_states(batch.shape[0]).curves
            t = self.get_t(batch)
            x_t = self.interpolate(x_start, conditional_targets_domain, t)
            reference_flow = self.get_reference_flow(x_start, conditional_targets_domain, t)

            model_flow = self.forward(x_t, t, condition_encoding)
            conditioned_x_0_hat = x_t + (1-t) * model_flow
            conditioned_x_0_hat_encoding = self.encode(conditioned_x_0_hat)
            conditional_consistency_loss = (F.normalize(conditioned_x_0_hat_encoding, dim=-1) * F.normalize(condition_encoding, dim=-1)).sum(dim=-1).mean() * 0.5 
            loss_dict["conditional consistency loss"] = 1 - conditional_consistency_loss

            model_flow_unconditioned = self.forward(x_t, t)

            loss_dict["conditional unconditional flow diff"] = (model_flow - model_flow_unconditioned).norm(dim=-1).mean().detach()
            conditional_loss = torch.nn.functional.mse_loss(reference_flow, model_flow)
            loss_dict["conditional loss"] = conditional_loss

            # contrastive divergence loss (cosine rescaled to [0, 1])
            condition_encoding_normalized = torch.nn.functional.normalize(condition_encoding, dim=-1)
            un_conditional_encoding_normalized = torch.nn.functional.normalize(un_conditional_encoding, dim=-1)

            similarity = (condition_encoding_normalized[:, None]
                        * un_conditional_encoding_normalized[None]).sum(dim=-1)

            eye = torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
            
            positive_samples = similarity[eye]
            negative_samples = similarity[~eye]

            # cosine similarity ∈ [-1, 1] → rescale to [0, 1]
            positive_scores = (positive_samples + 1.0) * 0.5
            negative_scores = (negative_samples + 1.0) * 0.5

            loss_dict["divergence loss positive"] = 1.0 - positive_scores.mean()
            loss_dict["divergence loss negative"] = negative_scores.mean()
            if not self.config.loss_function.include_contrastive_loss:
                loss_dict["divergence loss positive"]  = loss_dict["divergence loss positive"].detach()
                loss_dict["divergence loss negative"]  = loss_dict["divergence loss negative"].detach()
                
            if not self.config.loss_function.include_consistency_loss:
                loss_dict["conditional consistency loss"] = loss_dict["conditional consistency loss"].detach()

        # unconditional loss
        x_start = self.get_initial_states(batch.shape[0]).curves
        t = self.get_t(batch)
        x_t = self.interpolate(x_start, un_conditional_targets_domain, t)
        reference_flow = self.get_reference_flow(x_start, un_conditional_targets_domain, t)

        model_flow = self.forward(x_t, t)
        unconditional_loss = torch.nn.functional.mse_loss(reference_flow, model_flow)
        loss_dict["unconditional loss"] = unconditional_loss

        self.log_dict(
            loss_dict,
            prog_bar=True
            )
        wandb.log(loss_dict)
        return sum(loss_dict.values())
    
    def log_prob_normal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Log-probability of x under N(0, I)
        """
        d = x.shape[-1]
        return -0.5 * (x ** 2).sum(dim=-1) - 0.5 * d * torch.log(torch.tensor(2 * torch.pi, device=x.device))

    def divergence_approx(self, f, x, t, domain):
        eps = torch.randn_like(x)
        with torch.enable_grad():
            x_ = x.clone().requires_grad_(True)
            fx = f(t, x_, domain=domain)
            grad = torch.autograd.grad((fx * eps).sum(), x_, create_graph=True)[0]
        return (grad * eps).sum(dim=1)

    def log_likelihood(self, x_T: torch.Tensor, domain: str = "frequency", num_steps: int = 100, solver: str = "euler"):
        """
        Compute log-likelihood of x_T in given domain by integrating the log-Jacobian determinant
        backward through the flow and adding the base Normal(0,I) log-probability.
        """
        assert domain in self.inference_domains

        # Convert x_T to model's domain
        x_T_in_model_domain = self.tensor_to_domain(x_T, domain, self.domain)

        t = torch.linspace(1, 0, num_steps, device=x_T.device)

        def odefunc(t, states):
            x, logp = states
            v = self.call_model_in_domain(t, x, domain=self.domain)
            div = self.divergence_approx(self.call_model_in_domain, x, t, domain=self.domain)
            dx_dt = -v
            dlogp_dt = div
            return dx_dt, dlogp_dt

        logp_T = torch.zeros(x_T.shape[0], device=x_T.device)
        x_0, logp_0 = odeint(odefunc, (x_T_in_model_domain, logp_T), t, method=solver)
        x_0, logp_0 = x_0[-1], logp_0[-1]

        # base log-prob in model domain (frequency)
        base_log_prob = self.log_prob_normal(x_0)
        return base_log_prob + logp_0
    
    def kl_divergence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence of each sample to a standard normal N(0, I)
        using mean and std over the feature dimension.
        x: (B, D)
        returns: (B,) KL divergence values
        """
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        kl = 0.5 * ((std**2 + mu**2 - 1 - 2 * torch.log(std)).sum(dim=-1))
        return kl

    def inverse_sample_kl_divergence(self, samples: torch.Tensor, domain: str = "time", num_steps: int = 100, solver: str = "euler"):
        """
        Compute the KL divergence of samples (in any domain) after inverting them through the flow
        back to the base distribution in self.domain.

        samples: (B, D)
        domain: current domain of the samples
        returns: (B,) KL divergence to N(0, I)
        """
        # move samples to model domain
        samples_in_model_domain = self.tensor_to_domain(samples, domain, self.domain)

        # use existing inversion via domain_sample
        inverted = self.domain_sample(samples_in_model_domain, domain=self.domain, num_steps=num_steps, inverse=True, solver=solver)

        x_0 = inverted.curves
        mu = x_0.mean(dim=-1, keepdim=True)
        std = x_0.std(dim=-1, keepdim=True)
        kl = 0.5 * ((std**2 + mu**2 - 1 - 2 * torch.log(std)).sum(dim=-1))
        return kl

    def calculate_likelihood(self, curves: torch.Tensor):
        slices = self._get_batch_trajectory_slices(curves)
        return self.inverse_sample_kl_divergence(slices, domain="time")


