from .drw import DRWManager
import torch
from .parameter_distribution import AR1UniformParameterDistribution, AR2CorrelatedParameterDistribution
from ..model.flow_matching_model.gaussian_prior_flow_matching import GaussianPriorFlowMatching
from ..model.flow_matching_model.samples import Samples
ar1_parameter_distribution = AR1UniformParameterDistribution()
ar2_parameter_distribution = AR2CorrelatedParameterDistribution()

class NoiseModel:
    def generates_curves_like(self, reference_curves, parameters, num_curves):
        """
            will generate for num_curves for each reference curve
        """
        raise NotImplementedError

    def generate_curves(self, num_curves):
        """
            will generate num_curves
        """
        raise NotImplementedError

class AR1NoiseModel(NoiseModel):
    def __init__(self,drw_manager: DRWManager, num_samples=256, oracle_mode=False):
        self.num_samples = num_samples
        self.drw_manager = drw_manager
        self.oracle_mode = oracle_mode

    @torch.no_grad()
    def generates_curves_like(self, reference_curves, parameters, num_curves):
        if self.oracle_mode:
            phi = parameters.reshape(reference_curves.shape[0], -1)
        else:
            phi = self.drw_manager.estimate_parameters(reference_curves, 1)  # [B]
        B = phi.shape[0]

        phi_rep = phi[:, None].repeat(1, num_curves, 1).flatten(0,1)
        curves = self.drw_manager.simulate_from_params(phi_rep, self.num_samples)
        return curves.reshape(B, num_curves, -1).cpu()

    @torch.no_grad()
    def generate_curves(self, num_curves):
        return self.drw_manager.simulate(ar1_parameter_distribution, num_curves, num_samples=self.num_samples)
    
class StupidAR1NoiseModel(NoiseModel):
    def __init__(self,drw_manager: DRWManager, num_samples=256):
        self.num_samples = num_samples
        self.drw_manager = drw_manager

    @torch.no_grad()
    def generates_curves_like(self, reference_curves, parameters, num_curves):
        return self.generate_curves(reference_curves.shape[0] * num_curves)[0].reshape(reference_curves.shape[0], num_curves, -1)

    @torch.no_grad()
    def generate_curves(self, num_curves):
        return self.drw_manager.simulate(ar1_parameter_distribution, num_curves, num_samples=self.num_samples)
    
class AR2NoiseModel(NoiseModel):
    def __init__(self, drw_manager: DRWManager, num_samples = 256, oracle_mode=False):
        self.num_samples = num_samples
        self.drw_manager = drw_manager
        self.oracle_mode = oracle_mode

    @torch.no_grad()
    def generates_curves_like(self, reference_curves, parameters, num_curves):
        if self.oracle_mode:
            phi = parameters
        else:
            phi = self.drw_manager.estimate_parameters(reference_curves, 2).clip(-0.999, 0.999)  # [B, 2]
        B = phi.shape[0]
        phi_rep = phi[:, None].repeat(1, num_curves, 1).flatten(0,1)
        curves = self.drw_manager.simulate_from_params(phi_rep, self.num_samples)
        return curves.reshape(B, num_curves, -1).cpu()

    @torch.no_grad()
    def generate_curves(self, num_curves):
        return self.drw_manager.simulate(ar2_parameter_distribution, num_curves, num_samples=self.num_samples)
    
class FlowModelNoiseModel(NoiseModel):
    def __init__(self, model: GaussianPriorFlowMatching, num_samples = 256):
        self.model = model
        self.num_sample = num_samples
    
    @torch.no_grad()
    def generates_curves_like(self, reference_curves, parameters, num_curves):
        reference_curves = reference_curves.to(self.model.device)
        reference_curves = self.model.to_domain(Samples(reference_curves, domain="time", t=1)).curves
        condition = self.model.encode(reference_curves)
        condition = condition[:,None].repeat(1,num_curves,1).flatten(0,1)
        conditional_curves = self.model.to_domain(self.model.domain_sample(condition.shape[0], condition=condition), "time").curves
        conditional_curves = conditional_curves.reshape(reference_curves.shape[0], num_curves, -1)
        return conditional_curves.cpu()


    @torch.no_grad()
    def generate_curves(self, num_curves):
        return self.model.to_domain(self.model.domain_sample(num_curves), "time").curves.cpu()