import torch
from .parameter_distribution import ParametersDistribution

import torch

def interp1d_lastdim(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    x:   [B, N]
    idx: [B, M] continuous indices in [0, N-1] (or [0, N] with clamping)
    out: [B, M] linear interpolation along last dim
    """
    if x.ndim != 2 or idx.ndim != 2:
        raise ValueError(f"Expected x [B,N] and idx [B,M], got {x.shape=} {idx.shape=}")
    if x.shape[0] != idx.shape[0]:
        raise ValueError(f"Batch mismatch: {x.shape[0]=} vs {idx.shape[0]=}")

    B, N = x.shape
    M = idx.shape[1]

    # clamp to valid range for linear interp
    idx = idx.to(dtype=torch.float32, device=x.device)
    idx = idx.clamp(0.0, float(N - 1))

    i0 = torch.floor(idx).to(torch.long)                      # [B,M]
    i1 = (i0 + 1).clamp(max=N - 1)                            # [B,M]
    w1 = (idx - i0.to(idx.dtype))                             # [B,M]
    w0 = 1.0 - w1

    x0 = x.gather(1, i0)                                      # [B,M]
    x1 = x.gather(1, i1)                                      # [B,M]

    return w0 * x0 + w1 * x1


class DRWManager:
    def __init__(self, sigma=0.1, observation_noise = 0.01):
        """
        Docstring for __init__
        
        :param sigma: the random walk coeff added at every time step
        :param observation_noise: the overall observational noise added after the fact
        :param coeffs_max_sum: the max sum of the abs of all the coeffs
        """
        self.sigma = sigma
        self.observation_noise = observation_noise

    def simulate_from_params(self, parameters, num_samples=256, burn_in=100):
        if len(parameters.shape) == 1:
            parameters = parameters[:,None]
        assert len(parameters.shape) == 2
        order = parameters.shape[1]
        dw = torch.randn(parameters.shape[0], num_samples + burn_in)
        curves = torch.zeros_like(dw)
        curves[:,0] = dw[:,0] * self.sigma
        for i in range(1, dw.shape[1]):
            current_values = curves[:,i]
            for offset in range(order):
                if i - offset - 1 < 0:
                    break
                current_values = current_values + parameters[:, offset] * curves[:, i - offset - 1]
            current_values += dw[:,i] * self.sigma
            curves[:,i] = current_values
        curves = curves + torch.randn_like(curves) * self.observation_noise
        return curves[:,burn_in:]

    def simulate(self, parameter_distribution: ParametersDistribution, num_curves, num_samples=256, burn_in=100):
        parameters = parameter_distribution.sample(num_curves)
        return self.simulate_from_params(parameters, num_samples=num_samples, burn_in=burn_in), parameters

    def estimate_parameters(self, curves, order: int, eps=1e-8):
        x = curves - curves.mean(dim=-1, keepdim=True)                      # [B,N]
        p = int(order)
        gammas = torch.stack([(x[:, k:] * x[:, :-k]).mean(dim=-1) if k else (x * x).mean(dim=1)
                            for k in range(p + 1)], dim=-1)               # [B,p+1]

        idx = torch.arange(p, device=x.device)
        Gamma = gammas[:, (idx[:, None] - idx[None, :]).abs()]             # [B,p,p]
        g = gammas[:, 1:p+1]                                               # [B,p]

        Gamma = Gamma + eps * torch.eye(p, device=x.device, dtype=x.dtype)[None]
        phi = torch.linalg.solve(Gamma, g.unsqueeze(-1)).squeeze(-1)        # [B,p]
        # sigma2 = (gammas[:, 0] - (phi * g).sum(dim=1)).clamp_min(0.0)      # [B]

        return phi

    def subsample_curves(self, curves, dt_range=(3,7), gap_range=(30, 150), gap_every_range=(100, 200)):
        """
            this function will subsample the curves to be non uniform sampling, it will return the subsampled curves and the new sampling times
        """
        max_time = (curves.shape[1] - 1)
        new_sampling_times = []
        current_sampling_times = dt_range[0] + torch.rand(curves.shape[0]) * (dt_range[1] - dt_range[0])
        last_gap_time = torch.zeros_like(current_sampling_times)
        gap_times = gap_every_range[0] + torch.rand(curves.shape[0]) * (gap_every_range[1] - gap_every_range[0])
        while True:
            new_sampling_times.append(current_sampling_times.clone())
            current_sampling_times += dt_range[0] + torch.rand(curves.shape[0]) * (dt_range[1] - dt_range[0])
            gap_mask = current_sampling_times >= last_gap_time + gap_times
            masked_gap = current_sampling_times[gap_mask]
            current_sampling_times[gap_mask] += gap_range[0] + torch.rand_like(masked_gap) * (gap_range[1] - gap_range[0])
            last_gap_time[gap_mask] = current_sampling_times[gap_mask]
            if current_sampling_times.max() >= max_time:
                break
        new_sampling_times = torch.stack(new_sampling_times,dim=-1)
        interpolated_values = interp1d_lastdim(curves, new_sampling_times)
        return interpolated_values, new_sampling_times
