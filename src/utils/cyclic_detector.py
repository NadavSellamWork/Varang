import torch
from torch.utils.data import DataLoader
from .noise_models import NoiseModel
from .light_curve_sin_dataset import LightCurvesSinDataset
from tqdm import tqdm


def periodogram_max_stat(x: torch.Tensor, fmin_bin: int = 1) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"x must be [N] or [B, N], got shape {tuple(x.shape)}")

    x = x.to(dtype=torch.float64)
    x = x - x.mean(dim=1, keepdim=True)

    N = x.shape[1]
    # Hann window, broadcast over batch
    w = torch.hann_window(N, periodic=True, dtype=x.dtype, device=x.device).unsqueeze(0)

    X = torch.fft.rfft(x * w, dim=1)          # [B, N//2+1]
    per = X.abs().square()                    # [B, N//2+1]
    per = per[:, fmin_bin:]                   # drop low bins
    return per.max(dim=1).values              # [B]

class CyclicDetector:
    def __init__(self, dataset: LightCurvesSinDataset, tested_noise_model: NoiseModel, num_simulated_light_curves=300, fmin_bin=1):
        self.dataset = dataset
        self.tested_noise_model = tested_noise_model
        self.num_simulated_light_curves = int(num_simulated_light_curves)
        self.fmin_bin = int(fmin_bin)

    def get_curves_p_values(self, curves, parameters):
        obs_stat = periodogram_max_stat(curves)
        simulated_curves = self.tested_noise_model.generates_curves_like(curves, parameters, self.num_simulated_light_curves)
        sim_stats = periodogram_max_stat(simulated_curves.flatten(0,1)).reshape(list(simulated_curves.shape)[:-1])
        p_values = (1.0 + (sim_stats >= obs_stat[:, None]).sum(dim=-1)) / (self.num_simulated_light_curves + 1.0)
        return p_values

    def get_p_value_for_dataset(self, batch_size=100):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, drop_last=False, shuffle=False)
        overall_p_values, overall_labels = [], []
        for curves, parameters, labels in tqdm(dataloader):
            p_values = self.get_curves_p_values(curves, parameters)
            overall_p_values.append(p_values)
            overall_labels.append(labels)
        return torch.cat(overall_p_values, dim=0), torch.cat(overall_labels, dim=0)
    