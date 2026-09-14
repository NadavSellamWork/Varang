from .abstract_data_manager import AbstractDataManager
import numpy as np 
import os
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from ..amplitude_drw_curve_utils.utils import red_noise_psd
import wandb
import matplotlib.pyplot as plt

def periodogram(curves: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One-sided PSD from squared Fourier amplitudes.
    """
    N = curves.shape[-1]
    z = curves - curves.mean(dim=1, keepdim=True)

    F = torch.fft.rfft(z, n=N)
    psd = (dt / N) * (F.abs() ** 2)

    if N % 2 == 0:
        psd[:, 1:-1] *= 2.0
    else:
        psd[:, 1:] *= 2.0

    freqs = torch.fft.rfftfreq(N, d=dt)
    return freqs, psd

class AutoregressiveDataManager(AbstractDataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_curves = np.load(os.path.join(self.config.dataset_folder, "curves.npy"))
        self.curves = torch.tensor(data_curves, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.curves[:,None], batch_size=self.config.batch_size, shuffle=True, drop_last=True)
    
    def visualize_step(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()

        model_light_curves = pl_module.auto_regressive_generation(pl_module.config.rendering.num_sampled_curves, self.curves.shape[-1]).curves.squeeze(1)
        fig = plt.figure()
        freq, model_psd = periodogram(model_light_curves.cpu(), 1)
        freq, dataset_psd = periodogram(self.curves, 1)
        plot_slice = slice(1, None)
        plt.loglog(freq[plot_slice], dataset_psd.mean(dim=0)[plot_slice], label="dataset PSD")

        target_psd = red_noise_psd(freq.numpy(), decay=0.95, sigma=0.08)
        plt.loglog(freq[plot_slice], target_psd[plot_slice], label="target PSD")

        plt.loglog(freq[plot_slice], model_psd.mean(dim=0)[plot_slice], label="model PSD")
        plt.ylim(1e-5, 1e1)
        plt.legend()

        # === generate diagnostics ===
        trainer.logger.experiment.log({
            "autoregressive periodegrams": [wandb.Image(fig)],
        })

        plt.close("all")
        pl_module.train()
