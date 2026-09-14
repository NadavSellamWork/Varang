from .abstract_data_manager import AbstractDataManager
import numpy as np 
import os
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from src.utils.drw import DRWManager

def plot_coeff_kde(coeffs, grid_size=250, cmap="viridis", title="KDE of coefficients"):
    """
    coeffs: numpy array [B, 2]
    returns: matplotlib.figure.Figure
    """
    coeffs = np.asarray(coeffs)
    phi1 = coeffs[:, 0]
    phi2 = coeffs[:, 1]

    kde = gaussian_kde(coeffs.T)

    xgrid = np.linspace(-1, 1, grid_size)
    ygrid = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(7,6))
    img = ax.imshow(
        Z,
        origin='lower',
        extent=[xgrid.min(), xgrid.max(), ygrid.min(), ygrid.max()],
        aspect='auto',
        cmap=cmap
    )

    fig.colorbar(img, ax=ax, label="density")
    ax.set_xlabel("phi1")
    ax.set_ylabel("phi2")
    ax.set_title(title)

    return fig


class DRWAR2DataManager(AbstractDataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_curves = np.load(os.path.join(self.config.dataset_folder, "curves.npy"))
        self.curves = torch.tensor(data_curves, dtype=torch.float32)
        self.DRWManager = DRWManager(0.2, 0.0)
        data_ar2_coeffs = self.DRWManager.estimate_parameters(self.curves[torch.randperm(self.curves.shape[0])[:2000]], order=2)
        data_fig = plot_coeff_kde(data_ar2_coeffs)
        self.data_figure = wandb.Image(data_fig)
        plt.close("all")

    def train_dataloader(self):
        return DataLoader(self.curves[:,None], batch_size=self.config.batch_size, shuffle=True, drop_last=True)
    
    def visualize_step(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()

        # === generate diagnostics ===
        samples = pl_module.sample(pl_module.config.rendering.num_sampled_curves).cpu().squeeze()
        model_ar2_coeffs = self.DRWManager.estimate_parameters(samples, order=2)
        model_fig = plot_coeff_kde(model_ar2_coeffs)
        trainer.logger.experiment.log({
            "model ar2 estimated, coeffs": [wandb.Image(model_fig)],
            "data ar2 estimated, coeffs": [self.data_figure],
        })

        plt.close("all")
        pl_module.train()