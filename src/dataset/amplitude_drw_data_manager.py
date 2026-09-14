from .abstract_data_manager import AbstractDataManager
import numpy as np 
import os
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
from ..amplitude_drw_curve_utils.render_utils import render_amp_distribution_diagnostics ,render_avg_psd_vs_target
import wandb
import matplotlib.pyplot as plt

class AmplitudeDRWDataManager(AbstractDataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_curves = np.load(os.path.join(self.config.dataset_folder, "curves.npy"))
        self.curves = torch.tensor(data_curves, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.curves[:,None], batch_size=self.config.batch_size, shuffle=True, drop_last=True)
    
    def visualize_step(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()

        # === generate diagnostics ===
        samples = pl_module.sample(pl_module.config.rendering.num_sampled_curves).squeeze()
        psd_fig, _ = render_avg_psd_vs_target(samples.cpu().numpy())
        dist_fig, _ = render_amp_distribution_diagnostics(samples.cpu().numpy())

        # === log to W&B ===
        trainer.logger.experiment.log({
            "model psd": [wandb.Image(psd_fig)],
            "model distributions": [wandb.Image(dist_fig)],
        })

        plt.close("all")
        pl_module.train()