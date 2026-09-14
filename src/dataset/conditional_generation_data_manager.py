from .abstract_data_manager import AbstractDataManager
from ..model.flow_matching_model.samples import Samples
import numpy as np 
import os
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from src.utils.drw import DRWManager

class ConditionalGenerationDataManager(AbstractDataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_curves = np.load(os.path.join(self.config.dataset_folder, "curves.npy"))
        self.curves = torch.tensor(data_curves, dtype=torch.float32).reshape(data_curves.shape[0], 1, -1)
        self.DRWManager = DRWManager()
        self.estimated_coeffs = self.DRWManager.estimate_parameters(self.curves.squeeze(1), order=1).squeeze(-1)

    def train_dataloader(self):
        return DataLoader(self.curves, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
    
    def visualize_step(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()
        visualzied_coeffs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        eps = 0.01
        num_curves = pl_module.config.rendering.num_sampled_curves
        visualization_dict = {
            coeff: pl_module.to_domain(Samples(self.curves[((self.estimated_coeffs > coeff - eps) & (self.estimated_coeffs < coeff + eps))][:num_curves, :, 256:256+pl_module.config.unet_model.sequence_size].to(pl_module.device), "time", t=1)).curves
            for coeff in visualzied_coeffs
        }
        condition_dict = {
            coeff: pl_module.encode(curves)
            for coeff, curves in visualization_dict.items()
        }
        conditional_curves = {
            coeff: pl_module.to_domain(pl_module.domain_sample(condition.shape[0], condition=condition, num_steps=50))
            for coeff, condition in condition_dict.items()
        }
        conditional_curves_embedding = {
            coeff: pl_module.encode(curves.curves)
            for coeff, curves in conditional_curves.items()
        }
        fig_1 = plt.figure()
        for coeff in conditional_curves_embedding.keys():
            plt.scatter([coeff], [((torch.nn.functional.normalize(conditional_curves_embedding[coeff], dim=-1) * torch.nn.functional.normalize(condition_dict[coeff], dim=-1)).sum(dim=-1)).mean().item()])

        model_estimated_coeffs_dicts = {
            coeff: self.DRWManager.estimate_parameters(pl_module.to_domain(curves, "time").curves.cpu().squeeze(1), order=1) for 
            coeff, curves in conditional_curves.items()
        }
        fig_2 = plt.figure()
        for coeff, model_estimated_coeffs in model_estimated_coeffs_dicts.items():
            model_estimated_coeffs = model_estimated_coeffs.cpu()
            x = torch.zeros_like(model_estimated_coeffs) + coeff
            y = model_estimated_coeffs
            plt.scatter(x,y, c="black", alpha=0.005)
            plt.scatter([coeff],[model_estimated_coeffs.mean()], alpha=0.5)

        plt.xlabel("conditioned coeff")
        plt.ylabel("estimated mode curves coeff")
        x = list(model_estimated_coeffs_dicts.keys())
        plt.plot(x, x, c="red", alpha=0.1)
        trainer.logger.experiment.log({
            "model per coeff embedding alignment": [wandb.Image(fig_1)],
            "model coefficients": [wandb.Image(fig_2)],
        })
        plt.close("all")
        pl_module.train()