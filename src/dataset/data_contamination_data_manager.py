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
from src.utils.cyclic_detector import CyclicDetector
from src.utils.light_curve_sin_dataset import LightCurvesSinDataset
from src.utils.noise_models import AR1NoiseModel, StupidAR1NoiseModel, FlowModelNoiseModel
from src.utils.rendering import plot_rocs_from_items, plot_pvalue_histograms_from_items

class DataContaminationDataManager(AbstractDataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_curves = np.load(os.path.join(self.config.dataset_folder, "curves.npy"))
        outlier_curves = np.load(os.path.join(self.config.dataset_folder, "outliers.npy"))
        self.data = torch.tensor(data_curves, dtype=torch.float32)
        self.outlier_curves = torch.tensor(outlier_curves, dtype=torch.float32)
        self.curves = torch.cat([self.data, self.outlier_curves[:int(self.data.shape[0] * self.config.outlier_percentage)]], dim=0)
        self.curves = self.curves.reshape(self.curves.shape[0], 1, -1)
        self.DRWManager = DRWManager(
            sigma=0.2,
            observation_noise = 0.0
        )
        self.ar1_noise_model = AR1NoiseModel(self.DRWManager)
        self.stupid_ar1_noise_model = StupidAR1NoiseModel(self.DRWManager)
        self.detection_dataset = LightCurvesSinDataset(self.ar1_noise_model,
            snr_rms_range=(0.5, 0.8),
            cycle_length_range=(30, 100),
            dataset_size=500
            )
        self.estimated_coeffs = self.DRWManager.estimate_parameters(self.curves.squeeze( ), order=1)

    def train_dataloader(self):
        return DataLoader(self.curves, batch_size=self.config.batch_size, shuffle=True, drop_last=True)
    
    def visualize_step(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()
        flow_model_noise_model = FlowModelNoiseModel(pl_module, pl_module.config.unet_model.sequence_size)

        ar1_detector = CyclicDetector(self.detection_dataset, self.ar1_noise_model,  num_simulated_light_curves=50)
        ar1_stupid_detector = CyclicDetector(self.detection_dataset, self.stupid_ar1_noise_model, num_simulated_light_curves=50)
        flow_model_detector = CyclicDetector(self.detection_dataset, flow_model_noise_model, num_simulated_light_curves=50)
        flow_model_p_values, flow_model_labels = flow_model_detector.get_p_value_for_dataset(batch_size=10)
        ar1_p_values, ar1_labels = ar1_detector.get_p_value_for_dataset()
        ar1_stupid_p_values, ar1_stupid_labels = ar1_stupid_detector.get_p_value_for_dataset()

        roc_fig, _ = plot_rocs_from_items([("ar1", ar1_p_values, ar1_labels), ("ar1_stupid", ar1_stupid_p_values, ar1_stupid_labels), ("flow model", flow_model_p_values, flow_model_labels)])
        p_value_distribution_fig, _ = plot_pvalue_histograms_from_items([("ar1", ar1_p_values, ar1_labels), ("ar1_stupid", ar1_stupid_p_values, ar1_stupid_labels), ("flow model", flow_model_p_values, flow_model_labels)])

        trainer.logger.experiment.log({
            "roc curves": [wandb.Image(roc_fig)],
            "P-value distribution": [wandb.Image(p_value_distribution_fig)],
        })

        pl_module.train()