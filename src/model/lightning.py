from __future__ import annotations

import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from ..config import Config
from .nn import UNet1D, Encoder1D

def create_model_and_encoder(config):
    model = UNet1D(config)
    model_config = config.unet_model
    encoder = Encoder1D(config) if model_config.extra_cond_dim else None
    return model, encoder, model_config

    
class AbstractLightningWrapper(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.model, self.encoder, self.model_config = create_model_and_encoder(config)
        self.condition_vector_dim = self.model_config.extra_cond_dim
        self.has_conditional = self.condition_vector_dim > 0
        self.config = config
        self.data_channels = config.data.data_channels

    def _get_batch_trajectory_slices(self, batch):
        if batch.shape[-1] == self.model_config.sequence_size:
            return batch
        slice_index = torch.randint(0, batch.shape[-1] - self.model_config.sequence_size, (batch.shape[0], 1), device=batch.device)
        slice_index = slice_index + torch.arange(self.model_config.sequence_size, device=batch.device)[None]
        slice_index = slice_index[:,None].repeat(1,batch.shape[1],1)
        slices = batch.gather(-1, slice_index)
        return slices

    def training_step(self, batch, batch_idx):
        curve_slices = self._get_batch_trajectory_slices(batch)
        random_samples = torch.randn_like(curve_slices)
        t = torch.rand(batch.shape[0], 1, 1, device=batch.device)
        raise NotImplementedError("this part needs to be implemented for the diffusion or flow or whatever")
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.trainer.lr, weight_decay=self.config.trainer.weight_decay)

    def sample(self, num_samples):
        raise NotImplementedError()
    
    def calculate_likelihood(self, samples: torch.Tensor):
        return None

class SampleCallback(pl.Callback):
    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % pl_module.config.rendering.log_every_n_steps != 0 or trainer.global_step == 0:
            return
        trainer.datamodule.visualize_step(trainer, pl_module)

def get_trainer(config: Config):

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"outputs/{config.wandb.name}",       # folder to save in
        filename="step={step:06d}",          # naming scheme
        every_n_train_steps=config.rendering.log_every_n_steps,  # how often to save
        save_top_k=-1,                       # keep all checkpoints
        save_on_train_epoch_end=False        # don’t save at epoch end
    )

    callbacks = [
        SampleCallback(),
        checkpoint_callback
    ]

    logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name,
        config=config.__dict__,
        mode=config.wandb.wandb_mode
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=16,
        log_every_n_steps=config.rendering.log_every_n_steps,
        max_steps=config.trainer.max_steps,
        max_epochs=-1,
        callbacks = callbacks
    )
    return trainer