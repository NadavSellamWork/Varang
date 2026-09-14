# config.py
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal, List


class TrainerConfig(BaseModel):
    accelerator: str = "auto"
    devices: str = "auto"
    precision: str = "16"
    max_steps: int = 100000
    lr: float = 1e-3
    weight_decay: float = 0.05


class UNetModelConfig(BaseModel):
    # core
    sequence_size: int = 256
    harmonic_positional_encoding_power: int = 5
    t_encoder_num_layers: int = 1

    # UNet1D
    base_channels: int = 64
    channel_mults: List[int] = Field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2
    norm_groups: int = 8
    attn_resolutions: List[int] = Field(default_factory=list)
    attn_heads: int = 4
    final_tanh: bool = False

    # Conv hyperparams (typical for stride 2 up/down)
    down_kernel: int = 4
    down_stride: int = 2
    down_padding: int = 1
    up_kernel: int = 4
    up_stride: int = 2
    up_padding: int = 1

    # Positional encodings (input-domain)
    use_input_positional_encoding: bool = True
    xpos_encoder_num_layers: int = 1

    degrees_of_freedom: int = 10

    domain: Literal["frequency", "time"] = "frequency"
    include_other_domain: bool = False  # if True, NN inputs both domains, flow is in given domain

    # conditioning dims
    extra_cond_dim: int = 0
    global_cond_dim: int = 128

class DataConfig(BaseModel):
    dataset_folder: str = "datasets/drw_with_sin"
    data_class: str = "DRWSinDataManager"
    outlier_percentage: float = 0.0
    batch_size: int = 512
    data_channels: int = 1


class RenderingConfig(BaseModel):
    log_every_n_steps: int = 2000
    num_sampled_curves: int = 500


class WandbConfig(BaseModel):
    project: str = "example astrophysics training"
    name: str = "flow experiment"
    wandb_mode: Literal["online", "disabled"] = "online"


class LossFunction(BaseModel):
    include_contrastive_loss: bool = True
    include_consistency_loss: bool = True


class Config(BaseModel):
    trainer: TrainerConfig
    data: DataConfig
    rendering: RenderingConfig
    wandb: WandbConfig
    loss_function: LossFunction
    lightning_wrapper: str = "FlowMatchingModelWrapper"

    unet_model: UNetModelConfig
