# Varang: Learning Quasar Variability with Generative Models

This repository contains the code accompanying the paper *"Moving Beyond the DRW Model: Learning Quasar Variability with Generative Models"* (Nadav Bojan Sellam,Kevin Park,Alex Bronstein,Zoltan Haiman). The paper studies how misspecifying the noise model used as a null hypothesis in periodicity searches (for example, assuming quasar light curves follow a Damped Random Walk, or DRW) can bias the detection of periodic signals such as supermassive black hole binary candidates. Instead of assuming a fixed parametric noise model, we train a conditional flow matching generative model, Varang, that learns the statistical properties of stochastic light curve variability directly from data, and use it to build data driven null hypotheses for periodicity detection.

This document explains how to install the code, how to reproduce every experiment and figure in the paper, and how to use the codebase as a general toolkit for new datasets and new experiments of your own.

## Table of contents

1. Installation
2. Repository layout
3. Core concepts (light curves, domains, the flow matching model)
4. Reproducing the paper
5. Using the codebase for your own experiments
6. Working with a trained model
7. Configuration reference
8. Known limitations

## 1. Installation

The code was developed with Python 3.10+ and PyTorch, using PyTorch Lightning for training and Weights and Biases (wandb) for logging.

```bash
git clone https://github.com/NadavSellamWork/astro-diffusion.git
cd astro-diffusion
git checkout release

python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` lists the packages this code imports (PyTorch, `torchdiffeq` for solving the flow ODE, `pytorch-lightning`, `wandb`, `pydantic` for config validation, `numpy`, `scipy`, `matplotlib`, `tqdm`, and `jupyter` for the notebooks). Install a CUDA enabled build of PyTorch separately if you plan to train on a GPU, following the instructions at pytorch.org for your CUDA version, then install the rest of `requirements.txt`.

Weights and Biases logging is on by default (`wandb.wandb_mode: online` in the configs). If you do not want to log to a wandb account, either run `wandb offline` once in your shell, or set `wandb_mode: disabled` in the config you use.

## 2. Repository layout

```
configurations/    YAML experiment configs consumed by main.py
notebooks/          notebooks that build the synthetic datasets used for training
figures/             notebooks and images that reproduce the paper's figures
src/
  config.py          pydantic schema for the YAML configs
  dataset/            LightningDataModules, one per experiment family
  model/
    nn.py              the 1D UNet generator and the encoder network
    lightning.py        shared LightningModule base class and the training loop driver
    flow_matching_model/ the conditional flow matching model itself
  utils/
    drw.py              DRW / autoregressive light curve simulation and coefficient estimation
    noise_models.py      wrappers that turn a DRW, AR(1), AR(2), or trained flow model into a
                          "noise model" usable for periodicity detection
    light_curve_sin_dataset.py  injects synthetic periodic (sinusoidal) signals into light curves
    cyclic_detector.py   the periodicity detection statistic and p value computation
    parameter_distribution.py  distributions used to sample AR coefficients for a dataset
    rendering.py         ROC curve and p value histogram plotting
  amplitude_drw_curve_utils/  frequency domain DRW light curve generation used for the
                               amplitude distribution (Rayleigh vs Uniform) experiment
main.py              training entry point, reads a config and launches a PyTorch Lightning run
sweep.sh             runs every config in a folder sequentially, one after another
```

## 3. Core concepts

A few pieces of vocabulary are used throughout the code and this document.

**Light curve.** A 1D tensor of brightness values sampled at uniform time steps. All experiments in the paper use light curves of a fixed length (`unet_model.sequence_size`, 256 samples by default).

**Domain.** The model can operate in the time domain or the frequency domain (`unet_model.domain` in the config). In the frequency domain, `src/utils/fft.py` represents the real Fourier transform of a real signal as a real valued tensor of the same length (concatenating the real and imaginary parts), so the same 1D UNet architecture applies in either domain. Setting `include_other_domain: true` additionally feeds the other domain's representation into the network as extra input channels.

**Flow matching.** Instead of an explicit noise model, Varang defines a continuous normalizing flow that transports samples from a standard Gaussian to samples that look like real light curves, trained with the flow matching objective of Lipman et al. (2023). `src/model/flow_matching_model/gaussian_prior_flow_matching.py` implements this: the interpolation between noise and data, the loss, and the sampler (`torchdiffeq.odeint`, Euler by default).

**Conditioning, contrastive loss, and consistency loss.** For experiments that need conditional generation (produce a new light curve statistically consistent with an observed one), the model also trains an encoder that maps a light curve to a latent vector `z`. A contrastive loss pulls together embeddings of two segments of the same underlying process and pushes apart embeddings from different light curves, and a consistency loss ties the embedding of a freshly generated sample back to the conditioning embedding. Both are toggled independently through `loss_function.include_contrastive_loss` and `loss_function.include_consistency_loss` in the config, which is exactly how the paper's ablation in Fig. 6 was produced.

**Noise models and the cyclic detector.** For periodicity detection, `src/utils/noise_models.py` wraps a DRW/AR(1)/AR(2) parametric model or a trained flow model behind a common `NoiseModel` interface with two methods: `generate_curves(num_curves)` to draw unconditional realizations, and `generates_curves_like(reference_curves, parameters, num_curves)` to draw realizations conditioned on an observed curve. `src/utils/cyclic_detector.py`'s `CyclicDetector` then computes, for each observed light curve, the detection statistic `T(x) = max_k |x_hat_k|` (the largest Fourier amplitude) and a p value by comparing it against simulated null realizations from a chosen noise model, exactly as in section 4.4 of the paper.

## 4. Reproducing the paper

Every experiment in the paper follows the same three step pipeline:

1. **Build a dataset** of synthetic light curves (a notebook under `notebooks/datasets/`, saving a `curves.npy` file, and sometimes `outliers.npy` / `coeffs.npy`, under `datasets/<name>/`).
2. **Train** a model on that dataset with `python main.py --config=configurations/<config>.yaml`. Checkpoints, the resolved config, and W&B logs are written to `outputs/<wandb.name>/`.
3. **Render figures** from a trained checkpoint with the corresponding notebook under `figures/paper/`.

Below is the mapping from each experiment in the paper to the commands that reproduce it.

### Experiment 1: ensemble level noise structure beyond the power spectrum (Fig. 2 to 4)

Builds two datasets (true DRW with Rayleigh distributed Fourier amplitudes, and an ablation with the same mean/variance but Uniform amplitudes) and checks whether the model reproduces the full amplitude distribution, not just the power spectral density.

```bash
jupyter nbconvert --to notebook --execute notebooks/datasets/amplitude_distribution_drw.ipynb
python main.py --config=configurations/gaussian_prior_rayleigh_drw.yaml
python main.py --config=configurations/gaussian_prior_uniform_drw.yaml
```

Figures are produced from the trained checkpoints with the notebooks under `figures/model_uniform_gaussian_amplitudes/` and `figures/psd_fit/`.

### Experiment 2: recovering latent AR(2) structure without being told the parametric form (Fig. 5)

```bash
jupyter nbconvert --to notebook --execute notebooks/datasets/ar_process.ipynb   # builds datasets/ar2
python main.py --config=configurations/gaussian_prior_drw_ar2.yaml
```

`figures/model_ar2_fit/model_ar2_fit.ipynb` loads the resulting checkpoint, generates new light curves, fits AR(2) coefficients to them with the Yule Walker equations (`DRWManager.estimate_parameters`), and compares the recovered coefficient distribution to the ground truth one.

### Experiment 3: conditional generation and the effect of the contrastive/consistency losses (Fig. 6)

```bash
python main.py --config=configurations/gaussian_prior_conditional_generation.yaml
python main.py --config=configurations/gaussian_prior_conditional_generation_no_contrastive.yaml
python main.py --config=configurations/gaussian_prior_conditional_generation_no_consistency.yaml
```

These three configs share the `datasets/ar1` dataset (built by the "datasets" cell of `notebooks/datasets/ar_process.ipynb`) and differ only in `loss_function.include_contrastive_loss` / `include_consistency_loss`. `figures/conditional_generation/conditional_generation.ipynb` loads each checkpoint, conditions on light curves with known AR(1) coefficients, generates new realizations, and compares the recovered coefficient to the true one.

### Experiment 4: learned null hypotheses for periodic signal detection (Fig. 7 to 10)

This experiment does not need its own training config beyond the ones above; it reuses the conditional generation checkpoint (`gaussian prior - conditional generation`) as a data driven noise model and compares it against parametric AR(1)/AR(2) null hypotheses, using `CyclicDetector`. See `notebooks/datasets/ar_process.ipynb` for the AR(1) vs AR(2) comparison, and section 6 below for the code pattern to run this comparison against your own model or dataset.

### Experiment 5: training set contamination (Fig. 11 and 12)

```bash
python main.py --config=configurations/gaussian_prior_data_contamination_0_percent.yaml
python main.py --config=configurations/gaussian_prior_data_contamination_05_percent.yaml
python main.py --config=configurations/gaussian_prior_data_contamination_1_percent.yaml
python main.py --config=configurations/gaussian_prior_data_contamination_5_percent.yaml
python main.py --config=configurations/gaussian_prior_data_contamination_10_percent.yaml
python main.py --config=configurations/gaussian_prior_data_contamination_20_percent.yaml
```

(Or simply `./sweep.sh configurations` to run every config in the folder one after another; skip that if you only want a subset, since a full sweep trains every model in this README, which can take a long time.) Each config points at the same `datasets/ar1` curves but a different `outlier_percentage`, mixing in an increasing fraction of light curves from `datasets/ar1/outliers.npy` (light curves with an injected periodic signal). `figures/paper/data_contamination/data_contamination.ipynb` loads every checkpoint in `CONTAMINATION_RUNS`, runs `CyclicDetector` for each, and plots the ROC curves and p value histograms shown in the paper (Fig. 11, 12).

Note that the notebook in `figures/paper/data_contamination/` references checkpoint filenames such as `step=step=050000.ckpt`; the exact step number in your run depends on `trainer.max_steps` and `rendering.log_every_n_steps`, so adjust the checkpoint name in the notebook to match whichever checkpoint you want to load from your own `outputs/<name>/` folder.

## 5. Using the codebase for your own experiments

The codebase is organized so that adding a new dataset or a new experiment usually means writing one new class plus one new YAML file, not touching the training loop.

### 5.1 Generating your own synthetic dataset

Any dataset is just a NumPy array of shape `(num_curves, sequence_length)` saved as `curves.npy` inside a folder under `datasets/`. The simplest way to build one is with `DRWManager` in `src/utils/drw.py`, which simulates AR(p) processes (a DRW is the p=1 case):

```python
import numpy as np
import torch
from src.utils.drw import DRWManager
from src.utils.parameter_distribution import AR1UniformParameterDistribution

manager = DRWManager(sigma=0.2, observation_noise=0.0)
coeff_distribution = AR1UniformParameterDistribution(min_coeff=0.2, max_coeff=0.9)

curves, coeffs = manager.simulate(
    coeff_distribution,
    num_curves=100_000,
    num_samples=256,   # length of each light curve
    burn_in=100,        # discarded warmup samples so the process reaches stationarity
)

import os
os.makedirs("datasets/my_dataset", exist_ok=True)
np.save("datasets/my_dataset/curves.npy", curves.numpy())
np.save("datasets/my_dataset/coeffs.npy", coeffs.numpy())
```

For a dataset built directly in the frequency domain with a chosen power spectrum and amplitude distribution (as used for the DRW/Uniform amplitude ablation), use `generate_curve` in `src/amplitude_drw_curve_utils/utils.py`:

```python
from src.amplitude_drw_curve_utils.utils import generate_curve, rayleigh_distribution
import numpy as np
from tqdm import tqdm

curves = np.stack([
    generate_curve(256, distribution=rayleigh_distribution, decay=0.95, sigma_shape=0.08)
    for _ in tqdm(range(100_000))
])
np.save("datasets/my_frequency_dataset/curves.npy", curves)
```

To generate your own noise process entirely, write any function that returns a `(num_curves, sequence_length)` array and save it the same way; the rest of the pipeline only ever reads `curves.npy` (and, for the conditional/contamination experiments, `outliers.npy`), so the source of the curves does not matter.

### 5.2 Using real, observed light curves instead of synthetic ones

Nothing in the training pipeline requires the data to be synthetic. If you have real observed quasar light curves resampled onto a uniform time grid (all experiments in this release assume uniform sampling, see section 8), save them as a `curves.npy` array of shape `(num_curves, sequence_length)` the same way and point a config's `data.dataset_folder` at that folder. The one dataset manager that needs no extra metadata is `AmplitudeDRWDataManager`, so it is the simplest starting point for a new, real dataset; write your own thin subclass of `AbstractDataManager` (see 5.3) if you want a different diagnostic plot during training.

### 5.3 Writing a new dataset manager

Every experiment is driven by a subclass of `AbstractDataManager` (`src/dataset/abstract_data_manager.py`), which is a `pytorch_lightning.LightningDataModule` with two extra responsibilities: it loads `curves.npy` into `self.curves`, and it implements `visualize_step`, which is called periodically during training to log diagnostic plots to wandb. The simplest possible manager is:

```python
# src/dataset/my_experiment_data_manager.py
from .abstract_data_manager import AbstractDataManager
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

class MyExperimentDataManager(AbstractDataManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        curves = np.load(os.path.join(self.config.dataset_folder, "curves.npy"))
        self.curves = torch.tensor(curves, dtype=torch.float32)

    def train_dataloader(self):
        return DataLoader(self.curves[:, None], batch_size=self.config.batch_size, shuffle=True, drop_last=True)

    def visualize_step(self, trainer, pl_module):
        pl_module.eval()
        samples = pl_module.sample(pl_module.config.rendering.num_sampled_curves)
        # build whatever diagnostic plot you want here and log it with
        # trainer.logger.experiment.log({"my plot": [wandb.Image(fig)]})
        pl_module.train()
```

Register it in `src/dataset/__init__.py`:

```python
from .my_experiment_data_manager import MyExperimentDataManager
```

and reference it by class name from a config (`data.data_class: "MyExperimentDataManager"`); `main.py` looks it up with `getattr(datasets, config.data.data_class)`, so no other code needs to change.

### 5.4 Writing a new config

Copy an existing config close to what you need (`configurations/gaussian_prior_rayleigh_drw.yaml` is a reasonable unconditional starting point, `configurations/gaussian_prior_conditional_generation.yaml` a conditional one) and adjust the fields that matter for your run:

```yaml
trainer:
  accelerator: auto
  devices: auto
  precision: "16"
  max_steps: 100000
  lr: 1e-4
  weight_decay: 0.05

unet_model:
  sequence_size: 256           # must match the length of your curves.npy rows
  domain: frequency            # or "time"
  include_other_domain: true   # also feed the other domain in as extra channels
  extra_cond_dim: 0            # set > 0 (e.g. 128) to enable conditional generation

data:
  dataset_folder: "datasets/my_dataset"
  data_class: "MyExperimentDataManager"
  batch_size: 512

rendering:
  log_every_n_steps: 1000
  num_sampled_curves: 1000

wandb:
  project: "my project"
  name: "my run"
  wandb_mode: online

loss_function:
  include_contrastive_loss: true
  include_consistency_loss: true

lightning_wrapper: "GaussianPriorFlowMatching"
```

Then train with:

```bash
python main.py --config=configurations/my_config.yaml
```

`main.py` copies the resolved config into `outputs/<wandb.name>/config.yaml` at the start of the run, which is what the figure notebooks later use to reload the exact configuration used for training (see section 6.1). Setting `extra_cond_dim` to 0 disables the conditional encoder entirely and trains a purely unconditional generator; set it to a positive integer (128 in the paper's conditional configs) to train the encoder plus the contrastive and consistency losses alongside the generator.

### 5.5 Running many configs

`sweep.sh` runs every YAML file in a folder, one after another:

```bash
./sweep.sh configurations
# or point it at a folder containing only the configs you want to run:
mkdir -p my_sweep && cp configurations/gaussian_prior_rayleigh_drw.yaml my_sweep/
./sweep.sh my_sweep
```

## 6. Working with a trained model

### 6.1 Loading a checkpoint

```python
import yaml, torch
from pathlib import Path
from src.config import Config
import src.model.flow_matching_model as flow_matching

def load_model(output_name, checkpoint_name, device="cuda"):
    output_dir = Path("outputs") / output_name
    config = Config(**yaml.safe_load(open(output_dir / "config.yaml")))
    wrapper_cls = getattr(flow_matching, config.lightning_wrapper)
    model = wrapper_cls.load_from_checkpoint(output_dir / checkpoint_name, config=config, map_location=device)
    return model.eval().to(device)

model = load_model("gaussian prior - rayleigh drw", "step=step=100000.ckpt")
```

### 6.2 Unconditional sampling

```python
with torch.no_grad():
    curves = model.sample(num_samples=64)   # (64, data_channels, sequence_size) time domain tensor
```

### 6.3 Conditional sampling (given an observed light curve)

Requires a checkpoint trained with `extra_cond_dim > 0` (for example `gaussian_prior_conditional_generation.yaml`):

```python
from src.model.flow_matching_model.samples import Samples

observed_curves = observed_curves.to(model.device)  # (B, 1, sequence_size), time domain
observed_in_model_domain = model.to_domain(Samples(observed_curves, "time", t=1)).curves
condition = model.encode(observed_in_model_domain)

generated = model.domain_sample(condition.shape[0], condition=condition, num_steps=50)
generated_time_domain = model.to_domain(generated, "time").curves
```

`num_steps` controls the number of Euler integration steps used to solve the flow ODE; more steps trade compute for sample quality.

### 6.4 Using a trained model as a null hypothesis for periodicity detection

This reproduces the "generative model" curves in Fig. 7 to 12, wired up for your own light curves:

```python
from src.utils.noise_models import FlowModelNoiseModel
from src.utils.light_curve_sin_dataset import LightCurvesSinDataset
from src.utils.cyclic_detector import CyclicDetector

flow_noise_model = FlowModelNoiseModel(model, num_samples=256)

# LightCurvesSinDataset injects a synthetic sinusoid into half of a set of light curves,
# giving you a labeled positive/negative set to evaluate detection performance on. If you
# already have observed light curves and only want p values, call
# flow_noise_model.generates_curves_like(your_curves, parameters=None, num_curves=100)
# directly instead and compute your own detection statistic, or use
# CyclicDetector.get_curves_p_values(your_curves, parameters=None) below.

dataset = LightCurvesSinDataset(
    flow_noise_model,
    cycle_length_range=(30, 100),
    snr_rms_range=(0.5, 0.8),
    dataset_size=500,
)

detector = CyclicDetector(dataset, flow_noise_model, num_simulated_light_curves=100)
p_values, labels = detector.get_p_value_for_dataset(batch_size=20)
```

`p_values` is a p value per light curve under the null hypothesis that it contains no periodic signal, and `labels` marks which curves had a synthetic sinusoid injected. `src/utils/rendering.py` provides `plot_rocs_from_items` and `plot_pvalue_histograms_from_items` to turn one or more `(name, p_values, labels)` triples into the ROC and calibration plots used throughout the paper.

## 7. Configuration reference

The full schema lives in `src/config.py` (a `pydantic` model, so an invalid config raises a clear validation error at startup rather than failing deep inside training). The fields most worth knowing about:

`unet_model.domain`: `"frequency"` or `"time"`, which domain the flow is defined in.
`unet_model.include_other_domain`: also concatenate the other domain's representation as extra input channels (used in most of the paper's configs).
`unet_model.extra_cond_dim`: 0 disables conditioning, greater than 0 enables the encoder and conditional generation (the paper uses 128).
`unet_model.channel_mults`, `base_channels`, `num_res_blocks`, `attn_resolutions`: control the size of the 1D UNet.
`data.outlier_percentage`: only used by `DataContaminationDataManager`; the fraction of the training set replaced with light curves that contain an injected periodic signal.
`loss_function.include_contrastive_loss` / `include_consistency_loss`: toggle the two auxiliary losses described in section 3.
`trainer.max_steps`, `trainer.lr`: standard optimization hyperparameters (AdamW).
`rendering.log_every_n_steps`: how often (in training steps) `visualize_step` runs and a checkpoint is written.

## 8. Known limitations

These are called out explicitly in the paper's discussion section and apply equally to this codebase:

The model is trained on light curves of a single fixed length (`unet_model.sequence_size`, 256 samples in the released configs) and cannot directly generate or learn structure on longer baselines than that.
All experiments assume uniformly sampled light curves. Real time domain surveys typically produce non uniformly sampled data with observational gaps; extending the conditioning/architecture to handle irregular sampling is future work, not something this release supports out of the box. (`DRWManager.subsample_curves` in `src/utils/drw.py` can build non uniformly sampled curves for your own experimentation, but no data manager or model in this release consumes them.)
As shown in the contamination experiment (section 4, Fig. 11 to 12), a learned, data driven null hypothesis is only as reliable as the purity of its training set: if the training data itself contains the kind of periodic signal you are searching for, the model will treat that structure as part of the null and detection power degrades sharply, even at contamination levels as low as 1%. Curate training data accordingly.
