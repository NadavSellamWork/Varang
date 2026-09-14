import torch
from torch.utils.data import Dataset
from .noise_models import NoiseModel

# dataset
class LightCurvesSinDataset(Dataset):
    def __init__(
        self,
        noise_model: NoiseModel,
        cycle_length_range=(30, 100),
        snr_rms_range=(0.3, 0.7),   # signal_rms / noise_rms
        dataset_size=400,
        seed=42,
    ):
        self.noise_model = noise_model
        self.cycle_length_range = cycle_length_range
        self.snr_rms_range = snr_rms_range
        self.dataset_size = int(dataset_size)

        g = torch.Generator().manual_seed(seed)

        self.curves, self.parameters = self.noise_model.generate_curves(self.dataset_size)
        if not torch.is_tensor(self.curves):
            self.curves = torch.as_tensor(self.curves)

        self.curves = self.curves.float().cpu()

        self.sin_wave_mask = torch.zeros(self.dataset_size, dtype=torch.bool)
        self.sin_wave_mask[:self.dataset_size // 2] = True

        pos_idx = self.sin_wave_mask.nonzero(as_tuple=False).squeeze(-1)
        self.curves[pos_idx] = self.curves[pos_idx] + self.sin_waves_like(self.curves[pos_idx], generator=g)

    @torch.no_grad()
    def sin_waves_like(self, reference_tensor, generator=None):
        B, N = reference_tensor.shape

        # noise RMS per curve (center first)
        noise = reference_tensor - reference_tensor.mean(dim=1, keepdim=True)
        noise_rms = noise.pow(2).mean(dim=1).sqrt().clamp_min(1e-8)  # [B]

        # pick target SNR in RMS sense
        r0, r1 = self.snr_rms_range
        r = r0 + torch.rand(B, generator=generator) * (r1 - r0)      # [B]
        signal_rms = r * noise_rms                                    # [B]

        # sine params
        cyc0, cyc1 = self.cycle_length_range
        cyc = cyc0 + torch.rand(B, generator=generator) * (cyc1 - cyc0)
        phase = torch.rand(B, generator=generator) * 2 * torch.pi

        t = torch.arange(N, dtype=torch.float32)
        s = torch.sin(2 * torch.pi * t[None] / cyc[:, None] + phase[:, None])  # [B,N]

        # set peak amplitude A so that sine RMS matches signal_rms: RMS = A/sqrt(2)
        A = (2.0 ** 0.5) * signal_rms
        return A[:, None] * s

    def __getitem__(self, idx):
        return self.curves[idx], self.parameters[idx], bool(self.sin_wave_mask[idx])

    def __len__(self):
        return self.dataset_size