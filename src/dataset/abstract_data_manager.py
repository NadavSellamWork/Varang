from torch.utils.data import DataLoader
import pytorch_lightning as pl

class AbstractDataManager(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_channels = self.config.data_channels
        self.curves = None

    def get_reference_curves(self, num_curves=None):
        if num_curves is None:
            return self.curves[:]
        return self.curves[:num_curves]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def visualize_step(self, trainer: pl.Trainer, model):
        pass

