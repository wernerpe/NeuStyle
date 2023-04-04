from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .DatasetNeRFSynthetic import DatasetNeRFSynthetic

DATASETS = {
    "nerf_synthetic": DatasetNeRFSynthetic,
}


class DataModule(LightningDataModule):
    cfg: DictConfig

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    @property
    def dataset_name(self) -> str:
        (name,) = self.cfg.dataset.keys()
        return name

    def train_dataloader(self):
        return DataLoader(
            DATASETS[self.dataset_name](self.cfg.dataset[self.dataset_name], "train"),
            self.cfg.training.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            DATASETS[self.dataset_name](self.cfg.dataset[self.dataset_name], "val"),
            self.cfg.validation.batch_size,
        )
