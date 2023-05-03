from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .DatasetNeRFSynthetic import DatasetNeRFSynthetic
from .ValidationWrapper import ValidationWrapper

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
            shuffle=True,
            num_workers=self.cfg.training.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            ValidationWrapper(
                DATASETS[self.dataset_name](self.cfg.dataset[self.dataset_name], "val"),
                1,
            ),
            self.cfg.validation.batch_size,
            num_workers=self.cfg.validation.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            DATASETS[self.dataset_name](self.cfg.dataset[self.dataset_name], "val"),
            num_workers=self.cfg.validation.num_workers,
        )
