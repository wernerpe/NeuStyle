from typing import Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn, optim


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    cfg: DictConfig

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx: int):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.optim.lr)
        return {
            "optimizer": optimizer,
        }
