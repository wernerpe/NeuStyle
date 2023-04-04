from typing import Dict, Optional

from einops import repeat
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn, optim

from ..misc.sampling import sample_training_rays
from .ModelNeuS import ModelNeuS

MODELS: Dict[str, nn.Module] = {
    "neus": ModelNeuS,
}

class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    cfg: DictConfig

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # Set up the model.
        model_name, = cfg.model.keys()
        self.model = MODELS[model_name](cfg.model[model_name])

    def training_step(self, batch, batch_idx):
        num_rays = self.cfg.training.num_rays
        origins, directions = sample_training_rays(
            batch["extrinsics"],
            batch["intrinsics"],
            num_rays,
        )

        self.model(
            origins,
            directions,
            repeat(batch["near"], "b -> b r", r=num_rays),
            repeat(batch["far"], "b -> b r", r=num_rays),
        )
        a = 1
        a = 1


    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.optim.lr)
        return {
            "optimizer": optimizer,
        }
