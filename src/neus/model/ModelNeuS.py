from omegaconf import DictConfig
from torch import nn


class ModelNeuS(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

    def forward(self, model_input):
        pass