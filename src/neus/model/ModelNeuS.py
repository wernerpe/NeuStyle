from omegaconf import DictConfig
from torch import nn


class ModelNeuS(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.placeholder = nn.Linear(1, 1)

    def forward(self, model_input):
        a = 1
        a = 1
