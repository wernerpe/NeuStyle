from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from ..misc.sampling import sample_along_rays
from .components.PositionalEncoding import PositionalEncoding


class ModelNeuS(nn.Module):
    cfg_model: DictConfig

    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        self.cfg_model = cfg_model
        self.positional_encoding = PositionalEncoding(**cfg_model.positional_encoding)
        self.placeholder = nn.Sequential(
            nn.Linear(self.positional_encoding.d_out, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(
        self,
        origins: Float[Tensor, "batch ray 3"],
        directions: Float[Tensor, "batch ray 3"],
        near: Float[Tensor, "batch ray"],
        far: Float[Tensor, "batch ray"],
    ):
        # For now, simple NeRF without view dependence...
        sample_locations = sample_along_rays(
            origins, directions, near, far, self.cfg_model.num_coarse_samples
        )
        sample_locations = self.positional_encoding(sample_locations)
        density, color = self.placeholder(sample_locations).split((1, 3), dim=-1)

        a = 1
