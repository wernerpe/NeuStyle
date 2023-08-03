from einops import einsum
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from ..misc.sampling import sample_along_rays
from ..misc.volume_rendering import compute_volume_integral_weights
from .components.RandomSinusoidEncoding import RandomSinusoidEncoding


class ModelNeuS(nn.Module):
    cfg_model: DictConfig

    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        self.cfg_model = cfg_model
        self.positional_encoding = RandomSinusoidEncoding(3, 1024, 8.0, 6)
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
        depths, sample_locations = sample_along_rays(
            origins, directions, near, far, self.cfg_model.num_coarse_samples
        )
        sample_locations = self.positional_encoding(sample_locations)
        density, color = self.placeholder(sample_locations).split((1, 3), dim=-1)
        weights = compute_volume_integral_weights(depths, density[..., 0])
        return {
            "color": einsum(weights, color.sigmoid(), "b r s, b r s c -> b r c"),
            "depth": einsum(weights, depths, "b r s, b r s -> b r"),
            "alpha": einsum(weights, "b r s -> b r"),
        }
