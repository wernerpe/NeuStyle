from einops import einsum, repeat
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from ..misc.sampling import sample_along_rays
from ..misc.volume_rendering import compute_volume_integral_weights
from .components.ColorNetwork import ColorNetwork
from .components.SDFNetwork import SDFNetwork


class ModelNeuS(nn.Module):
    cfg_model: DictConfig

    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        self.cfg_model = cfg_model
        self.sdf_network = SDFNetwork(**cfg_model.sdf_network)
        self.color_network = ColorNetwork(**cfg_model.color_network)

    def forward(
        self,
        origins: Float[Tensor, "batch ray 3"],
        directions: Float[Tensor, "batch ray 3"],
        near: Float[Tensor, "batch ray"],
        far: Float[Tensor, "batch ray"],
    ):
        # For now, simple NeRF without view dependence...
        num_samples = self.cfg_model.num_coarse_samples
        depths, sample_locations = sample_along_rays(
            origins, directions, near, far, num_samples
        )
        sdf_and_features = self.sdf_network(sample_locations)
        sdf = sdf_and_features[..., 0]
        features = sdf_and_features[..., 1:]

        color = self.color_network(
            sample_locations,
            repeat(directions, "b r xyz -> b r s xyz", s=num_samples),
            repeat(directions, "b r xyz -> b r s xyz", s=num_samples),
            features,
        )

        weights = compute_volume_integral_weights(depths, sdf)
        return {
            "color": einsum(weights, color, "b r s, b r s c -> b r c"),
            "depth": einsum(weights, depths, "b r s, b r s -> b r"),
            "alpha": einsum(weights, "b r s -> b r"),
        }
