from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .components.ColorNetwork import ColorNetwork
from .components.RendererNeuS import RendererNeuS
from .components.SDFNetwork import SDFNetwork
from .components.SharpnessNetwork import SharpnessNetwork


class ModelNeuS(nn.Module):
    cfg_model: DictConfig

    def __init__(self, cfg_model: DictConfig) -> None:
        super().__init__()
        self.cfg_model = cfg_model
        self.sdf_network = SDFNetwork(**cfg_model.sdf_network)
        self.color_network = ColorNetwork(**cfg_model.color_network)
        self.sharpness_network = SharpnessNetwork(**cfg_model.sharpness_network)
        self.renderer = RendererNeuS(
            cfg_model.renderer,
            self.sdf_network,
            self.color_network,
            self.sharpness_network,
        )

    def forward(
        self,
        origins: Float[Tensor, "batch ray 3"],
        directions: Float[Tensor, "batch ray 3"],
        near: Float[Tensor, "batch ray"],
        far: Float[Tensor, "batch ray"],
        global_step: int,
    ):
        return self.renderer.render(
            origins,
            directions,
            near,
            far,
            min(1.0, global_step / self.cfg_model.cosine_annealing_end),
        )
