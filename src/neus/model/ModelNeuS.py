import mcubes
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn
from trimesh import Trimesh

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

    def generate_mesh(
        self,
        resolution: int = 192,
        batch_size: int = 16384,
    ) -> Trimesh:
        # Generate a 3D grid of coordinates.
        plus1 = resolution + 1
        device = next(iter(self.sdf_network.parameters())).device
        xyz = torch.linspace(-1, 1, plus1, dtype=torch.float32, device=device)
        xyz = torch.stack(torch.meshgrid(xyz, xyz, xyz, indexing="xy"), dim=-1)
        xyz = rearrange(xyz, "d0 d1 d2 xyz -> (d0 d1 d2) xyz")

        # Evaluate the SDF on the grid.
        sdf = [
            self.sdf_network.sdf(xyz_batch)["sdf"].detach().cpu()
            for xyz_batch in xyz.split(batch_size)
        ]
        sdf = rearrange(
            torch.cat(sdf, dim=0),
            "(d0 d1 d2) -> d0 d1 d2",
            d0=plus1,
            d1=plus1,
            d2=plus1,
        )

        # Get a mesh. Note the sign flip.
        vertices, faces = mcubes.marching_cubes(sdf.numpy(), 0)
        vertices = (vertices / resolution) * 2 - 1
        x, y, z = vertices.T
        vertices = np.stack([y, x, z], axis=-1)
        return Trimesh(vertices, faces)
