from dataclasses import dataclass
from typing import Protocol, TypedDict, runtime_checkable

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from ...misc.volume_rendering import compute_alpha_compositing_weights


class SignedDistance(TypedDict):
    sdf: Float[Tensor, " *batch"]
    feature: Float[Tensor, "*batch channel"]


@runtime_checkable
class SignedDistanceFunction(Protocol):
    def sdf(self, points: Float[Tensor, "*batch 3"]) -> SignedDistance:
        pass

    def gradient(self, points: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 3"]:
        pass


@runtime_checkable
class ColorFunction(Protocol):
    def forward(
        self,
        points: Float[Tensor, "*batch 3"],
        directions: Float[Tensor, "*batch 3"],
        normals: Float[Tensor, "*batch 3"],
        features: Float[Tensor, "*batch channel"],
    ) -> Float[Tensor, "*batch 3"]:
        pass


@runtime_checkable
class SharpnessFunction(Protocol):
    def forward(self, points: Float[Tensor, "*batch 3"]) -> Float[Tensor, " *batch"]:
        pass


class Output(TypedDict):
    color: Float[Tensor, "batch ray 3"]
    depth: Float[Tensor, "batch ray"]
    alpha: Float[Tensor, "batch ray"]
    error_eikonal: Float[Tensor, ""]


@dataclass
class RendererNeuS:
    cfg_renderer: DictConfig
    signed_distance_function: SignedDistanceFunction
    color_function: ColorFunction
    sharpness_function: SharpnessFunction

    def render(
        self,
        origins: Float[Tensor, "batch ray 3"],
        directions: Float[Tensor, "batch ray 3"],
        near: Float[Tensor, "batch ray"],
        far: Float[Tensor, "batch ray"],
        cosine_annealing_ratio: float,
    ) -> Output:
        # Broadcast inputs to (batch, ray, sample, xyz).
        origins = rearrange(origins, "b r xyz -> b r () xyz")
        directions = rearrange(directions, "b r xyz -> b r () xyz")
        near = rearrange(near, "b r -> b r () ()")
        far = rearrange(far, "b r -> b r () ()")

        # Generate coarse samples.
        num_samples = self.cfg_renderer.num_samples
        depths = torch.linspace(0, 1, num_samples, device=origins.device)
        depths = rearrange(depths, "s -> () () s ()")
        depths = near + (far - near) * depths

        # TODO: Add code for importance sampling.
        return self._render_core(origins, directions, depths, cosine_annealing_ratio)

    def _render_core(
        self,
        origins: Float[Tensor, "batch ray 1 3"],
        directions: Float[Tensor, "batch ray 1 3"],
        depths: Float[Tensor, "batch ray sample 1"],
        cosine_annealing_ratio: float,
    ) -> Output:
        # Compute section midpoints.
        deltas = depths[..., 1:, :] - depths[..., :-1, :]
        deltas = torch.cat([deltas, deltas.detach().mean(dim=-2, keepdim=True)], dim=-2)
        depths_middle = depths + deltas * 0.5
        points_middle = origins + depths_middle * directions
        directions_middle = directions.broadcast_to(points_middle.shape)

        # Sample the signed distance, color, and sharpness functions at the midpoints.
        sdf_outputs = self.signed_distance_function.sdf(points_middle)
        sdf_gradients = self.signed_distance_function.gradient(points_middle)
        color = self.color_function(
            points_middle,
            directions_middle,
            sdf_gradients,
            sdf_outputs["feature"],
        )
        sharpness = self.sharpness_function(points_middle).clip(1e-6, 1e6)

        # Convert signed distance to alpha values.
        cosine = (directions_middle * sdf_gradients).sum(dim=-1)
        cosine_annealed = -(
            F.relu(-cosine * 0.5 + 0.5) * (1 - cosine_annealing_ratio)
            + F.relu(-cosine) * cosine_annealing_ratio
        )
        sdf_delta = cosine_annealed * deltas[..., 0] * 0.5
        sdf_next = sdf_outputs["sdf"] + sdf_delta
        sdf_prev = sdf_outputs["sdf"] - sdf_delta
        cdf_next = (sdf_next * sharpness).sigmoid()
        cdf_prev = (sdf_prev * sharpness).sigmoid()
        alphas = ((cdf_prev - cdf_next + 1e-5) / (cdf_prev + 1e-5)).clip(min=0, max=1)

        # Do alpha compositing.
        weights = compute_alpha_compositing_weights(alphas)
        return {
            "color": einsum(weights, color, "b r s, b r s c -> b r c"),
            "depth": einsum(weights, depths_middle[..., 0], "b r s, b r s -> b r"),
            "alpha": einsum(weights, "b r s -> b r"),
            "error_eikonal": ((sdf_gradients - 1) ** 2).mean(),
        }
