from typing import Tuple

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .geometry import get_world_rays


def sample_training_rays(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    num_rays: int,
) -> Tuple[
    Float[Tensor, "batch ray 3"],  # origins
    Float[Tensor, "batch ray 3"],  # directions
]:
    batch, _, _ = extrinsics.shape
    coordinates = torch.rand(
        (batch, num_rays, 2), dtype=torch.float32, device=extrinsics.device
    )
    return get_world_rays(coordinates, extrinsics, intrinsics)


def sample_along_rays(
    origins: Float[Tensor, "batch ray 3"],
    directions: Float[Tensor, "batch ray 3"],
    near: Float[Tensor, "batch ray"],
    far: Float[Tensor, "batch ray"],
    num_samples: int,
) -> Float[Tensor, "batch ray sample 3"]:
    # Generate evenly spaced samples along the ray.
    spacing = torch.linspace(0, 1, num_samples, device=far.device, dtype=torch.float32)

    # Rearrange everything to (batch, ray, sample, xyz).
    origins = rearrange(origins, "b r xyz -> b r () xyz")
    directions = rearrange(directions, "b r xyz -> b r () xyz")
    spacing = rearrange(spacing, "s -> () () s ()")
    near = rearrange(near, "b r -> b r () ()")
    far = rearrange(far, "b r -> b r () ()")

    # Compute point locations along rays.
    return origins + directions * (near + spacing * (far - near))
