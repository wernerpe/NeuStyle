from typing import Tuple

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .geometry import get_world_rays


def sample_training_rays(
    image: Float[Tensor, "batch channel height width"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    num_rays: int,
) -> Tuple[
    Float[Tensor, "batch ray 3"],  # origins
    Float[Tensor, "batch ray 3"],  # directions
    Float[Tensor, "batch ray channel"],  # sampled color
]:
    # Sample coordinates.
    b, _, h, w = image.shape
    coordinates = torch.rand(
        (b, num_rays, 2), dtype=torch.float32, device=extrinsics.device
    )

    # Sample color at those coordinates. Pixel centers are at (x.5, y.5).
    x, y = coordinates.unbind(dim=-1)
    row = (x * w).type(torch.int64)
    col = (y * h).type(torch.int64)
    batch_index = rearrange(torch.arange(b, device=image.device), "b -> b ()")
    color = image[batch_index, :, row, col]

    return (*get_world_rays(coordinates, extrinsics, intrinsics), color)


def sample_along_rays(
    origins: Float[Tensor, "batch ray 3"],
    directions: Float[Tensor, "batch ray 3"],
    near: Float[Tensor, "batch ray"],
    far: Float[Tensor, "batch ray"],
    num_samples: int,
) -> Tuple[
    Float[Tensor, "batch ray sample"],  # depths
    Float[Tensor, "batch ray sample 3"],  # xyz coordinates
]:
    # Generate evenly spaced samples along the ray.
    spacing = torch.linspace(0, 1, num_samples, device=far.device, dtype=torch.float32)

    # Rearrange everything to (batch, ray, sample, xyz).
    origins = rearrange(origins, "b r xyz -> b r () xyz")
    directions = rearrange(directions, "b r xyz -> b r () xyz")
    spacing = rearrange(spacing, "s -> () () s ()")
    near = rearrange(near, "b r -> b r () ()")
    far = rearrange(far, "b r -> b r () ()")

    # Compute point locations along rays.
    depths = near + spacing * (far - near)
    return depths[..., 0], origins + directions * depths
