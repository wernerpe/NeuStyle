import torch
import torch.nn.functional as F
from einops import pack
from jaxtyping import Float
from torch import Tensor


def compute_alpha_compositing_weights(
    alpha: Float[Tensor, "batch ray sample"],
) -> Float[Tensor, "batch ray sample"]:
    # Compute occlusion for each sample.
    # The 1e-10 is from the original NeRF (and pixelNeRF).
    shifted_alpha, _ = pack(
        [torch.ones_like(alpha[..., :1]), 1 - alpha[..., :-1] + 1e-10], "b r *"
    )
    occlusion = torch.cumprod(shifted_alpha, dim=-1)

    # Combine alphas with occlusion effects to get the final weights.
    return alpha * occlusion


def compute_volume_integral_weights(
    depths: Float[Tensor, "batch ray sample"],
    densities: Float[Tensor, "batch ray sample"],
) -> Float[Tensor, "batch ray sample"]:
    # Compute distances between samples.
    deltas = depths[..., 1:] - depths[..., :-1]

    # Duplicate the last distance.
    deltas, _ = pack([deltas, deltas[..., :1]], "b r *")

    # Compute opacity for each sample.
    alpha = 1 - torch.exp(-F.relu(densities) * deltas)

    return compute_alpha_compositing_weights(alpha)
