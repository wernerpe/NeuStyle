from pathlib import Path

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor


def save_image(image: Float[Tensor, "batch channel height width"], path: Path):
    # Handle single-channel images.
    _, channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "b () h w -> b c h w", c=3)

    path.parent.mkdir(exist_ok=True, parents=True)
    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    image = rearrange(image, "b c h w -> h (b w) c")
    Image.fromarray(image.cpu().numpy()).save(path)
