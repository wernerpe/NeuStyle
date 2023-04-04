from pathlib import Path

import hydra
import torch
from einops import pack
from jaxtyping import Float, install_import_hook
from omegaconf import DictConfig
from torch import Tensor

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src.neus",),
    ("beartype", "beartype"),
):
    from src.neus.dataset.DataModule import DataModule
    from src.neus.misc.geometry import get_world_rays
    from src.neus.misc.image_io import save_image
    from src.neus.visualization.epipolar_lines import project_rays


@hydra.main(
    version_base=None,
    config_path="../../../config/neus",
    config_name="main",
)
def visualize_epipolar_lines(cfg: DictConfig):
    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(f"Saving outputs to {output_dir}")

    # Fetch two arbitrary frames.
    data_module = DataModule(cfg)
    loader = iter(data_module.train_dataloader())
    frame_left = next(loader)
    frame_right = next(loader)

    # Pick a random point in the image.
    b, _, h, w = frame_left["image"].shape
    coordinates = torch.rand((b, 1, 2)) * 0.5 + 0.25
    origins, directions = get_world_rays(
        coordinates,
        frame_left["extrinsics"],
        frame_left["intrinsics"],
    )
    projection = project_rays(
        origins,
        directions,
        frame_right["extrinsics"],
        frame_right["intrinsics"],
    )

    def round_pixel(pixel: Float[Tensor, "2"]):
        x, y = pixel
        x = (x * w).clip(min=0, max=w - 1).type(torch.int64)
        y = (y * h).clip(min=0, max=h - 1).type(torch.int64)
        return x, y

    # Draw the projections onto the image.
    visualization = []
    color = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    for i in range(b):
        image_left = frame_left["image"][i, :3]
        image_right = frame_right["image"][i, :3]
        x, y = round_pixel(coordinates[i, 0])
        xy_min = projection["xy_min"][i, 0]
        xy_max = projection["xy_max"][i, 0]

        # Draw a marker on the left image.
        r = 15
        image_left[
            :, (y - r).clip(min=0) : (y + r + 1), (x - r).clip(min=0) : (x + r + 1)
        ] = color[:, None, None]

        # Draw a line on the right image.
        r = 3
        for t in torch.linspace(0, 1, 100):
            x, y = round_pixel(xy_min + (xy_max - xy_min) * t)
            image_right[
                :, (y - r).clip(min=0) : (y + r + 1), (x - r).clip(min=0) : (x + r + 1)
            ] = color[:, None, None]

        visualization.append(pack([image_left, image_right], "c h *")[0])

    visualization = pack(visualization, "c * w")[0]
    save_image(visualization[None], Path("visualization.png"))


if __name__ == "__main__":
    visualize_epipolar_lines()
