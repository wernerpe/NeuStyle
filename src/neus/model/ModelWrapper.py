from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning_fabric.utilities.apply_func import apply_to_collection
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn, optim

from ..misc.collation import collate
from ..misc.geometry import get_world_rays
from ..misc.sampling import sample_image_grid, sample_training_rays
from ..visualization.color_map import apply_color_map_to_image
from .ModelNeuS import ModelNeuS

MODELS: Dict[str, nn.Module] = {
    "neus": ModelNeuS,
}


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    model: nn.Module
    cfg: DictConfig

    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # Set up the model.
        (model_name,) = cfg.model.keys()
        self.model = MODELS[model_name](cfg.model[model_name])

    def training_step(self, batch, batch_idx):
        # Generate training rays from the images.
        num_rays = self.cfg.training.num_rays
        origins, directions, color = sample_training_rays(
            batch["image"],
            batch["extrinsics"],
            batch["intrinsics"],
            num_rays,
        )

        output = self.model(
            origins,
            directions,
            repeat(batch["near"], "b -> b r", r=num_rays),
            repeat(batch["far"], "b -> b r", r=num_rays),
            self.global_step,
        )

        loss_color = ((output["color"] - color[..., :3]) ** 2).mean()
        loss_eikonal = 0.1 * output["error_eikonal"]

        self.log("train/loss_color", loss_color)
        self.log("train/loss_eikonal", loss_eikonal)

        return loss_color + loss_eikonal

    def validation_step(self, batch, batch_idx):
        if self.cfg.wandb.mode == "disabled":
            return

        scale = self.cfg.validation.preview_image_scale
        _, _, h, w = batch["image"].shape
        h = int(h * scale)
        w = int(w * scale)
        coordinates, predicted = self.render_image(
            batch["extrinsics"],
            batch["intrinsics"],
            batch["near"],
            batch["far"],
            (h, w),
        )

        # Sample ground-truth pixel locations.
        ground_truth = F.grid_sample(
            batch["image"],
            coordinates * 2 - 1,
            mode="bilinear",
            align_corners=False,
        ).cpu()

        # First row of visualization: RGB comparison.
        row_rgb = pack([predicted["color"], ground_truth[:, :3]], "b c h *")[0]

        # Second row of visualization: mask comaprison.
        row_mask = pack([predicted["alpha"], ground_truth[:, 3]], "b h *")[0]
        row_mask = repeat(row_mask, "b h w -> b c h w", c=3)

        # Third row of visualization: depth and normals.
        depth = predicted["depth"]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-10)
        depth = apply_color_map_to_image(depth)
        normals = predicted["normal"] * 0.5 + 0.5
        row_extra = pack([depth, normals], "b c h *")[0]

        visualization = pack([row_rgb, row_mask, row_extra], "b c * w")[0]
        visualization = visualization.clip(min=0, max=1)

        self.logger.log_image(
            "comparison",
            [rearrange(visualization, "b c h w -> (b h) w c").cpu().numpy()],
        )

    def render_image(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        near: Float[Tensor, " batch"],
        far: Float[Tensor, " batch"],
        height_width: Tuple[int, int],
    ) -> Tuple[Float[Tensor, "batch height width xy"], dict]:
        # Generate image rays.
        h, w = height_width
        b, _, _ = extrinsics.shape
        grid_coordinates, _ = sample_image_grid(h, w, extrinsics.device)
        grid_coordinates = repeat(grid_coordinates, "h w xy -> b h w xy", b=b)
        origins, directions = get_world_rays(
            rearrange(grid_coordinates, "b h w xy -> b (h w) xy"),
            extrinsics,
            intrinsics,
        )

        # Render image in batches.
        num_rays = self.cfg.validation.num_rays
        bundle = zip(origins.split(num_rays, dim=1), directions.split(num_rays, dim=1))
        output = [
            apply_to_collection(
                self.model(
                    origins_batch,
                    directions_batch,
                    repeat(near, "b -> b r", r=origins_batch.shape[1]),
                    repeat(far, "b -> b r", r=origins_batch.shape[1]),
                    self.global_step,
                ),
                Tensor,
                lambda x: x.cpu(),
            )
            for origins_batch, directions_batch in bundle
        ]

        # Drop elements are aren't image-like.
        output = [
            {
                k: v
                for k, v in batch.items()
                if k in ("color", "alpha", "depth", "normal")
            }
            for batch in output
        ]

        return grid_coordinates, collate(
            output,
            lambda x: rearrange(
                torch.cat(x, dim=1), "b (h w) ... -> b ... h w", h=h, w=w
            ),
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.optim.lr)
        return {
            "optimizer": optimizer,
        }
