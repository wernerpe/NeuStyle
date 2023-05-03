import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat
from jaxtyping import Float, Int
from lightning_fabric.utilities.apply_func import apply_to_collection
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor, nn, optim

from ..misc.collation import collate
from ..misc.deformrays import map_rays_to_neus
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

        # Dump a mesh.
        mesh = self.model.generate_mesh()
        mesh.export("latest_mesh.stl")

    def predict_step(self, batch, batch_idx):
        # Render a deformed image.
        scale = self.cfg.validation.preview_image_scale
        _, _, h, w = batch["image"].shape
        h = int(h * scale)
        w = int(w * scale)

        # Load the original and deformed meshes.
        with open(self.cfg.rendering.deformation, "rb") as f:
            v_deformed, v_undeformed, faces, rotations = pickle.load(f)

        coordinates, deformed = self.render_deformed_image(
            batch["extrinsics"],
            batch["intrinsics"],
            (h, w),
            v_undeformed,
            v_deformed,
            faces,
            rotations,
        )
        coordinates, undeformed = self.render_image(
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
        row_rgb = pack(
            [deformed["color"], undeformed["color"], ground_truth[:, :3]], "b c h *"
        )[0]

        # Second row of visualization: mask comaprison.
        row_mask = pack(
            [deformed["alpha"], undeformed["alpha"], ground_truth[:, 3]], "b h *"
        )[0]
        row_mask = repeat(row_mask, "b h w -> b c h w", c=3)

        # Third row of visualization: normals.
        row_normals = pack(
            [
                deformed["normal"] * 0.5 + 0.5,
                undeformed["normal"] * 0.5 + 0.5,
                torch.zeros_like(undeformed["normal"]),
            ],
            "b c h *",
        )[0]

        visualization = pack([row_rgb, row_mask, row_normals], "b c * w")[0]
        visualization = visualization.clip(min=0, max=1)

        self.logger.log_image(
            "comparison",
            [rearrange(visualization, "b c h w -> (b h) w c").cpu().numpy()],
        )

    def render_deformed_image(
        self,
        extrinsics: Float[Tensor, "batch 4 4"],
        intrinsics: Float[Tensor, "batch 3 3"],
        height_width: Tuple[int, int],
        undeformed_vertices: Float[np.ndarray, "vertex 3"],
        deformed_vertices: Float[np.ndarray, "vertex 3"],
        faces: Int[np.ndarray, "face 3"],
        rotations: Float[np.ndarray, "3 3 vertex"],
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

        # Map rays to undeformed space.
        print("Deforming rays...")
        endpoints, _ = map_rays_to_neus(
            origins[0],
            directions[0],
            undeformed_vertices,
            deformed_vertices,
            faces,
            rotations,
            2,
            0.1,  # large enough segment to capture surface
        )
        print("Done deforming rays.")

        # Convert endpoints to rays.
        endpoints = torch.tensor(endpoints, device=origins.device, dtype=torch.float32)
        start, end = endpoints.unbind(dim=1)
        origins = start
        far = (end - start).norm(dim=-1, keepdim=True)
        directions = (end - start) / far
        near = torch.zeros_like(far)

        # Filter out NaNs (non-intersections).
        valid = ~directions.isnan().any(dim=-1)
        origins = origins[valid]
        directions = directions[valid]
        near = near[valid]
        far = far[valid]

        # Render image in batches.
        num_rays = self.cfg.validation.num_rays
        bundle = zip(
            origins[None].split(num_rays, dim=1),
            directions[None].split(num_rays, dim=1),
            near[None, :, 0].split(num_rays, dim=1),
            far[None, :, 0].split(num_rays, dim=1),
        )
        output = [
            apply_to_collection(
                self.model(
                    origins_batch,
                    directions_batch,
                    near_batch,
                    far_batch,
                    self.global_step,
                ),
                Tensor,
                lambda x: x.cpu(),
            )
            for origins_batch, directions_batch, near_batch, far_batch in bundle
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

        # Collate/unmask the rays.
        output = collate(output, lambda x: torch.cat(x, dim=1))

        def fix(x):
            assert x.shape[0] == 1
            fixed = torch.zeros((h * w, *x.shape[2:]), dtype=x.dtype, device=x.device)
            fixed[valid] = x[0]
            return rearrange(fixed[None], "b (h w) ... -> b ... h w", h=h, w=w)

        output = {key: fix(value) for key, value in output.items()}

        # Render on white background.
        output["color"] = output["color"] + (1 - output["alpha"][:, None])

        return grid_coordinates, output

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

        output = collate(
            output,
            lambda x: rearrange(
                torch.cat(x, dim=1), "b (h w) ... -> b ... h w", h=h, w=w
            ),
        )

        # Render on white background.
        output["color"] = output["color"] + (1 - output["alpha"][:, None])

        return grid_coordinates, output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.cfg.training.optim.lr)
        return {
            "optimizer": optimizer,
        }
