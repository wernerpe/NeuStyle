import json
from math import tan
from pathlib import Path

import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from .types import Stage


class DatasetNeRFSynthetic(Dataset):
    cfg_dataset: DictConfig
    images: Float[Tensor, "batch channel height width"]
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]

    def __init__(self, cfg_dataset: DictConfig, stage: Stage) -> None:
        super().__init__()
        self.cfg_dataset = cfg_dataset
        path = Path(cfg_dataset.path) / cfg_dataset.scene

        # Load the metadata.
        transforms = tf.ToTensor()
        with (path / f"transforms_{stage}.json").open("r") as f:
            metadata = json.load(f)

        # This converts the extrinsics to OpenCV style.
        conversion = torch.eye(4, dtype=torch.float32)
        conversion[1:3, 1:3] *= -1

        # Read the images and extrinsics.
        images = []
        extrinsics = []
        for frame in tqdm(metadata["frames"], "Loading frames"):
            extrinsics.append(
                torch.tensor(frame["transform_matrix"], dtype=torch.float32)
                @ conversion
            )
            images.append(transforms(Image.open(path / f"{frame['file_path']}.png")))
        self.images = torch.stack(images)
        self.extrinsics = torch.stack(extrinsics)

        # Convert the intrinsics to (normalized) OpenCV style.
        camera_angle_x = float(metadata["camera_angle_x"])
        focal_length = 0.5 / tan(0.5 * camera_angle_x)
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics[:2, :2] *= focal_length
        intrinsics[:2, 2] = 0.5
        self.intrinsics = repeat(intrinsics, "i j -> b i j", b=self.extrinsics.shape[0])

    @property
    def num_images(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int):
        return self.images[index % self.num_images]

    def __len__(self) -> int:
        return self.num_images * self.cfg_dataset.repetitions_per_epoch
