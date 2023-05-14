import numpy as np
import torch
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import Dataset


@torch.no_grad()
def generate_spin(
    t: float,
    device: torch.device,
    elevation: float,
    radius: float,
) -> Float[Tensor, "4 4"]:
    # Translate back along the camera's look vector.
    tf_translation = torch.eye(4, dtype=torch.float32, device=device)
    tf_translation[:2] *= -1
    tf_translation[2, 3] = -radius

    # Generate the transformation for the azimuth.
    azimuth = R.from_rotvec(np.array([0, t * 2 * np.pi, 0], dtype=np.float32))
    azimuth = torch.tensor(azimuth.as_matrix())
    tf_azimuth = torch.eye(4, dtype=torch.float32, device=device)
    tf_azimuth[:3, :3] = azimuth

    # Generate the transformation for the elevation.
    deg_elevation = np.deg2rad(elevation)
    elevation = R.from_rotvec(np.array([deg_elevation, 0, 0], dtype=np.float32))
    elevation = torch.tensor(elevation.as_matrix())
    tf_elevation = torch.eye(4, dtype=torch.float32, device=device)
    tf_elevation[:3, :3] = elevation

    # This rotates the entire scene so that -Y is the up vector.
    global_rotation = torch.eye(4, dtype=torch.float32)
    global_rotation[1:3, 1:3] = 0
    global_rotation[2, 1] = -1
    global_rotation[1, 2] = 1

    return global_rotation.inverse() @ tf_azimuth @ tf_elevation @ tf_translation


class SpinWrapper(Dataset):
    dataset: Dataset
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        item = self.dataset[0]
        item["extrinsics"] = generate_spin(
            index / self.length,
            item["extrinsics"].device,
            30.0,
            1.2,
        )
        item["near"] = torch.tensor(0.2, dtype=torch.float32)
        item["far"] = torch.tensor(2.2, dtype=torch.float32)
        return item
