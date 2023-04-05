import torch
from jaxtyping import Float
from torch import Tensor, nn


class SharpnessNetwork(nn.Module):
    def __init__(self, initial_value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(initial_value, dtype=torch.float32))

    def forward(self, points: Float[Tensor, "*batch 3"]) -> Float[Tensor, " *batch"]:
        return torch.exp(10 * self.value).broadcast_to(points.shape[:-1])
