import torch
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .PositionalEncoding import PositionalEncoding


class ColorNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        d_hidden,
        n_hidden_layers,
        positional_encoding: DictConfig,
        weight_norm: bool,
    ):
        super().__init__()

        self.directional_encoding = PositionalEncoding(**positional_encoding)

        # Define the first layer.
        layers = [
            nn.Linear(6 + d_feature + self.directional_encoding.d_out, d_hidden),
            nn.ReLU(),
        ]

        # Define the hidden layers.
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(d_hidden, d_hidden))
            layers.append(nn.ReLU())

        # Define the final layer.
        layers.append(nn.Linear(d_hidden, 3))
        layers.append(nn.Sigmoid())

        # Apply weight norm to linear layers if desired.
        if weight_norm:
            layers = [
                nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer
                for layer in layers
            ]

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        points: Float[Tensor, "*batch 3"],
        normals: Float[Tensor, "*batch 3"],
        view_dirs: Float[Tensor, "*batch 3"],
        feature_vectors: Float[Tensor, "*batch channel"],
    ) -> Float[Tensor, "*batch 3"]:
        view_dirs = self.directional_encoding(view_dirs)
        network_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        return self.network(network_input)
