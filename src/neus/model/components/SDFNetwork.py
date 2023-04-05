import numpy as np
import torch
from jaxtyping import Float
from omegaconf import DictConfig, ListConfig
from torch import Tensor, nn

from .PositionalEncoding import PositionalEncoding


class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_layers: int,
        skip_in: ListConfig,
        positional_encoding: DictConfig,
        bias: float,
        scale: float,
        geometric_init: bool,
        weight_norm: bool,
    ):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.positional_encoding = PositionalEncoding(**positional_encoding)

        dims[0] = self.positional_encoding.d_out

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.skip_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight,
                        mean=np.sqrt(np.pi) / np.sqrt(dims[layer]),
                        std=0.0001,
                    )
                    torch.nn.init.constant_(lin.bias, -bias)
                elif layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(layer), lin)

        self.activation = nn.Softplus(beta=100)

    def sdf(self, points: Float[Tensor, "*batch 3"]):
        points = self.positional_encoding(points * self.scale)

        x = points
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, points], dim=-1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)

        return {
            "sdf": x[..., 0] / self.scale,
            "feature": x[..., 1:],
        }

    @torch.set_grad_enabled(True)
    def gradient(self, points: Float[Tensor, "*batch 3"]):
        points.requires_grad_(True)
        signed_distance = self.sdf(points)["sdf"]
        return torch.autograd.grad(
            outputs=signed_distance,
            inputs=points,
            grad_outputs=torch.ones_like(signed_distance, requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
