import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        num_frequencies: int,
        d_in: int,
        largest_period: float,
        include_input: bool,
    ):
        super().__init__()

        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.d_in = d_in
        self.d_out = 2 * num_frequencies * d_in
        if include_input:
            self.d_out += d_in

        base_frequency = 2 * torch.pi / largest_period
        frequencies = base_frequency * 2 ** torch.arange(0, num_frequencies)
        frequencies = repeat(frequencies, "f -> f fn", fn=2)
        self.register_buffer("frequencies", frequencies, persistent=False)

        phases = torch.zeros_like(frequencies)
        phases[:, 0] = torch.pi * 0.5
        self.register_buffer("phases", phases, persistent=False)

    def forward(
        self,
        input: Float[Tensor, "*batch d_in"],
    ) -> Float[Tensor, "*batch d_out"]:
        embedded = repeat(input, "... ch -> ... ch f fn", f=self.num_frequencies, fn=2)
        embedded = torch.sin(embedded * self.frequencies + self.phases)
        embedded = rearrange(embedded, "... ch f fn -> ... (ch f fn)")

        if self.include_input:
            embedded = torch.cat((input, embedded), dim=-1)

        return embedded
