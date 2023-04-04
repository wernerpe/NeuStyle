from typing import Literal, TypedDict

from jaxtyping import Float
from torch import Tensor

Stage = Literal["train", "val"]


class Example(TypedDict):
    image: Float[Tensor, "channel height width"]
    extrinsics: Float[Tensor, "4 4"]
    intrinsics: Float[Tensor, "3 3"]
    near: Float[Tensor, ""]
    far: Float[Tensor, ""]
