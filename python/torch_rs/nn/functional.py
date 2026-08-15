"""Functional interface."""

import torch_rs as torch
from torch_rs import Tensor


def relu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    if inplace:
        raise NotImplementedError(
            "torch_rs.nn.functional.relu does not support inplace=True"
        )
    return torch.relu(input)
