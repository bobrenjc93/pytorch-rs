# mypy: allow-untyped-defs
import sys as _sys
import types as _types

import torch_rs as torch


_CUDNN_TENSOR_DTYPES = {torch.float32}


def is_available():
    r"""Return a bool indicating if CUDNN is currently available."""
    return torch._C._has_cudnn


def is_acceptable(tensor):
    if tensor.device.type != "cuda" or tensor.dtype not in _CUDNN_TENSOR_DTYPES:
        return False
    return torch._C._has_cudnn


class CudnnModule(_types.ModuleType):
    def __init__(self, module, name):
        super().__init__(name)
        self.m = module

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


# Match PyTorch's module-proxy identity and reload behavior without exposing
# any of its cuDNN configuration or execution surface.
_sys.modules[__name__] = CudnnModule(_sys.modules[__name__], __name__)
