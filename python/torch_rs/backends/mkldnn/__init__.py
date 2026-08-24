# mypy: allow-untyped-defs
import sys as _sys
import types as _types

import torch_rs as torch


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    return torch._C._has_mkldnn


class MkldnnModule(_types.ModuleType):
    def __init__(self, module, name):
        super().__init__(name)
        self.m = module

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    def is_available(self):
        return is_available()


# Preserve PyTorch's bound-method query and module replacement behavior while
# leaving its oneDNN configuration and execution surface unimplemented.
_sys.modules[__name__] = MkldnnModule(_sys.modules[__name__], __name__)
