# mypy: allow-untyped-defs
import sys as _sys

import torch_rs as torch
from torch_rs.backends import PropModule


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    return torch._C._has_mkldnn


class MkldnnModule(PropModule):
    def is_available(self):
        return is_available()


_module = MkldnnModule(_sys.modules[__name__], __name__)
_sys.modules[__name__] = _module
_parent_name = __name__.rpartition(".")[0]
_parent = _sys.modules.get(_parent_name)
if _parent is not None:
    setattr(_parent, "mkldnn", _module)
