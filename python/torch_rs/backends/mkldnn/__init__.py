# mypy: allow-untyped-defs
import sys as _sys

import torch_rs as torch
from torch_rs.backends import PropModule as _PropModule


def is_available():
    r"""Return whether PyTorch is built with MKL-DNN support."""
    return torch._C._has_mkldnn


class MkldnnModule(_PropModule):
    def is_available(self):
        return is_available()


_sys.modules[__name__] = MkldnnModule(_sys.modules[__name__], __name__)
