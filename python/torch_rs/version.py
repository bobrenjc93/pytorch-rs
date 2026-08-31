from typing import Optional

from torch_rs import __version__ as __version__


__all__ = ["__version__", "debug", "cuda", "git_version", "hip", "rocm", "xpu"]
debug = False
cuda: Optional[str] = None
git_version = None
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
