from typing import Optional

from torch_rs import __version__ as __version__


__all__ = ["__version__", "debug", "cuda", "hip", "rocm", "xpu"]
debug = False
cuda: Optional[str] = None
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
