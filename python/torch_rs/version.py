from typing import Optional

from torch_rs import __version__ as __version__


__all__ = ["__version__", "cuda", "hip", "rocm", "xpu"]
cuda: Optional[str] = None
hip: Optional[str] = None
rocm: Optional[str] = None
xpu: Optional[str] = None
