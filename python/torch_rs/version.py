from . import __version__ as __version__


__all__ = ["__version__", "cuda"]

# The native backend is currently a CPU-only build, so PyTorch's CUDA build
# version sentinel is the exact ``None`` singleton. This is build metadata and
# intentionally does not inspect drivers, devices, or the process environment.
cuda: str | None = None
