"""Optional runtime discovery and namespace wiring for the Rust CUDA backend.

Python locates wheel-installed libraries only; native Rust owns CUDA state,
allocations, transfers, and errors. TORCH_RS_CUDART overrides discovery for
both Python and standalone Rust callers.
"""
import importlib.util
import pathlib
from . import torch_rs as _native


def _candidate_libraries():
    candidates = []
    for package in ("nvidia.cuda_runtime", "nvidia.cu13"):
        try:
            spec = importlib.util.find_spec(package)
        except Exception:  # Optional package discovery must not prevent CPU imports.
            continue
        if spec is not None and spec.submodule_search_locations is not None:
            for location in spec.submodule_search_locations:
                candidates.extend(str(path) for path in sorted(
                    (pathlib.Path(location) / "lib").glob("libcudart.so*")
                ))
    return list(dict.fromkeys(candidates))


_native._cuda_configure_runtime(_candidate_libraries())
device_count = _native._cuda_device_count
is_initialized = _native._cuda_is_initialized


def is_available():
    return device_count() > 0
