import tempfile
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from torch_rs import _cuda_public_storage as storage


class CudaRuntimeDiscoveryTests(unittest.TestCase):
    def test_runtime_wheel_layouts_are_discovered_without_system_libraries(self):
        for package, library in (
            ("nvidia.cuda_runtime", "libcudart.so.12"),
            ("nvidia.cu13", "libcudart.so.13"),
        ):
            with self.subTest(package=package), tempfile.TemporaryDirectory(
                dir=Path(__file__).resolve().parents[1]
            ) as directory:
                package_dir = Path(directory) / package.replace(".", "/")
                library_dir = package_dir / "lib"
                library_dir.mkdir(parents=True)
                runtime = library_dir / library
                runtime.touch()

                def find_spec(name):
                    if name != package:
                        raise ModuleNotFoundError(name)
                    return SimpleNamespace(
                        submodule_search_locations=[str(package_dir), str(package_dir)]
                    )

                with (
                    patch.object(storage.importlib.util, "find_spec", side_effect=find_spec),
                ):
                    candidates = storage._candidate_libraries()

                self.assertEqual(candidates.count(str(runtime)), 1)
                self.assertEqual(
                    candidates,
                    [str(runtime)],
                )

    def test_explicit_missing_runtime_fails_closed_without_python_backend(self):
        script = """
import sys
import torch_rs as torch
assert 'torch' not in sys.modules
assert torch.cuda.device_count() == 0
assert torch.cuda.is_available() is False
try:
    torch.zeros((1,), device='cuda:0')
except RuntimeError as error:
    assert 'TORCH_RS_CUDART' in str(error), str(error)
else:
    raise AssertionError('invalid runtime unexpectedly allocated storage')
assert torch.cuda.is_initialized() is False
"""
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                env={**os.environ, "TORCH_RS_CUDART": str(Path(__file__).resolve().parents[1] / "target/missing-cudart.so")})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_runtime_packages_need_no_python_candidates(self):
        with (
            patch.object(storage.importlib.util, "find_spec", side_effect=ModuleNotFoundError),
        ):
            self.assertEqual(
                storage._candidate_libraries(),
                [],
            )


if __name__ == "__main__":
    unittest.main()
