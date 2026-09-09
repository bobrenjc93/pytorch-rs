import tempfile
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
                    patch.object(storage.ctypes.util, "find_library", return_value=None),
                    patch.object(storage.importlib.util, "find_spec", side_effect=find_spec),
                ):
                    candidates = storage._candidate_libraries()

                self.assertEqual(candidates.count(str(runtime)), 1)
                self.assertEqual(
                    candidates,
                    ["libcudart.so.13", "libcudart.so.12", "libcudart.so", str(runtime)],
                )

    def test_missing_runtime_packages_leave_system_candidates_available(self):
        with (
            patch.object(
                storage.ctypes.util, "find_library", return_value="libcudart.so.13"
            ),
            patch.object(storage.importlib.util, "find_spec", side_effect=ModuleNotFoundError),
        ):
            self.assertEqual(
                storage._candidate_libraries(),
                ["libcudart.so.13", "libcudart.so.12", "libcudart.so"],
            )


if __name__ == "__main__":
    unittest.main()
