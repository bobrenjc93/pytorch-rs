import importlib
import inspect
import os
import pickle
import subprocess
import sys
import textwrap
import threading
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """


class CudaIsAvailableTests(unittest.TestCase):
    def test_canonical_namespace_is_minimal_and_reports_cpu_backend(self):
        cuda = importlib.import_module("torch_rs.cuda")
        from torch_rs import cuda as package_cuda
        from torch_rs.cuda import is_available

        self.assertIs(torch.cuda, cuda)
        self.assertIs(package_cuda, cuda)
        self.assertIs(is_available, cuda.is_available)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertEqual(cuda.__all__, ["is_available"])
        self.assertNotIn("cuda", torch.__all__)

        self.assertIs(is_available(), False)
        self.assertIs(type(is_available()), bool)
        for unsupported in (
            "current_device",
            "device",
            "device_count",
            "empty_cache",
            "get_device_name",
            "init",
            "is_initialized",
            "manual_seed",
            "set_device",
            "synchronize",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertFalse(hasattr(cuda, unsupported))

    def test_query_does_not_import_torch_probe_libraries_or_mutate_environment(self):
        script = textwrap.dedent(
            """
            import ctypes
            import importlib.abc
            import os
            import sys


            class RejectAcceleratorImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    root = fullname.partition(".")[0]
                    if root in {"torch", "pynvml", "nvidia_smi"}:
                        raise AssertionError(
                            f"unexpected accelerator import: {fullname}"
                        )
                    return None


            def reject_dynamic_probe(*args, **kwargs):
                raise AssertionError(
                    f"unexpected dynamic-library probe: {args!r}"
                )


            sys.meta_path.insert(0, RejectAcceleratorImports())
            ctypes.CDLL = reject_dynamic_probe
            environment = os.environ.copy()

            import torch_rs

            assert torch_rs.cuda.is_available() is False
            assert torch_rs.cuda.is_available() is False
            assert os.environ == environment
            assert not any(
                name == "torch" or name.startswith("torch.")
                for name in sys.modules
            )
            """
        )
        environment = os.environ.copy()
        environment["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_metadata_documentation_signature_and_pickle_match_contract(self):
        cuda = torch.cuda
        function = cuda.is_available

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(function.__dict__, {})
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "() -> bool")

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.cuda", payload)
                restored = pickle.loads(payload)
                self.assertIs(restored, function)
                self.assertIs(restored(), False)

    def test_false_result_is_stable_across_threads(self):
        function = torch.cuda.is_available
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = tuple(function() for _ in range(100))
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [(False,) * 100] * worker_count)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.cuda.is_available
        cases = (
            (
                lambda: function(None),
                "is_available() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_available() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(unexpected=True),
                "is_available() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(None, unexpected=True),
                "is_available() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
