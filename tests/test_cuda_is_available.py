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


FUNCTION_DOC = """
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
if sys.version_info >= (3, 13):
    FUNCTION_DOC = textwrap.dedent(FUNCTION_DOC)


class CudaIsAvailableTests(unittest.TestCase):
    def test_canonical_cuda_namespace_is_minimal_and_cpu_only(self):
        cuda = importlib.import_module("torch_rs.cuda")
        from torch_rs.cuda import is_available

        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertEqual(cuda.__name__, "torch_rs.cuda")
        self.assertEqual(cuda.__package__, "torch_rs.cuda")
        self.assertEqual(cuda.__spec__.name, "torch_rs.cuda")
        self.assertIs(cuda.is_available, is_available)
        self.assertIs(cuda.is_available(), False)
        self.assertIs(type(cuda.is_available()), bool)
        self.assertEqual(cuda.__all__, ["is_available"])
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {"is_available"},
        )
        self.assertFalse(hasattr(cuda, "__getattr__"))

        cuda_wildcard = {}
        exec("from torch_rs.cuda import *", cuda_wildcard)
        self.assertIs(cuda_wildcard["is_available"], cuda.is_available)
        self.assertEqual(
            {name for name in cuda_wildcard if not name.startswith("_")},
            {"is_available"},
        )

        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("cuda", top_level_wildcard)
        self.assertNotIn("cuda", torch.__all__)

        tensor = torch.tensor([1.0])
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.is_cuda, False)

    def test_callable_signature_annotations_documentation_and_metadata(self):
        cuda = torch.cuda
        function = cuda.is_available

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(inspect.get_annotations(function), {"return": bool})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertIs(inspect.signature(function).return_annotation, bool)

    def test_pickling_restores_the_canonical_function_for_every_protocol(self):
        function = torch.cuda.is_available

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.cuda", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_result_is_stable_across_threads(self):
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
        for result in results:
            self.assertEqual(result, (False,) * 100)
            self.assertTrue(all(type(value) is bool for value in result))

    def test_query_imports_no_accelerator_stack_and_has_no_side_effects(self):
        script = textwrap.dedent(
            r"""
            import builtins
            import ctypes
            import ctypes.util
            import os
            import pathlib
            import sys

            blocked_roots = {"amdsmi", "pynvml", "torch"}
            original_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if level == 0 and name.partition(".")[0] in blocked_roots:
                    raise AssertionError(f"unexpected accelerator import: {name}")
                return original_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import
            import torch_rs

            assert "torch" not in sys.modules
            assert "pynvml" not in sys.modules
            assert "amdsmi" not in sys.modules

            def accelerator_mappings():
                try:
                    mappings = pathlib.Path("/proc/self/maps").read_text().splitlines()
                except OSError:
                    return ()
                markers = ("libcuda", "libnvidia-ml", "libamd_smi")
                return tuple(
                    line for line in mappings
                    if any(marker in line.lower() for marker in markers)
                )

            before_environment = os.environ.copy()
            before_modules = set(sys.modules)
            before_mappings = accelerator_mappings()
            audit_events = []

            def audit_hook(event, args):
                if event == "import" or event.startswith("ctypes."):
                    audit_events.append(event)

            sys.addaudithook(audit_hook)

            def forbidden_probe(*args, **kwargs):
                raise AssertionError("unexpected accelerator library probe")

            ctypes.CDLL = forbidden_probe
            ctypes.PyDLL = forbidden_probe
            ctypes.cdll.LoadLibrary = forbidden_probe
            ctypes.util.find_library = forbidden_probe

            result = torch_rs.cuda.is_available()
            assert result is False
            assert type(result) is bool
            assert os.environ == before_environment
            assert set(sys.modules) == before_modules
            assert accelerator_mappings() == before_mappings
            assert audit_events == []
            """
        )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

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
                lambda: function(device=0),
                "is_available() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: function(check_nvml=True),
                "is_available() got an unexpected keyword argument 'check_nvml'",
            ),
            (
                lambda: function(None, device=0),
                "is_available() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
