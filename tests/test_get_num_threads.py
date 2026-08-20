import contextlib
import os
import pickle
import subprocess
import sys
import threading
import types
import unittest
from unittest import mock

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


FUNCTION_DOC = """
get_num_threads() -> int

Returns the number of threads used for parallelizing CPU operations
"""


class GetNumThreadsTests(unittest.TestCase):
    def test_returns_exact_one_without_runtime_probes(self):
        function = torch.get_num_threads
        environments = (
            {},
            {"OMP_NUM_THREADS": "64"},
            {"MKL_NUM_THREADS": "32"},
            {
                "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )

        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch(
                        "os.cpu_count",
                        side_effect=AssertionError("CPU hardware was probed"),
                    ):
                        result = function()
                self.assertIs(type(result), int)
                self.assertIs(result, 1)

    def test_one_is_stable_across_threads_and_grad_modes(self):
        function = torch.get_num_threads

        def query():
            before = torch.is_grad_enabled()
            first = function()
            middle = torch.is_grad_enabled()
            second = function()
            after = torch.is_grad_enabled()
            return before, first, middle, second, after

        self.assertEqual(query(), (True, 1, True, 1, True))
        with torch.no_grad():
            self.assertEqual(query(), (False, 1, False, 1, False))
            with torch.no_grad():
                self.assertEqual(query(), (False, 1, False, 1, False))
        self.assertEqual(query(), (True, 1, True, 1, True))

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = query()
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
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    1,
                    expected_grad_state,
                    1,
                    expected_grad_state,
                ),
            )
            self.assertIs(type(result[1]), int)
            self.assertIs(type(result[3]), int)

    def test_builtin_ownership_documentation_exports_and_pickling(self):
        function = torch.get_num_threads
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "get_num_threads")
        self.assertEqual(function.__qualname__, "get_num_threads")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(repr(function), "<built-in function get_num_threads>")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.get_num_threads, function)
        assert_no_argument_signature(self, function, "()")

        self.assertEqual(torch.__all__.count("get_num_threads"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["get_num_threads"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.get_num_threads
        cases = (
            (
                lambda: function(None),
                "torch.get_num_threads() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.get_num_threads() takes no arguments (2 given)",
            ),
            (
                lambda: function(threads=None),
                "torch.get_num_threads() takes no keyword arguments",
            ),
            (
                lambda: function(None, threads=None),
                "torch.get_num_threads() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_mutation_interop_and_parallel_surfaces_remain_unsupported(self):
        unsupported = (
            "set_num_threads",
            "get_num_interop_threads",
            "set_num_interop_threads",
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(torch._C, name))
                self.assertNotIn(name, torch.__all__)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertTrue(set(unsupported).isdisjoint(wildcard_namespace))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(OMP_NUM_THREADS="64", MKL_NUM_THREADS="32")
import torch_rs as torch

result = torch.get_num_threads()
assert type(result) is int
assert result == 1
assert not hasattr(torch, "set_num_threads")
assert not hasattr(torch, "get_num_interop_threads")
assert not hasattr(torch, "set_num_interop_threads")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
