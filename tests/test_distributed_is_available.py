import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """


class DistributedIsAvailableTests(unittest.TestCase):
    def test_returns_exact_false_without_runtime_probes(self):
        function = torch.distributed.is_available

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_DISTRIBUTED": "0"},
            {"USE_DISTRIBUTED": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "USE_DISTRIBUTED": "unexpected",
                "WORLD_SIZE": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(function(), False)

    def test_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.distributed.is_available
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                    )
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
                    False,
                    expected_grad_state,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[3], False)

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        function = distributed.is_available

        self.assertIs(torch.distributed, distributed)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.distributed")
        self.assertIs(inspect.getmodule(function), distributed)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        function = distributed.is_available

        self.assertFalse(hasattr(distributed, "__all__"))

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

        direct_import = {}
        exec("from torch_rs.distributed import is_available", direct_import)
        self.assertIs(direct_import["is_available"], function)

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertEqual(
            {
                name
                for name in distributed_namespace
                if not name.startswith("__")
            },
            {
                "distributed_c10d",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "is_available",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            },
        )
        self.assertIs(distributed_namespace["is_available"], function)

        self.assertNotIn("distributed", torch.__all__)
        self.assertNotIn("get_pg_count", torch.__all__)
        self.assertNotIn("is_available", torch.__all__)
        self.assertNotIn("is_gloo_available", torch.__all__)
        self.assertNotIn("is_initialized", torch.__all__)
        self.assertNotIn("is_mpi_available", torch.__all__)
        self.assertNotIn("is_nccl_available", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("distributed", top_level_namespace)
        self.assertNotIn("get_pg_count", top_level_namespace)
        self.assertNotIn("is_available", top_level_namespace)
        self.assertNotIn("is_gloo_available", top_level_namespace)
        self.assertNotIn("is_initialized", top_level_namespace)
        self.assertNotIn("is_mpi_available", top_level_namespace)
        self.assertNotIn("is_nccl_available", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.distributed", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.distributed.is_available
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
                lambda: function(enabled=True),
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_every_other_distributed_api_remains_unsupported(self):
        distributed = torch.distributed

        self.assertEqual(
            {name for name in vars(distributed) if not name.startswith("_")},
            {
                "distributed_c10d",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "is_available",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            },
        )
        self.assertFalse(hasattr(torch, "get_pg_count"))
        self.assertFalse(hasattr(torch, "is_available"))
        self.assertFalse(hasattr(torch, "is_gloo_available"))
        self.assertFalse(hasattr(torch, "is_initialized"))
        self.assertFalse(hasattr(torch, "is_mpi_available"))
        self.assertFalse(hasattr(torch, "is_nccl_available"))

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
os.environ.update(
    USE_DISTRIBUTED="1",
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29500",
    RANK="0",
    WORLD_SIZE="1",
    CUDA_VISIBLE_DEVICES="0",
)
import torch_rs as torch

function = torch.distributed.is_available
assert function.__code__.co_names == ()
assert function() is False
assert torch.distributed.get_pg_count() == 0
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
