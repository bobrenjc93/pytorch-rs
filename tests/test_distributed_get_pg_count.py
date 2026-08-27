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


FUNCTION_DOC = "Return the number of process groups."


class DistributedGetPgCountTests(unittest.TestCase):
    def test_returns_exact_zero_without_runtime_probes(self):
        function = torch.distributed.get_pg_count
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertFalse(hasattr(distributed_c10d, "_world"))
        self.assertFalse(hasattr(distributed_c10d, "GroupMember"))

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
                    result = function()
                    self.assertIs(type(result), int)
                    self.assertEqual(result, 0)

    def test_zero_is_stable_across_threads_and_grad_modes(self):
        function = torch.distributed.get_pg_count
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = function()
                    middle_grad_state = torch.is_grad_enabled()
                    second = function()
                    results[index] = (
                        type(first),
                        first,
                        middle_grad_state,
                        type(second),
                        second,
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
            self.assertEqual(
                result,
                (
                    int,
                    0,
                    index % 2 == 0,
                    int,
                    0,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.get_pg_count

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.get_pg_count, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> int")
        self.assertEqual(function.__annotations__, {"return": int})
        self.assertEqual(typing.get_type_hints(function), {"return": int})
        self.assertEqual(function.__name__, "get_pg_count")
        self.assertEqual(function.__qualname__, "get_pg_count")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(inspect.getdoc(function), FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_pg_count

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
                "get_backend_config",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            ],
        )

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

        direct_import = {}
        exec("from torch_rs.distributed import get_pg_count", direct_import)
        self.assertIs(direct_import["get_pg_count"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import get_pg_count",
            owner_import,
        )
        self.assertIs(owner_import["get_pg_count"], function)

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
                "get_backend_config",
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
        self.assertIs(distributed_namespace["get_pg_count"], function)
        self.assertIs(
            distributed_namespace["distributed_c10d"], distributed_c10d
        )

        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertEqual(
            {name for name in owner_namespace if not name.startswith("__")},
            {
                "get_backend_config",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            },
        )
        self.assertIs(owner_namespace["get_pg_count"], function)

        self.assertNotIn("distributed", torch.__all__)
        self.assertNotIn("get_pg_count", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("distributed", top_level_namespace)
        self.assertNotIn("get_pg_count", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.distributed.get_pg_count
        cases = (
            (
                lambda: function(None),
                "get_pg_count() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_pg_count() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "get_pg_count() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "get_pg_count() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_existing_queries_are_preserved_and_execution_remains_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_gloo_available(), False)
        self.assertIs(distributed.is_initialized(), False)
        self.assertIs(distributed.is_mpi_available(), False)
        self.assertIs(distributed.is_nccl_available(), False)
        self.assertEqual(
            {name for name in vars(distributed) if not name.startswith("_")},
            {
                "distributed_c10d",
                "get_backend_config",
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
        self.assertEqual(
            {
                name
                for name in vars(distributed_c10d)
                if not name.startswith("_")
            },
            {
                "get_backend_config",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            },
        )
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))
        self.assertFalse(hasattr(torch, "get_pg_count"))

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

function = torch.distributed.get_pg_count
assert function.__code__.co_names == ()
result = function()
assert type(result) is int
assert result == 0
assert torch.distributed.is_available() is False
assert torch.distributed.is_initialized() is False
assert not hasattr(torch.distributed, "init_process_group")
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
