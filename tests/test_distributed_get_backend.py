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


DEFAULT_GROUP_ERROR = (
    "Default process group has not been initialized, please make sure to call "
    "init_process_group."
)
NON_NONE_GROUP_ERROR = (
    "torch_rs.distributed.get_backend() does not support non-None process groups"
)
FUNCTION_DOC = """Return the backend of the given process group.

Args:
    group (ProcessGroup, optional): The process group to work on. The
        default is the general main process group. If another specific group
        is specified, the calling process must be part of :attr:`group`.

Returns:
    The backend of the given process group as a lower case string."""
C10D_EXPORTS = [
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
]
DISTRIBUTED_PUBLIC = {
    "distributed_c10d",
    "get_backend",
    "get_rank",
    "get_world_size",
    "get_pg_count",
    "get_node_local_rank",
    "is_available",
    "is_gloo_available",
    "is_initialized",
    "is_mpi_available",
    "is_nccl_available",
    "is_ucc_available",
    "is_xccl_available",
}


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class UnprobeableDeviceAPI:
    def __getattribute__(self, name):
        raise AssertionError(f"device API was probed: {name}")


class UnprobeableProcessGroup:
    def __getattribute__(self, name):
        raise AssertionError(f"process group was probed: {name}")


class DistributedGetBackendTests(unittest.TestCase):
    def error_outcome(self, call):
        try:
            call()
        except BaseException as error:
            return type(error), str(error), error.args
        self.fail("expected the call to raise")

    def assert_default_group_error(self, call):
        self.assertEqual(
            self.error_outcome(call),
            (ValueError, DEFAULT_GROUP_ERROR, (DEFAULT_GROUP_ERROR,)),
        )

    def test_default_group_forms_repeat_without_runtime_or_environment_probes(self):
        function = torch.distributed.get_backend
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        self.assertTrue(
            {
                "_os",
                "environ",
                "is_initialized",
                "accelerator",
                "cuda",
                "device",
            }.isdisjoint(function.__code__.co_names)
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertFalse(hasattr(distributed_c10d, "Backend"))
        self.assertFalse(hasattr(distributed_c10d, "GroupMember"))
        self.assertFalse(hasattr(distributed_c10d, "ProcessGroup"))

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
                "WORLD_SIZE": "123",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for _ in range(3):
                        self.assert_default_group_error(function)
                        self.assert_default_group_error(lambda: function(None))
                        self.assert_default_group_error(
                            lambda: function(group=None)
                        )

        with (
            mock.patch.object(
                distributed_c10d, "is_initialized", side_effect=AssertionError
            ),
            mock.patch.object(
                distributed_c10d._os, "environ", UnreadableEnvironment()
            ),
            mock.patch.object(torch, "accelerator", UnprobeableDeviceAPI()),
        ):
            self.assert_default_group_error(function)
            self.assert_default_group_error(lambda: function(group=None))
            with self.assertRaises(NotImplementedError) as raised:
                function(UnprobeableProcessGroup())
            self.assertEqual(str(raised.exception), NON_NONE_GROUP_ERROR)

        self.assertIs(torch.distributed.is_initialized(), False)
        self.assertEqual(torch.distributed.get_pg_count(), 0)

    def test_error_is_stable_across_threads(self):
        function = torch.distributed.get_backend
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                outcomes = []
                for _ in range(3):
                    outcomes.append(self.error_outcome(function))
                    outcomes.append(
                        self.error_outcome(lambda: function(group=None))
                    )
                results[index] = tuple(outcomes)
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
        expected = (ValueError, DEFAULT_GROUP_ERROR, (DEFAULT_GROUP_ERROR,))
        self.assertEqual(results, [(expected,) * 6] * worker_count)

    def test_reload_preserves_the_default_group_contract(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        for _ in range(3):
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)
            function = distributed.get_backend

            self.assertIs(torch.distributed, distributed)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(distributed_c10d.get_backend, function)
            self.assert_default_group_error(function)
            self.assert_default_group_error(lambda: function(None))
            self.assert_default_group_error(lambda: function(group=None))
            self.assertIs(distributed.is_initialized(), False)
            self.assertEqual(distributed.get_pg_count(), 0)

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.get_backend

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.get_backend, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(group: torch_rs.distributed.distributed_c10d.ProcessGroup | "
            "None = None) -> torch_rs.distributed.distributed_c10d.Backend",
        )
        self.assertEqual(set(function.__annotations__), {"group", "return"})
        group_annotation = function.__annotations__["group"]
        process_group, none_type = typing.get_args(group_annotation)
        self.assertEqual(
            (
                process_group.__module__,
                process_group.__name__,
                process_group.__qualname__,
                none_type,
            ),
            (
                "torch_rs.distributed.distributed_c10d",
                "ProcessGroup",
                "ProcessGroup",
                type(None),
            ),
        )
        backend = function.__annotations__["return"]
        self.assertEqual(
            (backend.__module__, backend.__name__, backend.__qualname__),
            (
                "torch_rs.distributed.distributed_c10d",
                "Backend",
                "Backend",
            ),
        )
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "get_backend")
        self.assertEqual(function.__qualname__, "get_backend")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(inspect.getdoc(function), FUNCTION_DOC)
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(distributed_c10d, "Backend"))
        self.assertFalse(hasattr(distributed_c10d, "ProcessGroup"))

    def test_imports_wildcards_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_backend

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(distributed_c10d.__all__, C10D_EXPORTS)
        self.assertEqual(
            {
                name for name in vars(distributed) if not name.startswith("_")
            },
            DISTRIBUTED_PUBLIC,
        )
        self.assertEqual(
            {
                name
                for name in vars(distributed_c10d)
                if not name.startswith("_")
            },
            set(C10D_EXPORTS),
        )

        direct_import = {}
        exec("from torch_rs.distributed import get_backend", direct_import)
        self.assertIs(direct_import["get_backend"], function)
        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import get_backend",
            owner_import,
        )
        self.assertIs(owner_import["get_backend"], function)

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertEqual(
            {
                name
                for name in distributed_namespace
                if not name.startswith("__")
            },
            DISTRIBUTED_PUBLIC,
        )
        self.assertIs(distributed_namespace["get_backend"], function)
        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertEqual(
            {name for name in owner_namespace if not name.startswith("__")},
            set(C10D_EXPORTS),
        )
        self.assertIs(owner_namespace["get_backend"], function)

        self.assertNotIn("get_backend", torch.__all__)
        self.assertFalse(hasattr(torch, "get_backend"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("get_backend", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_and_non_none_groups_are_explicitly_unsupported(self):
        function = torch.distributed.get_backend

        cases = (
            (
                lambda: function(None, None),
                "get_backend() takes from 0 to 1 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "get_backend() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, group=None),
                "get_backend() got multiple values for argument 'group'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    self.error_outcome(call),
                    (TypeError, message, (message,)),
                )

        process_group_annotation = typing.get_args(
            function.__annotations__["group"]
        )[0]
        for group in (
            object(),
            False,
            0,
            "",
            UnprobeableProcessGroup(),
            process_group_annotation(),
        ):
            with self.subTest(group=type(group).__name__):
                with self.assertRaises(NotImplementedError) as raised:
                    function(group)
                self.assertEqual(str(raised.exception), NON_NONE_GROUP_ERROR)
                self.assertEqual(raised.exception.args, (NON_NONE_GROUP_ERROR,))

        self.assertIs(torch.distributed.is_initialized(), False)
        self.assertEqual(torch.distributed.get_pg_count(), 0)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = rf"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {{fullname}}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    USE_DISTRIBUTED="1",
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29500",
    RANK="9",
    WORLD_SIZE="123",
    CUDA_VISIBLE_DEVICES="0",
)
import torch_rs as torch

function = torch.distributed.get_backend
for call in (function, lambda: function(None), lambda: function(group=None)):
    try:
        call()
    except ValueError as error:
        assert str(error) == {DEFAULT_GROUP_ERROR!r}
        assert error.args == ({DEFAULT_GROUP_ERROR!r},)
    else:
        raise AssertionError("get_backend did not reject the missing group")
try:
    function(object())
except NotImplementedError as error:
    assert str(error) == {NON_NONE_GROUP_ERROR!r}
else:
    raise AssertionError("get_backend accepted a non-None group")
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
assert not hasattr(torch.distributed, "Backend")
assert not hasattr(torch.distributed, "ProcessGroup")
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
