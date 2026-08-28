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


DEFAULT_GROUP_ERROR = (
    "Default process group has not been initialized, please make sure to call "
    "init_process_group."
)
NON_NONE_GROUP_ERROR = (
    "torch_rs.distributed.get_world_size() does not support non-None process "
    "groups"
)
FUNCTION_DOC = """Return the number of processes in the current process group.

Args:
    group (ProcessGroup, optional): The process group to work on. If None,
        the default process group will be used.

Returns:
    The world size of the process group
    -1, if not part of the group"""


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class DistributedGetWorldSizeTests(unittest.TestCase):
    def assert_default_group_error(self, call):
        with self.assertRaises(ValueError) as raised:
            call()
        self.assertEqual(str(raised.exception), DEFAULT_GROUP_ERROR)
        self.assertEqual(raised.exception.args, (DEFAULT_GROUP_ERROR,))

    def test_default_group_forms_repeat_without_runtime_or_environment_probes(self):
        function = torch.distributed.get_world_size
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        self.assertNotIn("_os", function.__code__.co_names)
        self.assertNotIn("environ", function.__code__.co_names)
        self.assertNotIn("is_initialized", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
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

        with mock.patch.object(
            distributed_c10d._os, "environ", UnreadableEnvironment()
        ):
            self.assert_default_group_error(function)
            self.assert_default_group_error(lambda: function(group=None))
            with self.assertRaises(NotImplementedError) as raised:
                function(object())
            self.assertEqual(str(raised.exception), NON_NONE_GROUP_ERROR)

        self.assertIs(torch.distributed.is_initialized(), False)
        self.assertEqual(torch.distributed.get_pg_count(), 0)

    def test_error_is_stable_across_threads_and_grad_modes(self):
        function = torch.distributed.get_world_size
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    before = torch.is_grad_enabled()
                    outcomes = []
                    for call in (function, lambda: function(group=None)):
                        try:
                            call()
                        except BaseException as error:
                            outcomes.append(
                                (type(error), str(error), error.args)
                            )
                    results[index] = (
                        before,
                        tuple(outcomes),
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
        expected_errors = (
            (ValueError, DEFAULT_GROUP_ERROR, (DEFAULT_GROUP_ERROR,)),
            (ValueError, DEFAULT_GROUP_ERROR, (DEFAULT_GROUP_ERROR,)),
        )
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (expected_grad_state, expected_errors, expected_grad_state),
            )

    def test_reload_preserves_the_default_group_contract(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        for _ in range(3):
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)
            function = distributed.get_world_size

            self.assertIs(torch.distributed, distributed)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(distributed_c10d.get_world_size, function)
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
        function = distributed.get_world_size

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.get_world_size, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(group: torch_rs.distributed.distributed_c10d.ProcessGroup | "
            "None = None) -> int",
        )
        self.assertEqual(set(function.__annotations__), {"group", "return"})
        self.assertIs(function.__annotations__["return"], int)
        group_annotation = function.__annotations__["group"]
        process_group, none_type = typing.get_args(group_annotation)
        self.assertEqual(process_group.__name__, "ProcessGroup")
        self.assertEqual(process_group.__qualname__, "ProcessGroup")
        self.assertEqual(
            process_group.__module__,
            "torch_rs.distributed.distributed_c10d",
        )
        self.assertIs(none_type, type(None))
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "get_world_size")
        self.assertEqual(function.__qualname__, "get_world_size")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(inspect.getdoc(function), FUNCTION_DOC)
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(distributed_c10d, "ProcessGroup"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_world_size

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
                "destroy_process_group",
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
                "get_group_rank",
                "get_global_rank",
                "get_process_group_ranks",
                "get_node_local_rank",
            ],
        )

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

        direct_import = {}
        exec("from torch_rs.distributed import get_world_size", direct_import)
        self.assertIs(direct_import["get_world_size"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import get_world_size",
            owner_import,
        )
        self.assertIs(owner_import["get_world_size"], function)

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
                "destroy_process_group",
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
                "get_group_rank",
                "get_global_rank",
                "get_process_group_ranks",
                "get_node_local_rank",
            },
        )
        self.assertIs(distributed_namespace["get_world_size"], function)

        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertEqual(
            {name for name in owner_namespace if not name.startswith("__")},
            set(distributed_c10d.__all__),
        )
        self.assertIs(owner_namespace["get_world_size"], function)

        self.assertNotIn("distributed", torch.__all__)
        self.assertNotIn("get_world_size", torch.__all__)
        self.assertFalse(hasattr(torch, "get_world_size"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("distributed", top_level_namespace)
        self.assertNotIn("get_world_size", top_level_namespace)

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
        function = torch.distributed.get_world_size

        for call in (
            function,
            lambda: function(None),
            lambda: function(group=None),
        ):
            self.assert_default_group_error(call)

        cases = (
            (
                lambda: function(None, None),
                "get_world_size() takes from 0 to 1 positional arguments but "
                "2 were given",
            ),
            (
                lambda: function(enabled=True),
                "get_world_size() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, group=None),
                "get_world_size() got multiple values for argument 'group'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        class FakeProcessGroup:
            @property
            def size(self):
                raise AssertionError("process-group methods must not be read")

        process_group_annotation = typing.get_args(
            function.__annotations__["group"]
        )[0]
        for group in (
            object(),
            False,
            0,
            "",
            FakeProcessGroup(),
            process_group_annotation(),
        ):
            with self.subTest(group=type(group).__name__):
                with self.assertRaises(NotImplementedError) as raised:
                    function(group)
                self.assertEqual(str(raised.exception), NON_NONE_GROUP_ERROR)
                self.assertEqual(raised.exception.args, (NON_NONE_GROUP_ERROR,))

    def test_initialization_and_collectives_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))

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

function = torch.distributed.get_world_size
for call in (function, lambda: function(None), lambda: function(group=None)):
    try:
        call()
    except ValueError as error:
        assert str(error) == {DEFAULT_GROUP_ERROR!r}
        assert error.args == ({DEFAULT_GROUP_ERROR!r},)
    else:
        raise AssertionError("get_world_size did not reject the missing group")
try:
    function(object())
except NotImplementedError as error:
    assert str(error) == {NON_NONE_GROUP_ERROR!r}
else:
    raise AssertionError("get_world_size accepted a non-None group")
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
assert not hasattr(torch.distributed, "ProcessGroup")
assert not hasattr(torch.distributed, "init_process_group")
assert not hasattr(torch.distributed, "all_reduce")
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
