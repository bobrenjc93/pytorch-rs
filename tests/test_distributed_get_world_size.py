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


ERROR_MESSAGE = (
    "Default process group has not been initialized, please make sure to call "
    "init_process_group."
)
UNSUPPORTED_MESSAGE = (
    "torch_rs.distributed.get_world_size does not support non-None process groups"
)
FUNCTION_DOC = """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group
        -1, if not part of the group

    """


class DistributedGetWorldSizeTests(unittest.TestCase):
    def assert_default_group_error(self, call):
        with self.assertRaises(ValueError) as raised:
            call()
        self.assertIs(type(raised.exception), ValueError)
        self.assertEqual(str(raised.exception), ERROR_MESSAGE)
        self.assertEqual(raised.exception.args, (ERROR_MESSAGE,))
        return raised.exception

    def test_default_group_errors_without_environment_or_runtime_probes(self):
        function = torch.distributed.get_world_size
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        self.assertNotIn("_os", function.__code__.co_names)
        self.assertNotIn("environ", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertFalse(hasattr(distributed_c10d, "_world"))
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
                "WORLD_SIZE": "64",
            },
        )
        calls = (
            lambda: function(),
            lambda: function(None),
            lambda: function(group=None),
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    errors = [self.assert_default_group_error(call) for call in calls]
                    self.assertEqual(len({id(error) for error in errors}), len(errors))
                    self.assertIs(torch.distributed.is_initialized(), False)
                    self.assertEqual(torch.distributed.get_pg_count(), 0)

    def test_repeated_threaded_and_grad_mode_calls_are_stable(self):
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
                    grad_state = torch.is_grad_enabled()
                    outcomes = []
                    exceptions = []
                    for args, kwargs in (
                        ((), {}),
                        ((None,), {}),
                        ((), {"group": None}),
                        ((), {}),
                    ):
                        try:
                            function(*args, **kwargs)
                        except BaseException as error:
                            exceptions.append(error)
                            outcomes.append(
                                (type(error), str(error), error.args)
                            )
                        else:
                            outcomes.append((None, None, None))
                    results[index] = (
                        grad_state,
                        torch.is_grad_enabled(),
                        outcomes,
                        len({id(error) for error in exceptions}),
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
        expected_outcomes = [
            (ValueError, ERROR_MESSAGE, (ERROR_MESSAGE,))
        ] * 4
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    expected_grad_state,
                    expected_outcomes,
                    4,
                ),
            )

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
        group_annotation = function.__annotations__["group"]
        group_type, none_type = typing.get_args(group_annotation)
        self.assertIs(none_type, type(None))
        self.assertEqual(group_type.__name__, "ProcessGroup")
        self.assertEqual(group_type.__qualname__, "ProcessGroup")
        self.assertEqual(
            group_type.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(function.__annotations__["return"], int)
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "get_world_size")
        self.assertEqual(function.__qualname__, "get_world_size")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_world_size

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
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

        self.assertNotIn("get_world_size", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
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

    def test_argument_binding_and_non_none_groups_are_explicit(self):
        function = torch.distributed.get_world_size
        binding_cases = (
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
        for call, message in binding_cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        for group in (object(), 0, False, "world", mock.sentinel.process_group):
            for call in (
                lambda group=group: function(group),
                lambda group=group: function(group=group),
            ):
                with self.subTest(group=group, call=call):
                    with self.assertRaises(NotImplementedError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), UNSUPPORTED_MESSAGE)
                    self.assertEqual(
                        raised.exception.args, (UNSUPPORTED_MESSAGE,)
                    )

    def test_reload_keeps_the_default_state_and_refreshes_package_alias(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        original = distributed.get_world_size

        self.assert_default_group_error(original)
        self.assertIs(importlib.reload(distributed_c10d), distributed_c10d)
        reloaded = distributed_c10d.get_world_size
        self.assertIsNot(reloaded, original)
        self.assert_default_group_error(original)
        self.assert_default_group_error(reloaded)

        self.assertIs(importlib.reload(distributed), distributed)
        self.assertIs(distributed.get_world_size, reloaded)
        self.assertIs(torch.distributed.get_world_size, reloaded)
        self.assert_default_group_error(distributed.get_world_size)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)

    def test_initialization_collectives_and_top_level_alias_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertEqual(
            {name for name in vars(distributed) if not name.startswith("_")},
            {
                "distributed_c10d",
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
            set(distributed_c10d.__all__),
        )
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "get_rank",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))
        self.assertFalse(hasattr(torch, "get_world_size"))

    def test_importing_and_calling_does_not_import_pytorch_or_read_environment(self):
        script = r"""
import importlib
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class RejectEnvironment:
    def __getattribute__(self, name):
        raise AssertionError(f"environment access was attempted: {name}")

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

distributed_c10d = importlib.import_module("torch_rs.distributed.distributed_c10d")
distributed_c10d._os.environ = RejectEnvironment()
function = torch.distributed.get_world_size
for args, kwargs in (((), {}), ((None,), {}), ((), {"group": None})):
    try:
        function(*args, **kwargs)
    except ValueError as error:
        assert str(error) == (
            "Default process group has not been initialized, please make sure to "
            "call init_process_group."
        )
    else:
        raise AssertionError("get_world_size unexpectedly returned")
try:
    function(object())
except NotImplementedError:
    pass
else:
    raise AssertionError("a non-None process group was accepted")
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
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
