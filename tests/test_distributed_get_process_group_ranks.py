import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


DEFAULT_GROUP_ERROR = (
    "Default process group has not been initialized, please make sure to call "
    "init_process_group."
)
EMPTY_TENSOR_ERROR = "Boolean value of Tensor with no values is ambiguous"
MULTI_TENSOR_ERROR = "Boolean value of Tensor with more than one value is ambiguous"
FUNCTION_DOC = """Get all ranks associated with ``group``.

Args:
    group (Optional[ProcessGroup]): ProcessGroup to get all ranks from.
        If None, the default process group will be used.

Returns:
    List of global ranks ordered by group rank."""


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class DistributedGetProcessGroupRanksTests(unittest.TestCase):
    def assert_error(self, error_type, message, call):
        with self.assertRaises(error_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

    def assert_zero_process_group_state(self):
        self.assertIs(torch.distributed.is_initialized(), False)
        count = torch.distributed.get_pg_count()
        self.assertIs(type(count), int)
        self.assertEqual(count, 0)

    def test_falsey_groups_raise_the_default_group_error_without_host_probes(self):
        function = torch.distributed.get_process_group_ranks
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        for name in ("_world", "GroupMember", "ProcessGroup"):
            self.assertFalse(hasattr(distributed_c10d, name))
        for name in (
            "_os",
            "environ",
            "is_initialized",
            "get_pg_count",
            "accelerator",
            "cuda",
            "device_count",
        ):
            self.assertNotIn(name, function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        falsey_groups = (
            None,
            False,
            0,
            0.0,
            0j,
            "",
            b"",
            bytearray(),
            (),
            [],
            {},
            set(),
            range(0),
        )
        with (
            mock.patch.object(distributed_c10d._os, "environ", UnreadableEnvironment()),
            mock.patch.object(
                torch.accelerator,
                "_discover_accelerator",
                side_effect=AssertionError("accelerator probe was attempted"),
            ),
            mock.patch.object(
                torch.cpu,
                "current_device",
                side_effect=AssertionError("CPU device probe was attempted"),
            ),
        ):
            for group in falsey_groups:
                with self.subTest(group_type=type(group).__name__):
                    self.assert_error(
                        ValueError,
                        DEFAULT_GROUP_ERROR,
                        lambda group=group: function(group),
                    )
                    self.assert_error(
                        ValueError,
                        DEFAULT_GROUP_ERROR,
                        lambda group=group: function(group=group),
                    )
                    self.assert_zero_process_group_state()

            events = []

            class FalseyGroup:
                def __bool__(self):
                    events.append("bool")
                    return False

                def __hash__(self):
                    events.append("hash")
                    raise AssertionError("falsey group was hashed")

                def __repr__(self):
                    events.append("repr")
                    raise AssertionError("falsey group was represented")

            self.assert_error(
                ValueError,
                DEFAULT_GROUP_ERROR,
                lambda: function(FalseyGroup()),
            )
            self.assertEqual(events, ["bool"])

            events.clear()

            class EmptyGroup:
                def __len__(self):
                    events.append("len")
                    return 0

                def __hash__(self):
                    events.append("hash")
                    raise AssertionError("empty group was hashed")

            self.assert_error(
                ValueError,
                DEFAULT_GROUP_ERROR,
                lambda: function(EmptyGroup()),
            )
            self.assertEqual(events, ["len"])

        for values in (0.0, [0.0], [[0.0]]):
            with self.subTest(tensor_values=values):
                self.assert_error(
                    ValueError,
                    DEFAULT_GROUP_ERROR,
                    lambda values=values: function(torch.tensor(values)),
                )
        self.assert_zero_process_group_state()

    def test_truthy_groups_use_an_empty_registry_with_python_lookup_errors(self):
        function = torch.distributed.get_process_group_ranks

        for group in (True, 1, -100, 1.5, "group", b"group", (1,), object()):
            with self.subTest(group_type=type(group).__name__):
                with self.assertRaises(KeyError) as raised:
                    function(group)
                self.assertEqual(len(raised.exception.args), 1)
                self.assertIs(raised.exception.args[0], group)

        for group in ([1], {"rank": 0}, {1}, bytearray(b"group")):
            with self.subTest(group_type=type(group).__name__):
                with self.assertRaises(TypeError) as expected_raised:
                    {}[group]
                with self.assertRaises(TypeError) as actual_raised:
                    function(group)
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )
                self.assertEqual(
                    actual_raised.exception.args, expected_raised.exception.args
                )

        events = []

        class TracedGroup:
            def __bool__(self):
                events.append("bool")
                return True

            def __hash__(self):
                events.append("hash")
                return 123

            def __eq__(self, other):
                events.append(("eq", other))
                raise AssertionError("group equality was read")

            def __repr__(self):
                events.append("repr")
                return "traced-group"

        group = TracedGroup()
        with self.assertRaises(KeyError) as raised:
            function(group)
        self.assertIs(raised.exception.args[0], group)
        self.assertEqual(events, ["bool", "hash"])
        self.assertEqual(str(raised.exception), "traced-group")
        self.assertEqual(events, ["bool", "hash", "repr"])

        events.clear()

        class BrokenHash:
            def __bool__(self):
                events.append("bool")
                return True

            def __hash__(self):
                events.append("hash")
                raise RuntimeError("hash failed")

        self.assert_error(
            RuntimeError,
            "hash failed",
            lambda: function(BrokenHash()),
        )
        self.assertEqual(events, ["bool", "hash"])

        events.clear()

        class BrokenTruth:
            def __bool__(self):
                events.append("bool")
                raise RuntimeError("truth failed")

            def __hash__(self):
                events.append("hash")
                raise AssertionError("group was hashed after truth failed")

        self.assert_error(
            RuntimeError,
            "truth failed",
            lambda: function(BrokenTruth()),
        )
        self.assertEqual(events, ["bool"])

        true_tensor = torch.tensor(1.0)
        with self.assertRaises(KeyError) as raised:
            function(true_tensor)
        self.assertIs(raised.exception.args[0], true_tensor)
        self.assert_error(
            RuntimeError,
            EMPTY_TENSOR_ERROR,
            lambda: function(torch.tensor([])),
        )
        self.assert_error(
            RuntimeError,
            MULTI_TENSOR_ERROR,
            lambda: function(torch.tensor([1.0, 2.0])),
        )
        self.assert_zero_process_group_state()

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.get_process_group_ranks

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.get_process_group_ranks, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(group: torch_rs.distributed.distributed_c10d.ProcessGroup | "
            "None) -> list[int]",
        )
        self.assertEqual(set(function.__annotations__), {"group", "return"})
        process_group, none_type = typing.get_args(function.__annotations__["group"])
        self.assertEqual(process_group.__name__, "ProcessGroup")
        self.assertEqual(process_group.__qualname__, "ProcessGroup")
        self.assertEqual(
            process_group.__module__,
            "torch_rs.distributed.distributed_c10d",
        )
        self.assertIs(none_type, type(None))
        return_annotation = function.__annotations__["return"]
        self.assertIs(typing.get_origin(return_annotation), list)
        self.assertEqual(typing.get_args(return_annotation), (int,))
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "get_process_group_ranks")
        self.assertEqual(function.__qualname__, "get_process_group_ranks")
        self.assertEqual(function.__module__, "torch_rs.distributed.distributed_c10d")
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(inspect.getdoc(function), FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(distributed_c10d, "ProcessGroup"))

    def test_argument_errors_imports_wildcards_copy_pickle_and_reload(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_process_group_ranks

        cases = (
            (
                function,
                "get_process_group_ranks() missing 1 required positional "
                "argument: 'group'",
            ),
            (
                lambda: function(None, True),
                "get_process_group_ranks() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function(process_group=None),
                "get_process_group_ranks() got an unexpected keyword argument "
                "'process_group'",
            ),
            (
                lambda: function(None, group=None),
                "get_process_group_ranks() got multiple values for argument " "'group'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

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
                "get_process_group_ranks",
                "get_node_local_rank",
            ],
        )

        direct_import = {}
        exec(
            "from torch_rs.distributed import get_process_group_ranks",
            direct_import,
        )
        self.assertIs(direct_import["get_process_group_ranks"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "get_process_group_ranks",
            owner_import,
        )
        self.assertIs(owner_import["get_process_group_ranks"], function)

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertIs(distributed_namespace["get_process_group_ranks"], function)
        self.assertIs(distributed_namespace["distributed_c10d"], distributed_c10d)

        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertEqual(
            {name for name in owner_namespace if not name.startswith("__")},
            set(distributed_c10d.__all__),
        )
        self.assertIs(owner_namespace["get_process_group_ranks"], function)

        self.assertNotIn("get_process_group_ranks", torch.__all__)
        self.assertFalse(hasattr(torch, "get_process_group_ranks"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("get_process_group_ranks", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.distributed.distributed_c10d", payload)
                self.assertIs(pickle.loads(payload), function)

        for _ in range(3):
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)
            function = distributed.get_process_group_ranks
            self.assertIs(torch.distributed, distributed)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(distributed_c10d.get_process_group_ranks, function)
            self.assert_error(
                ValueError,
                DEFAULT_GROUP_ERROR,
                lambda: function(None),
            )
            self.assert_zero_process_group_state()

    def test_registered_groups_and_distributed_execution_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        process_group_annotation, _ = typing.get_args(
            distributed.get_process_group_ranks.__annotations__["group"]
        )
        group = process_group_annotation()

        with self.assertRaises(KeyError) as raised:
            distributed.get_process_group_ranks(group)
        self.assertIs(raised.exception.args[0], group)

        for name in (
            "GroupMember",
            "ProcessGroup",
            "_world",
            "all_reduce",
            "get_global_rank",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))
        self.assert_zero_process_group_state()

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = rf"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {{fullname}}")
        return None

def reject_probe(*args, **kwargs):
    raise AssertionError("device probe was attempted")

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

torch.accelerator._discover_accelerator = reject_probe
torch.cpu.current_device = reject_probe
function = torch.distributed.get_process_group_ranks
for group in (None, False, 0, "", [], {{}}):
    try:
        function(group)
    except ValueError as error:
        assert str(error) == {DEFAULT_GROUP_ERROR!r}
        assert error.args == ({DEFAULT_GROUP_ERROR!r},)
    else:
        raise AssertionError("falsey group did not raise")
for group in (True, "group", (1,)):
    try:
        function(group)
    except KeyError as error:
        assert error.args == (group,)
    else:
        raise AssertionError("unregistered group did not raise")
group = [1]
try:
    {{}}[group]
except TypeError as error:
    expected_unhashable_error = type(error), str(error), error.args
else:
    raise AssertionError("dict lookup accepted an unhashable group")
try:
    function(group)
except TypeError as error:
    assert (type(error), str(error), error.args) == expected_unhashable_error
else:
    raise AssertionError("unhashable group did not raise")
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
assert not hasattr(torch.distributed, "ProcessGroup")
assert not hasattr(torch.distributed, "new_group")
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
