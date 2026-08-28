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


INVALID_GROUP_SUFFIX = (
    " is not registered, please create group with torch.distributed.new_group API"
)
FUNCTION_DOC = """Translate a group rank into a global rank.

``group_rank`` must be part of `group` otherwise this raises RuntimeError.

Args:
    group (ProcessGroup): ProcessGroup to find the global rank from.
    group_rank (int): Group rank to query.

Returns:
    Global rank of ``group_rank`` relative to ``group``

N.B. calling this function on the default process group returns identity"""


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class DistributedGetGlobalRankTests(unittest.TestCase):
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

    def test_default_world_returns_the_supplied_rank_without_coercion(self):
        function = torch.distributed.get_global_rank
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        for name in ("_world", "GroupMember", "ProcessGroup"):
            self.assertFalse(hasattr(distributed_c10d, name))
        for name in (
            "_os",
            "environ",
            "int",
            "is_initialized",
            "get_pg_count",
        ):
            self.assertNotIn(name, function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        class RankProbe:
            def __init__(self):
                self.events = []

            def __bool__(self):
                self.events.append("bool")
                raise AssertionError("rank truth value was read")

            def __eq__(self, other):
                self.events.append(("eq", other))
                raise AssertionError("rank equality was read")

            def __hash__(self):
                self.events.append("hash")
                raise AssertionError("rank hash was read")

            def __index__(self):
                self.events.append("index")
                raise AssertionError("rank index was read")

            def __int__(self):
                self.events.append("int")
                raise AssertionError("rank integer value was read")

            def __repr__(self):
                self.events.append("repr")
                raise AssertionError("rank representation was read")

            def __str__(self):
                self.events.append("str")
                raise AssertionError("rank string was read")

        environments = (
            {},
            {"USE_DISTRIBUTED": "0"},
            {"USE_DISTRIBUTED": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "9",
                "USE_DISTRIBUTED": "unexpected",
                "WORLD_SIZE": "123",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    rank = RankProbe()
                    for call in (
                        lambda: function(None, rank),
                        lambda: function(group=None, group_rank=rank),
                        lambda: function(group_rank=rank, group=None),
                    ):
                        self.assertIs(call(), rank)
                    self.assertEqual(rank.events, [])
                    self.assert_zero_process_group_state()

        with mock.patch.object(
            distributed_c10d._os, "environ", UnreadableEnvironment()
        ):
            ranks = (
                None,
                False,
                True,
                -7,
                3.5,
                "rank",
                object(),
                [],
                {},
                torch.tensor([1.0, 2.0]),
            )
            for rank in ranks:
                with self.subTest(rank_type=type(rank).__name__):
                    self.assertIs(function(None, rank), rank)
        self.assert_zero_process_group_state()

    def test_default_world_identity_is_thread_safe_and_survives_reload(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        ranks = [object() for _ in range(worker_count)]
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = distributed.get_global_rank(None, ranks[index])
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
        for result, rank in zip(results, ranks):
            self.assertIs(result, rank)

        for _ in range(3):
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)
            function = distributed.get_global_rank
            rank = object()
            self.assertIs(torch.distributed, distributed)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(distributed_c10d.get_global_rank, function)
            self.assertIs(function(None, rank), rank)
            self.assert_zero_process_group_state()

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.get_global_rank

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.get_global_rank, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(group: torch_rs.distributed.distributed_c10d.ProcessGroup, "
            "group_rank: int) -> int",
        )
        self.assertEqual(
            set(function.__annotations__), {"group", "group_rank", "return"}
        )
        group_annotation = function.__annotations__["group"]
        self.assertEqual(group_annotation.__name__, "ProcessGroup")
        self.assertEqual(group_annotation.__qualname__, "ProcessGroup")
        self.assertEqual(
            group_annotation.__module__,
            "torch_rs.distributed.distributed_c10d",
        )
        self.assertIs(function.__annotations__["group_rank"], int)
        self.assertIs(function.__annotations__["return"], int)
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "get_global_rank")
        self.assertEqual(function.__qualname__, "get_global_rank")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(inspect.getdoc(function), FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(distributed_c10d, "ProcessGroup"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_global_rank

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
                "get_node_local_rank",
            ],
        )

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

        direct_import = {}
        exec("from torch_rs.distributed import get_global_rank", direct_import)
        self.assertIs(direct_import["get_global_rank"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import get_global_rank",
            owner_import,
        )
        self.assertIs(owner_import["get_global_rank"], function)

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertIs(distributed_namespace["get_global_rank"], function)
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
            set(distributed_c10d.__all__),
        )
        self.assertIs(owner_namespace["get_global_rank"], function)

        self.assertNotIn("get_global_rank", torch.__all__)
        self.assertFalse(hasattr(torch, "get_global_rank"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("get_global_rank", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_and_invalid_group_lookup_order(self):
        function = torch.distributed.get_global_rank

        cases = (
            (
                function,
                "get_global_rank() missing 2 required positional arguments: "
                "'group' and 'group_rank'",
            ),
            (
                lambda: function(None),
                "get_global_rank() missing 1 required positional argument: "
                "'group_rank'",
            ),
            (
                lambda: function(None, 1, 2),
                "get_global_rank() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: function(group=None, rank=1),
                "get_global_rank() got an unexpected keyword argument 'rank'",
            ),
            (
                lambda: function(None, 1, group=None),
                "get_global_rank() got multiple values for argument 'group'",
            ),
            (
                lambda: function(None, 1, group_rank=2),
                "get_global_rank() got multiple values for argument 'group_rank'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

        rank_events = []

        class RankProbe:
            def __bool__(self):
                rank_events.append("bool")
                raise AssertionError("group rank truth value was read")

            def __eq__(self, other):
                rank_events.append(("eq", other))
                raise AssertionError("group rank equality was read")

            def __hash__(self):
                rank_events.append("hash")
                raise AssertionError("group rank hash was read")

            def __index__(self):
                rank_events.append("index")
                raise AssertionError("group rank index was read")

            def __int__(self):
                rank_events.append("int")
                raise AssertionError("group rank integer value was read")

            def __repr__(self):
                rank_events.append("repr")
                raise AssertionError("group rank representation was read")

            def __str__(self):
                rank_events.append("str")
                raise AssertionError("group rank string was read")

        rank = RankProbe()
        for group in (False, True, 0, -100, "", "group"):
            with self.subTest(group=group):
                self.assert_error(
                    ValueError,
                    f"Group {group}{INVALID_GROUP_SUFFIX}",
                    lambda group=group: function(group, rank),
                )

        for group in ([], {}, set(), bytearray()):
            with self.subTest(unhashable=type(group).__name__):
                with self.assertRaises(TypeError) as expected_raised:
                    group in {}
                with self.assertRaises(TypeError) as actual_raised:
                    function(group, rank)
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )
                self.assertEqual(
                    actual_raised.exception.args, expected_raised.exception.args
                )

        events = []

        class TracedGroup:
            def __hash__(self):
                events.append(("hash",))
                return 123

            def __eq__(self, other):
                events.append(("eq", other))
                raise AssertionError("group equality was read")

            def __format__(self, format_spec):
                events.append(("format", format_spec))
                return "traced-group"

        self.assert_error(
            ValueError,
            f"Group traced-group{INVALID_GROUP_SUFFIX}",
            lambda: function(TracedGroup(), rank),
        )
        self.assertEqual(events, [("hash",), ("format", "")])

        events.clear()

        class BrokenHash:
            def __hash__(self):
                events.append(("hash",))
                raise RuntimeError("hash failed")

            def __eq__(self, other):
                events.append(("eq", other))
                raise AssertionError("group equality was read")

            def __format__(self, format_spec):
                events.append(("format", format_spec))
                raise AssertionError("group formatting was read")

        self.assert_error(
            RuntimeError,
            "hash failed",
            lambda: function(BrokenHash(), rank),
        )
        self.assertEqual(events, [("hash",)])

        events.clear()

        class BrokenFormat:
            def __hash__(self):
                events.append(("hash",))
                return 456

            def __eq__(self, other):
                events.append(("eq", other))
                raise AssertionError("group equality was read")

            def __format__(self, format_spec):
                events.append(("format", format_spec))
                raise RuntimeError("format failed")

        self.assert_error(
            RuntimeError,
            "format failed",
            lambda: function(BrokenFormat(), rank),
        )
        self.assertEqual(events, [("hash",), ("format", "")])
        self.assertEqual(rank_events, [])
        self.assert_zero_process_group_state()

    def test_registered_groups_and_distributed_execution_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        process_group_annotation = distributed.get_global_rank.__annotations__[
            "group"
        ]
        group = process_group_annotation()

        with self.assertRaises(ValueError) as raised:
            distributed.get_global_rank(group, 0)
        self.assertTrue(str(raised.exception).startswith("Group <"))
        self.assertTrue(str(raised.exception).endswith(INVALID_GROUP_SUFFIX))

        for name in (
            "GroupMember",
            "ProcessGroup",
            "_world",
            "all_reduce",
            "get_process_group_ranks",
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

class RankProbe:
    def __int__(self):
        raise AssertionError("rank was coerced")

class GroupProbe:
    def __hash__(self):
        return 123

    def __eq__(self, other):
        raise AssertionError("group equality was read")

    def __format__(self, format_spec):
        return "group-probe"

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

function = torch.distributed.get_global_rank
rank = RankProbe()
assert function(None, rank) is rank
assert function(group=None, group_rank=rank) is rank
try:
    function(GroupProbe(), rank)
except ValueError as error:
    expected = "Group group-probe" + {INVALID_GROUP_SUFFIX!r}
    assert str(error) == expected
    assert error.args == (expected,)
else:
    raise AssertionError("get_global_rank accepted an unregistered group")
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
