import contextlib
import copy
import importlib
import inspect
import pickle
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


DEFAULT_GROUP_ERROR = (
    "Default process group has not been initialized, please make sure to "
    "call init_process_group."
)
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

    def test_falsey_groups_use_the_uninitialized_default_without_probes(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_process_group_ranks

        self.assertNotIn("_os", function.__code__.co_names)
        self.assertNotIn("environ", function.__code__.co_names)
        self.assertNotIn("is_initialized", function.__code__.co_names)
        self.assertNotIn("get_pg_count", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        falsey_groups = (
            None,
            False,
            0,
            0.0,
            0j,
            "",
            (),
            [],
            {},
            set(),
            bytearray(),
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    distributed_c10d._os,
                    "environ",
                    UnreadableEnvironment(),
                )
            )
            for owner, name in (
                (distributed, "is_initialized"),
                (distributed, "get_pg_count"),
                (torch.accelerator, "is_available"),
                (torch.accelerator, "current_accelerator"),
                (torch.accelerator, "device_count"),
                (torch.cpu, "current_device"),
                (torch.cpu, "device_count"),
            ):
                stack.enter_context(
                    mock.patch.object(
                        owner,
                        name,
                        side_effect=AssertionError(f"probe was called: {name}"),
                    )
                )
            for group in falsey_groups:
                with self.subTest(group_type=type(group).__name__):
                    self.assert_error(
                        ValueError,
                        DEFAULT_GROUP_ERROR,
                        lambda group=group: function(group),
                    )

        events = []

        class FalseyGroup:
            def __bool__(self):
                events.append("bool")
                return False

            def __hash__(self):
                events.append("hash")
                raise AssertionError("falsey group was hashed")

        self.assert_error(
            ValueError,
            DEFAULT_GROUP_ERROR,
            lambda: function(FalseyGroup()),
        )
        self.assertEqual(events, ["bool"])
        self.assert_zero_process_group_state()

    def test_truthy_groups_use_empty_registry_lookup_semantics(self):
        function = torch.distributed.get_process_group_ranks

        for group in (True, 1, -100, "group", (1,), b"group"):
            with self.subTest(hashable=repr(group)):
                with self.assertRaises(KeyError) as expected_raised:
                    {}[group]
                with self.assertRaises(KeyError) as actual_raised:
                    function(group)
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )
                self.assertEqual(
                    actual_raised.exception.args, expected_raised.exception.args
                )

        for group in ([1], {"group": 1}, {1}, bytearray(b"group")):
            with self.subTest(unhashable=type(group).__name__):
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

        class TruthyGroup:
            def __bool__(self):
                events.append("bool")
                return True

            def __hash__(self):
                events.append("hash")
                return 123

            def __eq__(self, other):
                events.append(("eq", other))
                raise AssertionError("empty registry compared a key")

        group = TruthyGroup()
        with self.assertRaises(KeyError) as raised:
            function(group)
        self.assertIs(raised.exception.args[0], group)
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
        self.assert_zero_process_group_state()

    def test_signature_metadata_and_exports(self):
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
        process_group, none_type = typing.get_args(
            function.__annotations__["group"]
        )
        self.assertEqual(process_group.__name__, "ProcessGroup")
        self.assertEqual(process_group.__qualname__, "ProcessGroup")
        self.assertEqual(
            process_group.__module__,
            "torch_rs.distributed.distributed_c10d",
        )
        self.assertIs(none_type, type(None))
        self.assertEqual(function.__annotations__["return"], list[int])
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "get_process_group_ranks")
        self.assertEqual(function.__qualname__, "get_process_group_ranks")
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

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(distributed_c10d.__all__.count(function.__name__), 1)
        direct_import = {}
        exec(
            "from torch_rs.distributed import get_process_group_ranks",
            direct_import,
        )
        self.assertIs(direct_import[function.__name__], function)
        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "get_process_group_ranks",
            owner_import,
        )
        self.assertIs(owner_import[function.__name__], function)
        for module in (distributed, distributed_c10d):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace[function.__name__], function)

        self.assertNotIn(function.__name__, torch.__all__)
        self.assertFalse(hasattr(torch, function.__name__))
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)
        self.assert_zero_process_group_state()

    def test_argument_errors(self):
        function = torch.distributed.get_process_group_ranks
        cases = (
            (
                function,
                "get_process_group_ranks() missing 1 required positional "
                "argument: 'group'",
            ),
            (
                lambda: function(None, None),
                "get_process_group_ranks() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "get_process_group_ranks() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, group=None),
                "get_process_group_ranks() got multiple values for argument "
                "'group'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)
        self.assert_zero_process_group_state()


if __name__ == "__main__":
    unittest.main()
