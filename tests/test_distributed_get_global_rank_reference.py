import copy
import importlib
import inspect
import os
import pickle
import pickletools
import types
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedGetGlobalRankReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_global_rank differentials require pinned "
                "PyTorch 2.13.0"
            )

    def outcome(self, call):
        try:
            result = call()
        except BaseException as error:
            return "error", type(error), str(error), error.args
        return "return", type(result), result

    def assert_call_matches(self, actual_call, expected_call):
        self.assertEqual(self.outcome(actual_call), self.outcome(expected_call))

    def assert_zero_process_group_state(self):
        self.assertIs(torch.distributed.is_initialized(), False)
        self.assertEqual(torch.distributed.get_pg_count(), 0)
        self.assertIs(reference_torch.distributed.is_initialized(), False)
        self.assertEqual(reference_torch.distributed.get_pg_count(), 0)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def annotation_shape(self, annotation):
        return (
            annotation.__module__.replace("torch_rs", "torch"),
            annotation.__name__,
            annotation.__qualname__,
        )

    def test_default_world_identity_matches_without_initialization_or_coercion(self):
        actual = torch.distributed.get_global_rank
        expected = reference_torch.distributed.get_global_rank
        expected_c10d = reference_torch.distributed.distributed_c10d

        self.assertIs(expected_c10d.GroupMember.WORLD, None)
        self.assertEqual(expected_c10d._world.pg_group_ranks, {})

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
                "WORLD_SIZE": "123",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for function in (actual, expected):
                        rank = RankProbe()
                        self.assertIs(function(None, rank), rank)
                        self.assertIs(
                            function(group=None, group_rank=rank), rank
                        )
                        self.assertEqual(rank.events, [])

        values = (None, False, True, -7, 3.5, "rank", object(), [], {})
        for rank in values:
            with self.subTest(rank_type=type(rank).__name__):
                self.assertIs(actual(None, rank), rank)
                self.assertIs(expected(None, rank), rank)

        for module in (torch, reference_torch):
            rank = module.tensor([1.0, 2.0])
            self.assertIs(module.distributed.get_global_rank(None, rank), rank)
        self.assert_zero_process_group_state()

    def test_reload_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )

        for _ in range(3):
            actual_c10d = importlib.reload(actual_c10d)
            actual_distributed = importlib.reload(actual_distributed)
            actual = actual_distributed.get_global_rank
            expected = expected_distributed.get_global_rank
            rank = object()

            self.assertIs(torch.distributed, actual_distributed)
            self.assertIs(reference_torch.distributed, expected_distributed)
            self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
            self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
            self.assertIs(actual_c10d.get_global_rank, actual)
            self.assertIs(expected_c10d.get_global_rank, expected)
            self.assertIs(actual(None, rank), rank)
            self.assertIs(type(actual), types.FunctionType)
            self.assertIs(type(expected), types.FunctionType)
            self.assertEqual(
                str(inspect.signature(actual)).replace("torch_rs", "torch"),
                str(inspect.signature(expected)),
            )
            self.assertEqual(
                set(actual.__annotations__), set(expected.__annotations__)
            )
            self.assertEqual(
                self.annotation_shape(actual.__annotations__["group"]),
                self.annotation_shape(expected.__annotations__["group"]),
            )
            self.assertIs(actual.__annotations__["group_rank"], int)
            self.assertIs(expected.__annotations__["group_rank"], int)
            self.assertIs(actual.__annotations__["return"], int)
            self.assertIs(expected.__annotations__["return"], int)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__qualname__, expected.__qualname__)
            self.assertEqual(
                actual.__module__.replace("torch_rs", "torch"),
                expected.__module__,
            )
            self.assertIs(inspect.getmodule(actual), actual_c10d)
            self.assertIs(inspect.getmodule(expected), expected_c10d)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__defaults__, expected.__defaults__)
            self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
            self.assertEqual(actual.__dict__, expected.__dict__)
            self.assertEqual(
                hasattr(actual, "__text_signature__"),
                hasattr(expected, "__text_signature__"),
            )
            self.assertFalse(hasattr(actual_c10d, "ProcessGroup"))
            self.assert_zero_process_group_state()

    def test_imports_wildcards_copy_and_pickle_match_the_supported_scope(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.get_global_rank
        expected = expected_distributed.get_global_rank
        supported = {
            "GroupMember",
            "destroy_process_group",
            "get_backend_config",
            "get_backend",
            "get_rank",
            "get_world_size",
            "get_pg_count",
            "group",
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
        }

        self.assertEqual(
            hasattr(actual_distributed, "__all__"),
            hasattr(expected_distributed, "__all__"),
        )
        self.assertEqual(
            actual_c10d.__all__,
            [name for name in expected_c10d.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("get_global_rank"),
            reference_torch.__all__.count("get_global_rank"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_global_rank"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_global_rank", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)), expected
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_and_invalid_group_errors_match_pytorch_2_13(self):
        actual = torch.distributed.get_global_rank
        expected = reference_torch.distributed.get_global_rank
        calls = (
            lambda function: function(),
            lambda function: function(None),
            lambda function: function(None, 1, 2),
            lambda function: function(group=None, rank=1),
            lambda function: function(None, 1, group=None),
            lambda function: function(None, 1, group_rank=2),
            lambda function: function(False, object()),
            lambda function: function(True, object()),
            lambda function: function(0, object()),
            lambda function: function(-100, object()),
            lambda function: function("", object()),
            lambda function: function("group", object()),
            lambda function: function([], object()),
            lambda function: function({}, object()),
            lambda function: function(set(), object()),
            lambda function: function(bytearray(), object()),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                self.assert_call_matches(
                    lambda call=call: call(actual),
                    lambda call=call: call(expected),
                )

        def traced_outcome(function, behavior):
            events = []

            class TracedGroup:
                def __hash__(self):
                    events.append(("hash",))
                    if behavior == "hash_error":
                        raise RuntimeError("hash failed")
                    if behavior == "invalid_hash":
                        return "not-an-integer"
                    return 123

                def __eq__(self, other):
                    events.append(("eq", other))
                    raise RuntimeError("equality failed")

                def __format__(self, format_spec):
                    events.append(("format", format_spec))
                    if behavior == "format_error":
                        raise RuntimeError("format failed")
                    return "traced-group"

            return self.outcome(lambda: function(TracedGroup(), object())), events

        for behavior in ("normal", "hash_error", "invalid_hash", "format_error"):
            with self.subTest(behavior=behavior):
                self.assertEqual(
                    traced_outcome(actual, behavior),
                    traced_outcome(expected, behavior),
                )
        self.assert_zero_process_group_state()

    def test_registered_process_groups_remain_outside_the_supported_surface(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        self.assertIs(
            actual_distributed.get_global_rank, actual_c10d.get_global_rank
        )
        self.assertIs(
            expected_distributed.get_global_rank, expected_c10d.get_global_rank
        )
        for name in (
            "ProcessGroup",
            "all_reduce",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_distributed, name))
                self.assertTrue(hasattr(expected_c10d, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))
        self.assertFalse(hasattr(actual_c10d, "_world"))
        self.assert_zero_process_group_state()


if __name__ == "__main__":
    unittest.main()
