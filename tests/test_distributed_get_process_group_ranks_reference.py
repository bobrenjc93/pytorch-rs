import copy
import importlib
import inspect
import os
import pickle
import pickletools
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedGetProcessGroupRanksReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_process_group_ranks differentials require "
                "pinned PyTorch 2.13.0"
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

    def annotation_shape(self, annotation):
        process_group, none_type = typing.get_args(annotation)
        return (
            process_group.__module__.replace("torch_rs", "torch"),
            process_group.__name__,
            process_group.__qualname__,
            none_type,
        )

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

    def test_falsey_default_group_errors_match_without_initialization(self):
        actual = torch.distributed.get_process_group_ranks
        expected = reference_torch.distributed.get_process_group_ranks
        expected_c10d = reference_torch.distributed.distributed_c10d

        self.assertIs(expected_c10d.GroupMember.WORLD, None)
        self.assertEqual(expected_c10d._world.pg_group_ranks, {})

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
                    for group in falsey_groups:
                        with self.subTest(group_type=type(group).__name__):
                            self.assert_call_matches(
                                lambda group=group: actual(group),
                                lambda group=group: expected(group),
                            )
                            self.assert_call_matches(
                                lambda group=group: actual(group=group),
                                lambda group=group: expected(group=group),
                            )
                self.assert_zero_process_group_state()

        def traced_outcome(function, behavior):
            events = []

            class FalseyGroup:
                def __bool__(self):
                    events.append("bool")
                    if behavior == "truth_error":
                        raise RuntimeError("truth failed")
                    if behavior == "invalid_truth":
                        return 1
                    return False

                def __hash__(self):
                    events.append("hash")
                    raise AssertionError("falsey group was hashed")

                def __repr__(self):
                    events.append("repr")
                    raise AssertionError("falsey group was represented")

            return self.outcome(lambda: function(FalseyGroup())), events

        for behavior in ("false", "truth_error", "invalid_truth"):
            with self.subTest(behavior=behavior):
                self.assertEqual(
                    traced_outcome(actual, behavior),
                    traced_outcome(expected, behavior),
                )

        for values in (0.0, [0.0], [[0.0]], [], [1.0, 2.0]):
            with self.subTest(tensor_values=values):
                self.assert_call_matches(
                    lambda values=values: actual(torch.tensor(values)),
                    lambda values=values: expected(reference_torch.tensor(values)),
                )
        self.assert_zero_process_group_state()

    def test_truthy_empty_registry_lookup_errors_and_order_match(self):
        actual = torch.distributed.get_process_group_ranks
        expected = reference_torch.distributed.get_process_group_ranks

        calls = (
            lambda function: function(True),
            lambda function: function(1),
            lambda function: function(-100),
            lambda function: function(1.5),
            lambda function: function("group"),
            lambda function: function(b"group"),
            lambda function: function((1,)),
            lambda function: function([1]),
            lambda function: function({"rank": 0}),
            lambda function: function({1}),
            lambda function: function(bytearray(b"group")),
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
                def __bool__(self):
                    events.append("bool")
                    return True

                def __hash__(self):
                    events.append("hash")
                    if behavior == "hash_error":
                        raise RuntimeError("hash failed")
                    if behavior == "invalid_hash":
                        return "not-an-integer"
                    return 123

                def __eq__(self, other):
                    events.append(("eq", other))
                    raise RuntimeError("equality failed")

                def __repr__(self):
                    events.append("repr")
                    if behavior == "repr_error":
                        raise RuntimeError("repr failed")
                    return "traced-group"

            group = TracedGroup()
            try:
                function(group)
            except BaseException as error:
                has_group_argument = (
                    type(error) is KeyError
                    and len(error.args) == 1
                    and error.args[0] is group
                )
                try:
                    message = str(error)
                except BaseException as formatting_error:
                    result = (
                        "format_error",
                        type(error),
                        type(formatting_error),
                        str(formatting_error),
                        formatting_error.args,
                        has_group_argument,
                    )
                else:
                    result = (
                        "error",
                        type(error),
                        message,
                        has_group_argument,
                    )
            else:
                result = ("return",)
            return result, events

        for behavior in ("normal", "hash_error", "invalid_hash", "repr_error"):
            with self.subTest(behavior=behavior):
                self.assertEqual(
                    traced_outcome(actual, behavior),
                    traced_outcome(expected, behavior),
                )
        self.assert_zero_process_group_state()

    def test_required_argument_errors_match_pytorch_2_13(self):
        actual = torch.distributed.get_process_group_ranks
        expected = reference_torch.distributed.get_process_group_ranks
        calls = (
            lambda function: function(),
            lambda function: function(None, True),
            lambda function: function(process_group=None),
            lambda function: function(None, group=None),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                self.assert_call_matches(
                    lambda call=call: call(actual),
                    lambda call=call: call(expected),
                )
        self.assert_zero_process_group_state()

    def test_reload_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module("torch_rs.distributed.distributed_c10d")
        expected_c10d = importlib.import_module("torch.distributed.distributed_c10d")

        for _ in range(3):
            actual_c10d = importlib.reload(actual_c10d)
            actual_distributed = importlib.reload(actual_distributed)
            actual = actual_distributed.get_process_group_ranks
            expected = expected_distributed.get_process_group_ranks

            self.assertIs(torch.distributed, actual_distributed)
            self.assertIs(reference_torch.distributed, expected_distributed)
            self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
            self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
            self.assertIs(actual_c10d.get_process_group_ranks, actual)
            self.assertIs(expected_c10d.get_process_group_ranks, expected)
            self.assertIs(type(actual), types.FunctionType)
            self.assertIs(type(expected), types.FunctionType)
            self.assertEqual(
                str(inspect.signature(actual)).replace("torch_rs", "torch"),
                str(inspect.signature(expected)),
            )
            self.assertEqual(set(actual.__annotations__), set(expected.__annotations__))
            self.assertEqual(
                self.annotation_shape(actual.__annotations__["group"]),
                self.annotation_shape(expected.__annotations__["group"]),
            )
            self.assertEqual(
                typing.get_origin(actual.__annotations__["return"]),
                typing.get_origin(expected.__annotations__["return"]),
            )
            self.assertEqual(
                typing.get_args(actual.__annotations__["return"]),
                typing.get_args(expected.__annotations__["return"]),
            )
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
        actual = actual_distributed.get_process_group_ranks
        expected = expected_distributed.get_process_group_ranks
        supported = {
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
            torch.__all__.count("get_process_group_ranks"),
            reference_torch.__all__.count("get_process_group_ranks"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_process_group_ranks"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_process_group_ranks", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_registered_process_groups_remain_outside_the_supported_surface(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        self.assertIs(
            actual_distributed.get_process_group_ranks,
            actual_c10d.get_process_group_ranks,
        )
        self.assertIs(
            expected_distributed.get_process_group_ranks,
            expected_c10d.get_process_group_ranks,
        )
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "get_global_rank",
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
