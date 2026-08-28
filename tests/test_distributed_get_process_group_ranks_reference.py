import copy
import importlib
import inspect
import pickle
import pickletools
import typing
import unittest

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

    def test_signature_metadata_and_exports_match_pytorch_2_13(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.get_process_group_ranks
        expected = expected_distributed.get_process_group_ranks

        self.assertIs(actual_c10d.get_process_group_ranks, actual)
        self.assertIs(expected_c10d.get_process_group_ranks, expected)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            str(actual.__annotations__["group"]).replace("torch_rs", "torch"),
            str(expected.__annotations__["group"]),
        )
        self.assertEqual(
            actual.__annotations__["return"], expected.__annotations__["return"]
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
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
        self.assertEqual(
            typing.get_args(actual.__annotations__["group"])[1],
            typing.get_args(expected.__annotations__["group"])[1],
        )
        self.assertFalse(hasattr(actual_c10d, "ProcessGroup"))

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
            "get_global_rank",
            "get_process_group_ranks",
            "get_node_local_rank",
        }
        self.assertEqual(
            actual_c10d.__all__,
            [name for name in expected_c10d.__all__ if name in supported],
        )
        self.assertEqual(
            hasattr(actual_distributed, "__all__"),
            hasattr(expected_distributed, "__all__"),
        )
        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace[function.__name__], function)

        self.assertEqual(
            torch.__all__.count(actual.__name__),
            reference_torch.__all__.count(expected.__name__),
        )
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
        self.assert_zero_process_group_state()

    def test_falsey_default_group_errors_match_without_initialization(self):
        actual = torch.distributed.get_process_group_ranks
        expected = reference_torch.distributed.get_process_group_ranks
        expected_c10d = reference_torch.distributed.distributed_c10d

        self.assertEqual(expected_c10d._world.pg_group_ranks, {})
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
        for group in falsey_groups:
            with self.subTest(group_type=type(group).__name__):
                self.assert_call_matches(
                    lambda group=group: actual(group),
                    lambda group=group: expected(group),
                )

        def traced_outcome(function):
            events = []

            class FalseyGroup:
                def __bool__(self):
                    events.append("bool")
                    return False

                def __hash__(self):
                    events.append("hash")
                    raise AssertionError("falsey group was hashed")

            outcome = self.outcome(lambda: function(FalseyGroup()))
            return outcome, events

        self.assertEqual(traced_outcome(actual), traced_outcome(expected))
        self.assert_zero_process_group_state()

    def test_truthy_empty_registry_lookups_match_pytorch_2_13(self):
        actual = torch.distributed.get_process_group_ranks
        expected = reference_torch.distributed.get_process_group_ranks

        for group in (
            True,
            1,
            -100,
            "group",
            (1,),
            b"group",
            [1],
            {"group": 1},
            {1},
            bytearray(b"group"),
        ):
            with self.subTest(group_type=type(group).__name__):
                self.assert_call_matches(
                    lambda group=group: actual(group),
                    lambda group=group: expected(group),
                )

        def traced_outcome(function, behavior):
            events = []

            class TracedGroup:
                def __bool__(self):
                    events.append("bool")
                    if behavior == "truth_error":
                        raise RuntimeError("truth failed")
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
                    raise AssertionError("empty registry compared a key")

                def __repr__(self):
                    return "TracedGroup()"

            outcome = self.outcome(lambda: function(TracedGroup()))
            return outcome[:3], events

        for behavior in ("normal", "truth_error", "hash_error", "invalid_hash"):
            with self.subTest(behavior=behavior):
                self.assertEqual(
                    traced_outcome(actual, behavior),
                    traced_outcome(expected, behavior),
                )
        self.assert_zero_process_group_state()

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.distributed.get_process_group_ranks
        expected = reference_torch.distributed.get_process_group_ranks
        calls = (
            lambda function: function(),
            lambda function: function(None, None),
            lambda function: function(enabled=True),
            lambda function: function(None, group=None),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                self.assert_call_matches(
                    lambda call=call: call(actual),
                    lambda call=call: call(expected),
                )
        self.assert_zero_process_group_state()


if __name__ == "__main__":
    unittest.main()
