import contextlib
import copy
import importlib
import inspect
import os
import pickle
import pickletools
import threading
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
class DistributedDestroyProcessGroupReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.destroy_process_group differentials require "
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

    def threaded_outcome(self, module):
        function = module.distributed.destroy_process_group
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    before = module.is_grad_enabled()
                    worker_states[index] = (
                        before,
                        self.outcome(lambda: function(-100)),
                        self.outcome(function),
                        self.outcome(lambda: function(index)),
                        module.distributed.get_pg_count(),
                        module.distributed.is_initialized(),
                        module.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

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
        return worker_states

    def native_tensor_outcome(self, module, values, no_grad):
        context = module.no_grad() if no_grad else contextlib.nullcontext()
        with context:
            group = module.tensor(values, requires_grad=True)
            before = (
                group.tolist(),
                tuple(group.shape),
                group.requires_grad,
                group.is_leaf,
                module.is_grad_enabled(),
            )
            outcomes = tuple(
                self.outcome(
                    lambda: module.distributed.destroy_process_group(group)
                )
                for _ in range(3)
            )
            after = (
                group.tolist(),
                tuple(group.shape),
                group.requires_grad,
                group.is_leaf,
                module.is_grad_enabled(),
            )
            return (
                before,
                outcomes,
                after,
                module.distributed.get_pg_count(),
                module.distributed.is_initialized(),
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

    def annotation_shape(self, annotation):
        process_group, none_type = typing.get_args(annotation)
        return (
            process_group.__module__.replace("torch_rs", "torch"),
            process_group.__name__,
            process_group.__qualname__,
            none_type,
        )

    def test_uninitialized_behavior_matches_across_environment_threads_and_reload(self):
        actual = torch.distributed.destroy_process_group
        expected = reference_torch.distributed.destroy_process_group
        expected_c10d = reference_torch.distributed.distributed_c10d

        self.assertIs(expected_c10d.GroupMember.WORLD, None)
        self.assertEqual(expected_c10d.GroupMember.NON_GROUP_MEMBER, -100)
        environments = (
            {},
            {"USE_DISTRIBUTED": "0"},
            {"USE_DISTRIBUTED": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "WORLD_SIZE": "1",
            },
        )
        calls = (
            (lambda function: function()),
            (lambda function: function(None)),
            (lambda function: function(group=None)),
            (lambda function: function(-100)),
            (lambda function: function(group=-100)),
            (lambda function: function(-100.0)),
            (lambda function: function(object())),
            (lambda function: function(0)),
            (lambda function: function("group")),
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for _ in range(3):
                        for call in calls:
                            self.assert_call_matches(
                                lambda call=call: call(actual),
                                lambda call=call: call(expected),
                            )
                        self.assert_zero_process_group_state()

        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

        actual_distributed = torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_outcomes = tuple(
            self.outcome(lambda call=call: call(expected)) for call in calls
        )
        for _ in range(3):
            actual_c10d = importlib.reload(actual_c10d)
            actual_distributed = importlib.reload(actual_distributed)
            actual = actual_distributed.destroy_process_group
            self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
            self.assertIs(actual_c10d.destroy_process_group, actual)
            self.assertEqual(
                tuple(self.outcome(lambda call=call: call(actual)) for call in calls),
                expected_outcomes,
            )
            self.assert_zero_process_group_state()

    def test_native_tensor_group_behavior_matches_pytorch_2_13(self):
        values_cases = (
            -100.0,
            [-100.0],
            [[-100.0]],
            -99.0,
            [0.0],
            [[1.0]],
            [],
            [-100.0, 0.0],
            [[1.0, 2.0]],
        )
        for no_grad in (False, True):
            for values in values_cases:
                with self.subTest(no_grad=no_grad, values=values):
                    self.assertEqual(
                        self.native_tensor_outcome(torch, values, no_grad),
                        self.native_tensor_outcome(
                            reference_torch,
                            values,
                            no_grad,
                        ),
                    )
        self.assert_zero_process_group_state()

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.destroy_process_group
        expected = expected_distributed.destroy_process_group

        self.assertIs(torch.distributed, actual_distributed)
        self.assertIs(reference_torch.distributed, expected_distributed)
        self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
        self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
        self.assertIs(actual_c10d.destroy_process_group, actual)
        self.assertIs(expected_c10d.destroy_process_group, expected)
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
            self.annotation_shape(typing.get_type_hints(actual)["group"]),
            self.annotation_shape(typing.get_type_hints(expected)["group"]),
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
        self.assertFalse(hasattr(actual_c10d, "ProcessGroup"))

    def test_imports_copy_wildcards_and_pickle_match_supported_scope(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.destroy_process_group
        expected = expected_distributed.destroy_process_group
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
            hasattr(actual_distributed, "__all__"),
            hasattr(expected_distributed, "__all__"),
        )
        self.assertEqual(
            actual_c10d.__all__,
            [name for name in expected_c10d.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("destroy_process_group"),
            reference_torch.__all__.count("destroy_process_group"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["destroy_process_group"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("destroy_process_group", namespace)

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

    def test_argument_and_equality_behavior_matches_pytorch_2_13(self):
        actual = torch.distributed.destroy_process_group
        expected = reference_torch.distributed.destroy_process_group

        class SentinelEquivalent:
            def __eq__(self, other):
                return other == -100

        class BrokenEquality:
            def __eq__(self, other):
                raise RuntimeError("equality failed")

        class BrokenHash:
            def __eq__(self, other):
                return False

            def __hash__(self):
                raise RuntimeError("hash failed")

        cases = (
            (lambda function: function(None, None)),
            (lambda function: function(enabled=True)),
            (lambda function: function(None, group=None)),
            (lambda function: function(False)),
            (lambda function: function(True)),
            (lambda function: function(-1)),
            (lambda function: function("")),
            (lambda function: function(SentinelEquivalent())),
            (lambda function: function(BrokenEquality())),
            (lambda function: function([])),
            (lambda function: function({})),
            (lambda function: function(set())),
            (lambda function: function(bytearray())),
            (lambda function: function(BrokenHash())),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_call_matches(
                    lambda call=call: call(actual),
                    lambda call=call: call(expected),
                )

        def traced_hash_outcome(function):
            events = []

            class TracedHash:
                def __eq__(self, other):
                    events.append(("eq", other))
                    return False

                def __hash__(self):
                    events.append(("hash",))
                    return 123

            return self.outcome(lambda: function(TracedHash())), events

        self.assertEqual(traced_hash_outcome(actual), traced_hash_outcome(expected))
        self.assert_zero_process_group_state()

    def test_only_the_independent_lifecycle_api_is_added(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual_public = {
            name for name in vars(actual_distributed) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(expected_distributed) if not name.startswith("_")
        }

        self.assertIn("destroy_process_group", actual_public)
        self.assertIs(
            actual_distributed.destroy_process_group,
            actual_c10d.destroy_process_group,
        )
        unsupported = expected_public - actual_public
        self.assertTrue(unsupported)
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_distributed, name))

        for name in (
            "GroupMember",
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
        self.assert_zero_process_group_state()


if __name__ == "__main__":
    unittest.main()
