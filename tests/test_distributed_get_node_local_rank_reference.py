import contextlib
import copy
import importlib
import inspect
import os
import pickle
import pickletools
import sys
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
class DistributedGetNodeLocalRankReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_node_local_rank differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def threaded_outcome(self, module):
        function = module.distributed.get_node_local_rank
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = (
                        module.is_grad_enabled(),
                        type(function(index)).__name__,
                        function(index),
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

    def test_environment_and_fallback_conversion_match(self):
        actual = torch.distributed.get_node_local_rank
        expected = reference_torch.distributed.get_node_local_rank
        environment_cases = (
            ("0", None),
            ("-2", 99),
            ("+7", "unused"),
            (" 03 ", object()),
            ("1_024", []),
        )
        for value, fallback in environment_cases:
            with self.subTest(value=value, fallback=fallback):
                with mock.patch.dict(
                    os.environ,
                    {
                        "LOCAL_RANK": value,
                        "RANK": "23",
                        "WORLD_SIZE": "32",
                    },
                    clear=True,
                ):
                    environment = dict(os.environ)
                    actual_result = actual(fallback)
                    self.assertEqual(dict(os.environ), environment)
                    expected_result = expected(fallback)
                    self.assertEqual(dict(os.environ), environment)
                    self.assertIs(type(actual_result), int)
                    self.assertIs(type(expected_result), int)
                    self.assertEqual(actual_result, expected_result)

        fallback_cases = (0, -1, True, False, 2.8, "4", b"5")
        with mock.patch.dict(os.environ, {}, clear=True):
            environment = dict(os.environ)
            for fallback in fallback_cases:
                with self.subTest(fallback=fallback):
                    actual_result = actual(fallback)
                    self.assertEqual(dict(os.environ), environment)
                    expected_result = expected(fallback)
                    self.assertEqual(dict(os.environ), environment)
                    self.assertIs(type(actual_result), int)
                    self.assertIs(type(expected_result), int)
                    self.assertEqual(actual_result, expected_result)

    def test_missing_environment_conversion_and_argument_errors_match(self):
        actual = torch.distributed.get_node_local_rank
        expected = reference_torch.distributed.get_node_local_rank

        with mock.patch.dict(os.environ, {}, clear=True):
            cases = (
                (lambda: actual(), lambda: expected()),
                (lambda: actual("abc"), lambda: expected("abc")),
                (lambda: actual([]), lambda: expected([])),
                (lambda: actual(None, None), lambda: expected(None, None)),
                (lambda: actual(other=1), lambda: expected(other=1)),
                (
                    lambda: actual(1, fallback_rank=2),
                    lambda: expected(1, fallback_rank=2),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(case=case):
                    self.assert_error_matches(actual_call, expected_call)

        for value in ("", "abc", "1.5"):
            with self.subTest(environment_value=value):
                with mock.patch.dict(
                    os.environ, {"LOCAL_RANK": value}, clear=True
                ):
                    environment = dict(os.environ)
                    self.assert_error_matches(
                        lambda: actual(9), lambda: expected(9)
                    )
                    self.assertEqual(dict(os.environ), environment)

    def test_threading_and_grad_mode_independence_match(self):
        for environment_values in ({"LOCAL_RANK": "17"}, {}):
            with self.subTest(environment=environment_values):
                with mock.patch.dict(
                    os.environ, environment_values, clear=True
                ):
                    environment = dict(os.environ)
                    actual = self.threaded_outcome(torch)
                    self.assertEqual(dict(os.environ), environment)
                    expected = self.threaded_outcome(reference_torch)
                    self.assertEqual(dict(os.environ), environment)
                self.assertEqual(actual, expected)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.get_node_local_rank
        expected = expected_distributed.get_node_local_rank

        self.assertIs(torch.distributed, actual_distributed)
        self.assertIs(reference_torch.distributed, expected_distributed)
        self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
        self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
        self.assertIs(actual_c10d.get_node_local_rank, actual)
        self.assertIs(expected_c10d.get_node_local_rank, expected)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
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

    def test_imports_copy_wildcards_and_pickle_match(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.get_node_local_rank
        expected = expected_distributed.get_node_local_rank
        supported = {
            "get_node_local_rank",
            "get_pg_count",
            "is_gloo_available",
            "is_initialized",
            "is_mpi_available",
            "is_nccl_available",
            "is_ucc_available",
        }

        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"], actual_c10d
        )
        self.assertIs(
            sys.modules["torch.distributed.distributed_c10d"], expected_c10d
        )
        self.assertEqual(
            hasattr(actual_distributed, "__all__"),
            hasattr(expected_distributed, "__all__"),
        )
        self.assertEqual(
            actual_c10d.__all__,
            [name for name in expected_c10d.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("get_node_local_rank"),
            reference_torch.__all__.count("get_node_local_rank"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_node_local_rank"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_node_local_rank", namespace)

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

    def test_lookup_does_not_initialize_or_expand_distributed_execution(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d

        with mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=True):
            environment = dict(os.environ)
            self.assertEqual(actual_distributed.get_node_local_rank(), 3)
            self.assertEqual(expected_distributed.get_node_local_rank(), 3)
            self.assertEqual(dict(os.environ), environment)

        self.assertIs(actual_distributed.is_initialized(), False)
        self.assertIs(expected_distributed.is_initialized(), False)
        self.assertEqual(actual_distributed.get_pg_count(), 0)
        self.assertEqual(expected_distributed.get_pg_count(), 0)
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "get_rank",
            "get_world_size",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_distributed, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))


if __name__ == "__main__":
    unittest.main()
