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


_OMITTED = object()


class IntLike:
    def __int__(self):
        return 23


class IndexLike:
    def __index__(self):
        return -24


class BadInt:
    def __int__(self):
        return "25"


class RaisingInt:
    def __int__(self):
        raise LookupError("sentinel conversion failure")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedGetNodeLocalRankReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_node_local_rank differentials require pinned "
                "PyTorch 2.13.0"
            )

    def outcome(self, function, fallback_rank=_OMITTED):
        try:
            if fallback_rank is _OMITTED:
                result = function()
            else:
                result = function(fallback_rank)
        except BaseException as error:
            return (
                "error",
                type(error).__name__,
                str(error),
                error.args,
            )
        return "return", type(result).__name__, result

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

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

    def threaded_outcome(self, module, environment, fallback_rank):
        function = module.distributed.get_node_local_rank
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        with mock.patch.dict(os.environ, environment, clear=True):
            before = dict(os.environ)

            def worker(index):
                try:
                    context = (
                        module.no_grad()
                        if index % 2
                        else contextlib.nullcontext()
                    )
                    with context:
                        barrier.wait(timeout=10)
                        first = function(fallback_rank)
                        second = function(fallback_rank=fallback_rank)
                        results[index] = (
                            module.is_grad_enabled(),
                            type(first).__name__,
                            first,
                            type(second).__name__,
                            second,
                        )
                except BaseException as error:
                    errors.append((type(error).__name__, str(error), error.args))

            threads = [
                threading.Thread(target=worker, args=(index,))
                for index in range(worker_count)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(dict(os.environ), before)
        return errors, results

    def test_environment_fallback_precedence_and_errors_match(self):
        actual = torch.distributed.get_node_local_rank
        expected = reference_torch.distributed.get_node_local_rank
        cases = (
            ({}, _OMITTED),
            ({}, None),
            ({}, 0),
            ({}, -3),
            ({}, True),
            ({}, 4.9),
            ({}, " 10 "),
            ({}, b"11"),
            ({}, bytearray(b"12")),
            ({}, IntLike()),
            ({}, IndexLike()),
            ({}, BadInt()),
            ({}, RaisingInt()),
            ({}, object()),
            ({"LOCAL_RANK": "0"}, _OMITTED),
            ({"LOCAL_RANK": "-7"}, 999),
            ({"LOCAL_RANK": "+8"}, None),
            ({"LOCAL_RANK": " 9 "}, object()),
            ({"LOCAL_RANK": "٠"}, 999),
            ({"LOCAL_RANK": ""}, 5),
            ({"LOCAL_RANK": "4.5"}, 5),
            ({"LOCAL_RANK": "not-a-rank"}, 5),
        )

        for environment, fallback_rank in cases:
            with self.subTest(
                environment=environment, fallback_rank=fallback_rank
            ):
                with mock.patch.dict(os.environ, environment, clear=True):
                    before = dict(os.environ)
                    actual_outcome = self.outcome(actual, fallback_rank)
                    self.assertEqual(dict(os.environ), before)
                    expected_outcome = self.outcome(expected, fallback_rank)
                    self.assertEqual(dict(os.environ), before)
                self.assertEqual(actual_outcome, expected_outcome)

    def test_threaded_environment_and_fallback_reads_match(self):
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        self.assertEqual(expected_c10d._world.group_count, 0)

        for environment, fallback_rank in (
            ({"LOCAL_RANK": "31"}, -1),
            ({"RANK": "8"}, "32"),
        ):
            with self.subTest(
                environment=environment, fallback_rank=fallback_rank
            ):
                self.assertEqual(
                    self.threaded_outcome(torch, environment, fallback_rank),
                    self.threaded_outcome(
                        reference_torch, environment, fallback_rank
                    ),
                )

        self.assertIs(reference_torch.distributed.is_initialized(), False)
        self.assertEqual(expected_c10d._world.group_count, 0)

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
        self.assertEqual(
            typing.get_type_hints(actual), typing.get_type_hints(expected)
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

    def test_imports_copy_wildcards_and_pickle_match(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.get_node_local_rank
        expected = expected_distributed.get_node_local_rank
        supported = {
            "get_pg_count",
            "is_gloo_available",
            "is_initialized",
            "is_mpi_available",
            "is_nccl_available",
            "is_ucc_available",
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
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol)), expected
                )
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.distributed.get_node_local_rank
        expected = reference_torch.distributed.get_node_local_rank
        cases = (
            (lambda: actual(1, 2), lambda: expected(1, 2)),
            (
                lambda: actual(enabled=True),
                lambda: expected(enabled=True),
            ),
            (
                lambda: actual(1, fallback_rank=2),
                lambda: expected(1, fallback_rank=2),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_no_process_group_or_global_rank_surface_was_added(self):
        actual_distributed = torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_distributed = reference_torch.distributed
        expected_c10d = expected_distributed.distributed_c10d

        self.assertIs(actual_distributed.is_initialized(), False)
        self.assertEqual(actual_distributed.get_pg_count(), 0)
        for name in (
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
                self.assertTrue(hasattr(expected_c10d, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))


if __name__ == "__main__":
    unittest.main()
