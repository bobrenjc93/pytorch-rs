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


NON_NONE_GROUP_ERROR = (
    "torch_rs.distributed.get_backend_config() does not support non-None "
    "process groups"
)


class OpaqueProcessGroup:
    def __bool__(self):
        raise AssertionError("process-group truthiness must not be read")

    def __getattr__(self, name):
        raise AssertionError(f"process-group attribute was read: {name}")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedGetBackendConfigReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_backend_config differentials require pinned "
                "PyTorch 2.13.0"
            )

    def error_outcome(self, call):
        try:
            call()
        except BaseException as error:
            return type(error), str(error), error.args
        self.fail("expected the call to raise")

    def assert_error_matches(self, actual_call, expected_call):
        self.assertEqual(
            self.error_outcome(actual_call),
            self.error_outcome(expected_call),
        )

    def threaded_outcome(self, module):
        function = module.distributed.get_backend_config
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
                    outcomes = []
                    for call in (function, lambda: function(group=None)):
                        outcomes.append(self.error_outcome(call))
                    worker_states[index] = (
                        before,
                        tuple(outcomes),
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

    def class_shape(self, value):
        return (
            value.__module__.replace("torch_rs", "torch"),
            value.__name__,
            value.__qualname__,
        )

    def group_annotation_shape(self, annotation):
        process_group, none_type = typing.get_args(annotation)
        return self.class_shape(process_group), none_type

    def test_uninitialized_errors_match_across_environments_threads_and_reloads(self):
        actual = torch.distributed.get_backend_config
        expected = reference_torch.distributed.get_backend_config

        environments = (
            {},
            {"USE_DISTRIBUTED": "0"},
            {"USE_DISTRIBUTED": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "WORLD_SIZE": "123",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for _ in range(3):
                        self.assert_error_matches(actual, expected)
                        self.assert_error_matches(
                            lambda: actual(None), lambda: expected(None)
                        )
                        self.assert_error_matches(
                            lambda: actual(group=None),
                            lambda: expected(group=None),
                        )

        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

        actual_distributed = torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_outcome = self.error_outcome(expected)
        for _ in range(3):
            actual_c10d = importlib.reload(actual_c10d)
            actual_distributed = importlib.reload(actual_distributed)
            actual = actual_distributed.get_backend_config
            self.assertIs(actual_c10d.get_backend_config, actual)
            self.assertEqual(self.error_outcome(actual), expected_outcome)
            self.assertEqual(
                self.error_outcome(lambda: actual(group=None)),
                expected_outcome,
            )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.get_backend_config
        expected = expected_distributed.get_backend_config

        self.assertIs(actual_c10d.get_backend_config, actual)
        self.assertIs(expected_c10d.get_backend_config, expected)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(set(actual.__annotations__), set(expected.__annotations__))
        self.assertEqual(
            self.group_annotation_shape(actual.__annotations__["group"]),
            self.group_annotation_shape(expected.__annotations__["group"]),
        )
        self.assertIs(actual.__annotations__["return"], str)
        self.assertIs(expected.__annotations__["return"], str)
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
        actual = actual_distributed.get_backend_config
        expected = expected_distributed.get_backend_config
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
        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_backend_config"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_backend_config", namespace)

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

    def test_default_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.distributed.get_backend_config
        expected = reference_torch.distributed.get_backend_config
        cases = (
            (actual, expected),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(group=None), lambda: expected(group=None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, group=None),
                lambda: expected(None, group=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_non_none_groups_are_explicitly_unsupported_without_inspection(self):
        function = torch.distributed.get_backend_config
        process_group_annotation = typing.get_args(
            function.__annotations__["group"]
        )[0]
        for group in (
            object(),
            0,
            False,
            "group",
            OpaqueProcessGroup(),
            process_group_annotation(),
        ):
            with self.subTest(group=type(group).__name__):
                with self.assertRaises(NotImplementedError) as raised:
                    function(group)
                self.assertEqual(str(raised.exception), NON_NONE_GROUP_ERROR)
                self.assertEqual(raised.exception.args, (NON_NONE_GROUP_ERROR,))


if __name__ == "__main__":
    unittest.main()
