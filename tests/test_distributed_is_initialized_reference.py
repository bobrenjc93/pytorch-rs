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
class DistributedIsInitializedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.is_initialized differentials require pinned PyTorch 2.13.0"
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
        function = module.distributed.is_initialized
        baseline = function()
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
                        function(),
                        module.is_grad_enabled(),
                        function(),
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
        return baseline, worker_states

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

    def test_uninitialized_default_state_matches_across_environments_and_threads(self):
        actual = torch.distributed.is_initialized
        expected = reference_torch.distributed.is_initialized
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )

        self.assertIs(expected_c10d.GroupMember.WORLD, None)
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
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(actual(), False)
                    self.assertIs(expected(), False)

        actual_baseline, actual_workers = self.threaded_outcome(torch)
        expected_baseline, expected_workers = self.threaded_outcome(reference_torch)
        self.assertIs(actual_baseline, False)
        self.assertIs(expected_baseline, False)
        self.assertEqual(actual_workers, expected_workers)
        self.assertIs(expected_c10d.GroupMember.WORLD, None)

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.is_initialized
        expected = expected_distributed.is_initialized

        self.assertIs(torch.distributed, actual_distributed)
        self.assertIs(reference_torch.distributed, expected_distributed)
        self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
        self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
        self.assertIs(actual_c10d.is_initialized, actual)
        self.assertIs(expected_c10d.is_initialized, expected)
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

    def test_imports_copy_wildcards_and_pickle_match_the_supported_scope(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.is_initialized
        expected = expected_distributed.is_initialized

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
            [
                name
                for name in expected_c10d.__all__
                if name
                in {
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
                }
            ],
        )
        self.assertEqual(
            torch.__all__.count("distributed"),
            reference_torch.__all__.count("distributed"),
        )
        self.assertEqual(
            torch.__all__.count("is_initialized"),
            reference_torch.__all__.count("is_initialized"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["is_initialized"], function)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.distributed import *", actual_namespace)
        exec("from torch.distributed import *", expected_namespace)
        self.assertIs(actual_namespace["distributed_c10d"], actual_c10d)
        self.assertIs(expected_namespace["distributed_c10d"], expected_c10d)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("distributed", namespace)
            self.assertNotIn("is_initialized", namespace)

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

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.distributed.is_initialized
        expected = reference_torch.distributed.is_initialized
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(enabled=True), lambda: expected(enabled=True)),
            (
                lambda: actual(None, enabled=True),
                lambda: expected(None, enabled=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_process_group_and_all_other_distributed_apis_remain_unsupported(self):
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

        self.assertEqual(
            actual_public,
            {
                "distributed_c10d",
                "destroy_process_group",
                "get_backend_config",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "is_available",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_group_rank",
                "get_global_rank",
                "get_node_local_rank",
            },
        )
        self.assertEqual(
            {
                name for name in vars(actual_c10d) if not name.startswith("_")
            },
            {
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
            },
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
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(expected_distributed, name))
                self.assertTrue(hasattr(expected_c10d, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))


if __name__ == "__main__":
    unittest.main()
