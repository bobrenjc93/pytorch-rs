import contextlib
import copy
import importlib
import inspect
import os
import pickle
import pickletools
import subprocess
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
class DistributedGetWorldSizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.get_world_size differentials require pinned "
                "PyTorch 2.13.0"
            )

    def error_outcome(self, call):
        try:
            call()
        except BaseException as error:
            return type(error), str(error), error.args
        return None

    def assert_error_matches(self, actual_call, expected_call):
        actual = self.error_outcome(actual_call)
        expected = self.error_outcome(expected_call)
        self.assertIsNotNone(actual)
        self.assertEqual(actual, expected)

    def threaded_outcome(self, module):
        function = module.distributed.get_world_size
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    outcomes = []
                    for args, kwargs in (
                        ((), {}),
                        ((None,), {}),
                        ((), {"group": None}),
                        ((), {}),
                    ):
                        outcomes.append(
                            self.error_outcome(
                                lambda args=args, kwargs=kwargs: function(
                                    *args, **kwargs
                                )
                            )
                        )
                    worker_states[index] = (
                        module.is_grad_enabled(),
                        outcomes,
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

    def test_default_errors_match_environments_threads_and_grad_modes(self):
        actual = torch.distributed.get_world_size
        expected = reference_torch.distributed.get_world_size
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
                "WORLD_SIZE": "64",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for actual_call, expected_call in (
                        (lambda: actual(), lambda: expected()),
                        (lambda: actual(None), lambda: expected(None)),
                        (
                            lambda: actual(group=None),
                            lambda: expected(group=None),
                        ),
                    ):
                        self.assert_error_matches(actual_call, expected_call)

        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )
        self.assertIs(torch.distributed.is_initialized(), False)
        self.assertIs(reference_torch.distributed.is_initialized(), False)
        self.assertIs(expected_c10d.GroupMember.WORLD, None)

    def test_signature_documentation_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_distributed.get_world_size
        expected = expected_distributed.get_world_size

        self.assertIs(actual_c10d.get_world_size, actual)
        self.assertIs(expected_c10d.get_world_size, expected)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            str(actual.__annotations__).replace("torch_rs", "torch"),
            str(expected.__annotations__),
        )
        self.assertEqual(
            str(typing.get_type_hints(actual)).replace("torch_rs", "torch"),
            str(typing.get_type_hints(expected)),
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

    def test_imports_copy_wildcards_and_pickle_match(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_distributed.get_world_size
        expected = expected_distributed.get_world_size
        supported = {
            "get_world_size",
            "get_pg_count",
            "is_gloo_available",
            "is_initialized",
            "is_mpi_available",
            "is_nccl_available",
            "is_ucc_available",
            "is_xccl_available",
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
            torch.__all__.count("get_world_size"),
            reference_torch.__all__.count("get_world_size"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_world_size"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("get_world_size", namespace)

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
        actual = torch.distributed.get_world_size
        expected = reference_torch.distributed.get_world_size
        cases = (
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(enabled=True),
                lambda: expected(enabled=True),
            ),
            (
                lambda: actual(None, group=None),
                lambda: expected(None, group=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_module_reload_preserves_the_uninitialized_error(self):
        script = r"""
import importlib
import torch
import torch_rs

def outcome(function):
    results = []
    for args, kwargs in (((), {}), ((None,), {}), ((), {"group": None})):
        try:
            function(*args, **kwargs)
        except BaseException as error:
            results.append((type(error).__name__, str(error), error.args))
        else:
            results.append(None)
    return results

actual_module = importlib.import_module("torch_rs.distributed.distributed_c10d")
expected_module = importlib.import_module("torch.distributed.distributed_c10d")
actual_before = actual_module.get_world_size
expected_before = expected_module.get_world_size
actual_after = importlib.reload(actual_module).get_world_size
expected_after = importlib.reload(expected_module).get_world_size
assert actual_before is not actual_after
assert expected_before is not expected_after
assert outcome(actual_before) == outcome(expected_before)
assert outcome(actual_after) == outcome(expected_after)
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

    def test_non_none_groups_and_distributed_execution_remain_unsupported(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        for group in (
            object(),
            0,
            False,
            expected_c10d.GroupMember.NON_GROUP_MEMBER,
        ):
            with self.subTest(group=group):
                with self.assertRaises(NotImplementedError):
                    actual_distributed.get_world_size(group)

        actual_public = {
            name for name in vars(actual_distributed) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(expected_distributed) if not name.startswith("_")
        }
        self.assertIn("get_world_size", actual_public)
        unsupported = expected_public - actual_public
        self.assertTrue(unsupported)
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_distributed, name))

        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "get_rank",
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
