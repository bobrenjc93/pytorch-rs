import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


MISSING_LOCAL_RANK_ERROR = (
    "LOCAL_RANK is not in the environment. Consider passing fallback_rank to "
    "allow `get_node_local_rank` to work, assuming you are not running in a "
    "multi-device context and want the code to run locally instead."
)


class DistributedGetNodeLocalRankTests(unittest.TestCase):
    def test_environment_value_takes_precedence_and_is_not_mutated(self):
        function = torch.distributed.get_node_local_rank
        values = {
            "0": 0,
            "-2": -2,
            "+7": 7,
            " 03 ": 3,
            "1_024": 1024,
        }

        class UnusedFallback:
            def __int__(self):
                raise AssertionError("fallback was converted")

        for value, expected in values.items():
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ,
                    {
                        "LOCAL_RANK": value,
                        "RANK": "91",
                        "WORLD_SIZE": "128",
                    },
                    clear=True,
                ):
                    environment = dict(os.environ)
                    result = function(UnusedFallback())
                    self.assertIs(type(result), int)
                    self.assertEqual(result, expected)
                    self.assertEqual(dict(os.environ), environment)

    def test_fallback_is_converted_only_when_environment_is_missing(self):
        function = torch.distributed.get_node_local_rank

        class RankLike:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __int__(self):
                self.calls += 1
                return self.value

        cases = (
            (0, 0),
            (-1, -1),
            (True, 1),
            (False, 0),
            (2.8, 2),
            ("4", 4),
            (b"5", 5),
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            environment = dict(os.environ)
            for fallback, expected in cases:
                with self.subTest(fallback=fallback):
                    result = function(fallback)
                    self.assertIs(type(result), int)
                    self.assertEqual(result, expected)
                    self.assertEqual(dict(os.environ), environment)

            rank_like = RankLike(6)
            self.assertEqual(function(rank_like), 6)
            self.assertEqual(rank_like.calls, 1)
            self.assertEqual(dict(os.environ), environment)

    def test_missing_environment_and_conversion_errors_match_pytorch_2_13(self):
        function = torch.distributed.get_node_local_rank
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError) as raised:
                function()
            self.assertEqual(str(raised.exception), MISSING_LOCAL_RANK_ERROR)
            self.assertEqual(raised.exception.args, (MISSING_LOCAL_RANK_ERROR,))

            cases = (
                (
                    lambda: function("abc"),
                    ValueError,
                    "invalid literal for int() with base 10: 'abc'",
                ),
                (
                    lambda: function([]),
                    TypeError,
                    "int() argument must be a string, a bytes-like object or a "
                    "real number, not 'list'",
                ),
            )
            for call, error_type, message in cases:
                with self.subTest(message=message):
                    with self.assertRaises(error_type) as converted:
                        call()
                    self.assertEqual(str(converted.exception), message)
                    self.assertEqual(converted.exception.args, (message,))

        for value in ("", "abc", "1.5"):
            with self.subTest(environment_value=value):
                with mock.patch.dict(
                    os.environ, {"LOCAL_RANK": value}, clear=True
                ):
                    environment = dict(os.environ)
                    with self.assertRaises(ValueError) as raised:
                        function(9)
                    message = f"invalid literal for int() with base 10: {value!r}"
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertEqual(dict(os.environ), environment)

    def test_reads_are_stable_across_threads_and_grad_modes(self):
        function = torch.distributed.get_node_local_rank
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(index),
                        torch.is_grad_enabled(),
                        function(index + worker_count),
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

        with mock.patch.dict(os.environ, {"LOCAL_RANK": "11"}, clear=True):
            environment = dict(os.environ)
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
            self.assertEqual(dict(os.environ), environment)

        for index, result in enumerate(results):
            grad_enabled = index % 2 == 0
            self.assertEqual(
                result,
                (grad_enabled, 11, grad_enabled, 11, grad_enabled),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.get_node_local_rank

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.get_node_local_rank, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(fallback_rank: int | None = None) -> int",
        )
        self.assertEqual(
            function.__annotations__,
            {"fallback_rank": int | None, "return": int},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"fallback_rank": int | None, "return": int},
        )
        self.assertEqual(function.__name__, "get_node_local_rank")
        self.assertEqual(function.__qualname__, "get_node_local_rank")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertIn(
            "Return the local rank of the current process relative to the node.",
            inspect.getdoc(function),
        )
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_node_local_rank

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
                "get_pg_count",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "get_node_local_rank",
            ],
        )

        direct_import = {}
        exec(
            "from torch_rs.distributed import get_node_local_rank",
            direct_import,
        )
        self.assertIs(direct_import["get_node_local_rank"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "get_node_local_rank",
            owner_import,
        )
        self.assertIs(owner_import["get_node_local_rank"], function)

        for module in (distributed, distributed_c10d):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["get_node_local_rank"], function)

        self.assertNotIn("get_node_local_rank", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("get_node_local_rank", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.distributed.get_node_local_rank
        cases = (
            (
                lambda: function(None, None),
                "get_node_local_rank() takes from 0 to 1 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function(other=1),
                "get_node_local_rank() got an unexpected keyword argument "
                "'other'",
            ),
            (
                lambda: function(1, fallback_rank=2),
                "get_node_local_rank() got multiple values for argument "
                "'fallback_rank'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_execution_surface_remains_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)
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
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))
        self.assertFalse(hasattr(torch, "get_node_local_rank"))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.clear()
os.environ["LOCAL_RANK"] = "13"
import torch_rs as torch

function = torch.distributed.get_node_local_rank
environment = dict(os.environ)
assert function() == 13
assert function(99) == 13
assert dict(os.environ) == environment
del os.environ["LOCAL_RANK"]
environment = dict(os.environ)
assert function("14") == 14
assert dict(os.environ) == environment
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
assert not hasattr(torch.distributed, "get_rank")
assert not hasattr(torch.distributed, "get_world_size")
assert not hasattr(torch.distributed, "init_process_group")
assert not hasattr(torch.distributed, "all_reduce")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
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


if __name__ == "__main__":
    unittest.main()
