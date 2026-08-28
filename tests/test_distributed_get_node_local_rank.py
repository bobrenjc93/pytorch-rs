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


FUNCTION_DOC = """
    Return the local rank of the current process relative to the node.

    Semantically, this is a useful concept for mapping processes to devices.
    For example, on a node with 8 accelerator you could use the node local rank to decide
    which accelerator device to bind the process to.

    In practice, the actual assignment of node local ranks is handled by the process launcher outside of pytorch,
    and communicated via the `LOCAL_RANK` environment variable.

    Torchrun will automatically populate `LOCAL_RANK`, but other launchers may not.  If `LOCAL_RANK` is unspecified,
    this API will fall back to the provided kwarg 'fallback_rank' if specified, otherwise it will raise an error. The
    intent is to allow writing an application that runs either in single or multi device contexts without error.

    """
MISSING_LOCAL_RANK_ERROR = (
    "LOCAL_RANK is not in the environment. Consider passing fallback_rank to "
    "allow `get_node_local_rank` to work, assuming you are not running in a "
    "multi-device context and want the code to run locally instead."
)


class IntLike:
    def __int__(self):
        return 12


class IndexLike:
    def __index__(self):
        return -13


class RaisingInt:
    def __int__(self):
        raise LookupError("sentinel conversion failure")


class DistributedGetNodeLocalRankTests(unittest.TestCase):
    def test_environment_value_takes_precedence_and_is_not_mutated(self):
        function = torch.distributed.get_node_local_rank
        cases = (
            ("0", 0),
            ("-7", -7),
            ("+8", 8),
            (" 9 ", 9),
            ("٠", 0),
        )

        for value, expected in cases:
            with self.subTest(value=value):
                environment = {"LOCAL_RANK": value, "RANK": "91"}
                with mock.patch.dict(os.environ, environment, clear=True):
                    before = dict(os.environ)
                    result = function(fallback_rank=123)
                    self.assertIs(type(result), int)
                    self.assertEqual(result, expected)
                    self.assertEqual(dict(os.environ), before)

    def test_fallback_is_converted_with_builtin_int_and_not_retained(self):
        function = torch.distributed.get_node_local_rank
        cases = (
            (0, 0),
            (-3, -3),
            (True, 1),
            (False, 0),
            (4.9, 4),
            (" 10 ", 10),
            (b"11", 11),
            (bytearray(b"12"), 12),
            (IntLike(), 12),
            (IndexLike(), -13),
        )

        for fallback, expected in cases:
            with self.subTest(fallback=fallback):
                with mock.patch.dict(os.environ, {"RANK": "91"}, clear=True):
                    before = dict(os.environ)
                    result = function(fallback)
                    self.assertIs(type(result), int)
                    self.assertEqual(result, expected)
                    self.assertEqual(dict(os.environ), before)

    def test_missing_and_conversion_errors_match_the_contract(self):
        function = torch.distributed.get_node_local_rank

        with mock.patch.dict(os.environ, {}, clear=True):
            for call in (function, lambda: function(None)):
                with self.subTest(call=call):
                    with self.assertRaises(RuntimeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), MISSING_LOCAL_RANK_ERROR)
                    self.assertEqual(
                        raised.exception.args, (MISSING_LOCAL_RANK_ERROR,)
                    )

            with self.assertRaises(ValueError) as raised:
                function("3.5")
            self.assertEqual(
                raised.exception.args,
                ("invalid literal for int() with base 10: '3.5'",),
            )

            with self.assertRaises(LookupError) as raised:
                function(RaisingInt())
            self.assertEqual(raised.exception.args, ("sentinel conversion failure",))

        for value in ("", "4.5", "not-a-rank"):
            with self.subTest(value=value):
                with mock.patch.dict(
                    os.environ, {"LOCAL_RANK": value}, clear=True
                ):
                    before = dict(os.environ)
                    with self.assertRaises(ValueError):
                        function(fallback_rank=5)
                    self.assertEqual(dict(os.environ), before)

    def test_results_are_stable_across_threads_and_grad_modes(self):
        function = torch.distributed.get_node_local_rank

        for environment, fallback, expected in (
            ({"LOCAL_RANK": "17"}, -1, 17),
            ({"RANK": "4"}, "18", 18),
        ):
            with self.subTest(environment=environment, fallback=fallback):
                with mock.patch.dict(os.environ, environment, clear=True):
                    before = dict(os.environ)
                    worker_count = 8
                    barrier = threading.Barrier(worker_count)
                    results = [None] * worker_count
                    errors = []

                    def worker(index):
                        try:
                            context = (
                                torch.no_grad()
                                if index % 2
                                else contextlib.nullcontext()
                            )
                            with context:
                                barrier.wait(timeout=10)
                                first = function(fallback)
                                second = function(fallback_rank=fallback)
                                results[index] = (
                                    torch.is_grad_enabled(),
                                    type(first),
                                    first,
                                    type(second),
                                    second,
                                )
                        except BaseException as error:
                            errors.append(error)

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
                    self.assertEqual(dict(os.environ), before)
                    for index, result in enumerate(results):
                        self.assertEqual(
                            result,
                            (index % 2 == 0, int, expected, int, expected),
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
        self.assertEqual(
            inspect.getdoc(function), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_imports_copy_wildcards_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.get_node_local_rank

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
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
                "default_pg_timeout",
                "get_group_rank",
                "get_global_rank",
                "get_process_group_ranks",
                "get_node_local_rank",
            ],
        )

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

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

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertEqual(
            {
                name
                for name in distributed_namespace
                if not name.startswith("__")
            },
            {
                "constants",
                "distributed_c10d",
                "destroy_process_group",
                "get_backend_config",
                "get_backend",
                "get_rank",
                "get_world_size",
                "get_pg_count",
                "default_pg_timeout",
                "get_group_rank",
                "get_global_rank",
                "get_process_group_ranks",
                "get_node_local_rank",
                "is_available",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
            },
        )
        self.assertIs(distributed_namespace["get_node_local_rank"], function)

        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertEqual(
            {name for name in owner_namespace if not name.startswith("__")},
            set(distributed_c10d.__all__),
        )
        self.assertIs(owner_namespace["get_node_local_rank"], function)

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

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.distributed.get_node_local_rank
        cases = (
            (
                lambda: function(1, 2),
                "get_node_local_rank() takes from 0 to 1 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "get_node_local_rank() got an unexpected keyword argument "
                "'enabled'",
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

    def test_process_group_initialization_and_collectives_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)
        for name in (
            "ProcessGroup",
            "all_reduce",
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
import torch_rs as torch

function = torch.distributed.get_node_local_rank
assert function is torch.distributed.distributed_c10d.get_node_local_rank
before = dict(os.environ)
result = function(fallback_rank=22)
assert type(result) is int
assert result == int(os.environ.get("LOCAL_RANK", 22))
assert dict(os.environ) == before
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
assert not hasattr(torch.distributed, "init_process_group")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        environment = os.environ.copy()
        environment.pop("LOCAL_RANK", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
