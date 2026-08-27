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


NONE_ERROR = "Process group cannot be None"
INVALID_ERROR = "Invalid process group specified"
EMPTY_TENSOR_ERROR = "Boolean value of Tensor with no values is ambiguous"
MULTI_TENSOR_ERROR = (
    "Boolean value of Tensor with more than one value is ambiguous"
)
FUNCTION_DOC = """Destroy a given process group, and deinitialize the distributed package.

Args:
    group (ProcessGroup, optional): The process group to be destroyed, if
                                    group.WORLD is given, all process
                                    groups including the default one will
                                    be destroyed."""


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class DistributedDestroyProcessGroupTests(unittest.TestCase):
    def assert_error(self, error_type, message, call):
        with self.assertRaises(error_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

    def assert_zero_process_group_state(self):
        self.assertIs(torch.distributed.is_initialized(), False)
        count = torch.distributed.get_pg_count()
        self.assertIs(type(count), int)
        self.assertEqual(count, 0)

    def test_uninitialized_calls_repeat_without_runtime_or_environment_probes(self):
        function = torch.distributed.destroy_process_group
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )

        for name in ("_world", "GroupMember", "ProcessGroup"):
            self.assertFalse(hasattr(distributed_c10d, name))
        for name in ("_os", "environ", "is_initialized", "get_pg_count"):
            self.assertNotIn(name, function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_DISTRIBUTED": "0"},
            {"USE_DISTRIBUTED": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "USE_DISTRIBUTED": "unexpected",
                "WORLD_SIZE": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for _ in range(3):
                        self.assert_error(AssertionError, NONE_ERROR, function)
                        self.assert_error(
                            AssertionError,
                            NONE_ERROR,
                            lambda: function(None),
                        )
                        self.assert_error(
                            AssertionError,
                            NONE_ERROR,
                            lambda: function(group=None),
                        )
                        self.assertIsNone(function(-100))
                        self.assertIsNone(function(group=-100))
                        self.assertIsNone(function(-100.0))
                        self.assert_error(
                            ValueError,
                            INVALID_ERROR,
                            lambda: function(object()),
                        )
                        self.assert_zero_process_group_state()

        with mock.patch.object(
            distributed_c10d._os, "environ", UnreadableEnvironment()
        ):
            self.assertIsNone(function(-100))
            self.assert_error(AssertionError, NONE_ERROR, function)
            self.assert_error(
                ValueError,
                INVALID_ERROR,
                lambda: function("group"),
            )
        self.assert_zero_process_group_state()

    def test_state_is_stable_across_threads_and_grad_modes(self):
        function = torch.distributed.destroy_process_group
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def outcome(call):
            try:
                result = call()
                return type(result), result
            except BaseException as error:
                return type(error), str(error), error.args

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    before = torch.is_grad_enabled()
                    sentinel_result = function(-100)
                    none_outcome = outcome(function)
                    invalid_outcome = outcome(lambda: function(index))
                    results[index] = (
                        before,
                        sentinel_result,
                        none_outcome,
                        invalid_outcome,
                        torch.distributed.get_pg_count(),
                        torch.distributed.is_initialized(),
                        torch.is_grad_enabled(),
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
        for index, result in enumerate(results):
            grad_enabled = index % 2 == 0
            self.assertEqual(
                result,
                (
                    grad_enabled,
                    None,
                    (AssertionError, NONE_ERROR, (NONE_ERROR,)),
                    (ValueError, INVALID_ERROR, (INVALID_ERROR,)),
                    0,
                    False,
                    grad_enabled,
                ),
            )
        self.assert_zero_process_group_state()

    def test_native_tensor_groups_follow_sentinel_truthiness(self):
        function = torch.distributed.destroy_process_group

        for no_grad in (False, True):
            context = torch.no_grad() if no_grad else contextlib.nullcontext()
            with self.subTest(no_grad=no_grad), context:
                grad_enabled = not no_grad
                for values in (-100.0, [-100.0], [[-100.0]]):
                    with self.subTest(values=values):
                        group = torch.tensor(values, requires_grad=True)
                        before = (
                            group.tolist(),
                            group.shape,
                            group.requires_grad,
                            group.is_leaf,
                        )
                        for _ in range(3):
                            self.assertIsNone(function(group))
                        self.assertEqual(
                            (
                                group.tolist(),
                                group.shape,
                                group.requires_grad,
                                group.is_leaf,
                            ),
                            before,
                        )
                        self.assertIs(torch.is_grad_enabled(), grad_enabled)
                        self.assert_zero_process_group_state()

                for values in (-99.0, [0.0], [[1.0]]):
                    with self.subTest(values=values):
                        group = torch.tensor(values)
                        self.assert_error(
                            ValueError,
                            INVALID_ERROR,
                            lambda: function(group),
                        )
                        self.assertIs(torch.is_grad_enabled(), grad_enabled)
                        self.assert_zero_process_group_state()

                for values, message in (
                    ([], EMPTY_TENSOR_ERROR),
                    ([-100.0, 0.0], MULTI_TENSOR_ERROR),
                    ([[1.0, 2.0]], MULTI_TENSOR_ERROR),
                ):
                    with self.subTest(values=values):
                        group = torch.tensor(values)
                        self.assert_error(
                            RuntimeError,
                            message,
                            lambda: function(group),
                        )
                        self.assertIs(torch.is_grad_enabled(), grad_enabled)
                        self.assert_zero_process_group_state()

    def test_reload_preserves_the_uninitialized_contract(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        for _ in range(3):
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)
            function = distributed.destroy_process_group

            self.assertIs(torch.distributed, distributed)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(distributed_c10d.destroy_process_group, function)
            self.assertIsNone(function(-100))
            self.assert_error(AssertionError, NONE_ERROR, function)
            self.assert_error(
                ValueError,
                INVALID_ERROR,
                lambda: function(object()),
            )
            self.assert_zero_process_group_state()

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.destroy_process_group

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.destroy_process_group, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(group: torch_rs.distributed.distributed_c10d.ProcessGroup | "
            "None = None)",
        )
        self.assertEqual(set(function.__annotations__), {"group"})
        group_annotation = function.__annotations__["group"]
        process_group, none_type = typing.get_args(group_annotation)
        self.assertEqual(process_group.__name__, "ProcessGroup")
        self.assertEqual(process_group.__qualname__, "ProcessGroup")
        self.assertEqual(
            process_group.__module__,
            "torch_rs.distributed.distributed_c10d",
        )
        self.assertIs(none_type, type(None))
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "destroy_process_group")
        self.assertEqual(function.__qualname__, "destroy_process_group")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(inspect.getdoc(function), FUNCTION_DOC)
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(distributed_c10d, "ProcessGroup"))

    def test_imports_copy_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.destroy_process_group

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
                "get_group_rank",
                "get_node_local_rank",
            ],
        )

        direct_import = {}
        exec(
            "from torch_rs.distributed import destroy_process_group",
            direct_import,
        )
        self.assertIs(direct_import["destroy_process_group"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "destroy_process_group",
            owner_import,
        )
        self.assertIs(owner_import["destroy_process_group"], function)

        for module_name, module in (
            ("torch_rs.distributed", distributed),
            ("torch_rs.distributed.distributed_c10d", distributed_c10d),
        ):
            namespace = {}
            exec(f"from {module_name} import *", namespace)
            self.assertIs(namespace["destroy_process_group"], function)
            if module is distributed:
                self.assertIs(namespace["distributed_c10d"], distributed_c10d)

        self.assertNotIn("destroy_process_group", torch.__all__)
        self.assertFalse(hasattr(torch, "destroy_process_group"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("destroy_process_group", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_and_group_classification_match_pytorch_2_13(self):
        function = torch.distributed.destroy_process_group

        cases = (
            (
                lambda: function(None, None),
                "destroy_process_group() takes from 0 to 1 positional "
                "arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "destroy_process_group() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, group=None),
                "destroy_process_group() got multiple values for argument "
                "'group'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

        class FakeProcessGroup:
            @property
            def shutdown(self):
                raise AssertionError("process-group methods must not be read")

        process_group_annotation = typing.get_args(
            function.__annotations__["group"]
        )[0]
        for group in (
            object(),
            False,
            True,
            0,
            -1,
            "",
            FakeProcessGroup(),
            process_group_annotation(),
        ):
            with self.subTest(group=type(group).__name__):
                self.assert_error(
                    ValueError,
                    INVALID_ERROR,
                    lambda group=group: function(group),
                )

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

        class TracedHash:
            def __init__(self):
                self.events = []

            def __eq__(self, other):
                self.events.append(("eq", other))
                return False

            def __hash__(self):
                self.events.append(("hash",))
                return 123

        self.assertIsNone(function(SentinelEquivalent()))
        self.assert_error(
            RuntimeError,
            "equality failed",
            lambda: function(BrokenEquality()),
        )
        for group in ([], {}, set(), bytearray()):
            with self.subTest(unhashable=type(group).__name__):
                with self.assertRaises(TypeError) as expected_raised:
                    {}.get(group, None)
                with self.assertRaises(TypeError) as actual_raised:
                    function(group)
                self.assertEqual(
                    str(actual_raised.exception),
                    str(expected_raised.exception),
                )
                self.assertEqual(
                    actual_raised.exception.args,
                    expected_raised.exception.args,
                )
        self.assert_error(
            RuntimeError,
            "hash failed",
            lambda: function(BrokenHash()),
        )
        traced_group = TracedHash()
        self.assert_error(
            ValueError,
            INVALID_ERROR,
            lambda: function(traced_group),
        )
        self.assertEqual(traced_group.events, [("eq", -100), ("hash",)])
        self.assert_zero_process_group_state()

    def test_initialization_and_collectives_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertIs(distributed.is_available(), False)
        self.assert_zero_process_group_state()
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "init_process_group",
            "new_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = rf"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {{fullname}}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    USE_DISTRIBUTED="1",
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29500",
    RANK="0",
    WORLD_SIZE="1",
    CUDA_VISIBLE_DEVICES="0",
)
import torch_rs as torch

function = torch.distributed.destroy_process_group
for call in (function, lambda: function(None), lambda: function(group=None)):
    try:
        call()
    except AssertionError as error:
        assert str(error) == {NONE_ERROR!r}
        assert error.args == ({NONE_ERROR!r},)
    else:
        raise AssertionError("destroy_process_group accepted None")
assert function(-100) is None
try:
    function(object())
except ValueError as error:
    assert str(error) == {INVALID_ERROR!r}
    assert error.args == ({INVALID_ERROR!r},)
else:
    raise AssertionError("destroy_process_group accepted an invalid group")
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
assert not hasattr(torch.distributed, "ProcessGroup")
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
