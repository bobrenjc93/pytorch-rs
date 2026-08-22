import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_BACKENDS = ("gloo", "mpi", "nccl", "ucc", "xccl")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedIsBackendAvailableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.is_backend_available differentials require pinned "
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

    def test_supported_simple_name_routing_matches(self):
        for backend in SUPPORTED_BACKENDS:
            query_name = f"is_{backend}_available"
            for module in (torch, reference_torch):
                function = module.distributed.is_backend_available
                marker = object()
                for spelling in (
                    backend,
                    backend.upper(),
                    backend.title(),
                ):
                    with self.subTest(
                        module=module.__name__,
                        backend=backend,
                        spelling=spelling,
                    ):
                        with contextlib.ExitStack() as stack:
                            queries = {
                                name: stack.enter_context(
                                    mock.patch.object(
                                        module.distributed,
                                        f"is_{name}_available",
                                        return_value=(
                                            marker if name == backend else False
                                        ),
                                    )
                                )
                                for name in SUPPORTED_BACKENDS
                            }
                            self.assertIs(function(spelling), marker)
                        queries[backend].assert_called_once_with()
                        for other_backend, query in queries.items():
                            if other_backend != backend:
                                query.assert_not_called()

    def test_unknown_simple_names_and_type_errors_match(self):
        actual = torch.distributed.is_backend_available
        expected = reference_torch.distributed.is_backend_available

        for backend in ("", "unknown", "cpu", " gloo ", "gloo_"):
            with self.subTest(backend=backend):
                self.assertIs(actual(backend), False)
                self.assertIs(expected(backend), False)

        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
            (lambda: actual(b"gloo"), lambda: expected(b"gloo")),
            (lambda: actual(["gloo"]), lambda: expected(["gloo"])),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertIs(actual(backend="gloo"), False)
        self.assertIs(
            type(expected(backend="gloo")),
            bool,
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
        actual = actual_distributed.is_backend_available
        expected = expected_distributed.is_backend_available

        self.assertIs(torch.distributed, actual_distributed)
        self.assertIs(reference_torch.distributed, expected_distributed)
        self.assertIs(actual_distributed.distributed_c10d, actual_c10d)
        self.assertIs(expected_distributed.distributed_c10d, expected_c10d)
        self.assertIs(actual_c10d.is_backend_available, actual)
        self.assertIs(expected_c10d.is_backend_available, expected)
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
        actual = actual_distributed.is_backend_available
        expected = expected_distributed.is_backend_available
        supported = {
            "get_pg_count",
            "is_gloo_available",
            "is_initialized",
            "is_mpi_available",
            "is_backend_available",
            "is_nccl_available",
            "is_ucc_available",
            "is_xccl_available",
            "get_node_local_rank",
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
            torch.__all__.count("is_backend_available"),
            reference_torch.__all__.count("is_backend_available"),
        )

        for module, function in (
            (actual_distributed, actual),
            (expected_distributed, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["is_backend_available"], function)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("is_backend_available", namespace)

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
        actual = torch.distributed.is_backend_available
        expected = reference_torch.distributed.is_backend_available
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual("gloo", "mpi"),
                lambda: expected("gloo", "mpi"),
            ),
            (
                lambda: actual(enabled=True),
                lambda: expected(enabled=True),
            ),
            (
                lambda: actual("gloo", backend="mpi"),
                lambda: expected("gloo", backend="mpi"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_explicitly_unsupported_distributed_surface_stays_absent(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d

        for backend in (
            "cpu:gloo",
            "cuda:nccl",
            "gloo:nccl",
            "cpu:gloo,cuda:nccl",
        ):
            with self.subTest(backend=backend):
                self.assertIs(
                    actual_distributed.is_backend_available(backend), False
                )

        third_party_query = mock.Mock(return_value=True)
        with mock.patch.object(
            actual_distributed,
            "is_third_party_available",
            third_party_query,
            create=True,
        ):
            self.assertIs(
                actual_distributed.is_backend_available("third_party"), False
            )
        third_party_query.assert_not_called()

        for name in (
            "Backend",
            "BackendConfig",
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
                self.assertTrue(hasattr(expected_c10d, name))
                self.assertFalse(hasattr(actual_distributed, name))
                self.assertFalse(hasattr(actual_c10d, name))


if __name__ == "__main__":
    unittest.main()
