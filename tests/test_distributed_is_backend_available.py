import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest
from contextlib import ExitStack
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """
    Check backend availability.

    Checks if the given backend is available and supports the built-in backends or
    third-party backends through function ``Backend.register_backend``.

    Args:
        backend (str): Backend name.
    Returns:
        bool: Returns true if the backend is available otherwise false.
    """

BACKEND_QUERIES = {
    "gloo": "is_gloo_available",
    "mpi": "is_mpi_available",
    "nccl": "is_nccl_available",
    "ucc": "is_ucc_available",
    "xccl": "is_xccl_available",
}


class DistributedIsBackendAvailableTests(unittest.TestCase):
    def test_routes_supported_names_case_insensitively(self):
        distributed = torch.distributed
        function = distributed.is_backend_available

        for backend, query_name in BACKEND_QUERIES.items():
            for spelling in (backend, backend.upper(), backend.title()):
                with self.subTest(backend=backend, spelling=spelling):
                    with ExitStack() as stack:
                        queries = {
                            name: stack.enter_context(
                                mock.patch.object(
                                    distributed,
                                    name,
                                    return_value=name == query_name,
                                )
                            )
                            for name in BACKEND_QUERIES.values()
                        }
                        self.assertIs(function(spelling), True)
                    queries[query_name].assert_called_once_with()
                    for other_name, query in queries.items():
                        if other_name != query_name:
                            query.assert_not_called()

    def test_current_backend_queries_return_exact_false(self):
        distributed = torch.distributed

        for backend, query_name in BACKEND_QUERIES.items():
            with self.subTest(backend=backend):
                expected = getattr(distributed, query_name)()
                self.assertIs(expected, False)
                self.assertIs(distributed.is_backend_available(backend), expected)

    def test_unknown_and_composite_names_are_unsupported(self):
        distributed = torch.distributed

        with ExitStack() as stack:
            queries = [
                stack.enter_context(mock.patch.object(distributed, query_name))
                for query_name in BACKEND_QUERIES.values()
            ]
            for backend in (
                "",
                "bogus",
                "cpu",
                "cuda",
                " gloo ",
                "gloo:foo",
                "cpu:gloo,cuda:nccl",
            ):
                with self.subTest(backend=backend):
                    self.assertIs(distributed.is_backend_available(backend), False)
            for query in queries:
                query.assert_not_called()

        distributed.is_bogus_available = lambda: True
        try:
            self.assertIs(distributed.is_backend_available("bogus"), False)
        finally:
            del distributed.is_bogus_available

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.is_backend_available

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.is_backend_available, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(backend: str) -> bool")
        self.assertEqual(
            function.__annotations__, {"backend": str, "return": bool}
        )
        self.assertEqual(
            typing.get_type_hints(function), {"backend": str, "return": bool}
        )
        self.assertEqual(function.__name__, "is_backend_available")
        self.assertEqual(function.__qualname__, "is_backend_available")
        self.assertEqual(
            function.__module__, "torch_rs.distributed.distributed_c10d"
        )
        self.assertIs(inspect.getmodule(function), distributed_c10d)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copy_wildcards_and_pickle_use_the_canonical_module(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d
        function = distributed.is_backend_available

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
                "get_pg_count",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_backend_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            ],
        )

        direct_import = {}
        exec("from torch_rs.distributed import is_backend_available", direct_import)
        self.assertIs(direct_import["is_backend_available"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "is_backend_available",
            owner_import,
        )
        self.assertIs(owner_import["is_backend_available"], function)

        for module in (distributed, distributed_c10d):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["is_backend_available"], function)

        self.assertNotIn("is_backend_available", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("is_backend_available", top_level_namespace)

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
        function = torch.distributed.is_backend_available
        cases = (
            (
                lambda: function(),
                "is_backend_available() missing 1 required positional argument: "
                "'backend'",
            ),
            (
                lambda: function("gloo", "nccl"),
                "is_backend_available() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(name="gloo"),
                "is_backend_available() got an unexpected keyword argument 'name'",
            ),
            (
                lambda: function("gloo", backend="nccl"),
                "is_backend_available() got multiple values for argument 'backend'",
            ),
            (
                lambda: function(None),
                "'NoneType' object has no attribute 'lower'",
            ),
            (
                lambda: function(1),
                "'int' object has no attribute 'lower'",
            ),
            (
                lambda: function(b"gloo"),
                "a bytes-like object is required, not 'str'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises((TypeError, AttributeError)) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(backend="gloo"), False)

    def test_execution_registration_and_hardware_probing_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertEqual(
            {name for name in vars(distributed) if not name.startswith("_")},
            {
                "distributed_c10d",
                "get_pg_count",
                "is_available",
                "is_gloo_available",
                "is_initialized",
                "is_mpi_available",
                "is_backend_available",
                "is_nccl_available",
                "is_ucc_available",
                "is_xccl_available",
                "get_node_local_rank",
            },
        )
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
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))
        self.assertFalse(hasattr(torch, "is_backend_available"))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.distributed.is_backend_available
for backend in ("gloo", "GLOO", "mpi", "nccl", "ucc", "xccl"):
    assert function(backend) is False
assert function("unknown") is False
assert function("cpu:gloo,cuda:nccl") is False
assert not hasattr(torch.distributed, "Backend")
assert not hasattr(torch.distributed, "init_process_group")
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
