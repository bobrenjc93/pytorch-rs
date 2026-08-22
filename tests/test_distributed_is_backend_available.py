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
    Check backend availability.

    Checks if the given backend is available and supports the built-in backends or
    third-party backends through function ``Backend.register_backend``.

    Args:
        backend (str): Backend name.
    Returns:
        bool: Returns true if the backend is available otherwise false.
    """
SUPPORTED_BACKENDS = ("gloo", "mpi", "nccl", "ucc", "xccl")


class DistributedIsBackendAvailableTests(unittest.TestCase):
    def test_routes_supported_names_case_insensitively(self):
        distributed = torch.distributed
        function = distributed.is_backend_available

        for backend in SUPPORTED_BACKENDS:
            query_name = f"is_{backend}_available"
            marker = object()
            for spelling in (backend, backend.upper(), backend.title()):
                with self.subTest(backend=backend, spelling=spelling):
                    with contextlib.ExitStack() as stack:
                        queries = {
                            name: stack.enter_context(
                                mock.patch.object(
                                    distributed,
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

    def test_unknown_simple_and_composite_names_are_unavailable(self):
        distributed = torch.distributed
        function = distributed.is_backend_available
        queries = [
            mock.patch.object(distributed, f"is_{backend}_available")
            for backend in SUPPORTED_BACKENDS
        ]

        with contextlib.ExitStack() as stack:
            query_mocks = [stack.enter_context(query) for query in queries]
            for backend in (
                "",
                "unknown",
                "cpu",
                " gloo ",
                "gloo_",
                ":",
                "cpu:gloo",
                "cuda:nccl",
                "gloo:nccl",
                "cpu:gloo,cuda:nccl",
            ):
                with self.subTest(backend=backend):
                    self.assertIs(function(backend), False)

            for query in query_mocks:
                query.assert_not_called()

        third_party_query = mock.Mock(return_value=True)
        with mock.patch.object(
            distributed,
            "is_third_party_available",
            third_party_query,
            create=True,
        ):
            self.assertIs(function("third_party"), False)
        third_party_query.assert_not_called()

    def test_results_are_stable_without_environment_or_grad_side_effects(self):
        function = torch.distributed.is_backend_available
        environments = (
            {},
            {"USE_DISTRIBUTED": "1", "USE_GLOO": "1", "USE_NCCL": "1"},
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
                                results[index] = (
                                    torch.is_grad_enabled(),
                                    function("GLOO"),
                                    function("unknown"),
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
                        self.assertEqual(result, (index % 2 == 0, False, False))

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
        self.assertEqual(
            str(inspect.signature(function)), "(backend: str) -> bool"
        )
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
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

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

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

        for module_name in (
            "torch_rs.distributed",
            "torch_rs.distributed.distributed_c10d",
        ):
            namespace = {}
            exec(f"from {module_name} import is_backend_available", namespace)
            self.assertIs(namespace["is_backend_available"], function)

            wildcard_namespace = {}
            exec(f"from {module_name} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["is_backend_available"], function)

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

    def test_argument_and_type_errors_match_pytorch_2_13(self):
        function = torch.distributed.is_backend_available
        cases = (
            (
                lambda: function(),
                "is_backend_available() missing 1 required positional argument: "
                "'backend'",
            ),
            (
                lambda: function("gloo", "mpi"),
                "is_backend_available() takes 1 positional argument but 2 were "
                "given",
            ),
            (
                lambda: function(enabled=True),
                "is_backend_available() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function("gloo", backend="mpi"),
                "is_backend_available() got multiple values for argument "
                "'backend'",
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
                with self.assertRaises((AttributeError, TypeError)) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(backend="gloo"), False)

    def test_distributed_execution_and_registration_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertIs(distributed.is_available(), False)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)
        for backend in SUPPORTED_BACKENDS:
            self.assertIs(distributed.is_backend_available(backend), False)

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
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    USE_DISTRIBUTED="1",
    USE_GLOO="1",
    USE_MPI="1",
    USE_NCCL="1",
    USE_UCC="1",
    USE_XCCL="1",
    CUDA_VISIBLE_DEVICES="0",
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29500",
    RANK="0",
    WORLD_SIZE="1",
)
import torch_rs as torch

function = torch.distributed.is_backend_available
for backend in ("gloo", "MPI", "NcCl", "ucc", "XCCL"):
    assert function(backend) is False
for backend in ("unknown", "cpu:gloo", "cuda:nccl"):
    assert function(backend) is False
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
