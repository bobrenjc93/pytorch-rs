import copy
import importlib
import importlib.util
import inspect
import os
import pickle
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """
    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).

    The existence of ``TORCHELASTIC_RUN_ID`` environment
    variable is used as a proxy to determine whether the current process
    was launched with torchelastic. This is a reasonable proxy since
    ``TORCHELASTIC_RUN_ID`` maps to the rendezvous id which is always a
    non-null value indicating the job id for peer discovery purposes..
    """


class DistributedIsTorchelasticLaunchedTests(unittest.TestCase):
    def test_returns_exact_bool_from_run_id_presence(self):
        function = torch.distributed.is_torchelastic_launched

        environments = (
            ({}, False),
            ({"TORCHELASTIC_RUN_ID": ""}, True),
            ({"TORCHELASTIC_RUN_ID": "0"}, True),
            ({"TORCHELASTIC_RUN_ID": "false"}, True),
            ({"TORCHELASTIC_RUN_ID": "job-123"}, True),
            (
                {
                    "LOCAL_RANK": "0",
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29500",
                    "RANK": "0",
                    "TORCHELASTIC_RESTART_COUNT": "4",
                    "WORLD_SIZE": "1",
                },
                False,
            ),
        )
        for environment, expected in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, expected)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIs(function(), False)
            os.environ["TORCHELASTIC_RUN_ID"] = ""
            self.assertIs(function(), True)
            os.environ["TORCHELASTIC_RUN_ID"] = "next-run"
            self.assertIs(function(), True)
            del os.environ["TORCHELASTIC_RUN_ID"]
            self.assertIs(function(), False)

    def test_queries_only_the_canonical_environment_key(self):
        function = torch.distributed.is_torchelastic_launched

        with mock.patch.object(os, "getenv", return_value=None) as getenv:
            self.assertIs(function(), False)
            getenv.assert_called_once_with("TORCHELASTIC_RUN_ID")

        marker = object()
        with mock.patch.object(os, "getenv", return_value=marker) as getenv:
            self.assertIs(function(), True)
            getenv.assert_called_once_with("TORCHELASTIC_RUN_ID")

    def test_signature_annotations_documentation_and_module_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        function = distributed.is_torchelastic_launched

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.is_torchelastic_launched, function)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(
            sys.modules["torch_rs.distributed.distributed_c10d"],
            distributed_c10d,
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_torchelastic_launched")
        self.assertEqual(function.__qualname__, "is_torchelastic_launched")
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
        function = distributed.is_torchelastic_launched

        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(
            distributed_c10d.__all__,
            [
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            ],
        )

        package_import = {}
        exec("from torch_rs import distributed", package_import)
        self.assertIs(package_import["distributed"], distributed)

        direct_import = {}
        exec(
            "from torch_rs.distributed import is_torchelastic_launched",
            direct_import,
        )
        self.assertIs(direct_import["is_torchelastic_launched"], function)

        owner_import = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "is_torchelastic_launched",
            owner_import,
        )
        self.assertIs(owner_import["is_torchelastic_launched"], function)

        distributed_namespace = {}
        exec("from torch_rs.distributed import *", distributed_namespace)
        self.assertEqual(
            {
                name
                for name in distributed_namespace
                if not name.startswith("__")
            },
            {
                "distributed_c10d",
                "is_available",
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            },
        )
        self.assertIs(
            distributed_namespace["is_torchelastic_launched"], function
        )
        self.assertIs(
            distributed_namespace["distributed_c10d"], distributed_c10d
        )

        owner_namespace = {}
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        self.assertEqual(
            {name for name in owner_namespace if not name.startswith("__")},
            {
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            },
        )
        self.assertIs(owner_namespace["is_torchelastic_launched"], function)

        self.assertNotIn("distributed", torch.__all__)
        self.assertNotIn("is_torchelastic_launched", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("distributed", top_level_namespace)
        self.assertNotIn("is_torchelastic_launched", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    b"torch_rs.distributed.distributed_c10d", payload
                )
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.distributed.is_torchelastic_launched
        cases = (
            (
                lambda: function(None),
                "is_torchelastic_launched() takes 0 positional arguments "
                "but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_torchelastic_launched() takes 0 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_torchelastic_launched() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_torchelastic_launched() got an unexpected keyword "
                "argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_elastic_execution_process_groups_and_collectives_remain_unsupported(self):
        distributed = torch.distributed
        distributed_c10d = distributed.distributed_c10d

        self.assertEqual(
            {name for name in vars(distributed) if not name.startswith("_")},
            {
                "distributed_c10d",
                "is_available",
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            },
        )
        self.assertEqual(
            {
                name
                for name in vars(distributed_c10d)
                if not name.startswith("_")
            },
            {
                "is_initialized",
                "is_nccl_available",
                "is_torchelastic_launched",
            },
        )
        for name in (
            "GroupMember",
            "ProcessGroup",
            "all_reduce",
            "destroy_process_group",
            "elastic",
            "get_rank",
            "get_world_size",
            "init_process_group",
            "new_group",
            "run",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(distributed, name))
                self.assertFalse(hasattr(distributed_c10d, name))
        self.assertIsNone(
            importlib.util.find_spec("torch_rs.distributed.elastic")
        )
        self.assertNotIn("torch_rs.distributed.elastic", sys.modules)
        self.assertFalse(hasattr(torch, "is_torchelastic_launched"))

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_RUN_ID": "run"}, clear=True
        ):
            self.assertIs(distributed.is_torchelastic_launched(), True)
            self.assertIs(distributed.is_available(), False)
            self.assertIs(distributed.is_initialized(), False)
            self.assertIs(distributed.is_nccl_available(), False)

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
os.environ.pop("TORCHELASTIC_RUN_ID", None)
import torch_rs as torch

function = torch.distributed.is_torchelastic_launched
assert function() is False
os.environ["TORCHELASTIC_RUN_ID"] = ""
assert function() is True
os.environ["TORCHELASTIC_RUN_ID"] = "run-id"
assert function() is True
del os.environ["TORCHELASTIC_RUN_ID"]
assert function() is False
assert torch.distributed.is_available() is False
assert torch.distributed.is_initialized() is False
assert torch.distributed.is_nccl_available() is False
assert not hasattr(torch.distributed, "elastic")
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
