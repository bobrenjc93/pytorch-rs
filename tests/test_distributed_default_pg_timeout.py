import copy
from datetime import timedelta
import importlib
import os
import pickle
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch_rs as torch


class DistributedDefaultPgTimeoutTests(unittest.TestCase):
    def test_exact_value_and_canonical_identity(self):
        distributed = importlib.import_module("torch_rs.distributed")
        constants = importlib.import_module("torch_rs.distributed.constants")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        timeout = constants.default_pg_timeout

        self.assertIs(type(timeout), timedelta)
        self.assertEqual(timeout, timedelta(minutes=30))
        self.assertEqual(
            (timeout.days, timeout.seconds, timeout.microseconds),
            (0, 1800, 0),
        )
        self.assertEqual(timeout.total_seconds(), 1800.0)
        self.assertIs(timeout, constants._DEFAULT_PG_TIMEOUT)
        self.assertIs(timeout, distributed_c10d.default_pg_timeout)
        self.assertIs(timeout, distributed.default_pg_timeout)
        self.assertIs(timeout, torch.distributed.default_pg_timeout)

    def test_module_metadata_and_imports(self):
        distributed = torch.distributed
        constants = importlib.import_module("torch_rs.distributed.constants")
        distributed_c10d = distributed.distributed_c10d
        timeout = distributed.default_pg_timeout

        self.assertIs(type(constants), types.ModuleType)
        self.assertIs(distributed.constants, constants)
        self.assertIs(sys.modules["torch_rs.distributed.constants"], constants)
        self.assertIsNone(constants.__doc__)
        self.assertEqual(constants.__name__, "torch_rs.distributed.constants")
        self.assertEqual(constants.__package__, "torch_rs.distributed")
        self.assertEqual(constants.__all__, ["default_pg_timeout"])
        self.assertEqual(constants.__annotations__, {"default_pg_timeout": timedelta})
        self.assertEqual(
            {name for name in vars(constants) if not name.startswith("_")},
            {"timedelta", "default_pg_timeout"},
        )
        self.assertFalse(hasattr(distributed, "__all__"))
        self.assertEqual(distributed_c10d.__all__.count("default_pg_timeout"), 1)

        package_import = {}
        constants_import = {}
        direct_import = {}
        owner_import = {}
        exec("from torch_rs.distributed import constants", package_import)
        exec(
            "import torch_rs.distributed.constants as constants",
            constants_import,
        )
        exec(
            "from torch_rs.distributed import default_pg_timeout",
            direct_import,
        )
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "default_pg_timeout",
            owner_import,
        )
        self.assertIs(package_import["constants"], constants)
        self.assertIs(constants_import["constants"], constants)
        self.assertIs(direct_import["default_pg_timeout"], timeout)
        self.assertIs(owner_import["default_pg_timeout"], timeout)

        constants_namespace = {}
        distributed_namespace = {}
        owner_namespace = {}
        top_level_namespace = {}
        exec("from torch_rs.distributed.constants import *", constants_namespace)
        exec("from torch_rs.distributed import *", distributed_namespace)
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_namespace,
        )
        exec("from torch_rs import *", top_level_namespace)
        self.assertEqual(
            [name for name in constants_namespace if name != "__builtins__"],
            ["default_pg_timeout"],
        )
        self.assertIs(constants_namespace["default_pg_timeout"], timeout)
        self.assertIs(distributed_namespace["constants"], constants)
        self.assertIs(distributed_namespace["default_pg_timeout"], timeout)
        self.assertIs(owner_namespace["default_pg_timeout"], timeout)
        self.assertNotIn("distributed", torch.__all__)
        self.assertNotIn("default_pg_timeout", torch.__all__)
        self.assertNotIn("constants", top_level_namespace)
        self.assertNotIn("default_pg_timeout", top_level_namespace)

    def test_copy_and_pickle_follow_timedelta_behavior(self):
        timeout = torch.distributed.default_pg_timeout

        for copied in (copy.copy(timeout), copy.deepcopy(timeout)):
            self.assertIsNot(copied, timeout)
            self.assertIs(type(copied), timedelta)
            self.assertEqual(copied, timeout)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(timeout, protocol=protocol)
                restored = pickle.loads(payload)
                self.assertIsNot(restored, timeout)
                self.assertIs(type(restored), timedelta)
                self.assertEqual(restored, timeout)
                self.assertIn(b"datetime", payload)
                self.assertIn(b"timedelta", payload)
                self.assertNotIn(b"torch_rs", payload)

    def test_reload_restores_exports_without_recreating_the_timeout(self):
        distributed = torch.distributed
        constants = distributed.constants
        distributed_c10d = distributed.distributed_c10d
        timeout = distributed.default_pg_timeout
        constants_namespace = constants.__dict__
        constants_all = constants.__all__
        constants_annotations = constants.__annotations__

        constants.default_pg_timeout = "stale"
        constants._DEFAULT_PG_TIMEOUT = "stale"
        constants.__all__ = []
        constants.__annotations__["default_pg_timeout"] = str
        distributed_c10d.default_pg_timeout = "stale"
        distributed.default_pg_timeout = "stale"

        self.assertIs(importlib.reload(constants), constants)
        self.assertIs(constants.__dict__, constants_namespace)
        self.assertIsNot(constants.__all__, constants_all)
        self.assertIs(constants.__annotations__, constants_annotations)
        self.assertEqual(constants.__all__, ["default_pg_timeout"])
        self.assertEqual(constants.__annotations__, {"default_pg_timeout": timedelta})
        self.assertIs(constants.default_pg_timeout, timeout)
        self.assertIs(constants._DEFAULT_PG_TIMEOUT, timeout)

        self.assertIs(importlib.reload(distributed_c10d), distributed_c10d)
        self.assertIs(distributed_c10d.default_pg_timeout, timeout)
        self.assertIs(importlib.reload(distributed), distributed)
        self.assertIs(distributed.constants, constants)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed.default_pg_timeout, timeout)
        self.assertIs(torch.distributed.default_pg_timeout, timeout)

    def test_metadata_does_not_consult_environment_or_process_groups(self):
        distributed = torch.distributed
        constants = distributed.constants
        distributed_c10d = distributed.distributed_c10d
        timeout = distributed.default_pg_timeout

        environments = (
            {},
            {"USE_DISTRIBUTED": "1"},
            {
                "BACKEND": "nccl",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "WORLD_SIZE": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(constants.default_pg_timeout, timeout)
                    self.assertIs(distributed.default_pg_timeout, timeout)

        with (
            mock.patch.object(
                distributed_c10d,
                "is_initialized",
                side_effect=AssertionError("process-group state was queried"),
            ),
            mock.patch.object(
                distributed_c10d,
                "get_backend",
                side_effect=AssertionError("backend execution was queried"),
            ),
        ):
            self.assertIs(constants.default_pg_timeout, timeout)
            self.assertIs(distributed.default_pg_timeout, timeout)

        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)

    def test_import_does_not_import_pytorch_or_initialize_distributed(self):
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
    BACKEND="nccl",
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29500",
    RANK="0",
    WORLD_SIZE="1",
)
from datetime import timedelta
import torch_rs as torch
from torch_rs.distributed.constants import default_pg_timeout

assert type(default_pg_timeout) is timedelta
assert default_pg_timeout == timedelta(minutes=30)
assert default_pg_timeout is torch.distributed.default_pg_timeout
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
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
