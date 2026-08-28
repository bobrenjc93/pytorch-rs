import copy
import importlib
import os
import pickle
import subprocess
import sys
import threading
import types
import unittest
from datetime import timedelta
from unittest import mock

import torch_rs as torch


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class DistributedDefaultPgTimeoutTests(unittest.TestCase):
    def test_value_is_an_exact_thirty_minute_timedelta(self):
        value = torch.distributed.constants.default_pg_timeout

        self.assertIs(type(value), timedelta)
        self.assertEqual(value, timedelta(minutes=30))
        self.assertEqual(
            (value.days, value.seconds, value.microseconds),
            (0, 1800, 0),
        )
        self.assertEqual(value.total_seconds(), 1800.0)

    def test_module_direct_and_wildcard_imports_share_one_object(self):
        distributed = importlib.import_module("torch_rs.distributed")
        constants = importlib.import_module("torch_rs.distributed.constants")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        value = constants.default_pg_timeout

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.constants, constants)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(constants._DEFAULT_PG_TIMEOUT, value)
        self.assertIs(distributed_c10d.default_pg_timeout, value)
        self.assertIs(distributed.default_pg_timeout, value)
        self.assertIs(sys.modules["torch_rs.distributed"], distributed)
        self.assertIs(sys.modules["torch_rs.distributed.constants"], constants)
        self.assertIs(type(constants), types.ModuleType)
        self.assertEqual(constants.__name__, "torch_rs.distributed.constants")
        self.assertEqual(constants.__package__, "torch_rs.distributed")
        self.assertEqual(constants.__all__, ["default_pg_timeout"])
        self.assertEqual(
            constants.__annotations__, {"default_pg_timeout": timedelta}
        )
        self.assertEqual(
            {name for name in vars(constants) if not name.startswith("_")},
            {"timedelta", "default_pg_timeout"},
        )
        self.assertEqual(distributed_c10d.__all__.count("default_pg_timeout"), 1)

        package_import = {}
        module_import = {}
        constant_import = {}
        owner_import = {}
        reexport_import = {}
        constants_wildcard = {}
        owner_wildcard = {}
        distributed_wildcard = {}
        exec("from torch_rs.distributed import constants", package_import)
        exec(
            "import torch_rs.distributed.constants as constants",
            module_import,
        )
        exec(
            "from torch_rs.distributed.constants import default_pg_timeout",
            constant_import,
        )
        exec(
            "from torch_rs.distributed.distributed_c10d import "
            "default_pg_timeout",
            owner_import,
        )
        exec(
            "from torch_rs.distributed import default_pg_timeout",
            reexport_import,
        )
        exec("from torch_rs.distributed.constants import *", constants_wildcard)
        exec(
            "from torch_rs.distributed.distributed_c10d import *",
            owner_wildcard,
        )
        exec("from torch_rs.distributed import *", distributed_wildcard)

        self.assertIs(package_import["constants"], constants)
        self.assertIs(module_import["constants"], constants)
        self.assertIs(constant_import["default_pg_timeout"], value)
        self.assertIs(owner_import["default_pg_timeout"], value)
        self.assertIs(reexport_import["default_pg_timeout"], value)
        self.assertEqual(
            {
                name
                for name in constants_wildcard
                if not name.startswith("__")
            },
            {"default_pg_timeout"},
        )
        self.assertIs(constants_wildcard["default_pg_timeout"], value)
        self.assertIs(owner_wildcard["default_pg_timeout"], value)
        self.assertIs(distributed_wildcard["constants"], constants)
        self.assertIs(distributed_wildcard["default_pg_timeout"], value)

        self.assertNotIn("default_pg_timeout", torch.__all__)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("default_pg_timeout", top_level_wildcard)

    def test_reload_restores_bindings_and_preserves_canonical_identity(self):
        distributed = torch.distributed
        constants = distributed.constants
        distributed_c10d = distributed.distributed_c10d
        value = constants.default_pg_timeout
        constants_namespace = constants.__dict__
        distributed_namespace = distributed.__dict__

        constants.default_pg_timeout = timedelta(seconds=1)
        constants._DEFAULT_PG_TIMEOUT = timedelta(seconds=2)
        reloaded_constants = importlib.reload(constants)

        self.assertIs(reloaded_constants, constants)
        self.assertIs(constants.__dict__, constants_namespace)
        self.assertIs(constants._DEFAULT_PG_TIMEOUT, value)
        self.assertIs(constants.default_pg_timeout, value)
        self.assertIs(distributed.constants, constants)
        self.assertIs(distributed.default_pg_timeout, value)

        distributed_c10d.default_pg_timeout = timedelta(seconds=3)
        distributed.default_pg_timeout = timedelta(seconds=4)
        reloaded_c10d = importlib.reload(distributed_c10d)
        reloaded_distributed = importlib.reload(distributed)

        self.assertIs(reloaded_c10d, distributed_c10d)
        self.assertIs(reloaded_distributed, distributed)
        self.assertIs(distributed.__dict__, distributed_namespace)
        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.constants, constants)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(distributed_c10d.default_pg_timeout, value)
        self.assertIs(distributed.default_pg_timeout, value)
        self.assertIs(constants.default_pg_timeout, value)

    def test_copy_and_pickle_follow_timedelta_value_semantics(self):
        value = torch.distributed.default_pg_timeout
        expected = timedelta(minutes=30)

        for copied in (copy.copy(value), copy.deepcopy(value)):
            self.assertIs(type(copied), timedelta)
            self.assertEqual(copied, value)
            self.assertIsNot(copied, value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(value, protocol=protocol)
                self.assertEqual(
                    payload,
                    pickle.dumps(expected, protocol=protocol),
                )
                restored = pickle.loads(payload)
                self.assertIs(type(restored), timedelta)
                self.assertEqual(restored, value)
                self.assertIsNot(restored, value)

    def test_alias_identity_is_stable_across_threads(self):
        distributed = torch.distributed
        constants = distributed.constants
        distributed_c10d = distributed.distributed_c10d
        value = distributed.default_pg_timeout
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    constants.default_pg_timeout,
                    distributed_c10d.default_pg_timeout,
                    distributed.default_pg_timeout,
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
        for result in results:
            self.assertTrue(all(alias is value for alias in result))

    def test_access_does_not_observe_environment_or_distributed_runtime(self):
        distributed = torch.distributed
        constants = distributed.constants
        distributed_c10d = distributed.distributed_c10d
        value = distributed.default_pg_timeout
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
                    self.assertIs(distributed.default_pg_timeout, value)
                    self.assertIs(
                        distributed.constants.default_pg_timeout,
                        value,
                    )
                    self.assertIs(
                        distributed_c10d.default_pg_timeout,
                        value,
                    )

        with mock.patch.object(os, "environ", UnreadableEnvironment()):
            constants = importlib.reload(constants)
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)

        self.assertIs(constants.default_pg_timeout, value)
        self.assertIs(distributed_c10d.default_pg_timeout, value)
        self.assertIs(distributed.default_pg_timeout, value)

        with mock.patch.object(os, "environ", UnreadableEnvironment()):
            with mock.patch.object(
                distributed,
                "is_initialized",
                side_effect=AssertionError("process-group state was queried"),
            ):
                with mock.patch.object(
                    distributed,
                    "get_backend",
                    side_effect=AssertionError("backend execution was queried"),
                ):
                    self.assertIs(distributed.default_pg_timeout, value)
                    self.assertIs(
                        distributed.constants.default_pg_timeout,
                        value,
                    )
                    self.assertIs(
                        distributed_c10d.default_pg_timeout,
                        value,
                    )

    def test_import_does_not_import_pytorch_or_initialize_distributed(self):
        script = r"""
import os
import sys
from datetime import timedelta

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
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
from torch_rs.distributed import default_pg_timeout
from torch_rs.distributed.constants import default_pg_timeout as direct

assert default_pg_timeout is direct
assert direct is torch.distributed.default_pg_timeout
assert direct is torch.distributed.constants.default_pg_timeout
assert direct is torch.distributed.distributed_c10d.default_pg_timeout
assert type(direct) is timedelta
assert direct == timedelta(minutes=30)
assert torch.distributed.is_available() is False
assert torch.distributed.is_initialized() is False
assert torch.distributed.get_pg_count() == 0
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
