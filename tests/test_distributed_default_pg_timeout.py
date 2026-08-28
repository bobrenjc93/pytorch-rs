import copy
import datetime
import importlib
import os
import pickle
import subprocess
import sys
import threading
import unittest
from unittest import mock

import torch_rs as torch


EXPECTED_TIMEOUT = datetime.timedelta(minutes=30)


class UnreadableEnvironment:
    def __contains__(self, key):
        raise AssertionError(f"environment membership was read: {key}")

    def __getitem__(self, key):
        raise AssertionError(f"environment value was read: {key}")

    def get(self, key, default=None):
        raise AssertionError(f"environment value was read: {key}")


class DistributedDefaultPgTimeoutTests(unittest.TestCase):
    def modules_and_value(self):
        distributed = importlib.import_module("torch_rs.distributed")
        constants = importlib.import_module("torch_rs.distributed.constants")
        distributed_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        return (
            distributed,
            constants,
            distributed_c10d,
            constants.default_pg_timeout,
        )

    def test_exact_timedelta_value_and_canonical_identity(self):
        distributed, constants, distributed_c10d, timeout = (
            self.modules_and_value()
        )

        self.assertIs(type(timeout), datetime.timedelta)
        self.assertEqual(timeout, EXPECTED_TIMEOUT)
        self.assertEqual(timeout.days, 0)
        self.assertEqual(timeout.seconds, 30 * 60)
        self.assertEqual(timeout.microseconds, 0)
        self.assertEqual(timeout.total_seconds(), 1800.0)
        self.assertIs(constants._DEFAULT_PG_TIMEOUT, timeout)
        self.assertIs(distributed_c10d.default_pg_timeout, timeout)
        self.assertIs(distributed.default_pg_timeout, timeout)
        self.assertIs(torch.distributed.default_pg_timeout, timeout)

    def test_direct_and_wildcard_imports_share_the_same_object(self):
        distributed, constants, distributed_c10d, timeout = (
            self.modules_and_value()
        )

        self.assertIs(torch.distributed, distributed)
        self.assertIs(distributed.constants, constants)
        self.assertIs(distributed.distributed_c10d, distributed_c10d)
        self.assertIs(sys.modules[constants.__name__], constants)
        self.assertIs(sys.modules[distributed_c10d.__name__], distributed_c10d)
        self.assertEqual(constants.__all__, ["default_pg_timeout"])
        self.assertEqual(
            constants.__annotations__,
            {"default_pg_timeout": datetime.timedelta},
        )
        self.assertEqual(
            {name for name in vars(constants) if not name.startswith("_")},
            {"default_pg_timeout", "timedelta"},
        )
        self.assertEqual(distributed_c10d.__all__.count("default_pg_timeout"), 1)

        for statement, name in (
            (
                "from torch_rs.distributed.constants import default_pg_timeout",
                "default_pg_timeout",
            ),
            (
                "from torch_rs.distributed.distributed_c10d import "
                "default_pg_timeout",
                "default_pg_timeout",
            ),
            (
                "from torch_rs.distributed import default_pg_timeout",
                "default_pg_timeout",
            ),
        ):
            with self.subTest(statement=statement):
                namespace = {}
                exec(statement, namespace)
                self.assertIs(namespace[name], timeout)

        for module in (constants, distributed_c10d, distributed):
            with self.subTest(module=module.__name__):
                namespace = {}
                exec(f"from {module.__name__} import *", namespace)
                self.assertIs(namespace["default_pg_timeout"], timeout)

        self.assertNotIn("default_pg_timeout", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("default_pg_timeout", top_level_namespace)

    def test_reload_restores_and_preserves_the_canonical_object(self):
        distributed, constants, distributed_c10d, timeout = (
            self.modules_and_value()
        )
        direct_import = timeout

        replacement = datetime.timedelta(seconds=1)
        constants.default_pg_timeout = replacement
        constants._DEFAULT_PG_TIMEOUT = replacement
        self.assertIsNot(constants.default_pg_timeout, timeout)

        for _ in range(3):
            constants = importlib.reload(constants)
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)

            self.assertIs(constants.default_pg_timeout, timeout)
            self.assertIs(constants._DEFAULT_PG_TIMEOUT, timeout)
            self.assertIs(distributed_c10d.default_pg_timeout, timeout)
            self.assertIs(distributed.default_pg_timeout, timeout)
            self.assertIs(distributed.constants, constants)
            self.assertIs(distributed.distributed_c10d, distributed_c10d)
            self.assertIs(torch.distributed, distributed)
            self.assertIs(direct_import, timeout)

    def test_copy_deepcopy_and_pickle_match_timedelta_semantics(self):
        timeout = torch.distributed.default_pg_timeout

        for copier in (copy.copy, copy.deepcopy):
            with self.subTest(copier=copier.__name__):
                copied = copier(timeout)
                self.assertIs(type(copied), datetime.timedelta)
                self.assertEqual(copied, timeout)
                self.assertIsNot(copied, timeout)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(timeout, protocol=protocol)
                self.assertEqual(
                    payload,
                    pickle.dumps(EXPECTED_TIMEOUT, protocol=protocol),
                )
                restored = pickle.loads(payload)
                self.assertIs(type(restored), datetime.timedelta)
                self.assertEqual(restored, timeout)
                self.assertIsNot(restored, timeout)

    def test_identity_is_stable_across_threads_and_environments(self):
        distributed, constants, distributed_c10d, timeout = (
            self.modules_and_value()
        )
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
                "WORLD_SIZE": "8",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(constants.default_pg_timeout, timeout)
                    self.assertIs(distributed_c10d.default_pg_timeout, timeout)
                    self.assertIs(distributed.default_pg_timeout, timeout)
                    self.assertIs(distributed.is_initialized(), False)
                    self.assertEqual(distributed.get_pg_count(), 0)

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count

        def worker(index):
            barrier.wait(timeout=10)
            results[index] = (
                constants.default_pg_timeout,
                distributed_c10d.default_pg_timeout,
                distributed.default_pg_timeout,
            )

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        for result in results:
            self.assertEqual(result, (timeout, timeout, timeout))
            self.assertTrue(all(value is timeout for value in result))

    def test_reloads_do_not_probe_environment_or_initialize_a_backend(self):
        distributed, constants, distributed_c10d, timeout = (
            self.modules_and_value()
        )

        with mock.patch.object(os, "environ", UnreadableEnvironment()):
            constants = importlib.reload(constants)
            distributed_c10d = importlib.reload(distributed_c10d)
            distributed = importlib.reload(distributed)

        self.assertIs(distributed.default_pg_timeout, timeout)
        self.assertIs(distributed.is_initialized(), False)
        self.assertEqual(distributed.get_pg_count(), 0)

    def test_import_does_not_import_pytorch_or_enable_distributed_execution(self):
        script = r"""
import datetime
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
    MASTER_ADDR="127.0.0.1",
    MASTER_PORT="29500",
    RANK="0",
    WORLD_SIZE="8",
    CUDA_VISIBLE_DEVICES="0",
)

from torch_rs.distributed.constants import default_pg_timeout
import torch_rs.distributed as distributed

assert type(default_pg_timeout) is datetime.timedelta
assert default_pg_timeout == datetime.timedelta(minutes=30)
assert distributed.default_pg_timeout is default_pg_timeout
assert distributed.distributed_c10d.default_pg_timeout is default_pg_timeout
assert distributed.is_initialized() is False
assert distributed.get_pg_count() == 0
assert not hasattr(distributed, "init_process_group")
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
