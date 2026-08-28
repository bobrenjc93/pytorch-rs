import copy
from datetime import timedelta
import importlib
import pickle
import sys
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DistributedDefaultPgTimeoutReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "distributed.default_pg_timeout differentials require pinned "
                "PyTorch 2.13.0"
            )

    def modules(self, root):
        distributed = root.distributed
        constants = importlib.import_module(f"{root.__name__}.distributed.constants")
        distributed_c10d = importlib.import_module(
            f"{root.__name__}.distributed.distributed_c10d"
        )
        return distributed, constants, distributed_c10d

    def test_value_type_and_identity_match_pytorch_2_13(self):
        for root in (torch, reference_torch):
            with self.subTest(package=root.__name__):
                distributed, constants, distributed_c10d = self.modules(root)
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

    def test_module_metadata_and_exports_match_supported_scope(self):
        actual_dist, actual_constants, actual_c10d = self.modules(torch)
        expected_dist, expected_constants, expected_c10d = self.modules(
            reference_torch
        )

        self.assertIs(torch.distributed, actual_dist)
        self.assertIs(reference_torch.distributed, expected_dist)
        self.assertIs(actual_dist.constants, actual_constants)
        self.assertIs(expected_dist.constants, expected_constants)
        self.assertIs(
            sys.modules["torch_rs.distributed.constants"], actual_constants
        )
        self.assertIs(sys.modules["torch.distributed.constants"], expected_constants)
        self.assertEqual(actual_constants.__doc__, expected_constants.__doc__)
        self.assertEqual(
            actual_constants.__all__,
            [
                name
                for name in expected_constants.__all__
                if name == "default_pg_timeout"
            ],
        )
        self.assertEqual(
            actual_constants.__annotations__,
            {
                name: annotation
                for name, annotation in expected_constants.__annotations__.items()
                if name == "default_pg_timeout"
            },
        )
        self.assertEqual(
            {name for name in vars(actual_constants) if not name.startswith("_")},
            {
                name
                for name in vars(expected_constants)
                if name in {"timedelta", "default_pg_timeout"}
            },
        )
        self.assertEqual(
            hasattr(actual_dist, "__all__"),
            hasattr(expected_dist, "__all__"),
        )
        self.assertEqual(
            actual_c10d.__all__.count("default_pg_timeout"),
            expected_c10d.__all__.count("default_pg_timeout"),
        )

        for package_name, distributed, constants, distributed_c10d in (
            ("torch_rs", actual_dist, actual_constants, actual_c10d),
            ("torch", expected_dist, expected_constants, expected_c10d),
        ):
            package_import = {}
            constants_import = {}
            direct_import = {}
            owner_import = {}
            constants_wildcard = {}
            distributed_wildcard = {}
            owner_wildcard = {}
            exec(
                f"from {package_name}.distributed import constants",
                package_import,
            )
            exec(
                f"import {package_name}.distributed.constants as constants",
                constants_import,
            )
            exec(
                f"from {package_name}.distributed import default_pg_timeout",
                direct_import,
            )
            exec(
                f"from {package_name}.distributed.distributed_c10d import "
                "default_pg_timeout",
                owner_import,
            )
            exec(
                f"from {package_name}.distributed.constants import *",
                constants_wildcard,
            )
            exec(
                f"from {package_name}.distributed import *",
                distributed_wildcard,
            )
            exec(
                f"from {package_name}.distributed.distributed_c10d import *",
                owner_wildcard,
            )
            timeout = distributed.default_pg_timeout
            self.assertIs(package_import["constants"], constants)
            self.assertIs(constants_import["constants"], constants)
            self.assertIs(direct_import["default_pg_timeout"], timeout)
            self.assertIs(owner_import["default_pg_timeout"], timeout)
            self.assertIs(constants_wildcard["default_pg_timeout"], timeout)
            self.assertIs(distributed_wildcard["constants"], constants)
            self.assertIs(distributed_wildcard["default_pg_timeout"], timeout)
            self.assertIs(owner_wildcard["default_pg_timeout"], timeout)

    def test_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.distributed.default_pg_timeout
        expected = reference_torch.distributed.default_pg_timeout

        for operation in (copy.copy, copy.deepcopy):
            actual_copy = operation(actual)
            expected_copy = operation(expected)
            self.assertEqual(actual_copy, expected_copy)
            self.assertIs(type(actual_copy), type(expected_copy))
            self.assertEqual(actual_copy is actual, expected_copy is expected)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_payload = pickle.dumps(actual, protocol=protocol)
                expected_payload = pickle.dumps(expected, protocol=protocol)
                self.assertEqual(actual_payload, expected_payload)
                actual_restored = pickle.loads(actual_payload)
                expected_restored = pickle.loads(expected_payload)
                self.assertEqual(actual_restored, expected_restored)
                self.assertIs(type(actual_restored), type(expected_restored))
                self.assertEqual(
                    actual_restored is actual,
                    expected_restored is expected,
                )

    def reload_contract(self, root):
        distributed, constants, distributed_c10d = self.modules(root)
        timeout = distributed.default_pg_timeout
        namespace = constants.__dict__
        old_all = constants.__all__
        old_annotations = constants.__annotations__

        constants.default_pg_timeout = "stale"
        constants._DEFAULT_PG_TIMEOUT = "stale"
        constants.__all__ = []
        constants.__annotations__["default_pg_timeout"] = str
        reloaded = importlib.reload(constants)

        return (
            reloaded is constants,
            constants.__dict__ is namespace,
            constants.__all__ is not old_all,
            constants.__annotations__ is old_annotations,
            constants.__all__,
            constants.__annotations__.get("default_pg_timeout"),
            constants.default_pg_timeout is timeout,
            constants._DEFAULT_PG_TIMEOUT is timeout,
            distributed.default_pg_timeout is timeout,
            distributed_c10d.default_pg_timeout is timeout,
        )

    def test_constants_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_contract(torch)
        expected = self.reload_contract(reference_torch)

        self.assertEqual(actual[:4], expected[:4])
        self.assertEqual(
            actual[4],
            [name for name in expected[4] if name == "default_pg_timeout"],
        )
        self.assertIs(actual[5], expected[5])
        self.assertEqual(actual[6:], expected[6:])


if __name__ == "__main__":
    unittest.main()
