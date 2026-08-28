import copy
import importlib
import pickle
import pickletools
import sys
import types
import unittest
from datetime import timedelta

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

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            shape.append((opcode.name, argument))
        return shape

    def test_value_module_metadata_and_identity_match(self):
        actual_distributed = importlib.import_module("torch_rs.distributed")
        expected_distributed = importlib.import_module("torch.distributed")
        actual_constants = importlib.import_module(
            "torch_rs.distributed.constants"
        )
        expected_constants = importlib.import_module(
            "torch.distributed.constants"
        )
        actual_c10d = importlib.import_module(
            "torch_rs.distributed.distributed_c10d"
        )
        expected_c10d = importlib.import_module(
            "torch.distributed.distributed_c10d"
        )
        actual = actual_constants.default_pg_timeout
        expected = expected_constants.default_pg_timeout

        self.assertIs(torch.distributed, actual_distributed)
        self.assertIs(reference_torch.distributed, expected_distributed)
        self.assertIs(actual_distributed.constants, actual_constants)
        self.assertIs(expected_distributed.constants, expected_constants)
        self.assertIs(actual_constants._DEFAULT_PG_TIMEOUT, actual)
        self.assertIs(expected_constants._DEFAULT_PG_TIMEOUT, expected)
        self.assertIs(actual_c10d.default_pg_timeout, actual)
        self.assertIs(expected_c10d.default_pg_timeout, expected)
        self.assertIs(sys.modules[actual_constants.__name__], actual_constants)
        self.assertIs(sys.modules[expected_constants.__name__], expected_constants)
        self.assertIs(type(actual_constants), types.ModuleType)
        self.assertIs(type(expected_constants), types.ModuleType)
        self.assertEqual(
            actual_constants.__name__.replace("torch_rs", "torch"),
            expected_constants.__name__,
        )
        self.assertEqual(
            actual_constants.__package__.replace("torch_rs", "torch"),
            expected_constants.__package__,
        )
        self.assertEqual(
            actual_constants.__all__,
            [
                name
                for name in expected_constants.__all__
                if name == "default_pg_timeout"
            ],
        )
        self.assertEqual(actual_constants.__doc__, expected_constants.__doc__)
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
        self.assertIs(type(actual), type(expected))
        self.assertIs(type(actual), timedelta)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, timedelta(minutes=30))
        self.assertEqual(
            actual_c10d.__all__.count("default_pg_timeout"),
            expected_c10d.__all__.count("default_pg_timeout"),
        )
        self.assertIs(actual_distributed.default_pg_timeout, actual)
        self.assertIs(expected_distributed.default_pg_timeout, expected)

    def test_direct_wildcard_copy_and_pickle_behavior_matches(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_constants = actual_distributed.constants
        expected_constants = expected_distributed.constants
        actual_c10d = actual_distributed.distributed_c10d
        expected_c10d = expected_distributed.distributed_c10d
        actual = actual_constants.default_pg_timeout
        expected = expected_constants.default_pg_timeout

        for package_name, constants, c10d, distributed, value in (
            (
                "torch_rs",
                actual_constants,
                actual_c10d,
                actual_distributed,
                actual,
            ),
            (
                "torch",
                expected_constants,
                expected_c10d,
                expected_distributed,
                expected,
            ),
        ):
            module_import = {}
            constant_import = {}
            owner_import = {}
            reexport_import = {}
            constants_wildcard = {}
            owner_wildcard = {}
            distributed_wildcard = {}
            exec(
                f"import {package_name}.distributed.constants as constants",
                module_import,
            )
            exec(
                f"from {package_name}.distributed.constants import "
                "default_pg_timeout",
                constant_import,
            )
            exec(
                f"from {package_name}.distributed.distributed_c10d import "
                "default_pg_timeout",
                owner_import,
            )
            exec(
                f"from {package_name}.distributed import default_pg_timeout",
                reexport_import,
            )
            exec(
                f"from {package_name}.distributed.constants import *",
                constants_wildcard,
            )
            exec(
                f"from {package_name}.distributed.distributed_c10d import *",
                owner_wildcard,
            )
            exec(
                f"from {package_name}.distributed import *",
                distributed_wildcard,
            )

            self.assertIs(module_import["constants"], constants)
            self.assertIs(c10d.default_pg_timeout, value)
            self.assertIs(constant_import["default_pg_timeout"], value)
            self.assertIs(owner_import["default_pg_timeout"], value)
            self.assertIs(reexport_import["default_pg_timeout"], value)
            self.assertIs(constants_wildcard["default_pg_timeout"], value)
            self.assertIs(owner_wildcard["default_pg_timeout"], value)
            self.assertIs(distributed_wildcard["constants"], constants)
            self.assertIs(distributed_wildcard["default_pg_timeout"], value)

        for copier in (copy.copy, copy.deepcopy):
            actual_copy = copier(actual)
            expected_copy = copier(expected)
            self.assertIs(type(actual_copy), type(expected_copy))
            self.assertEqual(actual_copy, expected_copy)
            self.assertEqual(actual_copy is actual, expected_copy is expected)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_payload = pickle.dumps(actual, protocol=protocol)
                expected_payload = pickle.dumps(expected, protocol=protocol)
                self.assertEqual(actual_payload, expected_payload)
                actual_restored = pickle.loads(actual_payload)
                expected_restored = pickle.loads(expected_payload)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(type(actual_restored), type(expected_restored))
                self.assertEqual(actual_restored, expected_restored)
                self.assertEqual(
                    actual_restored is actual,
                    expected_restored is expected,
                )

    def test_constants_reload_behavior_matches_pytorch(self):
        actual_distributed = torch.distributed
        expected_distributed = reference_torch.distributed
        actual_constants = actual_distributed.constants
        expected_constants = expected_distributed.constants
        actual_c10d = actual_distributed.distributed_c10d
        actual = actual_constants.default_pg_timeout
        expected = expected_constants.default_pg_timeout
        actual_namespace = actual_constants.__dict__
        expected_namespace = expected_constants.__dict__
        actual_all = actual_constants.__all__
        expected_all = expected_constants.__all__

        actual_constants.default_pg_timeout = timedelta(seconds=1)
        actual_constants._DEFAULT_PG_TIMEOUT = timedelta(seconds=2)
        expected_constants.default_pg_timeout = timedelta(seconds=3)
        expected_constants._DEFAULT_PG_TIMEOUT = timedelta(seconds=4)
        actual_reloaded = importlib.reload(actual_constants)
        expected_reloaded = importlib.reload(expected_constants)
        actual_c10d = importlib.reload(actual_c10d)
        actual_distributed = importlib.reload(actual_distributed)

        self.assertEqual(actual_reloaded is actual_constants, True)
        self.assertEqual(expected_reloaded is expected_constants, True)
        self.assertEqual(
            actual_constants.__dict__ is actual_namespace,
            expected_constants.__dict__ is expected_namespace,
        )
        self.assertEqual(
            actual_constants.__all__ is actual_all,
            expected_constants.__all__ is expected_all,
        )
        self.assertIs(actual_constants.default_pg_timeout, actual)
        self.assertIs(expected_constants.default_pg_timeout, expected)
        self.assertIs(actual_c10d.default_pg_timeout, actual)
        self.assertIs(actual_distributed.default_pg_timeout, actual)
        self.assertIs(expected_distributed.default_pg_timeout, expected)


if __name__ == "__main__":
    unittest.main()
