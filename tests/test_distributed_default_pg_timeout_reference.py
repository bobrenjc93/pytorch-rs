import copy
import datetime
import importlib
import os
import pickle
import pickletools
import unittest
from unittest import mock

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

    def modules_and_values(self):
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
        return (
            actual_distributed,
            expected_distributed,
            actual_constants,
            expected_constants,
            actual_c10d,
            expected_c10d,
            actual_constants.default_pg_timeout,
            expected_constants.default_pg_timeout,
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

    def test_value_type_fields_and_identity_match(self):
        (
            actual_distributed,
            expected_distributed,
            actual_constants,
            expected_constants,
            actual_c10d,
            expected_c10d,
            actual,
            expected,
        ) = self.modules_and_values()

        self.assertIs(type(actual), type(expected))
        self.assertIs(type(actual), datetime.timedelta)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, datetime.timedelta(minutes=30))
        self.assertEqual(
            (actual.days, actual.seconds, actual.microseconds),
            (expected.days, expected.seconds, expected.microseconds),
        )
        self.assertIs(actual_constants._DEFAULT_PG_TIMEOUT, actual)
        self.assertIs(expected_constants._DEFAULT_PG_TIMEOUT, expected)
        self.assertIs(actual_c10d.default_pg_timeout, actual)
        self.assertIs(expected_c10d.default_pg_timeout, expected)
        self.assertIs(actual_distributed.default_pg_timeout, actual)
        self.assertIs(expected_distributed.default_pg_timeout, expected)

    def test_module_metadata_and_import_forms_match_supported_scope(self):
        (
            actual_distributed,
            expected_distributed,
            actual_constants,
            expected_constants,
            actual_c10d,
            expected_c10d,
            actual,
            expected,
        ) = self.modules_and_values()

        self.assertIs(actual_distributed.constants, actual_constants)
        self.assertIs(expected_distributed.constants, expected_constants)
        self.assertEqual(
            actual_constants.__all__,
            [
                name
                for name in expected_constants.__all__
                if name == "default_pg_timeout"
            ],
        )
        self.assertEqual(
            actual_constants.__annotations__["default_pg_timeout"],
            expected_constants.__annotations__["default_pg_timeout"],
        )
        self.assertEqual(actual_constants.__doc__, expected_constants.__doc__)
        self.assertEqual(
            {
                name
                for name in vars(actual_constants)
                if not name.startswith("_")
            },
            {
                name
                for name in vars(expected_constants)
                if name in {"default_pg_timeout", "timedelta"}
            },
        )
        self.assertEqual(
            actual_c10d.__all__.count("default_pg_timeout"),
            expected_c10d.__all__.count("default_pg_timeout"),
        )

        for module, value in (
            (actual_constants, actual),
            (expected_constants, expected),
            (actual_c10d, actual),
            (expected_c10d, expected),
            (actual_distributed, actual),
            (expected_distributed, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import default_pg_timeout", namespace)
            self.assertIs(namespace["default_pg_timeout"], value)

            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["default_pg_timeout"], value)

        self.assertEqual(
            torch.__all__.count("default_pg_timeout"),
            reference_torch.__all__.count("default_pg_timeout"),
        )

    def test_reload_behavior_matches_the_canonical_reference_object(self):
        (
            actual_distributed,
            _,
            actual_constants,
            expected_constants,
            actual_c10d,
            _,
            actual,
            expected,
        ) = self.modules_and_values()

        replacement = datetime.timedelta(seconds=1)
        actual_constants.default_pg_timeout = replacement
        actual_constants._DEFAULT_PG_TIMEOUT = replacement
        expected_constants.default_pg_timeout = replacement
        expected_constants._DEFAULT_PG_TIMEOUT = replacement
        for _ in range(3):
            actual_constants = importlib.reload(actual_constants)
            expected_constants = importlib.reload(expected_constants)
            actual_c10d = importlib.reload(actual_c10d)
            actual_distributed = importlib.reload(actual_distributed)

            self.assertIs(actual_constants.default_pg_timeout, actual)
            self.assertIs(expected_constants.default_pg_timeout, expected)
            self.assertIs(actual_c10d.default_pg_timeout, actual)
            self.assertIs(actual_distributed.default_pg_timeout, actual)
            self.assertIs(actual_distributed.constants, actual_constants)
            self.assertIs(actual_distributed.distributed_c10d, actual_c10d)

    def test_copy_deepcopy_and_pickle_match(self):
        *_, actual, expected = self.modules_and_values()

        for copier in (copy.copy, copy.deepcopy):
            with self.subTest(copier=copier.__name__):
                actual_copy = copier(actual)
                expected_copy = copier(expected)
                self.assertIs(type(actual_copy), type(expected_copy))
                self.assertEqual(actual_copy, expected_copy)
                self.assertEqual(
                    actual_copy is actual,
                    expected_copy is expected,
                )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_payload = pickle.dumps(actual, protocol=protocol)
                expected_payload = pickle.dumps(expected, protocol=protocol)
                self.assertEqual(actual_payload, expected_payload)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                actual_restored = pickle.loads(actual_payload)
                expected_restored = pickle.loads(expected_payload)
                self.assertIs(type(actual_restored), type(expected_restored))
                self.assertEqual(actual_restored, expected_restored)
                self.assertEqual(
                    actual_restored is actual,
                    expected_restored is expected,
                )

    def test_environment_and_process_group_state_do_not_affect_the_value(self):
        (
            actual_distributed,
            expected_distributed,
            actual_constants,
            expected_constants,
            _,
            _,
            actual,
            expected,
        ) = self.modules_and_values()
        environments = (
            {},
            {"USE_DISTRIBUTED": "0"},
            {"USE_DISTRIBUTED": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
                "RANK": "0",
                "WORLD_SIZE": "8",
            },
        )

        self.assertIs(actual_distributed.is_initialized(), False)
        self.assertIs(expected_distributed.is_initialized(), False)
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(actual_constants.default_pg_timeout, actual)
                    self.assertIs(expected_constants.default_pg_timeout, expected)
                    self.assertEqual(actual, expected)
                    self.assertIs(actual_distributed.is_initialized(), False)
                    self.assertIs(expected_distributed.is_initialized(), False)


if __name__ == "__main__":
    unittest.main()
