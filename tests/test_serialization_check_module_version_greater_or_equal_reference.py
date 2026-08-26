import importlib
import inspect
import types
import unittest
import warnings
from decimal import Decimal

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SerializationCheckModuleVersionGreaterOrEqualReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "serialization module-version differentials require pinned "
                "PyTorch 2.13.0"
            )

    def module(self, version, name="example"):
        return types.SimpleNamespace(__name__=name, __version__=version)

    def call_outcome(self, function, version, requirement):
        try:
            result = function(self.module(version), requirement)
        except Exception as error:
            cause = error.__cause__
            return (
                "error",
                type(error).__module__,
                type(error).__qualname__,
                str(error),
                error.args,
                None if cause is None else type(cause).__module__,
                None if cause is None else type(cause).__qualname__,
                None if cause is None else str(cause),
                error.__context__ is cause,
                error.__suppress_context__,
            )
        return (
            "return",
            result,
            type(result).__module__,
            type(result).__qualname__,
        )

    def warning_outcome(self, function, version, requirement):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1
            result = function(self.module(version), requirement, False)
        return (
            result,
            tuple(
                (
                    type(warning.message).__module__,
                    type(warning.message).__qualname__,
                    str(warning.message),
                    warning.category.__module__,
                    warning.category.__qualname__,
                    warning.filename == __file__,
                    warning.lineno == expected_line,
                )
                for warning in caught
            ),
        )

    def serialization_state(self, module):
        serialization = module.serialization
        return (
            serialization.get_crc32_options(),
            serialization.get_default_load_endianness(),
            serialization.get_default_mmap_options(),
        )

    def test_well_formed_field_coercion_and_comparison_match_pytorch_2_13(self):
        cases = (
            ("1.2.3", (1, 2, 3)),
            ("1.2.3", (1, 2, 4)),
            ("1.10.0", (1, 2)),
            ("2.5.9+cu130", (2, 5)),
            ("2.5", (2.0, 5.0)),
            ("2.5", ("2", "10")),
            ("0.0", (False, False)),
            ("1.25", (Decimal("1"), Decimal("2.5"))),
            ("anything", ()),
        )
        actual = torch.serialization.check_module_version_greater_or_equal
        expected = (
            reference_torch.serialization.check_module_version_greater_or_equal
        )
        for version, requirement in cases:
            with self.subTest(version=version, requirement=requirement):
                self.assertEqual(
                    self.call_outcome(actual, version, requirement),
                    self.call_outcome(expected, version, requirement),
                )

    def test_malformed_errors_and_warnings_match_without_state_changes(self):
        actual = torch.serialization.check_module_version_greater_or_equal
        expected = (
            reference_torch.serialization.check_module_version_greater_or_equal
        )
        malformed_cases = (
            ("1.bad", (1, 2)),
            ("1", (1, 0)),
            ("1.2", [1, 2]),
        )
        for version, requirement in malformed_cases:
            with self.subTest(version=version, requirement=requirement):
                actual_state = self.serialization_state(torch)
                expected_state = self.serialization_state(reference_torch)

                self.assertEqual(
                    self.call_outcome(actual, version, requirement),
                    self.call_outcome(expected, version, requirement),
                )
                self.assertEqual(
                    self.warning_outcome(actual, version, requirement),
                    self.warning_outcome(expected, version, requirement),
                )
                self.assertEqual(self.serialization_state(torch), actual_state)
                self.assertEqual(
                    self.serialization_state(reference_torch),
                    expected_state,
                )

    def test_signature_documentation_identity_and_exports_match(self):
        actual_module = importlib.import_module("torch_rs.serialization")
        expected_module = importlib.import_module("torch.serialization")
        actual = actual_module.check_module_version_greater_or_equal
        expected = expected_module.check_module_version_greater_or_equal

        self.assertIs(torch.serialization, actual_module)
        self.assertIs(reference_torch.serialization, expected_module)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        supported_names = {
            "check_module_version_greater_or_equal",
            "LoadEndianness",
            "get_crc32_options",
            "set_crc32_options",
            "get_default_load_endianness",
            "set_default_load_endianness",
            "get_default_mmap_options",
            "set_default_mmap_options",
        }
        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported_names],
        )
        self.assertEqual(
            torch.__all__.count(actual.__name__),
            reference_torch.__all__.count(expected.__name__),
        )
        self.assertFalse(hasattr(torch, actual.__name__))
        self.assertFalse(hasattr(reference_torch, expected.__name__))

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.serialization import *", actual_namespace)
        exec("from torch.serialization import *", expected_namespace)
        self.assertIs(actual_namespace[actual.__name__], actual)
        self.assertIs(expected_namespace[expected.__name__], expected)


if __name__ == "__main__":
    unittest.main()
