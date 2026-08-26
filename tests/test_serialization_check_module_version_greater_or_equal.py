import copy
import functools
import importlib
import inspect
import pickle
import types
import unittest
import warnings

import torch_rs as torch


FUNCTION_DOC = """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """


@functools.total_ordering
class NumericVersionField:
    converted_values = []

    def __init__(self, value):
        self.converted_values.append(value)
        self.value = int(value)

    def __eq__(self, other):
        if not isinstance(other, NumericVersionField):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other):
        if not isinstance(other, NumericVersionField):
            return NotImplemented
        return self.value < other.value


class SerializationCheckModuleVersionGreaterOrEqualTests(unittest.TestCase):
    def setUp(self):
        self.serialization = torch.serialization
        self.function = self.serialization.check_module_version_greater_or_equal

    def module(self, version, name="example"):
        return types.SimpleNamespace(__name__=name, __version__=version)

    def serialization_state(self):
        return (
            self.serialization.get_crc32_options(),
            self.serialization.get_default_load_endianness(),
            self.serialization.get_default_mmap_options(),
        )

    def assert_state_is(self, expected):
        actual = self.serialization_state()
        for expected_value, actual_value in zip(expected, actual):
            self.assertIs(actual_value, expected_value)

    def test_well_formed_versions_use_requirement_field_types_and_tuple_ordering(self):
        cases = (
            ("1.2.3", (1, 2, 3), True),
            ("1.2.3", (1, 2, 4), False),
            ("1.10.0", (1, 2), True),
            ("2.5.9+cu130", (2, 5), True),
            ("2.5", (2.0, 5.0), True),
            ("2.5", ("2", "10"), True),
            ("0.0", (False, False), True),
            ("anything", (), True),
        )
        for version, requirement, expected in cases:
            with self.subTest(version=version, requirement=requirement):
                result = self.function(self.module(version), requirement)
                self.assertIs(result, expected)

        self.assertIs(
            self.function(
                module=self.module("3.4"),
                req_version_tuple=(3, 4),
                error_if_malformed=True,
            ),
            True,
        )

        requirement = (NumericVersionField(1), NumericVersionField(5))
        NumericVersionField.converted_values.clear()
        self.assertIs(self.function(self.module("1.6"), requirement), True)
        self.assertEqual(NumericVersionField.converted_values, ["1", "6"])

    def test_malformed_version_raises_runtime_error_with_original_cause(self):
        module = self.module("1.bad", name="demo")
        expected_message = (
            "'demo' module version string is malformed '1.bad' and cannot be "
            "compared with tuple (1, 2)"
        )
        state = self.serialization_state()

        with self.assertRaises(RuntimeError) as raised:
            self.function(module, (1, 2))

        error = raised.exception
        self.assertEqual(str(error), expected_message)
        self.assertEqual(error.args, (expected_message,))
        self.assertIsInstance(error.__cause__, ValueError)
        self.assertEqual(
            str(error.__cause__),
            "invalid literal for int() with base 10: 'bad'",
        )
        self.assertIs(error.__context__, error.__cause__)
        self.assertTrue(error.__suppress_context__)
        self.assert_state_is(state)

    def test_malformed_version_warns_at_the_caller_and_assumes_success(self):
        module = self.module("1.bad", name="demo")
        expected_message = (
            "'demo' module version string is malformed '1.bad' and cannot be "
            "compared with tuple (1, 2), but continuing assuming that requirement "
            "is met"
        )
        state = self.serialization_state()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expected_line = inspect.currentframe().f_lineno + 1
            result = self.function(module, (1, 2), error_if_malformed=False)

        self.assertIs(result, True)
        self.assertEqual(len(caught), 1)
        warning = caught[0]
        self.assertIs(warning.category, UserWarning)
        self.assertIs(type(warning.message), UserWarning)
        self.assertEqual(str(warning.message), expected_message)
        self.assertEqual(warning.filename, __file__)
        self.assertEqual(warning.lineno, expected_line)
        self.assert_state_is(state)

    def test_signature_documentation_identity_and_exports(self):
        serialization = importlib.import_module("torch_rs.serialization")
        function = serialization.check_module_version_greater_or_equal

        self.assertIs(serialization, self.serialization)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "check_module_version_greater_or_equal")
        self.assertEqual(function.__qualname__, function.__name__)
        self.assertEqual(function.__module__, "torch_rs.serialization")
        self.assertIs(inspect.getmodule(function), serialization)
        self.assertEqual(
            str(inspect.signature(function)),
            "(module, req_version_tuple, error_if_malformed=True)",
        )
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__defaults__, (True,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)

        self.assertEqual(
            serialization.__all__,
            [
                "check_module_version_greater_or_equal",
                "LoadEndianness",
                "get_crc32_options",
                "set_crc32_options",
                "get_default_load_endianness",
                "set_default_load_endianness",
                "get_default_mmap_options",
                "set_default_mmap_options",
            ],
        )
        direct_import = {}
        exec(
            "from torch_rs.serialization import "
            "check_module_version_greater_or_equal",
            direct_import,
        )
        self.assertIs(direct_import[function.__name__], function)

        wildcard_import = {}
        exec("from torch_rs.serialization import *", wildcard_import)
        self.assertIs(wildcard_import[function.__name__], function)

        package_wildcard = {}
        exec("from torch_rs import *", package_wildcard)
        self.assertNotIn(function.__name__, package_wildcard)
        self.assertFalse(hasattr(torch, function.__name__))
        self.assertNotIn(function.__name__, torch.__all__)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)


if __name__ == "__main__":
    unittest.main()
