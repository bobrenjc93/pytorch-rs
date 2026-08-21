import copy
import importlib
import inspect
import pickle
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitFinalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.Final differentials require pinned PyTorch 2.13.0"
            )

    def call_outcome(self, call):
        try:
            return "success", call()
        except Exception as error:
            return "error", type(error), str(error), error.args

    def test_alias_identity_and_module_placement_match(self):
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")

        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(actual_jit.Final, typing.Final)
        self.assertIs(expected_jit.Final, typing.Final)
        self.assertIs(actual_jit.Final, actual_internal.Final)
        self.assertIs(expected_jit.Final, expected_internal.Final)

    def test_subscription_origin_and_arguments_match(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final

        self.assertIsNone(typing.get_origin(actual))
        self.assertEqual(typing.get_args(actual), typing.get_args(expected))
        for parameter in (int, str, list[int], None, ...):
            with self.subTest(parameter=parameter):
                actual_outcome = self.call_outcome(lambda: actual[parameter])
                expected_outcome = self.call_outcome(lambda: expected[parameter])
                self.assertEqual(actual_outcome, expected_outcome)
                if actual_outcome[0] == "error":
                    continue

                actual_value = actual_outcome[1]
                expected_value = expected_outcome[1]
                self.assertIs(actual_value, expected_value)
                self.assertIs(
                    typing.get_origin(actual_value),
                    typing.get_origin(expected_value),
                )
                self.assertEqual(
                    typing.get_args(actual_value), typing.get_args(expected_value)
                )
                self.assertEqual(repr(actual_value), repr(expected_value))
                self.assertEqual(str(actual_value), str(expected_value))

    def test_instantiation_arity_and_runtime_type_errors_match(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        calls = (
            lambda marker: marker(),
            lambda marker: marker(1),
            lambda marker: marker(1, 2),
            lambda marker: marker(value=1),
            lambda marker: marker[int, str],
            lambda marker: marker[()],
            lambda marker: isinstance(1, marker),
            lambda marker: issubclass(int, marker),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                actual_outcome = self.call_outcome(lambda: call(actual))
                expected_outcome = self.call_outcome(lambda: call(expected))
                self.assertEqual(actual_outcome, expected_outcome)
                self.assertEqual(actual_outcome[0], "error")

    def test_nested_subscription_outcome_matches(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final

        self.assertEqual(
            self.call_outcome(lambda: actual[actual[int]]),
            self.call_outcome(lambda: expected[expected[int]]),
        )

    def test_metadata_matches(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final

        self.assertIs(type(actual), type(expected))
        self.assertEqual(repr(actual), repr(expected))
        self.assertEqual(str(actual), str(expected))
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual._name, expected._name)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertIs(inspect.getmodule(actual), inspect.getmodule(expected))
        self.assertEqual(hasattr(actual, "__dict__"), hasattr(expected, "__dict__"))

        actual_parameterized = actual[int]
        expected_parameterized = expected[int]
        self.assertIs(type(actual_parameterized), type(expected_parameterized))
        self.assertEqual(
            actual_parameterized.__module__, expected_parameterized.__module__
        )
        self.assertIs(
            typing.get_origin(actual_parameterized),
            typing.get_origin(expected_parameterized),
        )
        self.assertEqual(
            typing.get_args(actual_parameterized),
            typing.get_args(expected_parameterized),
        )

    def test_copying_and_pickling_match(self):
        for actual, expected in (
            (torch.jit.Final, reference_torch.jit.Final),
            (torch.jit.Final[int], reference_torch.jit.Final[int]),
        ):
            with self.subTest(value=actual, operation="copy"):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(expected), expected)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=actual, protocol=protocol):
                    actual_payload = pickle.dumps(actual, protocol=protocol)
                    expected_payload = pickle.dumps(expected, protocol=protocol)
                    self.assertEqual(actual_payload, expected_payload)
                    self.assertIs(pickle.loads(actual_payload), actual)
                    self.assertIs(pickle.loads(expected_payload), expected)

    def test_exports_and_unsupported_boundary_match(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "script_if_tracing",
            "unused",
        }

        self.assertNotIn("Final", actual_jit.__all__)
        self.assertNotIn("Final", expected_jit.__all__)
        self.assertEqual(
            actual_jit.__all__,
            [name for name in expected_jit.__all__ if name in wildcard_supported],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            {*wildcard_supported, "Final", "is_scripting", "is_tracing"},
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertNotIn("Final", actual_namespace)
        self.assertNotIn("Final", expected_namespace)

        expected_public = {
            name for name in vars(expected_jit) if not name.startswith("_")
        }
        for name in (
            "CompilationUnit",
            "ScriptModule",
            "interface",
            "script",
            "trace",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(actual_jit, name))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
