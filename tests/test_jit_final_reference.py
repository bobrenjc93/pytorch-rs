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

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_identity_metadata_and_internal_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = actual_jit.Final
        expected = expected_jit.Final

        self.assertIs(actual, typing.Final)
        self.assertIs(expected, typing.Final)
        self.assertIs(actual_internal.Final, actual)
        self.assertIs(expected_internal.Final, expected)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(repr(actual), repr(expected))
        self.assertEqual(str(actual), str(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertIs(inspect.getmodule(actual), inspect.getmodule(expected))
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertIs(typing.get_origin(actual), typing.get_origin(expected))
        self.assertEqual(typing.get_args(actual), typing.get_args(expected))
        for attribute in ("__origin__", "__args__", "__parameters__"):
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    hasattr(actual, attribute), hasattr(expected, attribute)
                )

    def test_subscription_origin_arguments_and_annotations_match(self):
        actual_marker = torch.jit.Final
        expected_marker = reference_torch.jit.Final
        for argument in (int, list[str], tuple[int, ...]):
            with self.subTest(argument=argument):
                actual = actual_marker[argument]
                expected = expected_marker[argument]
                self.assertIs(actual, expected)
                self.assertEqual(repr(actual), repr(expected))
                self.assertEqual(str(actual), str(expected))
                self.assertIs(typing.get_origin(actual), typing.get_origin(expected))
                self.assertEqual(typing.get_args(actual), typing.get_args(expected))
                self.assertEqual(actual.__module__, expected.__module__)

        class ActualConfiguration:
            retries: actual_marker[int] = 3

        class ExpectedConfiguration:
            retries: expected_marker[int] = 3

        self.assertEqual(
            ActualConfiguration.__annotations__, ExpectedConfiguration.__annotations__
        )
        self.assertEqual(
            typing.get_type_hints(ActualConfiguration),
            typing.get_type_hints(ExpectedConfiguration),
        )

    def test_copy_pickle_instantiation_and_arity_match(self):
        actual_values = (
            torch.jit.Final,
            torch.jit.Final[int],
            torch.jit.Final[list[str]],
        )
        expected_values = (
            reference_torch.jit.Final,
            reference_torch.jit.Final[int],
            reference_torch.jit.Final[list[str]],
        )
        for actual, expected in zip(actual_values, expected_values):
            with self.subTest(value=actual):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=actual, protocol=protocol):
                        actual_payload = pickle.dumps(actual, protocol=protocol)
                        expected_payload = pickle.dumps(expected, protocol=protocol)
                        self.assertEqual(actual_payload, expected_payload)
                        self.assertIs(pickle.loads(actual_payload), actual)
                        self.assertIs(pickle.loads(expected_payload), expected)

        cases = (
            lambda marker: marker(),
            lambda marker: marker(int),
            lambda marker: marker[int](),
            lambda marker: marker[()],
            lambda marker: marker[int, str],
            lambda marker: marker[int,],
            lambda marker: marker[int][str],
        )
        for case, operation in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: operation(torch.jit.Final),
                    lambda: operation(reference_torch.jit.Final),
                )

    def test_export_shape_matches_while_compilation_remains_unsupported(self):
        wildcard_supported = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "script_if_tracing",
            "unused",
        }
        public_supported = {
            *wildcard_supported,
            "Final",
            "is_scripting",
            "is_tracing",
        }

        self.assertNotIn("Final", torch.jit.__all__)
        self.assertNotIn("Final", reference_torch.jit.__all__)
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            public_supported,
        )

        actual_explicit = {}
        expected_explicit = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.jit import Final", actual_explicit)
        exec("from torch.jit import Final", expected_explicit)
        exec("from torch_rs.jit import *", actual_wildcard)
        exec("from torch.jit import *", expected_wildcard)
        self.assertIs(actual_explicit["Final"], typing.Final)
        self.assertIs(expected_explicit["Final"], typing.Final)
        self.assertNotIn("Final", actual_wildcard)
        self.assertNotIn("Final", expected_wildcard)
        self.assertEqual(
            {name for name in actual_wildcard if not name.startswith("__")},
            wildcard_supported,
        )

        for module in (torch, reference_torch):
            self.assertNotIn("Final", module.__all__)
            self.assertFalse(hasattr(module, "Final"))

        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
            "script_method",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
                self.assertTrue(hasattr(reference_torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertTrue(hasattr(reference_torch, "compile"))


if __name__ == "__main__":
    unittest.main()
