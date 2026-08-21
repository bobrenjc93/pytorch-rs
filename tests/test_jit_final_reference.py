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

    def assert_outcome_matches(self, actual_call, expected_call):
        def capture(call):
            try:
                return None, call()
            except Exception as error:
                return (type(error), str(error), error.args), None

        actual_error, actual_result = capture(actual_call)
        expected_error, expected_result = capture(expected_call)
        self.assertEqual(actual_error, expected_error)
        if actual_error is None:
            self.assertIs(actual_result, expected_result)

    def test_identity_and_metadata_match_pytorch_2_13(self):
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual = actual_jit.Final
        expected = expected_jit.Final

        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(actual, typing.Final)
        self.assertIs(expected, typing.Final)
        self.assertIs(type(actual), type(expected))
        self.assertIs(inspect.getmodule(actual), inspect.getmodule(expected))
        for attribute in (
            "__module__",
            "__name__",
            "__qualname__",
            "__doc__",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(
                    getattr(actual, attribute), getattr(expected, attribute)
                )
        self.assertEqual(str(actual), str(expected))
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(typing.get_origin(actual), typing.get_origin(expected))
        self.assertEqual(typing.get_args(actual), typing.get_args(expected))

    def test_subscription_origin_arguments_and_annotations_match(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        for argument in (
            int,
            str,
            list[int],
            tuple[int, str],
            typing.Literal[1, "two"],
        ):
            with self.subTest(argument=argument):
                actual_annotation = actual[argument]
                expected_annotation = expected[argument]
                self.assertIs(actual_annotation, expected_annotation)
                self.assertIs(
                    typing.get_origin(actual_annotation),
                    typing.get_origin(expected_annotation),
                )
                self.assertEqual(
                    typing.get_args(actual_annotation),
                    typing.get_args(expected_annotation),
                )
                for attribute in (
                    "__origin__",
                    "__args__",
                    "__parameters__",
                    "__module__",
                    "__name__",
                    "__qualname__",
                ):
                    with self.subTest(argument=argument, attribute=attribute):
                        self.assertEqual(
                            getattr(actual_annotation, attribute),
                            getattr(expected_annotation, attribute),
                        )

        class ActualConfiguration:
            retries: torch.jit.Final[int] = 3

        class ExpectedConfiguration:
            retries: reference_torch.jit.Final[int] = 3

        self.assertEqual(
            ActualConfiguration.__annotations__, ExpectedConfiguration.__annotations__
        )
        self.assertEqual(
            typing.get_type_hints(ActualConfiguration),
            typing.get_type_hints(ExpectedConfiguration),
        )
        actual_configuration = ActualConfiguration()
        expected_configuration = ExpectedConfiguration()
        actual_configuration.retries = 4
        expected_configuration.retries = 4
        self.assertEqual(actual_configuration.retries, expected_configuration.retries)

    def test_instantiation_and_subscription_outcomes_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(int), lambda: expected(int)),
            (lambda: actual(value=int), lambda: expected(value=int)),
            (lambda: actual[()], lambda: expected[()]),
            (lambda: actual[int, str], lambda: expected[int, str]),
            (lambda: actual[actual[int]], lambda: expected[expected[int]]),
            (lambda: actual[int][str], lambda: expected[int][str]),
        )
        for actual_call, expected_call in cases:
            with self.subTest(actual_call=actual_call, expected_call=expected_call):
                self.assert_outcome_matches(actual_call, expected_call)

    def test_namespace_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final

        self.assertIn("Final", vars(torch.jit))
        self.assertIn("Final", vars(reference_torch.jit))
        self.assertNotIn("Final", torch.jit.__all__)
        self.assertNotIn("Final", reference_torch.jit.__all__)
        self.assertEqual(
            torch.__all__.count("Final"), reference_torch.__all__.count("Final")
        )
        self.assertFalse(hasattr(torch, "Final"))
        self.assertFalse(hasattr(reference_torch, "Final"))

        actual_explicit = {}
        expected_explicit = {}
        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.jit import Final", actual_explicit)
        exec("from torch.jit import Final", expected_explicit)
        exec("from torch_rs.jit import *", actual_wildcard)
        exec("from torch.jit import *", expected_wildcard)
        self.assertIs(actual_explicit["Final"], actual)
        self.assertIs(expected_explicit["Final"], expected)
        self.assertNotIn("Final", actual_wildcard)
        self.assertNotIn("Final", expected_wildcard)

        for actual_value, expected_value in (
            (actual, expected),
            (actual[int], expected[int]),
            (actual[tuple[str, int]], expected[tuple[str, int]]),
        ):
            with self.subTest(value=actual_value):
                self.assertIs(copy.copy(actual_value), actual_value)
                self.assertIs(copy.copy(expected_value), expected_value)
                self.assertIs(copy.deepcopy(actual_value), actual_value)
                self.assertIs(copy.deepcopy(expected_value), expected_value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=actual_value, protocol=protocol):
                        actual_payload = pickle.dumps(actual_value, protocol=protocol)
                        expected_payload = pickle.dumps(
                            expected_value, protocol=protocol
                        )
                        self.assertEqual(actual_payload, expected_payload)
                        self.assertIs(pickle.loads(actual_payload), actual_value)
                        self.assertIs(pickle.loads(expected_payload), expected_value)

    def test_supported_scope_remains_eager_only(self):
        self.assertIs(torch.jit.Final, reference_torch.jit.Final)
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
                self.assertTrue(hasattr(reference_torch.jit, name))
                self.assertFalse(hasattr(torch.jit, name))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
