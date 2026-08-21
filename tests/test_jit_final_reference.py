import copy
import importlib
import inspect
import pickle
import pickletools
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

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            shape.append((opcode.name, argument))
        return shape

    def test_internal_and_public_aliases_are_exact_typing_final(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(torch._jit_internal, actual_internal)
        self.assertIs(reference_torch._jit_internal, expected_internal)
        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(actual_jit.Final, typing.Final)
        self.assertIs(expected_jit.Final, typing.Final)

    def test_subscription_origin_and_arguments_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final

        self.assertEqual(
            (typing.get_origin(actual), typing.get_args(actual)),
            (typing.get_origin(expected), typing.get_args(expected)),
        )
        for argument in (int, list[int], typing.Any, None, ...):
            with self.subTest(argument=argument):
                actual_value = actual[argument]
                expected_value = expected[argument]
                self.assertIs(actual_value, expected_value)
                self.assertEqual(repr(actual_value), repr(expected_value))
                self.assertEqual(str(actual_value), str(expected_value))
                self.assertIs(typing.get_origin(actual_value), actual)
                self.assertIs(typing.get_origin(expected_value), expected)
                self.assertEqual(
                    typing.get_args(actual_value),
                    typing.get_args(expected_value),
                )

    def test_instantiation_and_arity_errors_match_pytorch_2_13(self):
        cases = (
            lambda marker: marker(),
            lambda marker: marker(int),
            lambda marker: marker(item=int),
            lambda marker: marker[()],
            lambda marker: marker[int, str],
            lambda marker: marker[int, str, bytes],
            lambda marker: marker[int][str],
        )

        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch.jit.Final),
                    lambda: call(reference_torch.jit.Final),
                )

    def test_metadata_matches_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final

        self.assertIs(type(actual), type(expected))
        self.assertEqual(repr(actual), repr(expected))
        self.assertEqual(str(actual), str(expected))
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        for name in (
            "__name__",
            "__qualname__",
            "__module__",
            "__doc__",
            "_name",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(actual, name), getattr(expected, name))
        self.assertIs(inspect.getmodule(actual), typing)
        self.assertIs(inspect.getmodule(expected), typing)

        actual_value = actual[list[int]]
        expected_value = expected[list[int]]
        for name in (
            "__name__",
            "__qualname__",
            "__module__",
            "__doc__",
            "_name",
            "__origin__",
            "__args__",
            "__parameters__",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(actual_value, name),
                    getattr(expected_value, name),
                )

    def test_copy_and_pickle_match_pytorch_2_13(self):
        pairs = (
            (torch.jit.Final, reference_torch.jit.Final),
            (torch.jit.Final[int], reference_torch.jit.Final[int]),
            (torch.jit.Final[list[int]], reference_torch.jit.Final[list[int]]),
        )
        for actual, expected in pairs:
            with self.subTest(value=actual):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=actual, protocol=protocol):
                        self.assertEqual(
                            self.pickle_shape(actual, protocol),
                            self.pickle_shape(expected, protocol),
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(actual, protocol)), actual
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol)), expected
                        )

    def test_exports_and_unsupported_boundary_match(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit

        self.assertNotIn("Final", actual_jit.__all__)
        self.assertNotIn("Final", expected_jit.__all__)
        self.assertIn("Final", vars(actual_jit))
        self.assertIn("Final", vars(expected_jit))

        actual_explicit = {}
        expected_explicit = {}
        exec("from torch_rs.jit import Final", actual_explicit)
        exec("from torch.jit import Final", expected_explicit)
        self.assertIs(actual_explicit["Final"], actual_jit.Final)
        self.assertIs(expected_explicit["Final"], expected_jit.Final)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.jit import *", actual_wildcard)
        exec("from torch.jit import *", expected_wildcard)
        self.assertNotIn("Final", actual_wildcard)
        self.assertNotIn("Final", expected_wildcard)

        self.assertEqual(
            torch.__all__.count("Final"),
            reference_torch.__all__.count("Final"),
        )
        self.assertFalse(hasattr(torch, "Final"))
        self.assertFalse(hasattr(reference_torch, "Final"))
        self.assertTrue(callable(reference_torch.jit.script))
        self.assertTrue(callable(reference_torch.jit.trace))
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
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
