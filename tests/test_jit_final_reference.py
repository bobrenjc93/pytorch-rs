import copy
import importlib
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

    def test_identity_and_metadata_match_pytorch_2_13(self):
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")

        self.assertIs(torch._jit_internal, actual_internal)
        self.assertIs(reference_torch._jit_internal, expected_internal)
        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(actual_jit.Final, typing.Final)
        self.assertIs(expected_jit.Final, typing.Final)

        actual = actual_jit.Final
        expected = expected_jit.Final
        self.assertIs(type(actual), type(expected))
        self.assertEqual(repr(actual), repr(expected))
        self.assertEqual(str(actual), str(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__doc__, expected.__doc__)

    def test_subscription_origin_and_arguments_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        for argument in (
            int,
            list[str],
            None,
            int | str,
            typing.Any,
        ):
            with self.subTest(argument=argument):
                actual_alias = actual[argument]
                expected_alias = expected[argument]
                self.assertEqual(actual_alias, expected_alias)
                self.assertIs(
                    typing.get_origin(actual_alias),
                    typing.get_origin(expected_alias),
                )
                self.assertEqual(
                    typing.get_args(actual_alias),
                    typing.get_args(expected_alias),
                )
                self.assertEqual(repr(actual_alias), repr(expected_alias))
                self.assertEqual(actual_alias.__dict__, expected_alias.__dict__)

    def test_instantiation_and_arity_errors_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        cases = (
            lambda marker: marker(),
            lambda marker: marker(int),
            lambda marker: marker(value=int),
            lambda marker: marker[int](),
            lambda marker: marker[int](1),
            lambda marker: marker[int](value=1),
            lambda marker: marker[()],
            lambda marker: marker[int, str],
            lambda marker: marker[int][str],
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_copying_and_pickling_match_pytorch_2_13(self):
        actual_values = (
            torch.jit.Final,
            torch.jit.Final[int],
            torch.jit.Final[list[str]],
            torch.jit.Final[None],
        )
        expected_values = (
            reference_torch.jit.Final,
            reference_torch.jit.Final[int],
            reference_torch.jit.Final[list[str]],
            reference_torch.jit.Final[None],
        )
        for actual, expected in zip(actual_values, expected_values, strict=True):
            with self.subTest(value=actual):
                self.assertEqual(actual, expected)

                actual_copy = copy.copy(actual)
                expected_copy = copy.copy(expected)
                self.assertEqual(actual_copy, expected_copy)
                self.assertEqual(
                    actual_copy is actual,
                    expected_copy is expected,
                )

                actual_deepcopy = copy.deepcopy(actual)
                expected_deepcopy = copy.deepcopy(expected)
                self.assertEqual(actual_deepcopy, expected_deepcopy)
                self.assertEqual(
                    actual_deepcopy is actual,
                    expected_deepcopy is expected,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(protocol=protocol):
                        actual_payload = pickle.dumps(actual, protocol=protocol)
                        expected_payload = pickle.dumps(expected, protocol=protocol)
                        actual_restored = pickle.loads(actual_payload)
                        expected_restored = pickle.loads(expected_payload)
                        self.assertEqual(actual_payload, expected_payload)
                        self.assertEqual(actual_restored, expected_restored)
                        self.assertEqual(
                            actual_restored is actual,
                            expected_restored is expected,
                        )

    def test_exports_and_unsupported_boundary_match_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        self.assertNotIn("Final", actual_jit.__all__)
        self.assertNotIn("Final", expected_jit.__all__)
        self.assertFalse(hasattr(torch._jit_internal, "__all__"))
        self.assertFalse(hasattr(reference_torch._jit_internal, "__all__"))

        actual_explicit = {}
        expected_explicit = {}
        exec("from torch_rs.jit import Final", actual_explicit)
        exec("from torch.jit import Final", expected_explicit)
        self.assertIs(actual_explicit["Final"], typing.Final)
        self.assertIs(expected_explicit["Final"], typing.Final)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.jit import *", actual_wildcard)
        exec("from torch.jit import *", expected_wildcard)
        self.assertNotIn("Final", actual_wildcard)
        self.assertNotIn("Final", expected_wildcard)

        self.assertFalse(hasattr(torch, "Final"))
        self.assertFalse(hasattr(reference_torch, "Final"))
        self.assertNotIn("Final", torch.__all__)
        self.assertNotIn("Final", reference_torch.__all__)
        self.assertIs(torch.jit.is_scripting(), False)
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
                self.assertFalse(hasattr(actual_jit, name))
                self.assertTrue(hasattr(expected_jit, name))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertTrue(hasattr(reference_torch, "compile"))


if __name__ == "__main__":
    unittest.main()
