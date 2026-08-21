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

    def metadata(self, value):
        return (
            type(value),
            value.__name__,
            value.__qualname__,
            value.__module__,
            value.__doc__,
            str(inspect.signature(value)),
            inspect.getmodule(value),
            str(value),
            repr(value),
            typing.get_origin(value),
            typing.get_args(value),
        )

    def test_alias_identity_and_metadata_match_pytorch_2_13(self):
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")

        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(actual_jit.Final, typing.Final)
        self.assertIs(expected_jit.Final, typing.Final)
        self.assertEqual(
            self.metadata(actual_jit.Final),
            self.metadata(expected_jit.Final),
        )

    def test_subscription_origin_and_arguments_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        for argument in (int, list[str], int | str, ..., "ForwardDeclared", [int]):
            with self.subTest(argument=argument):
                try:
                    expected_value = expected[argument]
                except Exception as expected_error:
                    with self.assertRaises(type(expected_error)) as actual_raised:
                        actual[argument]
                    self.assertIs(
                        type(actual_raised.exception), type(expected_error)
                    )
                    self.assertEqual(
                        str(actual_raised.exception), str(expected_error)
                    )
                    self.assertEqual(
                        actual_raised.exception.args, expected_error.args
                    )
                    continue

                actual_value = actual[argument]
                self.assertEqual(actual_value, expected_value)
                self.assertIs(
                    typing.get_origin(actual_value),
                    typing.get_origin(expected_value),
                )
                self.assertEqual(
                    typing.get_args(actual_value),
                    typing.get_args(expected_value),
                )
                self.assertEqual(
                    self.metadata(actual_value),
                    self.metadata(expected_value),
                )

    def test_instantiation_and_arity_errors_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        cases = (
            lambda marker: marker(),
            lambda marker: marker(int),
            lambda marker: marker(value=int),
            lambda marker: marker[()],
            lambda marker: marker[int, str],
            lambda marker: marker[int,],
            lambda marker: marker[int][str],
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_copying_and_pickling_match_pytorch_2_13(self):
        for actual, expected in (
            (torch.jit.Final, reference_torch.jit.Final),
            (torch.jit.Final[int], reference_torch.jit.Final[int]),
        ):
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
                            pickle.loads(pickle.dumps(actual, protocol=protocol)),
                            actual,
                        )
                        self.assertIs(
                            pickle.loads(pickle.dumps(expected, protocol=protocol)),
                            expected,
                        )

    def test_exports_match_the_supported_eager_surface(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit

        self.assertNotIn("Final", actual_jit.__all__)
        self.assertNotIn("Final", expected_jit.__all__)
        self.assertIn("Final", vars(actual_jit))
        self.assertIn("Final", vars(expected_jit))

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertNotIn("Final", actual_namespace)
        self.assertNotIn("Final", expected_namespace)

        actual_internal_namespace = {}
        expected_internal_namespace = {}
        exec("from torch_rs._jit_internal import *", actual_internal_namespace)
        exec("from torch._jit_internal import *", expected_internal_namespace)
        self.assertIs(actual_internal_namespace["Final"], typing.Final)
        self.assertIs(expected_internal_namespace["Final"], typing.Final)

        for module in (torch, reference_torch):
            self.assertNotIn("Final", module.__all__)
            self.assertFalse(hasattr(module, "Final"))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "interface",
            "script",
            "script_method",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
