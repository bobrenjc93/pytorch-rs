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

    def metadata(self, marker):
        return (
            type(marker),
            repr(marker),
            str(marker),
            marker.__module__,
            marker.__name__,
            marker.__qualname__,
            marker.__doc__,
            str(inspect.signature(marker)),
            typing.get_origin(marker),
            typing.get_args(marker),
        )

    def subscription_outcome(self, marker, argument):
        value = marker[argument]
        return (
            type(value),
            repr(value),
            str(value),
            typing.get_origin(value) is marker,
            typing.get_args(value),
            value == marker[argument],
            value is marker[argument],
        )

    def test_alias_identity_and_metadata_match_pytorch_2_13(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")

        self.assertIs(actual_jit.Final, typing.Final)
        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_jit.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(actual_jit.Final, expected_jit.Final)
        self.assertEqual(
            self.metadata(actual_jit.Final),
            self.metadata(expected_jit.Final),
        )

    def test_subscription_origin_and_arguments_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        for argument in (int, list[str], None, Ellipsis, typing.Dict[str, int]):
            with self.subTest(argument=argument):
                self.assertEqual(
                    self.subscription_outcome(actual, argument),
                    self.subscription_outcome(expected, argument),
                )

    def test_instantiation_and_arity_errors_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        calls = (
            lambda marker: marker(),
            lambda marker: marker(int),
            lambda marker: marker(value=int),
            lambda marker: marker[int](),
            lambda marker: marker[int](1, value=2),
            lambda marker: marker[()],
            lambda marker: marker[(int,)],
            lambda marker: marker[int, str],
            lambda marker: marker[int, str, float],
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_copying_and_pickling_match_pytorch_2_13(self):
        actual = torch.jit.Final
        expected = reference_torch.jit.Final
        factories = (
            lambda marker: marker,
            lambda marker: marker[int],
            lambda marker: marker[list[int]],
            lambda marker: marker[None],
        )

        for factory in factories:
            actual_value = factory(actual)
            expected_value = factory(expected)
            with self.subTest(value=actual_value):
                self.assertEqual(
                    copy.copy(actual_value) is actual_value,
                    copy.copy(expected_value) is expected_value,
                )
                self.assertEqual(
                    copy.deepcopy(actual_value) is actual_value,
                    copy.deepcopy(expected_value) is expected_value,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    actual_payload = pickle.dumps(actual_value, protocol=protocol)
                    expected_payload = pickle.dumps(expected_value, protocol=protocol)
                    self.assertEqual(actual_payload, expected_payload)
                    self.assertEqual(
                        pickle.loads(actual_payload),
                        pickle.loads(expected_payload),
                    )

    def test_namespace_and_unsupported_compiler_boundary_match(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit

        self.assertIn("Final", vars(actual_jit))
        self.assertIn("Final", vars(expected_jit))
        self.assertEqual("Final" in actual_jit.__all__, "Final" in expected_jit.__all__)
        self.assertNotIn("Final", actual_jit.__all__)

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
                self.assertTrue(hasattr(expected_jit, name))
                self.assertFalse(hasattr(actual_jit, name))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
