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
            raise AssertionError("jit.Final differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def metadata(self, value):
        return (
            type(value),
            value.__module__,
            value.__name__,
            value.__qualname__,
            repr(value),
            str(value),
            typing.get_origin(value),
            typing.get_args(value),
            getattr(value, "__origin__", None),
            getattr(value, "__args__", None),
            getattr(value, "__parameters__", None),
        )

    def test_alias_identity_and_metadata_match(self):
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")

        self.assertIs(actual_internal.Final, typing.Final)
        self.assertIs(expected_internal.Final, typing.Final)
        self.assertIs(torch.jit.Final, typing.Final)
        self.assertIs(reference_torch.jit.Final, typing.Final)
        self.assertIs(torch.jit.Final, actual_internal.Final)
        self.assertIs(reference_torch.jit.Final, expected_internal.Final)
        self.assertEqual(
            self.metadata(torch.jit.Final),
            self.metadata(reference_torch.jit.Final),
        )

    def test_subscription_origin_and_arguments_match(self):
        subscriptions = (
            lambda final: final[int],
            lambda final: final[list[int]],
            lambda final: final[tuple[str, int]],
        )
        for subscribe in subscriptions:
            with self.subTest(subscribe=subscribe):
                actual = subscribe(torch.jit.Final)
                expected = subscribe(reference_torch.jit.Final)
                self.assertIs(actual, expected)
                self.assertEqual(self.metadata(actual), self.metadata(expected))

    def test_instantiation_and_arity_errors_match(self):
        cases = (
            lambda final: final(),
            lambda final: final(int),
            lambda final: final[int](),
            lambda final: final[()],
            lambda final: final[int, str],
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch.jit.Final),
                    lambda: call(reference_torch.jit.Final),
                )

    def test_copy_and_pickle_match(self):
        pairs = (
            (torch.jit.Final, reference_torch.jit.Final),
            (torch.jit.Final[int], reference_torch.jit.Final[int]),
            (
                torch.jit.Final[list[int]],
                reference_torch.jit.Final[list[int]],
            ),
        )
        for actual, expected in pairs:
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

    def test_exports_and_unsupported_compilation_boundary_match_scope(self):
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)

        self.assertNotIn("Final", torch.jit.__all__)
        self.assertNotIn("Final", reference_torch.jit.__all__)
        self.assertNotIn("Final", actual_namespace)
        self.assertNotIn("Final", expected_namespace)
        self.assertNotIn("Final", torch.__all__)
        self.assertNotIn("Final", reference_torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))
        self.assertFalse(hasattr(reference_torch, "Final"))

        for name in ("script", "trace"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch.jit, name))
                self.assertFalse(hasattr(torch.jit, name))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
