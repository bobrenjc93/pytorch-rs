import copy
import importlib
import pickle
import typing
import unittest

import torch_rs as torch


class JitFinalTests(unittest.TestCase):
    def test_alias_identity_and_metadata(self):
        internal = importlib.import_module("torch_rs._jit_internal")

        self.assertIs(torch._jit_internal, internal)
        self.assertIs(internal.Final, typing.Final)
        self.assertIs(torch.jit.Final, typing.Final)
        self.assertIs(torch.jit.Final, internal.Final)
        self.assertIs(type(torch.jit.Final), type(typing.Final))
        self.assertEqual(torch.jit.Final.__module__, "typing")
        self.assertEqual(torch.jit.Final.__name__, "Final")
        self.assertEqual(torch.jit.Final.__qualname__, "Final")
        self.assertEqual(repr(torch.jit.Final), "typing.Final")
        self.assertEqual(str(torch.jit.Final), "typing.Final")
        self.assertIsNone(typing.get_origin(torch.jit.Final))
        self.assertEqual(typing.get_args(torch.jit.Final), ())

    def test_subscription_origin_and_arguments(self):
        cases = (
            (int, (int,), "typing.Final[int]"),
            (list[int], (list[int],), "typing.Final[list[int]]"),
        )
        for argument, expected_args, expected_repr in cases:
            with self.subTest(argument=argument):
                annotation = torch.jit.Final[argument]
                self.assertIs(annotation, typing.Final[argument])
                self.assertIs(typing.get_origin(annotation), typing.Final)
                self.assertEqual(typing.get_args(annotation), expected_args)
                self.assertIs(annotation.__origin__, typing.Final)
                self.assertEqual(annotation.__args__, expected_args)
                self.assertEqual(annotation.__parameters__, ())
                self.assertEqual(repr(annotation), expected_repr)

    def test_instantiation_and_subscription_errors(self):
        calls = (
            lambda: torch.jit.Final(),
            lambda: torch.jit.Final(int),
            lambda: torch.jit.Final[int](),
            lambda: torch.jit.Final[()],
            lambda: torch.jit.Final[int, str],
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_copy_and_pickle_preserve_canonical_typing_objects(self):
        for value in (
            torch.jit.Final,
            torch.jit.Final[int],
            torch.jit.Final[list[int]],
        ):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=value, protocol=protocol):
                        payload = pickle.dumps(value, protocol=protocol)
                        self.assertNotIn(b"torch_rs", payload)
                        self.assertIs(pickle.loads(payload), value)

    def test_exports_and_unsupported_compilation_boundary(self):
        internal_namespace = {}
        jit_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs._jit_internal import Final", internal_namespace)
        exec("from torch_rs.jit import Final", jit_namespace)
        exec("from torch_rs.jit import *", wildcard_namespace)

        self.assertIs(internal_namespace["Final"], typing.Final)
        self.assertIs(jit_namespace["Final"], typing.Final)
        self.assertIn("Final", vars(torch.jit))
        self.assertNotIn("Final", torch.jit.__all__)
        self.assertNotIn("Final", wildcard_namespace)
        self.assertNotIn("Final", torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))

        for name in ("script", "trace"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
