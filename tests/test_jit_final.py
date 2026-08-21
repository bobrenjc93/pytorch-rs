import copy
import importlib
import inspect
import pickle
import typing
import unittest

import torch_rs as torch


class JitFinalTests(unittest.TestCase):
    def assert_error_matches_typing(self, operation):
        with self.assertRaises(Exception) as actual_raised:
            operation(torch.jit.Final)
        with self.assertRaises(Exception) as expected_raised:
            operation(typing.Final)

        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_exact_typing_alias_and_metadata(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        marker = jit.Final

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(marker, typing.Final)
        self.assertIs(internal.Final, typing.Final)
        self.assertIs(marker, internal.Final)
        self.assertIs(type(marker), type(typing.Final))
        self.assertEqual(repr(marker), "typing.Final")
        self.assertEqual(str(marker), "typing.Final")
        self.assertEqual(marker.__name__, "Final")
        self.assertEqual(marker.__qualname__, "Final")
        self.assertEqual(marker.__module__, "typing")
        self.assertIs(marker.__doc__, typing.Final.__doc__)
        self.assertIs(inspect.getmodule(marker), typing)
        self.assertEqual(
            str(inspect.signature(marker)), str(inspect.signature(typing.Final))
        )
        self.assertIsNone(typing.get_origin(marker))
        self.assertEqual(typing.get_args(marker), ())
        for attribute in ("__origin__", "__args__", "__parameters__"):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(marker, attribute))

    def test_subscription_origin_arguments_and_annotations(self):
        marker = torch.jit.Final
        for argument in (int, list[str], tuple[int, ...]):
            with self.subTest(argument=argument):
                alias = marker[argument]
                self.assertIs(alias, typing.Final[argument])
                self.assertIs(typing.get_origin(alias), marker)
                self.assertEqual(typing.get_args(alias), (argument,))
                self.assertEqual(repr(alias), repr(typing.Final[argument]))
                self.assertEqual(str(alias), str(typing.Final[argument]))
                self.assertEqual(alias.__module__, "typing")

        class Configuration:
            retries: marker[int] = 3

        self.assertEqual(
            Configuration.__annotations__, {"retries": typing.Final[int]}
        )
        self.assertEqual(
            typing.get_type_hints(Configuration), {"retries": typing.Final[int]}
        )
        self.assertEqual(Configuration.retries, 3)

    def test_copy_and_pickle_preserve_canonical_typing_objects(self):
        for value in (
            torch.jit.Final,
            torch.jit.Final[int],
            torch.jit.Final[list[str]],
        ):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=value, protocol=protocol):
                        payload = pickle.dumps(value, protocol=protocol)
                        self.assertIn(b"typing", payload)
                        self.assertIs(pickle.loads(payload), value)

    def test_instantiation_and_subscription_arity_errors_match_typing(self):
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
                self.assert_error_matches_typing(operation)

    def test_explicit_exports_exclude_wildcards_and_compilation(self):
        internal_namespace = {}
        jit_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs._jit_internal import Final", internal_namespace)
        exec("from torch_rs.jit import Final", jit_namespace)
        exec("from torch_rs.jit import *", wildcard_namespace)

        self.assertIs(internal_namespace["Final"], typing.Final)
        self.assertIs(jit_namespace["Final"], typing.Final)
        self.assertNotIn("Final", torch.jit.__all__)
        self.assertNotIn("Final", wildcard_namespace)
        self.assertNotIn("Final", torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))

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
