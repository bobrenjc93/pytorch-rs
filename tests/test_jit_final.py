import copy
import importlib
import inspect
import pickle
import typing
import unittest

import torch_rs as torch


class JitFinalTests(unittest.TestCase):
    def test_internal_and_jit_names_are_the_typing_final_singleton(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        jit = importlib.import_module("torch_rs.jit")
        marker = typing.Final

        self.assertIs(torch._jit_internal, internal)
        self.assertIs(torch.jit, jit)
        self.assertIs(internal.Final, marker)
        self.assertIs(jit.Final, marker)
        self.assertIs(torch.jit.Final, torch._jit_internal.Final)
        self.assertIs(type(jit.Final), type(marker))
        self.assertIs(inspect.getmodule(jit.Final), typing)
        self.assertEqual(jit.Final.__module__, "typing")
        self.assertEqual(jit.Final.__name__, "Final")
        self.assertEqual(jit.Final.__qualname__, "Final")
        self.assertEqual(jit.Final.__doc__, marker.__doc__)
        self.assertEqual(str(jit.Final), "typing.Final")
        self.assertEqual(repr(jit.Final), "typing.Final")
        self.assertIsNone(typing.get_origin(jit.Final))
        self.assertEqual(typing.get_args(jit.Final), ())

    def test_subscription_origin_arguments_and_eager_annotations(self):
        marker = torch.jit.Final
        for argument in (
            int,
            str,
            list[int],
            tuple[int, str],
            typing.Literal[1, "two"],
        ):
            with self.subTest(argument=argument):
                annotation = marker[argument]
                self.assertIs(annotation, typing.Final[argument])
                self.assertIs(typing.get_origin(annotation), typing.Final)
                self.assertEqual(typing.get_args(annotation), (argument,))
                self.assertIs(annotation.__origin__, typing.Final)
                self.assertEqual(annotation.__args__, (argument,))
                self.assertEqual(annotation.__parameters__, ())
                self.assertEqual(annotation.__module__, "typing")
                self.assertEqual(annotation.__name__, "Final")
                self.assertEqual(annotation.__qualname__, "Final")

        class Configuration:
            retries: torch.jit.Final[int] = 3

        self.assertIs(Configuration.__annotations__["retries"], typing.Final[int])
        self.assertIs(
            typing.get_type_hints(Configuration)["retries"], typing.Final[int]
        )
        configuration = Configuration()
        configuration.retries = 4
        self.assertEqual(configuration.retries, 4)

    def test_instantiation_and_invalid_subscriptions_use_typing_errors(self):
        marker = torch.jit.Final
        cases = (
            (lambda: marker(), "Cannot instantiate typing.Final"),
            (lambda: marker(int), "Cannot instantiate typing.Final"),
            (lambda: marker(value=int), "Cannot instantiate typing.Final"),
            (
                lambda: marker[()],
                "typing.Final accepts only single type. Got ().",
            ),
            (
                lambda: marker[int, str],
                "typing.Final accepts only single type. Got "
                "(<class 'int'>, <class 'str'>).",
            ),
            (
                lambda: marker[marker[int]],
                "typing.Final[int] is not valid as type argument",
            ),
            (
                lambda: marker[int][str],
                "typing.Final[int] is not a generic class",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_namespace_copying_and_pickling_preserve_the_typing_alias(self):
        marker = torch.jit.Final

        self.assertIn("Final", vars(torch.jit))
        self.assertNotIn("Final", torch.jit.__all__)
        self.assertNotIn("Final", torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))

        internal_namespace = {}
        jit_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs._jit_internal import Final", internal_namespace)
        exec("from torch_rs.jit import Final", jit_namespace)
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertIs(internal_namespace["Final"], typing.Final)
        self.assertIs(jit_namespace["Final"], typing.Final)
        self.assertNotIn("Final", wildcard_namespace)

        for value in (marker, marker[int], marker[tuple[str, int]]):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=value, protocol=protocol):
                        payload = pickle.dumps(value, protocol=protocol)
                        self.assertNotIn(b"torch_rs", payload)
                        self.assertIs(pickle.loads(payload), value)

    def test_final_does_not_enable_scripting_or_compilation(self):
        class Configuration:
            value: torch.jit.Final[int] = 1

        self.assertEqual(Configuration().value, 1)
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
