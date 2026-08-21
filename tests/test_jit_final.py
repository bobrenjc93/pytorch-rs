import copy
import importlib
import inspect
import pickle
import typing
import unittest

import torch_rs as torch


class JitFinalTests(unittest.TestCase):
    def test_is_the_standard_library_final_object(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(internal.Final, typing.Final)
        self.assertIs(jit.Final, typing.Final)
        self.assertIs(jit.Final, internal.Final)

    def test_subscription_origin_and_arguments(self):
        marker = torch.jit.Final

        self.assertIs(marker[int], typing.Final[int])
        self.assertIs(marker[int], marker[int])
        for argument in (int, list[int], typing.Any, None):
            with self.subTest(argument=argument):
                value = marker[argument]
                expected_argument = type(None) if argument is None else argument
                self.assertIs(value, typing.Final[argument])
                self.assertIs(typing.get_origin(value), marker)
                self.assertEqual(typing.get_args(value), (expected_argument,))

        self.assertIsNone(typing.get_origin(marker))
        self.assertEqual(typing.get_args(marker), ())

    def test_instantiation_and_subscription_arity_errors(self):
        marker = torch.jit.Final
        cases = (
            (lambda: marker(), "Cannot instantiate typing.Final"),
            (lambda: marker(int), "Cannot instantiate typing.Final"),
            (lambda: marker(item=int), "Cannot instantiate typing.Final"),
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

    def test_metadata_matches_typing_final(self):
        marker = torch.jit.Final

        self.assertIs(type(marker), type(typing.Final))
        self.assertEqual(repr(marker), "typing.Final")
        self.assertEqual(str(marker), "typing.Final")
        self.assertEqual(str(inspect.signature(marker)), "(*args, **kwds)")
        self.assertEqual(marker.__name__, "Final")
        self.assertEqual(marker.__qualname__, "Final")
        self.assertEqual(marker.__module__, "typing")
        self.assertEqual(marker.__doc__, typing.Final.__doc__)
        self.assertEqual(marker._name, "Final")
        self.assertIs(inspect.getmodule(marker), typing)
        for name in ("__origin__", "__args__", "__parameters__"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError) as raised:
                    getattr(marker, name)
                self.assertEqual(raised.exception.args, (name,))

        subscribed = marker[list[int]]
        self.assertEqual(repr(subscribed), "typing.Final[list[int]]")
        self.assertEqual(subscribed.__name__, "Final")
        self.assertEqual(subscribed.__qualname__, "Final")
        self.assertEqual(subscribed.__module__, "typing")
        self.assertIsNone(subscribed.__doc__)
        self.assertIsNone(subscribed._name)
        self.assertIs(subscribed.__origin__, marker)
        self.assertEqual(subscribed.__args__, (list[int],))
        self.assertEqual(subscribed.__parameters__, ())

    def test_copy_and_pickle_preserve_canonical_objects(self):
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
        marker = torch.jit.Final

        self.assertIn("Final", vars(torch.jit))
        self.assertNotIn("Final", torch.jit.__all__)
        explicit_namespace = {}
        exec("from torch_rs.jit import Final", explicit_namespace)
        self.assertIs(explicit_namespace["Final"], marker)

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
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
