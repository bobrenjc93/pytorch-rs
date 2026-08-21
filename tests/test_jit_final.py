import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import typing
import unittest

import torch_rs as torch


class JitFinalTests(unittest.TestCase):
    def error_outcome(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception), raised.exception.args

    def test_both_jit_names_are_the_exact_typing_marker(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        jit = importlib.import_module("torch_rs.jit")

        self.assertIs(torch._jit_internal, internal)
        self.assertIs(torch.jit, jit)
        self.assertIs(internal.Final, typing.Final)
        self.assertIs(jit.Final, typing.Final)
        self.assertIs(jit.Final, internal.Final)

        internal_namespace = {}
        jit_namespace = {}
        exec("from torch_rs._jit_internal import Final", internal_namespace)
        exec("from torch_rs.jit import Final", jit_namespace)
        self.assertIs(internal_namespace["Final"], typing.Final)
        self.assertIs(jit_namespace["Final"], typing.Final)

        internal_wildcard_namespace = {}
        exec("from torch_rs._jit_internal import *", internal_wildcard_namespace)
        self.assertIs(internal_wildcard_namespace["Final"], typing.Final)

    def test_subscription_origin_and_arguments_are_typing_final_semantics(self):
        for marker in (torch._jit_internal.Final, torch.jit.Final):
            with self.subTest(marker=marker, parameter="unsubscripted"):
                self.assertIsNone(typing.get_origin(marker))
                self.assertEqual(typing.get_args(marker), ())

            for parameter in (int, str, list[int], None, ...):
                with self.subTest(marker=marker, parameter=parameter):
                    actual = marker[parameter]
                    expected = typing.Final[parameter]
                    self.assertIs(actual, expected)
                    self.assertIs(typing.get_origin(actual), typing.Final)
                    self.assertEqual(typing.get_args(actual), typing.get_args(expected))
                    self.assertEqual(repr(actual), repr(expected))
                    self.assertEqual(str(actual), str(expected))

    def test_instantiation_and_subscription_arity_errors_match_typing(self):
        calls = (
            lambda marker: marker(),
            lambda marker: marker(1),
            lambda marker: marker(1, 2),
            lambda marker: marker(value=1),
            lambda marker: marker[int, str],
            lambda marker: marker[()],
            lambda marker: marker[marker[int]],
        )
        for marker in (torch._jit_internal.Final, torch.jit.Final):
            for case, call in enumerate(calls):
                with self.subTest(marker=marker, case=case):
                    self.assertEqual(
                        self.error_outcome(lambda: call(marker)),
                        self.error_outcome(lambda: call(typing.Final)),
                    )

        for operation in (
            lambda marker: isinstance(1, marker),
            lambda marker: issubclass(int, marker),
        ):
            self.assertEqual(
                self.error_outcome(lambda: operation(torch.jit.Final)),
                self.error_outcome(lambda: operation(typing.Final)),
            )

    def test_metadata_is_owned_by_typing(self):
        marker = torch.jit.Final

        self.assertIs(type(marker), type(typing.Final))
        self.assertEqual(repr(marker), "typing.Final")
        self.assertEqual(str(marker), "typing.Final")
        self.assertEqual(marker.__module__, "typing")
        self.assertEqual(marker.__name__, "Final")
        self.assertEqual(marker.__qualname__, "Final")
        self.assertEqual(marker._name, "Final")
        self.assertEqual(marker.__doc__, typing.Final.__doc__)
        self.assertEqual(
            str(inspect.signature(marker)),
            str(inspect.signature(typing.Final)),
        )
        self.assertIs(inspect.getmodule(marker), typing)
        self.assertFalse(hasattr(marker, "__dict__"))

        parameterized = marker[int]
        self.assertEqual(parameterized.__module__, "typing")
        self.assertIs(typing.get_origin(parameterized), marker)
        self.assertEqual(typing.get_args(parameterized), (int,))

    def test_copying_and_pickling_preserve_canonical_typing_objects(self):
        for value in (torch.jit.Final, torch.jit.Final[int]):
            with self.subTest(value=value, operation="copy"):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=value, protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertNotIn(b"torch_rs", payload)
                    self.assertIn(b"typing", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_final_is_public_but_not_a_wildcard_export(self):
        jit = torch.jit
        supported_wildcard = {
            "Attribute",
            "annotate",
            "export",
            "ignore",
            "isinstance",
            "script_if_tracing",
            "unused",
        }

        self.assertNotIn("Final", jit.__all__)
        self.assertEqual(set(jit.__all__), supported_wildcard)
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {*supported_wildcard, "Final", "is_scripting", "is_tracing"},
        )

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if not name.startswith("__")},
            supported_wildcard,
        )
        self.assertNotIn("Final", wildcard_namespace)

        self.assertNotIn("Final", torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))
        for name in (
            "CompilationUnit",
            "ScriptModule",
            "interface",
            "script",
            "trace",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(jit, name))
        self.assertFalse(hasattr(torch, "compile"))

    def test_fresh_process_imports_preserve_the_alias_and_unsupported_boundary(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import typing
import torch_rs as torch
from torch_rs._jit_internal import Final as InternalFinal
from torch_rs.jit import Final as JitFinal

assert InternalFinal is typing.Final
assert JitFinal is typing.Final
assert torch._jit_internal.Final is torch.jit.Final
assert typing.get_origin(torch.jit.Final[int]) is typing.Final
assert typing.get_args(torch.jit.Final[int]) == (int,)
assert "Final" not in torch.jit.__all__
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "trace")
assert not hasattr(torch, "compile")
""",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
