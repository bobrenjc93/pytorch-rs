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
    def test_internal_and_public_aliases_are_exact_typing_final(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        jit = importlib.import_module("torch_rs.jit")

        self.assertIs(torch._jit_internal, internal)
        self.assertIs(torch.jit, jit)
        self.assertIs(internal.Final, typing.Final)
        self.assertIs(jit.Final, typing.Final)
        self.assertIs(jit.Final, internal.Final)

        direct_namespace = {}
        exec(
            "from torch_rs._jit_internal import Final as InternalFinal\n"
            "from torch_rs.jit import Final as JitFinal",
            direct_namespace,
        )
        self.assertIs(direct_namespace["InternalFinal"], typing.Final)
        self.assertIs(direct_namespace["JitFinal"], typing.Final)

    def test_subscription_origin_and_arguments_follow_typing_final(self):
        marker = torch.jit.Final

        self.assertIsNone(typing.get_origin(marker))
        self.assertEqual(typing.get_args(marker), ())

        cases = (
            int,
            list[str],
            int | str,
            ...,
            "ForwardDeclared",
            [int],
        )
        for argument in cases:
            with self.subTest(argument=argument):
                try:
                    expected = typing.Final[argument]
                except Exception as expected_error:
                    with self.assertRaises(type(expected_error)) as actual_raised:
                        marker[argument]
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

                actual = marker[argument]
                self.assertEqual(actual, expected)
                self.assertIs(typing.get_origin(actual), typing.Final)
                self.assertEqual(typing.get_args(actual), typing.get_args(expected))

    def test_instantiation_and_subscription_arity_errors(self):
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
                lambda: marker[int,],
                "typing.Final accepts only single type. Got (<class 'int'>,).",
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

    def test_metadata_matches_the_standard_library_object(self):
        marker = torch.jit.Final
        parameterized = marker[int]

        self.assertIs(type(marker), type(typing.Final))
        self.assertEqual(marker.__name__, "Final")
        self.assertEqual(marker.__qualname__, "Final")
        self.assertEqual(marker.__module__, "typing")
        self.assertEqual(marker.__doc__, typing.Final.__doc__)
        self.assertEqual(str(inspect.signature(marker)), "(*args, **kwds)")
        self.assertIs(inspect.getmodule(marker), typing)
        self.assertEqual(str(marker), "typing.Final")
        self.assertEqual(repr(marker), "typing.Final")

        self.assertEqual(parameterized.__name__, "Final")
        self.assertEqual(parameterized.__qualname__, "Final")
        self.assertEqual(parameterized.__module__, "typing")
        self.assertIs(parameterized.__origin__, marker)
        self.assertEqual(parameterized.__args__, (int,))
        self.assertEqual(parameterized.__parameters__, ())
        self.assertEqual(str(parameterized), "typing.Final[int]")
        self.assertEqual(repr(parameterized), "typing.Final[int]")

    def test_copying_and_pickling_preserve_standard_library_identities(self):
        for value in (torch.jit.Final, torch.jit.Final[int]):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(value=value, protocol=protocol):
                        payload = pickle.dumps(value, protocol=protocol)
                        self.assertIn(b"typing", payload)
                        self.assertNotIn(b"torch_rs", payload)
                        self.assertIs(pickle.loads(payload), value)

    def test_final_is_explicit_only_and_not_a_wildcard_export(self):
        jit = torch.jit
        internal = torch._jit_internal

        self.assertNotIn("Final", jit.__all__)
        self.assertIn("Final", vars(jit))
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertNotIn("Final", jit_namespace)

        self.assertFalse(hasattr(internal, "__all__"))
        internal_namespace = {}
        exec("from torch_rs._jit_internal import *", internal_namespace)
        self.assertIs(internal_namespace["Final"], typing.Final)

        self.assertNotIn("Final", torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("Final", top_level_namespace)

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
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
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))

    def test_importing_final_does_not_import_pytorch(self):
        script = r"""
import sys
import typing

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

class Configuration:
    limit: torch.jit.Final[int] = 3

assert torch._jit_internal.Final is typing.Final
assert torch.jit.Final is typing.Final
assert typing.get_origin(Configuration.__annotations__["limit"]) is typing.Final
assert typing.get_args(Configuration.__annotations__["limit"]) == (int,)
assert not hasattr(torch.jit, "script")
assert not hasattr(torch, "compile")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
