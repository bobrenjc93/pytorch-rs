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
    def test_internal_and_public_markers_are_the_exact_typing_object(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(internal.Final, typing.Final)
        self.assertIs(jit.Final, typing.Final)
        self.assertIs(jit.Final, internal.Final)

        internal_namespace = {}
        jit_namespace = {}
        exec("from torch_rs._jit_internal import Final", internal_namespace)
        exec("from torch_rs.jit import Final", jit_namespace)
        self.assertIs(internal_namespace["Final"], typing.Final)
        self.assertIs(jit_namespace["Final"], typing.Final)

    def test_subscription_origin_and_arguments_follow_typing_final(self):
        marker = torch.jit.Final
        cases = (
            (int, (int,)),
            (list[str], (list[str],)),
            (None, (type(None),)),
            (Ellipsis, (Ellipsis,)),
        )

        for argument, expected_arguments in cases:
            with self.subTest(argument=argument):
                value = marker[argument]
                self.assertEqual(value, typing.Final[argument])
                self.assertIs(typing.get_origin(value), typing.Final)
                self.assertEqual(typing.get_args(value), expected_arguments)

        self.assertIs(marker[int], marker[int])
        self.assertEqual(repr(marker[int]), "typing.Final[int]")
        self.assertEqual(str(marker[list[str]]), "typing.Final[list[str]]")

    def test_instantiation_and_subscription_arity_errors(self):
        marker = torch.jit.Final
        instantiation_cases = (
            lambda: marker(),
            lambda: marker(int),
            lambda: marker(value=int),
            lambda: marker[int](),
            lambda: marker[int](1, value=2),
        )
        for case, call in enumerate(instantiation_cases):
            with self.subTest(kind="instantiation", case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception), "Cannot instantiate typing.Final"
                )
                self.assertEqual(
                    raised.exception.args,
                    ("Cannot instantiate typing.Final",),
                )

        arity_cases = (
            (
                lambda: marker[()],
                "typing.Final accepts only single type. Got ().",
            ),
            (
                lambda: marker[(int,)],
                "typing.Final accepts only single type. Got (<class 'int'>,).",
            ),
            (
                lambda: marker[int, str],
                "typing.Final accepts only single type. "
                "Got (<class 'int'>, <class 'str'>).",
            ),
        )
        for call, message in arity_cases:
            with self.subTest(kind="subscription", message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_metadata_copying_and_pickling_are_owned_by_typing(self):
        marker = torch.jit.Final

        self.assertIs(type(marker), type(typing.Final))
        self.assertEqual(repr(marker), "typing.Final")
        self.assertEqual(str(marker), "typing.Final")
        self.assertEqual(marker.__module__, "typing")
        self.assertEqual(marker.__name__, "Final")
        self.assertEqual(marker.__qualname__, "Final")
        self.assertIs(marker.__doc__, typing.Final.__doc__)
        self.assertEqual(str(inspect.signature(marker)), "(*args, **kwds)")
        self.assertIsNone(typing.get_origin(marker))
        self.assertEqual(typing.get_args(marker), ())

        for value in (marker, marker[int], marker[list[int]]):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"typing", payload)
                    self.assertNotIn(b"torch_rs", payload)
                    self.assertIs(pickle.loads(payload), value)

    def test_namespace_matches_the_non_wildcard_pytorch_export(self):
        jit = torch.jit
        self.assertIn("Final", vars(jit))
        self.assertNotIn("Final", jit.__all__)
        self.assertEqual(jit.__all__.count("Final"), 0)

        wildcard_namespace = {}
        exec("from torch_rs.jit import *", wildcard_namespace)
        self.assertNotIn("Final", wildcard_namespace)

        self.assertNotIn("Final", torch.__all__)
        self.assertFalse(hasattr(torch, "Final"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("Final", top_level_namespace)

    def test_marker_is_eager_only_and_compilation_remains_unsupported(self):
        class Configuration:
            answer: torch.jit.Final[int] = 42

        self.assertIs(
            typing.get_type_hints(Configuration)["answer"],
            typing.Final[int],
        )
        configuration = Configuration()
        configuration.answer = 7
        self.assertEqual(configuration.answer, 7)

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

    def test_importing_the_alias_does_not_import_pytorch(self):
        script = r"""
import pickle
import sys
import typing

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
from torch_rs import _jit_internal
from torch_rs.jit import Final

assert Final is typing.Final
assert _jit_internal.Final is typing.Final
assert typing.get_origin(Final[int]) is typing.Final
assert pickle.loads(pickle.dumps(Final[int])) is Final[int]
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
