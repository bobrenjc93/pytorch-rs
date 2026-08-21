import copy
import importlib
import pickle
import subprocess
import sys
import typing
import unittest

import torch_rs as torch


class JitFinalTests(unittest.TestCase):
    def assert_type_error(self, call, message):
        with self.assertRaises(TypeError) as raised:
            call()
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

    def test_internal_and_jit_names_are_the_exact_typing_object(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        jit = importlib.import_module("torch_rs.jit")
        marker = typing.Final

        self.assertIs(torch._jit_internal, internal)
        self.assertIs(torch.jit, jit)
        self.assertIs(internal.Final, marker)
        self.assertIs(jit.Final, marker)
        self.assertIs(vars(internal)["Final"], marker)
        self.assertIs(vars(jit)["Final"], marker)
        self.assertIs(type(jit.Final), type(marker))

        self.assertEqual(repr(jit.Final), "typing.Final")
        self.assertEqual(str(jit.Final), "typing.Final")
        self.assertEqual(jit.Final.__name__, "Final")
        self.assertEqual(jit.Final.__qualname__, "Final")
        self.assertEqual(jit.Final.__module__, "typing")
        self.assertEqual(jit.Final.__doc__, marker.__doc__)

    def test_subscription_origin_arguments_and_eager_annotations(self):
        marker = torch.jit.Final
        cases = (
            (int, (int,)),
            (list[str], (list[str],)),
            (None, (type(None),)),
            (int | str, (int | str,)),
            (typing.Any, (typing.Any,)),
        )

        for argument, expected_args in cases:
            with self.subTest(argument=argument):
                alias = marker[argument]
                self.assertEqual(alias, typing.Final[argument])
                self.assertIs(typing.get_origin(alias), typing.Final)
                self.assertEqual(typing.get_args(alias), expected_args)
                self.assertIs(alias.__origin__, typing.Final)
                self.assertEqual(alias.__args__, expected_args)
                self.assertEqual(alias.__parameters__, ())
                self.assertEqual(alias.__name__, "Final")
                self.assertEqual(alias.__qualname__, "Final")
                self.assertEqual(alias.__module__, "typing")

        initial = object()

        class EagerConfiguration:
            value: marker[object] = initial

        self.assertEqual(
            typing.get_type_hints(EagerConfiguration)["value"],
            typing.Final[object],
        )
        configuration = EagerConfiguration()
        replacement = object()
        configuration.value = replacement
        self.assertIs(configuration.value, replacement)

    def test_instantiation_and_subscription_errors_match_typing_final(self):
        marker = torch.jit.Final
        for call in (
            lambda: marker(),
            lambda: marker(int),
            lambda: marker(value=int),
            lambda: marker[int](),
            lambda: marker[int](1),
            lambda: marker[int](value=1),
        ):
            with self.subTest(call=call):
                self.assert_type_error(call, "Cannot instantiate typing.Final")

        cases = (
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
                self.assert_type_error(call, message)

    def test_copying_and_pickling_preserve_canonical_typing_objects(self):
        cases = (
            (torch.jit.Final, typing.Final),
            (torch.jit.Final[int], typing.Final[int]),
            (torch.jit.Final[list[str]], typing.Final[list[str]]),
            (torch.jit.Final[None], typing.Final[None]),
        )
        for value, expected in cases:
            with self.subTest(value=value):
                copied = copy.copy(value)
                expected_copy = copy.copy(expected)
                self.assertEqual(copied, expected_copy)
                self.assertEqual(copied is value, expected_copy is expected)

                deepcopied = copy.deepcopy(value)
                expected_deepcopy = copy.deepcopy(expected)
                self.assertEqual(deepcopied, expected_deepcopy)
                self.assertEqual(
                    deepcopied is value,
                    expected_deepcopy is expected,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(protocol=protocol):
                        payload = pickle.dumps(value, protocol=protocol)
                        expected_payload = pickle.dumps(expected, protocol=protocol)
                        restored = pickle.loads(payload)
                        expected_restored = pickle.loads(expected_payload)
                        self.assertIn(b"typing", payload)
                        self.assertNotIn(b"torch_rs", payload)
                        self.assertEqual(payload, expected_payload)
                        self.assertEqual(restored, expected_restored)
                        self.assertEqual(
                            restored is value,
                            expected_restored is expected,
                        )

    def test_exports_and_unsupported_compilation_boundary(self):
        jit = torch.jit
        self.assertEqual(
            jit.__all__,
            [
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "script_if_tracing",
                "unused",
            ],
        )
        self.assertNotIn("Final", jit.__all__)
        self.assertFalse(hasattr(torch._jit_internal, "__all__"))
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "Attribute",
                "Final",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "is_scripting",
                "is_tracing",
                "script_if_tracing",
                "unused",
            },
        )

        explicit = {}
        exec("from torch_rs.jit import Final", explicit)
        self.assertIs(explicit["Final"], typing.Final)

        wildcard = {}
        exec("from torch_rs.jit import *", wildcard)
        self.assertNotIn("Final", wildcard)

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
                self.assertFalse(hasattr(jit, name))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertIs(jit.is_scripting(), False)

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
from torch_rs._jit_internal import Final as InternalFinal
from torch_rs.jit import Final as PublicFinal

assert InternalFinal is typing.Final
assert PublicFinal is typing.Final
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
