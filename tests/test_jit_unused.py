import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """


def picklable_target(value):
    return value


class JitUnusedTests(unittest.TestCase):
    def test_function_is_marked_and_returned_without_wrapping(self):
        def target(value, *, offset=1):
            """target documentation"""
            return value + offset

        target.custom_attribute = object()
        original_dict = dict(target.__dict__)

        result = torch.jit.unused(target)

        self.assertIs(result, target)
        self.assertEqual(result(4, offset=3), 7)
        self.assertEqual(result.__name__, "target")
        self.assertEqual(result.__qualname__, target.__qualname__)
        self.assertEqual(result.__doc__, "target documentation")
        self.assertIs(result.custom_attribute, original_dict["custom_attribute"])
        self.assertEqual(
            result._torchscript_modifier,
            "unused (ignored and replaced with raising of an exception)",
        )

        previous_marker = object()
        target._torchscript_modifier = previous_marker
        self.assertIs(torch.jit.unused(target), target)
        self.assertIsNot(target._torchscript_modifier, previous_marker)
        self.assertEqual(
            target._torchscript_modifier,
            "unused (ignored and replaced with raising of an exception)",
        )

    def test_property_marks_getter_and_setter_but_not_deleter(self):
        def getter(instance):
            return instance._value

        def setter(instance, value):
            instance._value = value

        def deleter(instance):
            del instance._value

        prop = property(getter, setter, deleter, "property documentation")
        result = torch.jit.unused(prop)

        self.assertIs(result, prop)
        self.assertIs(result.fget, getter)
        self.assertIs(result.fset, setter)
        self.assertIs(result.fdel, deleter)
        self.assertEqual(result.__doc__, "property documentation")
        self.assertEqual(
            getter._torchscript_modifier,
            "unused (ignored and replaced with raising of an exception)",
        )
        self.assertEqual(
            setter._torchscript_modifier,
            "unused (ignored and replaced with raising of an exception)",
        )
        self.assertFalse(hasattr(deleter, "_torchscript_modifier"))

        def read_only_getter(instance):
            return 1

        read_only = property(read_only_getter)
        self.assertIs(torch.jit.unused(read_only), read_only)
        self.assertEqual(
            read_only_getter._torchscript_modifier,
            "unused (ignored and replaced with raising of an exception)",
        )

    def test_signature_annotations_documentation_and_internal_ownership(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        function = torch.jit.unused

        self.assertIs(function, internal.unused)
        self.assertIs(sys.modules["torch_rs._jit_internal"], internal)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(fn: Callable[~_P, ~_R]) -> Callable[~_P, ~_R]",
        )
        expected_annotations = {
            "fn": typing.Callable[internal._P, internal._R],
            "return": typing.Callable[internal._P, internal._R],
        }
        self.assertEqual(function.__annotations__, expected_annotations)
        self.assertEqual(typing.get_type_hints(function), expected_annotations)
        self.assertEqual(function.__name__, "unused")
        self.assertEqual(function.__qualname__, "unused")
        self.assertEqual(function.__module__, "torch_rs._jit_internal")
        self.assertIs(inspect.getmodule(function), internal)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            internal.FunctionModifiers.UNUSED,
            "unused (ignored and replaced with raising of an exception)",
        )
        self.assertEqual(
            internal.FunctionModifiers.__module__, "torch_rs._jit_internal"
        )

        def keyword_target():
            return None

        self.assertIs(function(fn=keyword_target), keyword_target)
        self.assertIs(
            keyword_target._torchscript_modifier,
            internal.FunctionModifiers.UNUSED,
        )

    def test_exports_copy_and_pickle_use_the_canonical_internal_module(self):
        jit = torch.jit
        function = jit.unused

        self.assertEqual(jit.__all__, ["annotate", "unused"])
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {"annotate", "unused"},
        )
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            {"annotate", "unused"},
        )
        self.assertIs(jit_namespace["unused"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("unused", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("unused", top_level_namespace)
        self.assertFalse(hasattr(torch, "unused"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs._jit_internal", payload)
                self.assertIs(pickle.loads(payload), function)

        decorated = function(picklable_target)
        self.assertIs(copy.copy(decorated), decorated)
        self.assertIs(copy.deepcopy(decorated), decorated)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(decorated_protocol=protocol):
                restored = pickle.loads(pickle.dumps(decorated, protocol=protocol))
                self.assertIs(restored, decorated)
                self.assertEqual(
                    restored._torchscript_modifier,
                    "unused (ignored and replaced with raising of an exception)",
                )

    def test_rejects_invalid_calls_and_immutable_targets_with_exact_errors(self):
        function = torch.jit.unused
        immutable_attribute_suffix = (
            " and no __dict__ for setting new attributes"
            if sys.version_info >= (3, 14)
            else ""
        )

        def target():
            return None

        call_cases = (
            (
                lambda: function(),
                TypeError,
                "unused() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(target, target),
                TypeError,
                "unused() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(function=target),
                TypeError,
                "unused() got an unexpected keyword argument 'function'",
            ),
            (
                lambda: function(target, fn=target),
                TypeError,
                "unused() got multiple values for argument 'fn'",
            ),
            (
                lambda: function(None),
                AttributeError,
                "'NoneType' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(1),
                AttributeError,
                "'int' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(len),
                AttributeError,
                "'builtin_function_or_method' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(property()),
                AttributeError,
                "'NoneType' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
        )
        for call, exception_type, message in call_cases:
            with self.subTest(message=message):
                with self.assertRaises(exception_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_ignore_scripting_tracing_and_compilation_remain_unsupported(self):
        unsupported_jit_names = (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "ignore",
            "is_scripting",
            "is_tracing",
            "script",
            "script_if_tracing",
            "script_method",
            "trace",
            "trace_module",
        )
        for name in unsupported_jit_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))

        value = {"items": [1, 2]}
        self.assertIs(torch.jit.annotate(list[int], value), value)
        self.assertEqual(
            str(inspect.signature(torch.jit.annotate)), "(the_type, the_value)"
        )

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

@torch.jit.unused
def target(value):
    return value

assert target(3) == 3
assert target._torchscript_modifier == (
    "unused (ignored and replaced with raising of an exception)"
)
assert not hasattr(torch.jit, "ignore")
assert not hasattr(torch.jit, "script")
assert not hasattr(torch.jit, "trace")
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
