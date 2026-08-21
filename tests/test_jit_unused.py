import collections.abc
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


def _picklable_unused_function(value):
    return value


torch.jit.unused(_picklable_unused_function)


class JitUnusedTests(unittest.TestCase):
    def test_function_is_marked_in_place_without_changing_eager_behavior(self):
        sentinel = object()

        def function(value, *, option=sentinel):
            """function documentation"""
            return value, option

        function.custom_attribute = sentinel
        before = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__.copy(),
            function.__defaults__,
            function.__kwdefaults__.copy(),
        )

        result = torch.jit.unused(function)

        self.assertIs(result, function)
        self.assertEqual(
            (
                function.__name__,
                function.__qualname__,
                function.__doc__,
                function.__annotations__,
                function.__defaults__,
                function.__kwdefaults__,
            ),
            before,
        )
        self.assertIs(function.custom_attribute, sentinel)
        self.assertEqual(function("value"), ("value", sentinel))

        internal = importlib.import_module("torch_rs._jit_internal")
        self.assertIs(
            function._torchscript_modifier,
            internal.FunctionModifiers.UNUSED,
        )
        self.assertEqual(
            function._torchscript_modifier,
            "unused (ignored and replaced with raising of an exception)",
        )

        previous_modifier = object()
        function._torchscript_modifier = previous_modifier
        self.assertIs(torch.jit.unused(function), function)
        self.assertIs(
            function._torchscript_modifier,
            internal.FunctionModifiers.UNUSED,
        )

    def test_property_marks_getter_and_setter_and_returns_exact_property(self):
        def getter(instance):
            return instance._value

        def setter(instance, value):
            instance._value = value

        def deleter(instance):
            del instance._value

        deleter._torchscript_modifier = "leave unchanged"
        prop = property(getter, setter, deleter, "property documentation")
        before = (prop.fget, prop.fset, prop.fdel, prop.__doc__)

        result = torch.jit.unused(prop)

        self.assertIs(result, prop)
        self.assertEqual((prop.fget, prop.fset, prop.fdel, prop.__doc__), before)
        self.assertFalse(hasattr(prop, "_torchscript_modifier"))

        modifier = importlib.import_module(
            "torch_rs._jit_internal"
        ).FunctionModifiers.UNUSED
        self.assertIs(getter._torchscript_modifier, modifier)
        self.assertIs(setter._torchscript_modifier, modifier)
        self.assertEqual(deleter._torchscript_modifier, "leave unchanged")

        class Holder:
            value = prop

            def __init__(self):
                self._value = 3

        holder = Holder()
        self.assertEqual(holder.value, 3)
        holder.value = 7
        self.assertEqual(holder.value, 7)
        del holder.value
        self.assertFalse(hasattr(holder, "_value"))

        def read_only_getter(instance):
            return 11

        read_only = property(read_only_getter)
        self.assertIs(torch.jit.unused(read_only), read_only)
        self.assertIs(read_only_getter._torchscript_modifier, modifier)

    def test_signature_annotations_documentation_and_internal_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        function = jit.unused

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(function, internal.unused)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(fn: Callable[~_P, ~_R]) -> Callable[~_P, ~_R]",
        )
        self.assertEqual(set(function.__annotations__), {"fn", "return"})
        self.assertIs(
            typing.get_origin(function.__annotations__["fn"]),
            collections.abc.Callable,
        )
        self.assertEqual(
            function.__annotations__["fn"],
            function.__annotations__["return"],
        )
        parameters, result = typing.get_args(function.__annotations__["fn"])
        self.assertIs(parameters, internal._P)
        self.assertIs(result, internal._R)
        self.assertEqual(internal._P.__name__, "_P")
        self.assertEqual(internal._P.__module__, "torch_rs._jit_internal")
        self.assertEqual(internal._R.__name__, "_R")
        self.assertEqual(internal._R.__module__, "torch_rs._jit_internal")
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
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

        modifiers = internal.FunctionModifiers
        self.assertEqual(modifiers.__module__, "torch_rs._jit_internal")
        self.assertEqual(modifiers.__qualname__, "FunctionModifiers")
        self.assertEqual(modifiers.__annotations__, {})
        self.assertEqual(
            modifiers.UNUSED,
            "unused (ignored and replaced with raising of an exception)",
        )

        def keyword_target():
            return None

        self.assertIs(function(fn=keyword_target), keyword_target)
        self.assertIs(
            keyword_target._torchscript_modifier,
            modifiers.UNUSED,
        )

    def test_exports_copy_and_pickle_use_the_canonical_internal_module(self):
        jit = torch.jit
        function = jit.unused
        internal = torch._jit_internal

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
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            {
                "Attribute",
                "annotate",
                "export",
                "ignore",
                "isinstance",
                "script_if_tracing",
                "unused",
            },
        )
        self.assertIs(jit_namespace["unused"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("unused", torch.__all__)
        self.assertNotIn("_jit_internal", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("unused", top_level_namespace)
        self.assertNotIn("_jit_internal", top_level_namespace)
        self.assertFalse(hasattr(torch, "unused"))

        for value in (function, internal.FunctionModifiers, _picklable_unused_function):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIs(pickle.loads(payload), value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs._jit_internal", payload)

        def getter(instance):
            return instance

        prop = torch.jit.unused(property(getter))
        self.assertIs(copy.copy(prop), prop)
        self.assertIs(copy.deepcopy(prop), prop)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(property_protocol=protocol):
                with self.assertRaisesRegex(
                    TypeError, "^cannot pickle 'property' object$"
                ):
                    pickle.dumps(prop, protocol=protocol)

    def test_rejects_invalid_calls_with_pytorch_2_13_errors(self):
        function = torch.jit.unused
        immutable_attribute_suffix = (
            " and no __dict__ for setting new attributes"
            if sys.version_info >= (3, 14)
            else ""
        )

        class Example:
            def method(self):
                return None

        cases = (
            (
                lambda: function(),
                TypeError,
                "unused() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(lambda: None, lambda: None),
                TypeError,
                "unused() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(function=lambda: None),
                TypeError,
                "unused() got an unexpected keyword argument 'function'",
            ),
            (
                lambda: function(lambda: None, fn=lambda: None),
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
                lambda: function(Example().method),
                AttributeError,
                "'method' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(property()),
                AttributeError,
                "'NoneType' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
        )
        for call, exception_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(exception_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
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

        value = {"items": [1, 2]}
        self.assertIs(torch.jit.annotate(list[int], value), value)

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

def function(value):
    return value

decorated = torch.jit.unused(function)
assert decorated is function
assert decorated("value") == "value"
assert decorated._torchscript_modifier == (
    "unused (ignored and replaced with raising of an exception)"
)

class Example:
    @torch.jit.unused
    @property
    def value(self):
        return 3

assert Example().value == 3
assert hasattr(torch.jit, "ignore")
assert torch.jit.annotate(int, decorated) is decorated
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
