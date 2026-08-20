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


UNUSED_MARKER = "unused (ignored and replaced with raising of an exception)"

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


@torch.jit.unused
def picklable_decorated(value):
    return value


class JitUnusedTests(unittest.TestCase):
    def test_function_is_marked_in_place_and_remains_eager(self):
        calls = []

        def function(value, *, scale=1):
            calls.append((value, scale))
            return value * scale

        function.custom_attribute = "preserved"
        function._torchscript_modifier = object()
        result = torch.jit.unused(function)

        self.assertIs(result, function)
        self.assertEqual(
            function.__dict__,
            {
                "custom_attribute": "preserved",
                "_torchscript_modifier": UNUSED_MARKER,
            },
        )
        self.assertEqual(result(4, scale=3), 12)
        self.assertEqual(calls, [(4, 3)])

        keyword_function = lambda: "keyword call"
        self.assertIs(torch.jit.unused(fn=keyword_function), keyword_function)
        self.assertEqual(keyword_function(), "keyword call")
        self.assertEqual(keyword_function._torchscript_modifier, UNUSED_MARKER)

    def test_attr_settable_callable_objects_follow_function_semantics(self):
        class CallableObject:
            def __init__(self):
                self.calls = []

            def __call__(self, value):
                self.calls.append(value)
                return value + 1

        callable_object = CallableObject()
        result = torch.jit.unused(callable_object)

        self.assertIs(result, callable_object)
        self.assertEqual(callable_object._torchscript_modifier, UNUSED_MARKER)
        self.assertEqual(result(5), 6)
        self.assertEqual(callable_object.calls, [5])

    def test_property_getter_and_setter_are_marked_but_deleter_is_not(self):
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
        self.assertEqual(getter._torchscript_modifier, UNUSED_MARKER)
        self.assertEqual(setter._torchscript_modifier, UNUSED_MARKER)
        self.assertFalse(hasattr(deleter, "_torchscript_modifier"))
        self.assertFalse(hasattr(result, "_torchscript_modifier"))

        class Owner:
            value = result

            def __init__(self):
                self._value = 2

        owner = Owner()
        self.assertEqual(owner.value, 2)
        owner.value = 7
        self.assertEqual(owner.value, 7)
        del owner.value
        self.assertFalse(hasattr(owner, "_value"))

    def test_getter_only_property_and_decorator_syntax(self):
        class Owner:
            def __init__(self, value):
                self._value = value

            @torch.jit.unused
            @property
            def value(self):
                return self._value

        descriptor = inspect.getattr_static(Owner, "value")
        self.assertIs(type(descriptor), property)
        self.assertIsNone(descriptor.fset)
        self.assertIsNone(descriptor.fdel)
        self.assertEqual(descriptor.fget._torchscript_modifier, UNUSED_MARKER)
        self.assertEqual(Owner(11).value, 11)

    def test_marker_and_callable_are_owned_by_package_jit_internal(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        function = torch.jit.unused

        self.assertIs(torch._jit_internal, internal)
        self.assertIs(function, internal.unused)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__module__, "torch_rs._jit_internal")
        self.assertEqual(function.__name__, "unused")
        self.assertEqual(function.__qualname__, "unused")
        self.assertIs(inspect.getmodule(function), internal)
        self.assertEqual(
            internal.FunctionModifiers.__module__, "torch_rs._jit_internal"
        )
        self.assertEqual(internal.FunctionModifiers.UNUSED, UNUSED_MARKER)
        self.assertIs(
            picklable_decorated._torchscript_modifier,
            internal.FunctionModifiers.UNUSED,
        )

        self.assertEqual(
            {
                name: value
                for name, value in vars(internal.FunctionModifiers).items()
                if name.isupper()
            },
            {
                "UNUSED": UNUSED_MARKER,
                "IGNORE": (
                    "ignore (leave as a call to Python, cannot be "
                    "torch.jit.save'd)"
                ),
                "EXPORT": "export (compile this function even if nothing calls it)",
                "DEFAULT": (
                    "default (compile if called from an exported function / forward)"
                ),
                "COPY_TO_SCRIPT_WRAPPER": (
                    "if this method is not scripted, copy the python method onto the "
                    "scripted model"
                ),
                "_DROP": (
                    "_drop (function is fully ignored, declaration can be "
                    "unscriptable)"
                ),
            },
        )

    def test_signature_annotations_and_documentation_match_pytorch_2_13(self):
        internal = importlib.import_module("torch_rs._jit_internal")
        function = torch.jit.unused

        self.assertEqual(
            str(inspect.signature(function)),
            "(fn: Callable[~_P, ~_R]) -> Callable[~_P, ~_R]",
        )
        self.assertEqual(
            repr(function.__annotations__),
            "{'fn': typing.Callable[~_P, ~_R], "
            "'return': typing.Callable[~_P, ~_R]}",
        )
        self.assertIs(
            function.__annotations__["fn"], function.__annotations__["return"]
        )
        parameter_spec, return_type = typing.get_args(
            function.__annotations__["fn"]
        )
        self.assertIs(parameter_spec, internal._P)
        self.assertIs(return_type, internal._R)
        self.assertEqual(parameter_spec.__name__, "_P")
        self.assertEqual(parameter_spec.__module__, "torch_rs._jit_internal")
        self.assertEqual(return_type.__name__, "_R")
        self.assertEqual(return_type.__module__, "torch_rs._jit_internal")
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(torch.jit.__doc__)

    def test_exports_copy_and_pickle_use_canonical_owners(self):
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

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(copy.copy(picklable_decorated), picklable_decorated)
        self.assertIs(copy.deepcopy(picklable_decorated), picklable_decorated)

        def getter(instance):
            return instance

        prop = torch.jit.unused(property(getter))
        self.assertIs(copy.copy(prop), prop)
        self.assertIs(copy.deepcopy(prop), prop)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs._jit_internal", payload)
                self.assertIs(pickle.loads(payload), function)
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(picklable_decorated, protocol=protocol)
                    ),
                    picklable_decorated,
                )
                self.assertEqual(
                    pickle.loads(
                        pickle.dumps(UNUSED_MARKER, protocol=protocol)
                    ),
                    UNUSED_MARKER,
                )

    def test_invalid_calls_and_targets_raise_pytorch_2_13_errors(self):
        function = torch.jit.unused
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
                lambda: function(other=1),
                TypeError,
                "unused() got an unexpected keyword argument 'other'",
            ),
            (
                lambda: function(fn=lambda: None, other=1),
                TypeError,
                "unused() got an unexpected keyword argument 'other'",
            ),
            (
                lambda: function(lambda: None, fn=lambda: None),
                TypeError,
                "unused() got multiple values for argument 'fn'",
            ),
            (
                lambda: function(1),
                AttributeError,
                "'int' object has no attribute '_torchscript_modifier'",
            ),
            (
                lambda: function(object()),
                AttributeError,
                "'object' object has no attribute '_torchscript_modifier'",
            ),
            (
                lambda: function(len),
                AttributeError,
                "'builtin_function_or_method' object has no attribute "
                "'_torchscript_modifier'",
            ),
            (
                lambda: function(property()),
                AttributeError,
                "'NoneType' object has no attribute '_torchscript_modifier'",
            ),
        )
        for call, exception_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(exception_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_annotate_remains_unchanged_and_compilers_remain_unsupported(self):
        value = {"items": [1, 2]}
        self.assertIs(torch.jit.annotate(list[int], value), value)
        self.assertEqual(
            str(inspect.signature(torch.jit.annotate)), "(the_type, the_value)"
        )
        self.assertEqual(torch.jit.annotate.__dict__, {})

        for name in (
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
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))

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
import torch_rs._jit_internal as internal

@torch.jit.unused
def function(value):
    return value + 1

def getter(instance):
    return instance._value

def setter(instance, value):
    instance._value = value

prop = property(getter, setter)
assert torch.jit.unused(prop) is prop
assert function(2) == 3
assert function._torchscript_modifier is internal.FunctionModifiers.UNUSED
assert getter._torchscript_modifier is internal.FunctionModifiers.UNUSED
assert setter._torchscript_modifier is internal.FunctionModifiers.UNUSED
assert torch.jit.unused is internal.unused
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
