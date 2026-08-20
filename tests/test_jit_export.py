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
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """


def _picklable_export_function(value):
    return value


torch.jit.export(_picklable_export_function)


class JitExportTests(unittest.TestCase):
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

        result = torch.jit.export(function)

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
            internal.FunctionModifiers.EXPORT,
        )
        self.assertEqual(
            function._torchscript_modifier,
            "export (compile this function even if nothing calls it)",
        )

    def test_methods_and_callable_objects_keep_their_eager_behavior(self):
        class Example:
            @torch.jit.export
            def method(self, value):
                return value + 1

        raw_method = Example.__dict__["method"]
        self.assertIs(Example.method, raw_method)
        self.assertEqual(Example().method(4), 5)

        class CallableTarget:
            def __call__(self, value):
                return value * 2

        target = CallableTarget()
        self.assertIs(torch.jit.export(target), target)
        self.assertEqual(target(6), 12)

        modifier = torch._jit_internal.FunctionModifiers.EXPORT
        self.assertIs(raw_method._torchscript_modifier, modifier)
        self.assertIs(target._torchscript_modifier, modifier)

    def test_existing_modifiers_are_overwritten(self):
        internal = torch._jit_internal

        def function():
            return "eager result"

        previous_modifier = object()
        function._torchscript_modifier = previous_modifier
        self.assertIs(torch.jit.export(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.EXPORT)

        self.assertIs(torch.jit.unused(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.UNUSED)
        self.assertIs(torch.jit.export(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.EXPORT)
        self.assertEqual(function(), "eager result")

    def test_signature_annotations_documentation_and_internal_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        function = jit.export

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(function, internal.export)
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
        self.assertEqual(function.__name__, "export")
        self.assertEqual(function.__qualname__, "export")
        self.assertEqual(function.__module__, "torch_rs._jit_internal")
        self.assertIs(inspect.getmodule(function), internal)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        def keyword_target():
            return None

        self.assertIs(function(fn=keyword_target), keyword_target)
        self.assertIs(
            keyword_target._torchscript_modifier,
            internal.FunctionModifiers.EXPORT,
        )

    def test_exports_copy_and_pickle_use_the_canonical_internal_module(self):
        jit = torch.jit
        function = jit.export
        internal = torch._jit_internal

        self.assertEqual(jit.__all__, ["annotate", "export", "unused"])
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {"annotate", "export", "unused"},
        )
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            {"annotate", "export", "unused"},
        )
        self.assertIs(jit_namespace["export"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("export", torch.__all__)
        self.assertNotIn("_jit_internal", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("export", top_level_namespace)
        self.assertNotIn("_jit_internal", top_level_namespace)
        self.assertFalse(hasattr(torch, "export"))

        for value in (function, internal.FunctionModifiers, _picklable_export_function):
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

    def test_rejects_invalid_calls_with_pytorch_2_13_errors(self):
        function = torch.jit.export
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
                "export() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(lambda: None, lambda: None),
                TypeError,
                "export() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(function=lambda: None),
                TypeError,
                "export() got an unexpected keyword argument 'function'",
            ),
            (
                lambda: function(lambda: None, fn=lambda: None),
                TypeError,
                "export() got multiple values for argument 'fn'",
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
                "'property' object has no attribute "
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

        value = {"items": [1, 2]}
        self.assertIs(torch.jit.annotate(list[int], value), value)

        def function():
            return "unchanged"

        self.assertIs(torch.jit.unused(function), function)
        self.assertEqual(function(), "unchanged")

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

decorated = torch.jit.export(function)
assert decorated is function
assert decorated("value") == "value"
assert decorated._torchscript_modifier == (
    "export (compile this function even if nothing calls it)"
)

class Example:
    @torch.jit.export
    def method(self, value):
        return value + 1

assert Example().method(2) == 3
assert torch.jit.annotate(int, decorated) is decorated
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
