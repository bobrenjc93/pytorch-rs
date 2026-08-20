import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest
import warnings

import torch_rs as torch


FUNCTION_DOC = """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb

                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x


        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """

DROP_WARNING = (
    "ignore(True) has been deprecated. TorchScript will now drop the function "
    "call on compilation. Use torch.jit.unused now. {}"
)
DROP_ON_EXPORT_WARNING = (
    "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop "
    "the function call on compilation. Use torch.jit.unused now. {}"
)


def _picklable_bare_ignore_function(value):
    return value


def _picklable_factory_ignore_function(value):
    return value


torch.jit.ignore(_picklable_bare_ignore_function)
torch.jit.ignore()(_picklable_factory_ignore_function)


class JitIgnoreTests(unittest.TestCase):
    def test_bare_decorator_marks_exact_callable_without_changing_eager_behavior(
        self,
    ):
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

        result = torch.jit.ignore(function)

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
        modifier = torch._jit_internal.FunctionModifiers.IGNORE
        self.assertIs(function._torchscript_modifier, modifier)
        self.assertEqual(
            modifier,
            "ignore (leave as a call to Python, cannot be torch.jit.save'd)",
        )

        class CallableTarget:
            def __call__(self, value):
                return value * 2

        target = CallableTarget()
        self.assertIs(torch.jit.ignore(target), target)
        self.assertIs(target._torchscript_modifier, modifier)
        self.assertEqual(target(6), 12)

    def test_factory_forms_mark_exact_functions_and_methods(self):
        modifier = torch._jit_internal.FunctionModifiers.IGNORE
        factories = (
            lambda: torch.jit.ignore(),
            lambda: torch.jit.ignore(False),
            lambda: torch.jit.ignore(drop=False),
            lambda: torch.jit.ignore(drop_on_export=False),
            lambda: torch.jit.ignore(unrecognized_keyword="ignored"),
        )
        for make_factory in factories:
            with self.subTest(make_factory=make_factory):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    decorator = make_factory()

                self.assertEqual(caught, [])

                def function(value):
                    return value + 1

                self.assertIs(decorator(function), function)
                self.assertIs(function._torchscript_modifier, modifier)
                self.assertEqual(function(4), 5)

        class Example:
            @torch.jit.ignore()
            def method(self, value):
                return value + 3

        raw_method = Example.__dict__["method"]
        self.assertIs(Example.method, raw_method)
        self.assertIs(raw_method._torchscript_modifier, modifier)
        self.assertEqual(Example().method(2), 5)

    def test_drop_forms_warn_and_use_unused_modifier(self):
        cases = (
            (lambda: torch.jit.ignore(True), DROP_WARNING),
            (lambda: torch.jit.ignore(drop=True), DROP_WARNING),
            (
                lambda: torch.jit.ignore(drop_on_export=True),
                DROP_ON_EXPORT_WARNING,
            ),
            (
                lambda: torch.jit.ignore(False, drop_on_export=True),
                DROP_ON_EXPORT_WARNING,
            ),
            (
                lambda: torch.jit.ignore(True, drop_on_export=True),
                DROP_ON_EXPORT_WARNING,
            ),
            (lambda: torch.jit.ignore(True, drop_on_export=False), DROP_WARNING),
        )
        modifier = torch._jit_internal.FunctionModifiers.UNUSED
        for make_factory, message in cases:
            with self.subTest(message=message, make_factory=make_factory):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    decorator = make_factory()

                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, FutureWarning)
                self.assertEqual(str(caught[0].message), message)
                self.assertEqual(caught[0].message.args, (message,))
                self.assertEqual(caught[0].filename, __file__)

                def function(value):
                    return value + 1

                self.assertIs(decorator(function), function)
                self.assertIs(function._torchscript_modifier, modifier)
                self.assertEqual(function(4), 5)

    def test_legacy_drop_on_export_uses_truthiness_and_ignores_other_keywords(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            decorator = torch.jit.ignore(
                drop_on_export="legacy truthy value", future_option=object()
            )

        self.assertEqual(len(caught), 1)
        self.assertEqual(str(caught[0].message), DROP_ON_EXPORT_WARNING)

        def function():
            return "eager result"

        self.assertIs(decorator(function), function)
        self.assertIs(
            function._torchscript_modifier,
            torch._jit_internal.FunctionModifiers.UNUSED,
        )
        self.assertEqual(function(), "eager result")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            default_decorator = torch.jit.ignore(drop_on_export=0, typo=True)
        self.assertEqual(caught, [])
        self.assertIs(default_decorator(function), function)
        self.assertIs(
            function._torchscript_modifier,
            torch._jit_internal.FunctionModifiers.IGNORE,
        )

    def test_decorators_overwrite_existing_modifiers(self):
        internal = torch._jit_internal

        def function():
            return "eager result"

        function._torchscript_modifier = object()
        self.assertIs(torch.jit.ignore(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.IGNORE)

        self.assertIs(torch.jit.export(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.EXPORT)

        self.assertIs(torch.jit.ignore()(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.IGNORE)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.assertIs(torch.jit.ignore(drop=True)(function), function)
        self.assertIs(function._torchscript_modifier, internal.FunctionModifiers.UNUSED)
        self.assertEqual(function(), "eager result")

    def test_signature_documentation_and_internal_ownership(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        function = jit.ignore

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(function, internal.ignore)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(drop=False, **kwargs)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "ignore")
        self.assertEqual(function.__qualname__, "ignore")
        self.assertEqual(function.__module__, "torch_rs._jit_internal")
        self.assertIs(inspect.getmodule(function), internal)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (False,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        decorator = function()
        self.assertIs(type(decorator), types.FunctionType)
        self.assertEqual(str(inspect.signature(decorator)), "(fn)")
        self.assertEqual(decorator.__annotations__, {})
        self.assertEqual(decorator.__name__, "decorator")
        self.assertEqual(decorator.__qualname__, "ignore.<locals>.decorator")
        self.assertEqual(decorator.__module__, "torch_rs._jit_internal")
        self.assertIs(inspect.getmodule(decorator), internal)
        self.assertIsNone(decorator.__doc__)
        self.assertIsNone(decorator.__defaults__)
        self.assertIsNone(decorator.__kwdefaults__)
        self.assertEqual(decorator.__dict__, {})

    def test_exports_copy_and_pickle_use_canonical_modules(self):
        jit = torch.jit
        function = jit.ignore
        internal = torch._jit_internal

        self.assertEqual(jit.__all__, ["annotate", "export", "ignore", "unused"])
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {
                "annotate",
                "export",
                "ignore",
                "is_scripting",
                "is_tracing",
                "unused",
            },
        )
        jit_namespace = {}
        exec("from torch_rs.jit import *", jit_namespace)
        self.assertEqual(
            {name for name in jit_namespace if not name.startswith("__")},
            {"annotate", "export", "ignore", "unused"},
        )
        self.assertIs(jit_namespace["ignore"], function)

        self.assertNotIn("jit", torch.__all__)
        self.assertNotIn("ignore", torch.__all__)
        self.assertNotIn("_jit_internal", torch.__all__)
        self.assertFalse(hasattr(torch, "ignore"))

        for value in (
            function,
            internal.FunctionModifiers,
            _picklable_bare_ignore_function,
            _picklable_factory_ignore_function,
        ):
            with self.subTest(value=value):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(value, protocol=protocol)), value
                    )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            payload = pickle.dumps(function, protocol=protocol)
            self.assertIn(b"torch_rs._jit_internal", payload)

        factory = function()
        self.assertIs(copy.copy(factory), factory)
        self.assertIs(copy.deepcopy(factory), factory)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(factory_protocol=protocol):
                with self.assertRaises(Exception) as raised:
                    pickle.dumps(factory, protocol=protocol)
                self.assertIn("ignore.<locals>.decorator", str(raised.exception))

    def test_invalid_calls_match_pytorch_2_13_errors(self):
        function = torch.jit.ignore
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
                lambda: function(False, True),
                TypeError,
                "ignore() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: function(False, drop=True),
                TypeError,
                "ignore() got multiple values for argument 'drop'",
            ),
            (
                lambda: function(None),
                RuntimeError,
                "Argument to @torch.jit.ignore must be a bool or a function but got None",
            ),
            (
                lambda: function(1),
                RuntimeError,
                "Argument to @torch.jit.ignore must be a bool or a function but got 1",
            ),
            (
                lambda: function("invalid"),
                RuntimeError,
                "Argument to @torch.jit.ignore must be a bool or a function but got invalid",
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
                lambda: function()(None),
                AttributeError,
                "'NoneType' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function()(property()),
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

        with self.assertRaisesRegex(
            RuntimeError,
            r"^Argument to @torch\.jit\.ignore must be a bool or a function but got "
            r"<property object at 0x[0-9a-f]+>$",
        ):
            function(property())

    def test_scripting_tracing_and_compilation_remain_unsupported(self):
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
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

        self.assertIs(torch.jit.export(function), function)
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

def bare(value):
    return value

def factory(value):
    return value + 1

assert torch.jit.ignore(bare) is bare
assert torch.jit.ignore()(factory) is factory
assert bare._torchscript_modifier == (
    "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
)
assert factory._torchscript_modifier == bare._torchscript_modifier
assert bare("value") == "value"
assert factory(2) == 3
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
