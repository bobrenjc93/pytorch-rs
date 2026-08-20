import copy
import importlib
import inspect
import pickle
import sys
import types
import unittest
import warnings

import torch_rs as torch


DROP_WARNING = (
    "ignore(True) has been deprecated. TorchScript will now drop the function "
    "call on compilation. Use torch.jit.unused now. {}"
)
DROP_ON_EXPORT_WARNING = (
    "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop "
    "the function call on compilation. Use torch.jit.unused now. {}"
)


def _picklable_ignored_function(value):
    return value


torch.jit.ignore(_picklable_ignored_function)


class JitIgnoreTests(unittest.TestCase):
    def test_bare_form_marks_the_exact_callable_without_changing_eager_behavior(self):
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
        self.assertIs(
            function._torchscript_modifier,
            torch._jit_internal.FunctionModifiers.IGNORE,
        )

        function._torchscript_modifier = object()
        self.assertIs(torch.jit.ignore(drop=function), function)
        self.assertIs(
            function._torchscript_modifier,
            torch._jit_internal.FunctionModifiers.IGNORE,
        )

    def test_bare_and_factory_decorator_syntax_work_on_methods_and_callables(self):
        class Example:
            @torch.jit.ignore
            def bare_method(self, value):
                return value + 1

            @torch.jit.ignore()
            def factory_method(self, value):
                return value * 2

        modifier = torch._jit_internal.FunctionModifiers.IGNORE
        self.assertIs(Example.__dict__["bare_method"]._torchscript_modifier, modifier)
        self.assertIs(
            Example.__dict__["factory_method"]._torchscript_modifier, modifier
        )
        self.assertEqual(Example().bare_method(3), 4)
        self.assertEqual(Example().factory_method(3), 6)

        class CallableTarget:
            def __call__(self, value):
                return value - 1

        target = CallableTarget()
        self.assertIs(torch.jit.ignore(target), target)
        self.assertIs(target._torchscript_modifier, modifier)
        self.assertEqual(target(5), 4)

    def test_no_drop_factories_mark_ignore_and_accept_legacy_extra_keywords(self):
        factories = (
            lambda: torch.jit.ignore(),
            lambda: torch.jit.ignore(False),
            lambda: torch.jit.ignore(drop=False),
            lambda: torch.jit.ignore(drop_on_export=False),
            lambda: torch.jit.ignore(unrecognized="silently ignored"),
            lambda: torch.jit.ignore(drop=False, first=1, second=2),
        )
        modifier = torch._jit_internal.FunctionModifiers.IGNORE

        for case, factory in enumerate(factories):
            with self.subTest(case=case):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    decorator = factory()

                def function(value):
                    return value

                function._torchscript_modifier = object()
                result = decorator(function)
                self.assertEqual(caught, [])
                self.assertIs(result, function)
                self.assertIs(function._torchscript_modifier, modifier)
                self.assertEqual(function("eager"), "eager")

    def test_drop_and_drop_on_export_warn_and_mark_unused(self):
        cases = (
            ((True,), {}, DROP_WARNING),
            ((), {"drop": True}, DROP_WARNING),
            ((True,), {"drop_on_export": False}, DROP_WARNING),
            ((), {"drop_on_export": True}, DROP_ON_EXPORT_WARNING),
            ((False,), {"drop_on_export": True}, DROP_ON_EXPORT_WARNING),
            ((), {"drop_on_export": "truthy"}, DROP_ON_EXPORT_WARNING),
            (
                (),
                {"drop_on_export": True, "unrecognized": object()},
                DROP_ON_EXPORT_WARNING,
            ),
        )
        modifier = torch._jit_internal.FunctionModifiers.UNUSED

        for args, kwargs, message in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    decorator = torch.jit.ignore(*args, **kwargs)

                self.assertEqual(len(caught), 1)
                warning = caught[0]
                self.assertIs(warning.category, FutureWarning)
                self.assertIs(type(warning.message), FutureWarning)
                self.assertEqual(str(warning.message), message)
                self.assertEqual(warning.message.args, (message,))
                self.assertEqual(warning.filename, __file__)

                def function(value):
                    return value

                with warnings.catch_warnings(record=True) as apply_warnings:
                    warnings.simplefilter("always")
                    result = decorator(function)
                self.assertEqual(apply_warnings, [])
                self.assertIs(result, function)
                self.assertIs(function._torchscript_modifier, modifier)
                self.assertEqual(function("eager"), "eager")

    def test_existing_modifier_decorators_continue_to_interoperate(self):
        modifiers = torch._jit_internal.FunctionModifiers

        def function(value):
            return value

        self.assertIs(torch.jit.export(function), function)
        self.assertIs(function._torchscript_modifier, modifiers.EXPORT)
        self.assertIs(torch.jit.ignore(function), function)
        self.assertIs(function._torchscript_modifier, modifiers.IGNORE)
        self.assertIs(torch.jit.unused(function), function)
        self.assertIs(function._torchscript_modifier, modifiers.UNUSED)
        self.assertIs(torch.jit.ignore()(function), function)
        self.assertIs(function._torchscript_modifier, modifiers.IGNORE)
        self.assertEqual(function("eager"), "eager")

        value = object()
        self.assertIs(torch.jit.annotate(object, value), value)

    def test_function_and_factory_metadata_match_the_public_contract(self):
        jit = importlib.import_module("torch_rs.jit")
        internal = importlib.import_module("torch_rs._jit_internal")
        function = jit.ignore

        self.assertIs(torch.jit, jit)
        self.assertIs(torch._jit_internal, internal)
        self.assertIs(function, internal.ignore)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(drop=False, **kwargs)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__name__, "ignore")
        self.assertEqual(function.__qualname__, "ignore")
        self.assertEqual(function.__module__, "torch_rs._jit_internal")
        self.assertIs(inspect.getmodule(function), internal)
        self.assertIn("@torch.jit.ignore(drop=True)", function.__doc__)
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
        self.assertIsNone(decorator.__doc__)
        self.assertIsNone(decorator.__defaults__)
        self.assertIsNone(decorator.__kwdefaults__)
        self.assertEqual(decorator.__dict__, {})
        self.assertFalse(hasattr(decorator, "__text_signature__"))
        self.assertEqual(len(decorator.__closure__), 1)
        self.assertIs(decorator.__closure__[0].cell_contents, False)

    def test_exports_copying_and_pickling_use_the_canonical_modules(self):
        jit = torch.jit
        function = jit.ignore
        internal = torch._jit_internal

        self.assertEqual(jit.__all__, ["annotate", "export", "ignore", "unused"])
        self.assertEqual(
            {name for name in vars(jit) if not name.startswith("_")},
            {"annotate", "export", "ignore", "unused"},
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
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("jit", top_level_namespace)
        self.assertNotIn("ignore", top_level_namespace)
        self.assertNotIn("_jit_internal", top_level_namespace)
        self.assertFalse(hasattr(torch, "ignore"))

        for value in (function, internal.FunctionModifiers, _picklable_ignored_function):
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

        decorator = function()
        self.assertIs(copy.copy(decorator), decorator)
        self.assertIs(copy.deepcopy(decorator), decorator)
        pickle_error = (
            pickle.PicklingError
            if sys.version_info >= (3, 14)
            else AttributeError
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(decorator_protocol=protocol):
                with self.assertRaises(pickle_error) as raised:
                    pickle.dumps(decorator, protocol=protocol)
                self.assertIn("ignore.<locals>.decorator", str(raised.exception))

    def test_invalid_calls_match_pytorch_2_13_errors(self):
        function = torch.jit.ignore
        invalid_drop_cases = (
            (None, "None"),
            (1, "1"),
            (-1, "-1"),
            ("invalid", "invalid"),
        )
        for value, representation in invalid_drop_cases:
            message = (
                "Argument to @torch.jit.ignore must be a bool or a function but "
                f"got {representation}"
            )
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError) as raised:
                    function(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        call_errors = (
            (
                lambda: function(False, True),
                "ignore() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: function(False, drop=True),
                "ignore() got multiple values for argument 'drop'",
            ),
        )
        for call, message in call_errors:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        immutable_attribute_suffix = (
            " and no __dict__ for setting new attributes"
            if sys.version_info >= (3, 14)
            else ""
        )

        class Example:
            def method(self):
                return None

        invalid_targets = (
            (
                lambda: function(len),
                "'builtin_function_or_method' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function(Example().method),
                "'method' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function()(None),
                "'NoneType' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function()(1),
                "'int' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
            (
                lambda: function()(property()),
                "'property' object has no attribute "
                f"'_torchscript_modifier'{immutable_attribute_suffix}",
            ),
        )
        for call, message in invalid_targets:
            with self.subTest(message=message):
                with self.assertRaises(AttributeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_scripting_and_tracing_remain_unsupported(self):
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
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


if __name__ == "__main__":
    unittest.main()
