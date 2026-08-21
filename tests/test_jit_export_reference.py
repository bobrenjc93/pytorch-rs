import collections.abc
import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def _actual_picklable_export_function(value):
    return value


torch.jit.export(_actual_picklable_export_function)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitExportReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.export differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def annotation_shape(self, function):
        shape = {}
        for name, annotation in function.__annotations__.items():
            parameters, result = typing.get_args(annotation)
            shape[name] = (
                typing.get_origin(annotation) is collections.abc.Callable,
                type(parameters).__name__,
                parameters.__name__,
                parameters.__module__.replace("torch_rs", "torch"),
                type(result).__name__,
                result.__name__,
                result.__module__.replace("torch_rs", "torch"),
            )
        return shape

    def function_outcome(self, module):
        sentinel = object()

        def function(value, *, option=sentinel):
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
        result = module.jit.export(function)
        modifier = module._jit_internal.FunctionModifiers.EXPORT
        after = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        return (
            result is function,
            function("value") == ("value", sentinel),
            function.custom_attribute is sentinel,
            before == after,
            function._torchscript_modifier,
            function._torchscript_modifier is modifier,
            copy.copy(function) is function,
            copy.deepcopy(function) is function,
        )

    def method_and_callable_outcome(self, module):
        class Example:
            @module.jit.export
            def method(self, value):
                return value + 1

        class CallableTarget:
            def __call__(self, value):
                return value * 2

        target = CallableTarget()
        result = module.jit.export(target)
        modifier = module._jit_internal.FunctionModifiers.EXPORT
        raw_method = Example.__dict__["method"]
        return (
            Example.method is raw_method,
            Example().method(4),
            raw_method._torchscript_modifier,
            raw_method._torchscript_modifier is modifier,
            result is target,
            target(6),
            target._torchscript_modifier,
            target._torchscript_modifier is modifier,
        )

    def overwrite_outcome(self, module):
        def function():
            return "eager result"

        function._torchscript_modifier = object()
        first = module.jit.export(function)
        export_modifier = module._jit_internal.FunctionModifiers.EXPORT
        first_state = (
            first is function,
            function._torchscript_modifier,
            function._torchscript_modifier is export_modifier,
        )
        module.jit.unused(function)
        unused_modifier = module._jit_internal.FunctionModifiers.UNUSED
        unused_state = (
            function._torchscript_modifier,
            function._torchscript_modifier is unused_modifier,
        )
        second = module.jit.export(function)
        return (
            first_state,
            unused_state,
            second is function,
            function._torchscript_modifier,
            function._torchscript_modifier is export_modifier,
            function(),
        )

    def test_function_method_callable_and_overwrite_semantics_match(self):
        self.assertEqual(
            self.function_outcome(torch),
            self.function_outcome(reference_torch),
        )
        self.assertEqual(
            self.method_and_callable_outcome(torch),
            self.method_and_callable_outcome(reference_torch),
        )
        self.assertEqual(
            self.overwrite_outcome(torch),
            self.overwrite_outcome(reference_torch),
        )

    def test_signature_annotations_documentation_and_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = actual_jit.export
        expected = expected_jit.export

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(torch._jit_internal, actual_internal)
        self.assertIs(reference_torch._jit_internal, expected_internal)
        self.assertIs(actual, actual_internal.export)
        self.assertIs(expected, expected_internal.export)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(self.annotation_shape(actual), self.annotation_shape(expected))
        self.assertEqual(typing.get_type_hints(actual), actual.__annotations__)
        self.assertEqual(typing.get_type_hints(expected), expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_internal)
        self.assertIs(inspect.getmodule(expected), expected_internal)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.export
        expected = expected_jit.export

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name
                in {
                    "Attribute",
                    "annotate",
                    "export",
                    "ignore",
                    "isinstance",
                    "script_if_tracing",
                    "unused",
                }
            ],
        )
        self.assertEqual(
            torch.__all__.count("jit"), reference_torch.__all__.count("jit")
        )
        self.assertEqual(torch.__all__.count("export"), 0)
        self.assertEqual(reference_torch.__all__.count("export"), 1)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
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
        self.assertIs(actual_namespace["export"], actual)
        self.assertIs(expected_namespace["export"], expected)

        for actual_value, expected_value in (
            (actual, expected),
            (
                torch._jit_internal.FunctionModifiers,
                reference_torch._jit_internal.FunctionModifiers,
            ),
        ):
            self.assertIs(copy.copy(actual_value), actual_value)
            self.assertIs(copy.copy(expected_value), expected_value)
            self.assertIs(copy.deepcopy(actual_value), actual_value)
            self.assertIs(copy.deepcopy(expected_value), expected_value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=actual_value, protocol=protocol):
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_value, protocol)),
                        actual_value,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_value, protocol)),
                        expected_value,
                    )

        self.assertIs(
            _actual_picklable_export_function._torchscript_modifier,
            torch._jit_internal.FunctionModifiers.EXPORT,
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(
                    pickle.dumps(_actual_picklable_export_function, protocol)
                ),
                _actual_picklable_export_function,
            )

    def test_call_and_invalid_target_errors_match_pytorch_2_13(self):
        actual = torch.jit.export
        expected = reference_torch.jit.export

        class Example:
            def method(self):
                return None

        cases = (
            (lambda function: function()),
            (lambda function: function(lambda: None, lambda: None)),
            (lambda function: function(function=lambda: None)),
            (lambda function: function(lambda: None, fn=lambda: None)),
            (lambda function: function(None)),
            (lambda function: function(1)),
            (lambda function: function(len)),
            (lambda function: function(Example().method)),
            (lambda function: function(property())),
            (lambda function: function(property(1))),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_supported_boundary_remains_eager_jit_helpers_only(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {
                "Attribute",
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
        for name in (
            "script",
            "trace",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))

        self.assertIs(torch.jit.is_scripting(), False)

        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertFalse(hasattr(torch, "export"))

        actual_value = object()
        expected_value = object()
        self.assertIs(torch.jit.annotate(int, actual_value), actual_value)
        self.assertIs(
            reference_torch.jit.annotate(int, expected_value), expected_value
        )

        def actual_function():
            return "actual"

        def expected_function():
            return "expected"

        self.assertIs(torch.jit.unused(actual_function), actual_function)
        self.assertIs(
            reference_torch.jit.unused(expected_function), expected_function
        )
        self.assertEqual(actual_function(), "actual")
        self.assertEqual(expected_function(), "expected")


if __name__ == "__main__":
    unittest.main()
