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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitUnusedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.unused differentials require pinned PyTorch 2.13.0"
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
        result = module.jit.unused(function)
        modifier = module._jit_internal.FunctionModifiers.UNUSED
        return (
            result is function,
            function("value") == ("value", sentinel),
            function.custom_attribute is sentinel,
            function._torchscript_modifier,
            function._torchscript_modifier is modifier,
            copy.copy(function) is function,
            copy.deepcopy(function) is function,
        )

    def property_outcome(self, module):
        def getter(instance):
            return instance._value

        def setter(instance, value):
            instance._value = value

        def deleter(instance):
            del instance._value

        deleter._torchscript_modifier = "leave unchanged"
        prop = property(getter, setter, deleter, "property documentation")
        before = (prop.fget, prop.fset, prop.fdel, prop.__doc__)
        result = module.jit.unused(prop)
        modifier = module._jit_internal.FunctionModifiers.UNUSED

        class Holder:
            value = prop

            def __init__(self):
                self._value = 3

        holder = Holder()
        first = holder.value
        holder.value = 7
        second = holder.value
        del holder.value

        pickle_errors = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                pickle.dumps(prop, protocol=protocol)
            except Exception as error:
                pickle_errors.append(
                    (type(error).__name__, str(error), error.args)
                )

        return (
            result is prop,
            (prop.fget, prop.fset, prop.fdel, prop.__doc__) == before,
            getter._torchscript_modifier is modifier,
            setter._torchscript_modifier is modifier,
            deleter._torchscript_modifier,
            hasattr(prop, "_torchscript_modifier"),
            (first, second, hasattr(holder, "_value")),
            copy.copy(prop) is prop,
            copy.deepcopy(prop) is prop,
            pickle_errors,
        )

    def test_function_and_property_eager_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.function_outcome(torch),
            self.function_outcome(reference_torch),
        )
        self.assertEqual(
            self.property_outcome(torch),
            self.property_outcome(reference_torch),
        )

    def test_signature_annotations_documentation_and_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = actual_jit.unused
        expected = expected_jit.unused

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(torch._jit_internal, actual_internal)
        self.assertIs(reference_torch._jit_internal, expected_internal)
        self.assertIs(actual, actual_internal.unused)
        self.assertIs(expected, expected_internal.unused)
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

        actual_modifiers = actual_internal.FunctionModifiers
        expected_modifiers = expected_internal.FunctionModifiers
        self.assertEqual(
            actual_modifiers.__module__.replace("torch_rs", "torch"),
            expected_modifiers.__module__,
        )
        self.assertEqual(actual_modifiers.__name__, expected_modifiers.__name__)
        self.assertEqual(
            actual_modifiers.__qualname__, expected_modifiers.__qualname__
        )
        self.assertEqual(actual_modifiers.__doc__, expected_modifiers.__doc__)
        self.assertEqual(
            actual_modifiers.__annotations__, expected_modifiers.__annotations__
        )
        for name in (
            "UNUSED",
            "IGNORE",
            "EXPORT",
            "DEFAULT",
            "COPY_TO_SCRIPT_WRAPPER",
            "_DROP",
        ):
            with self.subTest(modifier=name):
                self.assertEqual(
                    getattr(actual_modifiers, name),
                    getattr(expected_modifiers, name),
                )

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.unused
        expected = expected_jit.unused

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name
                in {"annotate", "export", "ignore", "script_if_tracing", "unused"}
            ],
        )
        self.assertEqual(
            torch.__all__.count("jit"), reference_torch.__all__.count("jit")
        )
        self.assertEqual(
            torch.__all__.count("unused"),
            reference_torch.__all__.count("unused"),
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            {"annotate", "export", "ignore", "script_if_tracing", "unused"},
        )
        self.assertIs(actual_namespace["unused"], actual)
        self.assertIs(expected_namespace["unused"], expected)

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

    def test_call_and_invalid_target_errors_match_pytorch_2_13(self):
        actual = torch.jit.unused
        expected = reference_torch.jit.unused

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

    def test_supported_boundary_remains_eager_decorators_and_annotate_only(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {
                "annotate",
                "export",
                "ignore",
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

        actual_value = object()
        expected_value = object()
        self.assertIs(torch.jit.annotate(int, actual_value), actual_value)
        self.assertIs(
            reference_torch.jit.annotate(int, expected_value), expected_value
        )


if __name__ == "__main__":
    unittest.main()
