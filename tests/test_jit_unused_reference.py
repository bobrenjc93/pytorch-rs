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

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def eager_outcome(self, module):
        calls = []

        def function(value, *, scale=1):
            calls.append((value, scale))
            return value * scale

        function.existing = "preserved"
        original_function = function
        decorated_function = module.jit.unused(function)
        function_result = decorated_function(4, scale=3)

        class CallableObject:
            def __call__(self, value):
                return value + 1

        callable_object = CallableObject()
        decorated_callable = module.jit.unused(callable_object)

        def getter(instance):
            return instance._value

        def setter(instance, value):
            instance._value = value

        def deleter(instance):
            del instance._value

        prop = property(getter, setter, deleter, "property documentation")
        original_property = prop
        decorated_property = module.jit.unused(prop)

        class Owner:
            value = decorated_property

            def __init__(self):
                self._value = 5

        owner = Owner()
        property_before = owner.value
        owner.value = 9
        property_after = owner.value
        del owner.value

        return {
            "function_identity": decorated_function is original_function,
            "function_result": function_result,
            "function_calls": calls,
            "function_dict": dict(function.__dict__),
            "callable_identity": decorated_callable is callable_object,
            "callable_result": decorated_callable(7),
            "callable_marker": callable_object._torchscript_modifier,
            "property_identity": decorated_property is original_property,
            "property_accessors": (
                decorated_property.fget is getter,
                decorated_property.fset is setter,
                decorated_property.fdel is deleter,
            ),
            "property_doc": decorated_property.__doc__,
            "property_values": (property_before, property_after),
            "property_deleted": not hasattr(owner, "_value"),
            "getter_marker": getter._torchscript_modifier,
            "setter_marker": setter._torchscript_modifier,
            "deleter_marked": hasattr(deleter, "_torchscript_modifier"),
            "property_marked": hasattr(prop, "_torchscript_modifier"),
        }

    def test_function_callable_and_property_eager_semantics_match(self):
        self.assertEqual(
            self.eager_outcome(torch), self.eager_outcome(reference_torch)
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
        self.assertIs(actual, actual_internal.unused)
        self.assertIs(expected, expected_internal.unused)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(repr(actual.__annotations__), repr(expected.__annotations__))
        self.assertEqual(
            repr(typing.get_type_hints(actual)),
            repr(typing.get_type_hints(expected)),
        )
        self.assertEqual(
            actual.__annotations__["fn"] is actual.__annotations__["return"],
            expected.__annotations__["fn"] is expected.__annotations__["return"],
        )

        actual_parameter_spec, actual_return_type = typing.get_args(
            actual.__annotations__["fn"]
        )
        expected_parameter_spec, expected_return_type = typing.get_args(
            expected.__annotations__["fn"]
        )
        for actual_type, expected_type in (
            (actual_parameter_spec, expected_parameter_spec),
            (actual_return_type, expected_return_type),
        ):
            with self.subTest(type_name=expected_type.__name__):
                self.assertIs(type(actual_type), type(expected_type))
                self.assertEqual(actual_type.__name__, expected_type.__name__)
                self.assertEqual(
                    actual_type.__module__.replace("torch_rs", "torch"),
                    expected_type.__module__,
                )

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
        self.assertEqual(actual_jit.__doc__, expected_jit.__doc__)

        self.assertEqual(
            actual_internal.FunctionModifiers.__module__.replace(
                "torch_rs", "torch"
            ),
            expected_internal.FunctionModifiers.__module__,
        )
        self.assertEqual(
            actual_internal.FunctionModifiers.__doc__,
            expected_internal.FunctionModifiers.__doc__,
        )
        expected_modifiers = {
            name: value
            for name, value in vars(expected_internal.FunctionModifiers).items()
            if name.isupper()
        }
        actual_modifiers = {
            name: value
            for name, value in vars(actual_internal.FunctionModifiers).items()
            if name.isupper()
        }
        self.assertEqual(actual_modifiers, expected_modifiers)

        def function():
            pass

        self.assertIs(actual(function), function)
        self.assertIs(
            function._torchscript_modifier,
            actual_internal.FunctionModifiers.UNUSED,
        )
        self.assertEqual(
            function._torchscript_modifier,
            expected_internal.FunctionModifiers.UNUSED,
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
                if name in {"annotate", "unused"}
            ],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            {"annotate", "unused"},
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
            {"annotate", "unused"},
        )
        self.assertIs(actual_namespace["unused"], actual)
        self.assertIs(expected_namespace["unused"], expected)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)), expected
                )

    def test_call_and_target_errors_match_pytorch_2_13(self):
        actual = torch.jit.unused
        expected = reference_torch.jit.unused
        cases = (
            lambda decorator: decorator(),
            lambda decorator: decorator(lambda: None, lambda: None),
            lambda decorator: decorator(other=1),
            lambda decorator: decorator(fn=lambda: None, other=1),
            lambda decorator: decorator(lambda: None, fn=lambda: None),
            lambda decorator: decorator(1),
            lambda decorator: decorator(object()),
            lambda decorator: decorator(len),
            lambda decorator: decorator(property()),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_annotate_is_unchanged_and_compilation_surface_stays_unsupported(self):
        actual_value = {"items": [1, 2]}
        expected_value = {"items": [1, 2]}
        self.assertIs(torch.jit.annotate(list[int], actual_value), actual_value)
        self.assertIs(
            reference_torch.jit.annotate(list[int], expected_value), expected_value
        )
        self.assertEqual(
            str(inspect.signature(torch.jit.annotate)),
            str(inspect.signature(reference_torch.jit.annotate)),
        )

        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in (
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
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))

        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
