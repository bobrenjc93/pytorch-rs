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

    def function_outcome(self, module):
        def target(value, *, offset=1):
            """target documentation"""
            return value + offset

        result = module.jit.unused(target)
        return (
            result is target,
            result(4, offset=3),
            result.__name__,
            result.__qualname__.rsplit(".", 1)[-1],
            result.__doc__,
            dict(result.__dict__),
        )

    def property_outcome(self, module):
        def getter(instance):
            return instance._value

        def setter(instance, value):
            instance._value = value

        def deleter(instance):
            del instance._value

        prop = property(getter, setter, deleter, "property documentation")
        result = module.jit.unused(prop)
        return (
            result is prop,
            result.fget is getter,
            result.fset is setter,
            result.fdel is deleter,
            result.__doc__,
            dict(getter.__dict__),
            dict(setter.__dict__),
            dict(deleter.__dict__),
        )

    def test_function_and_property_semantics_match(self):
        self.assertEqual(
            self.function_outcome(torch), self.function_outcome(reference_torch)
        )
        self.assertEqual(
            self.property_outcome(torch), self.property_outcome(reference_torch)
        )

    def test_signature_annotations_documentation_and_ownership_match(self):
        actual_internal = importlib.import_module("torch_rs._jit_internal")
        expected_internal = importlib.import_module("torch._jit_internal")
        actual = torch.jit.unused
        expected = reference_torch.jit.unused

        self.assertIs(torch.jit.unused, actual_internal.unused)
        self.assertIs(reference_torch.jit.unused, expected_internal.unused)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(
            {name: repr(value) for name, value in actual.__annotations__.items()},
            {name: repr(value) for name, value in expected.__annotations__.items()},
        )
        self.assertEqual(
            {
                name: repr(value)
                for name, value in typing.get_type_hints(actual).items()
            },
            {
                name: repr(value)
                for name, value in typing.get_type_hints(expected).items()
            },
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

        modifier_names = (
            "UNUSED",
            "IGNORE",
            "EXPORT",
            "DEFAULT",
            "COPY_TO_SCRIPT_WRAPPER",
            "_DROP",
        )
        self.assertEqual(
            {
                name: getattr(actual_internal.FunctionModifiers, name)
                for name in modifier_names
            },
            {
                name: getattr(expected_internal.FunctionModifiers, name)
                for name in modifier_names
            },
        )
        self.assertEqual(
            actual_internal.FunctionModifiers.__module__.replace(
                "torch_rs", "torch"
            ),
            expected_internal.FunctionModifiers.__module__,
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

        def target():
            return None

        cases = (
            lambda function: function(),
            lambda function: function(target, target),
            lambda function: function(function=target),
            lambda function: function(target, fn=target),
            lambda function: function(None),
            lambda function: function(1),
            lambda function: function(len),
            lambda function: function(property()),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_supported_boundary_adds_only_eager_unused(self):
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {"annotate", "unused"},
        )
        for name in ("ignore", "script", "trace", "is_scripting", "is_tracing"):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))

        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertEqual(
            str(inspect.signature(torch.jit.annotate)),
            str(inspect.signature(reference_torch.jit.annotate)),
        )


if __name__ == "__main__":
    unittest.main()
