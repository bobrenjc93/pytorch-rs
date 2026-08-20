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


def _actual_picklable_constant_function(value):
    return value + 1


def _expected_picklable_constant_function(value):
    return value + 1


torch.compiler.assume_constant_result(_actual_picklable_constant_function)
if reference_torch is not None:
    reference_torch.compiler.assume_constant_result(
        _expected_picklable_constant_function
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAssumeConstantResultReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.assume_constant_result differentials require pinned "
                "PyTorch 2.13.0"
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

    def function_outcome(self, module):
        sentinel = object()
        calls = []

        def function(value, *, option=sentinel):
            """function documentation"""
            calls.append((value, option))
            return len(calls), value, option

        function.custom_attribute = sentinel
        before = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__.copy(),
            function.__defaults__,
            function.__kwdefaults__.copy(),
        )
        first = module.compiler.assume_constant_result(function)
        first_state = (
            first is function,
            function._dynamo_marked_constant is True,
            function.custom_attribute is sentinel,
        )
        first_result = function("first")
        second_result = function("second", option="value")
        eager_results = (
            (first_result[0], first_result[1], first_result[2] is sentinel),
            second_result,
            tuple(
                (value, option is sentinel if value == "first" else option)
                for value, option in calls
            ),
        )
        function._dynamo_marked_constant = sentinel
        second = module.compiler.assume_constant_result(function)
        third = module.compiler.assume_constant_result(second)
        after = (
            function.__name__,
            function.__qualname__,
            function.__doc__,
            function.__annotations__,
            function.__defaults__,
            function.__kwdefaults__,
        )
        return (
            first_state,
            eager_results,
            second is function,
            third is function,
            function._dynamo_marked_constant is True,
            before == after,
        )

    def method_callable_and_noncallable_outcome(self, module):
        class Example:
            @module.compiler.assume_constant_result
            def method(self, value):
                return value + 1

        class CallableTarget:
            def __init__(self):
                self.calls = []

            def __call__(self, value):
                self.calls.append(value)
                return len(self.calls), value * 2

        class WritableTarget:
            pass

        callable_target = CallableTarget()
        callable_target._dynamo_marked_constant = object()
        first = module.compiler.assume_constant_result(callable_target)
        second = module.compiler.assume_constant_result(first)

        writable_target = WritableTarget()
        writable_result = module.compiler.assume_constant_result(writable_target)
        raw_method = Example.__dict__["method"]
        return (
            Example.method is raw_method,
            raw_method._dynamo_marked_constant is True,
            Example().method(4),
            first is callable_target,
            second is callable_target,
            callable_target._dynamo_marked_constant is True,
            callable_target(3),
            callable_target(5),
            tuple(callable_target.calls),
            writable_result is writable_target,
            writable_target._dynamo_marked_constant is True,
        )

    def test_marking_eager_behavior_and_idempotence_match_pytorch_2_13(self):
        self.assertEqual(
            self.function_outcome(torch),
            self.function_outcome(reference_torch),
        )
        self.assertEqual(
            self.method_callable_and_noncallable_outcome(torch),
            self.method_callable_and_noncallable_outcome(reference_torch),
        )

    def test_signature_documentation_and_ownership_match(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.assume_constant_result
        expected = expected_compiler.assume_constant_result

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        def actual_target():
            return "actual"

        def expected_target():
            return "expected"

        self.assertIs(actual(fn=actual_target), actual_target)
        self.assertIs(expected(fn=expected_target), expected_target)
        self.assertIs(actual_target._dynamo_marked_constant, True)
        self.assertIs(expected_target._dynamo_marked_constant, True)

    def test_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.assume_constant_result
        expected = expected_compiler.assume_constant_result
        supported = {
            "assume_constant_result",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
        }

        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in supported],
        )
        for name in ("compiler", *supported):
            self.assertEqual(
                torch.__all__.count(name),
                reference_torch.__all__.count(name),
            )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.compiler import *", actual_namespace)
        exec("from torch.compiler import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            supported,
        )
        for name in supported:
            self.assertIs(actual_namespace[name], getattr(actual_compiler, name))
            self.assertIs(expected_namespace[name], getattr(expected_compiler, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("assume_constant_result", namespace)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        for value in (
            _actual_picklable_constant_function,
            _expected_picklable_constant_function,
        ):
            self.assertIs(value._dynamo_marked_constant, True)
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(pickle.loads(pickle.dumps(value, protocol)), value)

    def test_call_and_invalid_target_errors_match_pytorch_2_13(self):
        actual = torch.compiler.assume_constant_result
        expected = reference_torch.compiler.assume_constant_result

        class Example:
            def method(self):
                return None

        class SlottedCallable:
            __slots__ = ()

            def __call__(self):
                return None

        cases = (
            lambda function: function(),
            lambda function: function(lambda: None, lambda: None),
            lambda function: function(function=lambda: None),
            lambda function: function(lambda: None, fn=lambda: None),
            lambda function: function(None),
            lambda function: function(1),
            lambda function: function(len),
            lambda function: function(Example().method),
            lambda function: function(property()),
            lambda function: function(property(1)),
            lambda function: function(SlottedCallable()),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_existing_queries_and_unsupported_boundary_match(self):
        actual_queries = (
            torch.compiler.is_compiling,
            torch.compiler.is_dynamo_compiling,
            torch.compiler.is_exporting,
        )
        self.assertEqual(
            tuple(query() for query in actual_queries),
            (False, False, False),
        )

        def function(value):
            return value + 1

        torch.compiler.assume_constant_result(function)
        self.assertEqual(function(2), 3)
        self.assertEqual(
            tuple(query() for query in actual_queries),
            (False, False, False),
        )

        self.assertTrue(callable(reference_torch.compile))
        self.assertTrue(hasattr(reference_torch, "export"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))

        supported = {
            "assume_constant_result",
            "is_compiling",
            "is_dynamo_compiling",
            "is_exporting",
        }
        unsupported = set(reference_torch.compiler.__all__) - supported
        self.assertTrue(unsupported)
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))


if __name__ == "__main__":
    unittest.main()
