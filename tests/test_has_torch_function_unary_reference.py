import copy
import inspect
import pickle
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class HasTorchFunctionUnaryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "has_torch_function_unary differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_tensor_and_custom_override_results_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_unary
        expected = reference_torch.overrides.has_torch_function_unary

        class ClassOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        class PlainOverride:
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        class StaticOverride:
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                return NotImplemented

        class NoneOverride:
            __torch_function__ = None

        class FalseOverride:
            __torch_function__ = False

        class MissingOverride:
            pass

        shared_cases = (
            ClassOverride(),
            ClassOverride,
            PlainOverride(),
            StaticOverride(),
            NoneOverride(),
            FalseOverride(),
            MissingOverride(),
            None,
            1,
            1.0,
            [],
            (),
            {},
            object(),
            int,
        )
        for value in shared_cases:
            with self.subTest(value=repr(value)):
                self.assertIs(actual(value), expected(value))

        paired_cases = (
            (torch.tensor([1.0]), reference_torch.tensor([1.0])),
            (torch.Tensor, reference_torch.Tensor),
            (torch.Tensor.__base__, reference_torch.Tensor.__base__),
        )
        for actual_value, expected_value in paired_cases:
            with self.subTest(value=repr(actual_value)):
                self.assertIs(actual(actual_value), expected(expected_value))

    def test_disabled_handlers_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_unary
        expected = reference_torch.overrides.has_torch_function_unary
        disabled_handler = reference_torch._C._disabled_torch_function_impl

        class DisabledOverride:
            __torch_function__ = disabled_handler

        class DisabledDescriptor:
            def __init__(self):
                self.calls = 0

            def __get__(self, instance, owner):
                self.calls += 1
                return disabled_handler

        for value in (DisabledOverride(), DisabledOverride):
            with self.subTest(value=repr(value)):
                self.assertIs(actual(value), False)
                self.assertIs(actual(value), expected(value))

        actual_descriptor = DisabledDescriptor()
        expected_descriptor = DisabledDescriptor()

        class ActualDisabledDescriptor:
            __torch_function__ = actual_descriptor

        class ExpectedDisabledDescriptor:
            __torch_function__ = expected_descriptor

        self.assertIs(actual(ActualDisabledDescriptor()), False)
        self.assertIs(expected(ExpectedDisabledDescriptor()), False)
        self.assertEqual(actual_descriptor.calls, expected_descriptor.calls)
        self.assertEqual(actual_descriptor.calls, 1)

    def test_failing_descriptor_behavior_matches_pytorch_2_13(self):
        def outcomes(function):
            class RaisingDescriptor:
                def __init__(self):
                    self.calls = []

                def __get__(self, instance, owner):
                    self.calls.append((instance is None, owner.__name__))
                    raise RuntimeError("descriptor failed")

            descriptor = RaisingDescriptor()

            class BrokenOverride:
                __torch_function__ = descriptor

            class BrokenDynamicOverride:
                calls = 0

                def __getattribute__(self, name):
                    if name == "__torch_function__":
                        type(self).calls += 1
                        raise RuntimeError("dynamic lookup failed")
                    return object.__getattribute__(self, name)

            results = (
                function(BrokenOverride()),
                function(BrokenOverride),
                function(BrokenDynamicOverride()),
            )
            return results, descriptor.calls, BrokenDynamicOverride.calls

        self.assertEqual(
            outcomes(torch.overrides.has_torch_function_unary),
            outcomes(reference_torch.overrides.has_torch_function_unary),
        )

    def test_mode_behavior_matches_pytorch_2_13(self):
        def outcomes(module):
            function = module.overrides.has_torch_function_unary
            tensor = module.tensor([1.0])

            class RaisingDescriptor:
                def __init__(self):
                    self.calls = 0

                def __get__(self, instance, owner):
                    self.calls += 1
                    raise RuntimeError("descriptor should not be read")

            descriptor = RaisingDescriptor()

            class BrokenOverride:
                __torch_function__ = descriptor

            class Mode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    return NotImplemented

            before = (
                function(tensor),
                function(module.Tensor),
                function(1),
                function(BrokenOverride()),
                descriptor.calls,
            )
            with Mode():
                during = (
                    function(tensor),
                    function(module.Tensor),
                    function(1),
                    function(BrokenOverride()),
                    descriptor.calls,
                )
            after = function(tensor), descriptor.calls
            return before, during, after

        self.assertEqual(outcomes(torch), outcomes(reference_torch))

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_unary
        expected = reference_torch.overrides.has_torch_function_unary

        self.assertIs(actual, torch._C._has_torch_function_unary)
        self.assertIs(expected, reference_torch._C._has_torch_function_unary)
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch._C"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertEqual(actual.__reduce__(), expected.__reduce__())

        for function in (actual, expected):
            if function.__text_signature__ is None:
                with self.assertRaises(ValueError):
                    inspect.signature(function)
            else:
                self.assertEqual(str(inspect.signature(function)), "(object, /)")
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_unary
        expected = reference_torch.overrides.has_torch_function_unary
        cases = (
            (lambda function: function()),
            (lambda function: function(None, None)),
            (lambda function: function(input=None)),
            (lambda function: function(None, unexpected=True)),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

        self.assertIs(actual(None, **{}), expected(None, **{}))


if __name__ == "__main__":
    unittest.main()
