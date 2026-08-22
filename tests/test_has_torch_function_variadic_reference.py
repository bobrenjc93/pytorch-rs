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
class HasTorchFunctionVariadicReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "has_torch_function_variadic differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_zero_multiple_tensor_and_custom_results_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic

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
            (),
            (None,),
            (None, 1, object()),
            (ClassOverride(),),
            (None, ClassOverride),
            (PlainOverride(),),
            (StaticOverride(),),
            (NoneOverride(),),
            (FalseOverride(),),
            (MissingOverride(),),
            ((ClassOverride(),),),
        )
        for arguments in shared_cases:
            with self.subTest(arguments=repr(arguments)):
                self.assertIs(actual(*arguments), expected(*arguments))

        paired_cases = (
            (
                (torch.tensor([1.0]),),
                (reference_torch.tensor([1.0]),),
            ),
            (
                (None, torch.tensor([1.0]), 1),
                (None, reference_torch.tensor([1.0]), 1),
            ),
            ((torch.Tensor,), (reference_torch.Tensor,)),
            (
                (None, torch.Tensor.__base__),
                (None, reference_torch.Tensor.__base__),
            ),
        )
        for actual_arguments, expected_arguments in paired_cases:
            with self.subTest(arguments=repr(actual_arguments)):
                self.assertIs(
                    actual(*actual_arguments),
                    expected(*expected_arguments),
                )

    def test_exact_cuda_tensor_matches_exact_cpu_tensor(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic
        actual_cpu = torch.tensor([1.0])
        expected_cpu = reference_torch.tensor([1.0])
        expected_cuda = reference_torch.tensor([1.0], device="cuda")

        self.assertIs(actual(actual_cpu), expected(expected_cpu))
        self.assertIs(actual(actual_cpu), expected(expected_cuda))
        self.assertIs(actual(None, actual_cpu), expected(None, expected_cuda))

    def test_disabled_handlers_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic
        disabled_handler = reference_torch._C._disabled_torch_function_impl

        class DisabledOverride:
            __torch_function__ = disabled_handler

        class ActiveOverride:
            __torch_function__ = None

        for arguments in (
            (DisabledOverride(),),
            (None, DisabledOverride(), object()),
            (DisabledOverride(), ActiveOverride()),
        ):
            with self.subTest(arguments=repr(arguments)):
                self.assertIs(actual(*arguments), expected(*arguments))

        class DisabledDescriptor:
            def __init__(self):
                self.calls = 0

            def __get__(self, instance, owner):
                self.calls += 1
                return disabled_handler

        actual_descriptor = DisabledDescriptor()
        expected_descriptor = DisabledDescriptor()

        class ActualDisabledDescriptor:
            __torch_function__ = actual_descriptor

        class ExpectedDisabledDescriptor:
            __torch_function__ = expected_descriptor

        self.assertIs(actual(None, ActualDisabledDescriptor()), False)
        self.assertIs(expected(None, ExpectedDisabledDescriptor()), False)
        self.assertEqual(actual_descriptor.calls, expected_descriptor.calls)
        self.assertEqual(actual_descriptor.calls, 1)

    def test_descriptor_failures_and_short_circuit_match_pytorch_2_13(self):
        def outcomes(function):
            events = []

            class Descriptor:
                def __init__(self, name, *, raises=False):
                    self.name = name
                    self.raises = raises

                def __get__(self, instance, owner):
                    events.append((self.name, instance is None, owner.__name__))
                    if self.raises:
                        raise RuntimeError(f"{self.name} failed")
                    return None

            class BrokenFirst:
                __torch_function__ = Descriptor("broken-first", raises=True)

            class Override:
                __torch_function__ = Descriptor("override")

            class Unreachable:
                __torch_function__ = Descriptor("unreachable", raises=True)

            results = (
                function(object(), BrokenFirst(), Override(), Unreachable()),
                tuple(events),
            )
            events.clear()
            class_result = function(object(), Override)
            class_events = tuple(events)
            events.clear()
            false_result = function(BrokenFirst(), object())
            false_events = tuple(events)
            return results, class_result, class_events, false_result, false_events

        self.assertEqual(
            outcomes(torch.overrides.has_torch_function_variadic),
            outcomes(reference_torch.overrides.has_torch_function_variadic),
        )

    def test_mode_behavior_matches_pytorch_2_13(self):
        def outcomes(module):
            function = module.overrides.has_torch_function_variadic
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
                function(),
                function(tensor, 1),
                function(1, BrokenOverride()),
                descriptor.calls,
            )
            with Mode():
                during = (
                    function(),
                    function(tensor),
                    function(1),
                    function(1, BrokenOverride()),
                    descriptor.calls,
                )
            after = function(tensor), descriptor.calls
            return before, during, after

        self.assertEqual(outcomes(torch), outcomes(reference_torch))

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic

        self.assertIs(actual, torch._C._has_torch_function_variadic)
        self.assertIs(expected, reference_torch._C._has_torch_function_variadic)
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
            with self.assertRaises(ValueError):
                inspect.signature(function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic
        cases = (
            lambda function: function(input=None),
            lambda function: function(objects=()),
            lambda function: function(None, unexpected=True),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

        self.assertIs(actual(**{}), expected(**{}))
        self.assertIs(actual(None, **{}), expected(None, **{}))


if __name__ == "__main__":
    unittest.main()
