import copy
import ctypes
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
            (None, 1, 1.0, [], (), {}, object(), int),
            (ClassOverride(),),
            (None, ClassOverride),
            (PlainOverride(), MissingOverride()),
            (StaticOverride(),),
            (NoneOverride(),),
            (FalseOverride(),),
            (MissingOverride(), None),
        )
        for arguments in shared_cases:
            with self.subTest(arguments=repr(arguments)):
                self.assertIs(actual(*arguments), expected(*arguments))

        paired_cases = (
            ((torch.tensor([1.0]),), (reference_torch.tensor([1.0]),)),
            (
                (None, torch.tensor([1.0]), torch.tensor([2.0])),
                (
                    None,
                    reference_torch.tensor([1.0]),
                    reference_torch.tensor([2.0]),
                ),
            ),
            ((torch.Tensor,), (reference_torch.Tensor,)),
            (
                (torch.Tensor.__base__, torch.tensor([1.0])),
                (
                    reference_torch.Tensor.__base__,
                    reference_torch.tensor([1.0]),
                ),
            ),
        )
        for actual_arguments, expected_arguments in paired_cases:
            with self.subTest(arguments=repr(actual_arguments)):
                self.assertIs(
                    actual(*actual_arguments),
                    expected(*expected_arguments),
                )

    def test_disabled_handlers_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic
        disabled_handler = reference_torch._C._disabled_torch_function_impl

        class DisabledOverride:
            __torch_function__ = disabled_handler

        class Override:
            __torch_function__ = None

        shared_cases = (
            (DisabledOverride(),),
            (None, DisabledOverride(), 1),
            (DisabledOverride(), Override()),
            (Override(), DisabledOverride()),
        )
        for arguments in shared_cases:
            with self.subTest(arguments=repr(arguments)):
                self.assertIs(actual(*arguments), expected(*arguments))

        def outcomes(function):
            events = []

            class DisabledDescriptor:
                def __get__(self, instance, owner):
                    events.append("disabled")
                    return disabled_handler

            class OverrideDescriptor:
                def __get__(self, instance, owner):
                    events.append("override")
                    return None

            class SkippedDescriptor:
                def __get__(self, instance, owner):
                    events.append("skipped")
                    raise RuntimeError("should not be read")

            class Disabled:
                __torch_function__ = DisabledDescriptor()

            class Found:
                __torch_function__ = OverrideDescriptor()

            class Skipped:
                __torch_function__ = SkippedDescriptor()

            result = function(Disabled(), Found(), Skipped())
            return result, events

        self.assertEqual(outcomes(actual), outcomes(expected))
        self.assertEqual(outcomes(actual), (True, ["disabled", "override"]))

    def test_descriptor_failure_and_short_circuit_match_pytorch_2_13(self):
        def outcomes(function):
            events = []

            class Descriptor:
                def __init__(self, name, raises=False):
                    self.name = name
                    self.raises = raises

                def __get__(self, instance, owner):
                    events.append((self.name, instance is None, owner.__name__))
                    if self.raises:
                        raise RuntimeError(f"{self.name} failed")
                    return None

            class Broken:
                __torch_function__ = Descriptor("broken", raises=True)

            class Override:
                __torch_function__ = Descriptor("override")

            class Skipped:
                __torch_function__ = Descriptor("skipped", raises=True)

            first = function(Broken(), 1, Override(), Skipped())
            first_events = events.copy()
            events.clear()
            second = function(Override(), Skipped(), Broken())
            second_events = events.copy()
            events.clear()
            third = function(Broken(), Broken)
            third_events = events.copy()
            return (
                first,
                first_events,
                second,
                second_events,
                third,
                third_events,
            )

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

            broken = BrokenOverride()
            before = (
                function(),
                function(tensor, 1),
                function(broken),
                descriptor.calls,
            )
            with Mode():
                during = (
                    function(),
                    function(tensor),
                    function(module.Tensor),
                    function(module.Tensor.__base__),
                    function(1, None, broken),
                    descriptor.calls,
                )
            after = function(tensor, 1), function(broken), descriptor.calls
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

        get_flags = ctypes.pythonapi.PyCFunction_GetFlags
        get_flags.argtypes = [ctypes.py_object]
        get_flags.restype = ctypes.c_int
        self.assertTrue(get_flags(actual) & 0x0080)
        self.assertTrue(get_flags(expected) & 0x0080)

        for function in (actual, expected):
            if function.__text_signature__ is None:
                with self.assertRaises(ValueError):
                    inspect.signature(function)
            else:
                self.assertEqual(str(inspect.signature(function)), "(*args)")
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_keyword_errors_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function_variadic
        expected = reference_torch.overrides.has_torch_function_variadic

        class RaisingLookup:
            calls = 0

            def __getattribute__(self, name):
                if name == "__torch_function__":
                    type(self).calls += 1
                    raise RuntimeError("argument should not be probed")
                return object.__getattribute__(self, name)

        cases = (
            lambda function: function(input=None),
            lambda function: function(None, unexpected=True),
            lambda function: function(RaisingLookup(), unexpected=True),
        )
        for call in cases:
            with self.subTest(call=call):
                before = RaisingLookup.calls
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )
                self.assertEqual(RaisingLookup.calls, before)

        self.assertIs(actual(None, **{}), expected(None, **{}))


if __name__ == "__main__":
    unittest.main()
