import json
import pickle
import re
import subprocess
import sys
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelPositiveOverrideReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("positive override differentials require PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def override_observation(self, module, keyword):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = (
            module.positive(value)
            if keyword is None
            else module.positive(**{keyword: value})
        )
        function, types, args, kwargs = Override.calls[0]
        return (
            result is marker,
            function is module.positive,
            types == (Override,),
            args == ((value,) if keyword is None else ()),
            kwargs is None if keyword is None else kwargs == {keyword: value},
        )

    def mode_observation(self, module, keyword):
        tensor = module.tensor([1.0])
        marker = object()

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = Mode()
        with mode:
            result = (
                module.positive(tensor)
                if keyword is None
                else module.positive(**{keyword: tensor})
            )
        function, types, args, kwargs = mode.calls[0]
        return (
            result is marker,
            function is module.positive,
            types,
            args == ((tensor,) if keyword is None else ()),
            kwargs is None if keyword is None else kwargs == {keyword: tensor},
        )

    def test_object_and_mode_dispatch_match_pytorch_2_13(self):
        for keyword in (None, "input", "x", "a", "x1"):
            with self.subTest(dispatch="object", keyword=keyword):
                self.assertEqual(
                    self.override_observation(torch, keyword),
                    self.override_observation(reference_torch, keyword),
                )
            with self.subTest(dispatch="mode", keyword=keyword):
                self.assertEqual(
                    self.mode_observation(torch, keyword),
                    self.mode_observation(reference_torch, keyword),
                )

    def special_override_observation(self, module):
        marker = object()
        instance_calls = []

        class InstanceAssigned:
            pass

        instance = InstanceAssigned()

        def instance_handler(func, types, args=(), kwargs=None):
            instance_calls.append((func, types, args, kwargs))
            return marker

        instance.__torch_function__ = instance_handler
        instance_result = module.positive(x1=instance)

        class_calls = []

        class ClassObject:
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                class_calls.append((func, types, args, kwargs))
                return marker

        class_result = module.positive(ClassObject)

        descriptor_calls = []

        class StatefulDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                resolution = self.lookups

                def handler(func, types, args=(), kwargs=None):
                    descriptor_calls.append((func, types, args, kwargs))
                    return resolution

                if resolution > 2:
                    raise AttributeError("descriptor was resolved more than twice")
                return handler

        descriptor = StatefulDescriptor()

        class DescriptorOverride:
            __torch_function__ = descriptor

        descriptor_value = DescriptorOverride()
        descriptor_result = module.positive(descriptor_value)
        return (
            instance_result is marker,
            instance_calls
            == [
                (
                    module.positive,
                    (InstanceAssigned,),
                    (),
                    {"x1": instance},
                )
            ],
            class_result is marker,
            class_calls
            == [(module.positive, (ClassObject,), (ClassObject,), None)],
            descriptor_result,
            descriptor.lookups,
            descriptor_calls
            == [
                (
                    module.positive,
                    (DescriptorOverride,),
                    (descriptor_value,),
                    None,
                )
            ],
        )

    def test_instance_class_and_descriptor_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.special_override_observation(torch),
            self.special_override_observation(reference_torch),
        )

    def test_override_binding_precedence_matches_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("invalid calls must not dispatch")

        actual = Override()
        expected = Override()
        cases = (
            (
                lambda: torch.positive(actual, actual),
                lambda: reference_torch.positive(expected, expected),
            ),
            (
                lambda: torch.positive(actual, input=actual),
                lambda: reference_torch.positive(expected, input=expected),
            ),
            (
                lambda: torch.positive(actual, extra=True),
                lambda: reference_torch.positive(expected, extra=True),
            ),
            (
                lambda: torch.positive(extra=actual, input=actual),
                lambda: reference_torch.positive(extra=expected, input=expected),
            ),
            (
                lambda: torch.positive(extra=actual),
                lambda: reference_torch.positive(extra=expected),
            ),
            (
                lambda: torch.positive(a=actual, x=actual),
                lambda: reference_torch.positive(a=expected, x=expected),
            ),
            (
                lambda: torch.positive(x=actual, a=actual),
                lambda: reference_torch.positive(x=expected, a=expected),
            ),
            (
                lambda: torch.positive(input=actual, x1=actual),
                lambda: reference_torch.positive(input=expected, x1=expected),
            ),
            (
                lambda: torch.positive(x=actual, x1=actual),
                lambda: reference_torch.positive(x=expected, x1=expected),
            ),
            (
                lambda: torch.positive(x1=actual, x=actual),
                lambda: reference_torch.positive(x1=expected, x=expected),
            ),
            (
                lambda: torch.positive(x=1, x1=actual),
                lambda: reference_torch.positive(x=1, x1=expected),
            ),
            (
                lambda: torch.positive(x1=1, a=actual),
                lambda: reference_torch.positive(x1=1, a=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def forwarding_observation(self, module):
        tensor = module.tensor([1.0])
        order = []

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with Mode("lower"):
            with Mode("upper"):
                result = module.positive(tensor)
        return result is tensor, order

    def test_nested_mode_forwarding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.forwarding_observation(torch),
            self.forwarding_observation(reference_torch),
        )

    def mode_error_observation(self, module, call):
        tensor = module.tensor([1.0])

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = Mode()
        try:
            with mode:
                call(module, tensor)
        except Exception as error:
            return type(error).__name__, str(error), len(mode.calls)
        return None

    def test_mode_binding_precedence_matches_pytorch_2_13(self):
        cases = (
            lambda module, tensor: module.positive(),
            lambda module, tensor: module.positive(tensor, tensor),
            lambda module, tensor: module.positive(extra=tensor),
            lambda module, tensor: module.positive(1, extra=True),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(
                    self.mode_error_observation(torch, call),
                    self.mode_error_observation(reference_torch, call),
                )

    def test_mode_not_implemented_fallback_matches_pytorch_2_13(self):
        def observation(module):
            marker = object()
            calls = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    calls.append("override")
                    return marker

            class Mode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    calls.append("mode")
                    return NotImplemented

            with Mode():
                result = module.positive(Override())
            return result is marker, calls

        self.assertEqual(observation(torch), observation(reference_torch))

    def late_resolution_observation(self, module):
        marker = object()
        events = []

        class StatefulDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                events.append(("lookup", self.lookups))
                if self.lookups > 1:
                    raise RuntimeError("late override resolution must not run")
                return lambda func, types, args=(), kwargs=None: "override"

        descriptor = StatefulDescriptor()

        class Override:
            __torch_function__ = descriptor

        value = Override()

        class AcceptingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "mode",
                        types == (Override,),
                        args == (value,),
                        kwargs is None,
                    )
                )
                return marker

        with AcceptingMode():
            accepting_result = module.positive(value)

        calls = []

        class MutableOverride:
            pass

        mutable = MutableOverride()

        def original_handler(func, types, args=(), kwargs=None):
            calls.append("original")
            return "original"

        def replacement_handler(func, types, args=(), kwargs=None):
            calls.append("replacement")
            return marker

        mutable.__torch_function__ = original_handler

        class MutatingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append("mode")
                mutable.__torch_function__ = replacement_handler
                return NotImplemented

        with MutatingMode():
            mutating_result = module.positive(mutable)

        return (
            accepting_result is marker,
            descriptor.lookups,
            events,
            mutating_result is marker,
            calls,
        )

    def test_mode_order_and_late_override_resolution_match_pytorch_2_13(self):
        self.assertEqual(
            self.late_resolution_observation(torch),
            self.late_resolution_observation(reference_torch),
        )

    def retry_and_mode_reresolution_observation(self, module):
        override_calls = []

        class RetryDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                resolution = self.lookups
                if resolution == 1:
                    raise AttributeError("transient probe failure")

                def handler(func, dispatch_types, args=(), kwargs=None):
                    override_calls.append(
                        (
                            resolution,
                            func is module.positive,
                            dispatch_types == (Override,),
                            args == (value,),
                            kwargs is None,
                        )
                    )
                    return resolution

                return handler

        retry_descriptor = RetryDescriptor()

        class Override:
            __torch_function__ = retry_descriptor

        value = Override()
        override_result = module.positive(value)

        mode_calls = []

        class CallableHandler:
            def __init__(self, receiver, resolution):
                self.__self__ = receiver
                self.receiver = receiver
                self.resolution = resolution

            def __call__(self, func, dispatch_types, args=(), kwargs=None):
                mode_calls.append(
                    (
                        self.resolution,
                        self.__self__ is self.receiver,
                        func is module.positive,
                        dispatch_types,
                        len(args),
                        kwargs is None,
                    )
                )
                return self.resolution

        class StatefulModeDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                return CallableHandler(instance, self.lookups)

        mode_descriptor = StatefulModeDescriptor()

        class Mode(module.overrides.TorchFunctionMode):
            __torch_function__ = mode_descriptor

        tensor = module.tensor([1.0])
        with Mode():
            mode_result = module.positive(tensor)

        return (
            override_result,
            retry_descriptor.lookups,
            override_calls,
            mode_result,
            mode_descriptor.lookups,
            mode_calls,
        )

    def test_fallback_probe_and_mode_reresolution_match_pytorch_2_13(self):
        self.assertEqual(
            self.retry_and_mode_reresolution_observation(torch),
            self.retry_and_mode_reresolution_observation(reference_torch),
        )

    def mode_handler_form_observation(self, module):
        tensor = module.tensor([1.0])
        calls = []

        class ClassMethodMode(module.overrides.TorchFunctionMode):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append("classmethod")
                return object()

        class StaticMethodMode(module.overrides.TorchFunctionMode):
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                calls.append("staticmethod")
                return object()

        class InstanceAssignedMode(module.overrides.TorchFunctionMode):
            pass

        class RaisingSelfHandler:
            @property
            def __self__(self):
                raise RuntimeError("suppressed self lookup")

            def __call__(self, func, types, args=(), kwargs=None):
                calls.append("raising-self")
                return object()

        class RaisingSelfDescriptor:
            def __get__(self, instance, owner):
                return RaisingSelfHandler()

        class RaisingSelfMode(module.overrides.TorchFunctionMode):
            __torch_function__ = RaisingSelfDescriptor()

        instance_assigned = InstanceAssignedMode()

        def instance_handler(func, types, args=(), kwargs=None):
            calls.append("instance")
            return object()

        instance_assigned.__torch_function__ = instance_handler
        errors = []
        for mode in (
            ClassMethodMode(),
            StaticMethodMode(),
            instance_assigned,
            RaisingSelfMode(),
        ):
            try:
                with mode:
                    module.positive(tensor)
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                errors.append(None)

        class LateFailureDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                if self.lookups > 1:
                    raise RuntimeError("late override failure")
                return lambda func, types, args=(), kwargs=None: object()

        descriptor = LateFailureDescriptor()

        class Override:
            __torch_function__ = descriptor

        try:
            with ClassMethodMode():
                module.positive(Override())
        except Exception as error:
            precedence_error = (type(error).__name__, str(error))
        else:
            precedence_error = None
        return errors, calls, precedence_error, descriptor.lookups

    def test_mode_handler_form_validation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_handler_form_observation(torch),
            self.mode_handler_form_observation(reference_torch),
        )

    def warning_observation(self, module_name, scenario):
        source = r'''
import importlib
import json
import sys
import warnings

module = importlib.import_module(sys.argv[1])
scenario = sys.argv[2]
results = []

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    if scenario == "plain":
        class ModernOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "modern"

        class PlainOverride:
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return "plain"

        class AcceptingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return "mode"

        results.append(module.positive(ModernOverride()))
        with AcceptingMode():
            results.append(module.positive(PlainOverride()))
        results.append(module.positive(PlainOverride()))
        results.append(module.positive(PlainOverride()))
    elif scenario == "class-object":
        class ClassObject:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return "class-object"

        results.append(module.positive(ClassObject))
        results.append(module.positive(ClassObject))
    elif scenario == "callable-descriptor":
        class CallableHandler:
            def __init__(self, receiver):
                self.__self__ = receiver

            def __call__(self, func, types, args=(), kwargs=None):
                return "callable-descriptor"

        class Descriptor:
            def __get__(self, instance, owner):
                return CallableHandler(instance)

        class Override:
            __torch_function__ = Descriptor()

        results.append(module.positive(Override()))
        results.append(module.positive(Override()))
    else:
        raise AssertionError(scenario)

print(json.dumps({
    "results": results,
    "warnings": [
        [warning.category.__name__, str(warning.message)]
        for warning in caught
    ],
}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", source, module_name, scenario],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        return json.loads(completed.stdout)

    def test_plain_method_override_warnings_match_pytorch_2_13(self):
        for scenario in ("plain", "class-object", "callable-descriptor"):
            with self.subTest(scenario=scenario):
                actual = self.warning_observation("torch_rs", scenario)
                expected = self.warning_observation("torch", scenario)
                self.assertEqual(actual, expected)
                self.assertEqual(len(expected["warnings"]), 1)

    def test_not_implemented_error_matches_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.positive(Override()),
            lambda: reference_torch.positive(Override()),
        )

    def owner_observation(self, module):
        function = module.positive
        owner = function.__reduce__()[1][0]
        self.assertTrue(owner.__flags__ & (1 << 8))
        errors = []
        for action in (
            lambda: setattr(owner, "positive", None),
            lambda: delattr(owner, "positive"),
            lambda: setattr(owner, "marker", object()),
            lambda: delattr(owner, "marker"),
        ):
            try:
                action()
            except Exception as error:
                errors.append(
                    (
                        type(error).__name__,
                        str(error).replace("torch_rs._C", "torch._C"),
                    )
                )
            else:
                errors.append(None)
        return (
            errors,
            owner.positive is function,
            pickle.loads(pickle.dumps(function)) is function,
        )

    def test_owner_immutability_and_serialization_match_pytorch_2_13(self):
        self.assertEqual(
            self.owner_observation(torch),
            self.owner_observation(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
