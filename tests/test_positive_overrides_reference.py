import pickle
import re
import unittest
import warnings

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
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                class_calls.append((func, types, args, kwargs))
                return marker

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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
