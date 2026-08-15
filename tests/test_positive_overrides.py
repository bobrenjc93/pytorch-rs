import pickle
import re
import subprocess
import sys
import types
import unittest

import torch_rs as torch


class TopLevelPositiveOverrideTests(unittest.TestCase):
    def make_override(self, result):
        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return result

        return Override(), Override

    def test_torch_function_override_receives_original_call_forms(self):
        marker = object()
        value, override_type = self.make_override(marker)
        calls = (
            ("positional", lambda: torch.positive(value), (value,), None),
            (
                "input",
                lambda: torch.positive(input=value),
                (),
                {"input": value},
            ),
            ("x", lambda: torch.positive(x=value), (), {"x": value}),
            ("a", lambda: torch.positive(a=value), (), {"a": value}),
            ("x1", lambda: torch.positive(x1=value), (), {"x1": value}),
        )

        for form, call, expected_args, expected_kwargs in calls:
            with self.subTest(form=form):
                override_type.calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(override_type.calls), 1)
                function, types, args, kwargs = override_type.calls[0]
                self.assertIs(function, torch.positive)
                self.assertEqual(types, (override_type,))
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

    def test_binding_errors_are_resolved_before_override_dispatch(self):
        marker = object()
        value, override_type = self.make_override(marker)
        cases = (
            (
                lambda: torch.positive(value, value),
                "positive() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.positive(value, input=value),
                "positive() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.positive(value, extra=True),
                "positive() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.positive(extra=value, input=value),
                "positive() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.positive(extra=value),
                'positive() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.positive(a=value, x=value),
                "positive() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.positive(x=value, a=value),
                "positive() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.positive(input=value, x1=value),
                "positive() got an unexpected keyword argument 'x1'",
            ),
            (
                lambda: torch.positive(x=value, x1=value),
                "positive() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.positive(x1=value, x=value),
                "positive() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                override_type.calls.clear()
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()
                self.assertEqual(override_type.calls, [])

    def test_instance_class_and_stateful_descriptor_overrides(self):
        marker = object()
        instance_calls = []

        class InstanceAssigned:
            pass

        instance = InstanceAssigned()

        def instance_handler(func, types, args=(), kwargs=None):
            instance_calls.append((func, types, args, kwargs))
            return marker

        instance.__torch_function__ = instance_handler
        self.assertIs(torch.positive(x1=instance), marker)
        self.assertEqual(
            instance_calls,
            [
                (
                    torch.positive,
                    (InstanceAssigned,),
                    (),
                    {"x1": instance},
                )
            ],
        )

        class_calls = []

        class ClassObject:
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                class_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.positive(ClassObject), marker)
        self.assertEqual(
            class_calls,
            [(torch.positive, (ClassObject,), (ClassObject,), None)],
        )

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

        value = DescriptorOverride()
        self.assertEqual(torch.positive(value), 2)
        self.assertEqual(descriptor.lookups, 2)
        self.assertEqual(
            descriptor_calls,
            [(torch.positive, (DescriptorOverride,), (value,), None)],
        )

    def test_failed_override_probe_is_retried_before_type_rejection(self):
        calls = []

        class RetryDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                resolution = self.lookups
                if resolution == 1:
                    raise AttributeError("transient probe failure")

                def handler(func, dispatch_types, args=(), kwargs=None):
                    calls.append((resolution, func, dispatch_types, args, kwargs))
                    return resolution

                return handler

        descriptor = RetryDescriptor()

        class Override:
            __torch_function__ = descriptor

        value = Override()
        self.assertEqual(torch.positive(value), 3)
        self.assertEqual(descriptor.lookups, 3)
        self.assertEqual(
            calls,
            [(3, torch.positive, (Override,), (value,), None)],
        )

    def test_torch_function_modes_receive_calls_and_forward_to_lower_modes(self):
        tensor = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            ("positional", lambda: torch.positive(tensor), (tensor,), None),
            (
                "input",
                lambda: torch.positive(input=tensor),
                (),
                {"input": tensor},
            ),
            ("x", lambda: torch.positive(x=tensor), (), {"x": tensor}),
            ("a", lambda: torch.positive(a=tensor), (), {"a": tensor}),
            ("x1", lambda: torch.positive(x1=tensor), (), {"x1": tensor}),
        )
        for form, call, expected_args, expected_kwargs in calls:
            with self.subTest(form=form):
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                self.assertEqual(len(mode.calls), 1)
                function, types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.positive)
                self.assertEqual(types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                self.assertIs(torch.positive(tensor), tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(torch.positive(tensor), tensor)

    def test_binding_errors_are_resolved_before_mode_dispatch(self):
        tensor = torch.tensor([1.0])

        class Mode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        cases = (
            (
                lambda: torch.positive(),
                'positive() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.positive(tensor, tensor),
                "positive() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.positive(extra=tensor),
                'positive() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.positive(1, extra=True),
                "positive(): argument 'input' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                mode = Mode()
                with mode:
                    with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                        call()
                self.assertEqual(mode.calls, [])

    def test_mode_not_implemented_falls_through_to_object_override(self):
        marker = object()
        value, override_type = self.make_override(marker)

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return NotImplemented

        mode = DecliningMode()
        with mode:
            self.assertIs(torch.positive(value), marker)
        self.assertEqual(len(mode.calls), 1)
        self.assertEqual(len(override_type.calls), 1)
        self.assertEqual(mode.calls[0][1], (override_type,))

    def test_mode_dispatch_precedes_late_override_resolution(self):
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

                def handler(func, types, args=(), kwargs=None):
                    raise AssertionError("the accepting mode must handle the call")

                return handler

        descriptor = StatefulDescriptor()

        class Override:
            __torch_function__ = descriptor

        value = Override()

        class AcceptingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(("mode", types, args, kwargs))
                return marker

        with AcceptingMode():
            self.assertIs(torch.positive(value), marker)
        self.assertEqual(descriptor.lookups, 1)
        self.assertEqual(events[0], ("lookup", 1))
        self.assertEqual(events[1][0], "mode")

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

        class MutatingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append("mode")
                mutable.__torch_function__ = replacement_handler
                return NotImplemented

        with MutatingMode():
            self.assertIs(torch.positive(mutable), marker)
        self.assertEqual(calls, ["mode", "replacement"])

    def test_mode_handler_is_resolved_again_for_invocation(self):
        calls = []

        class StatefulModeDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                resolution = self.lookups

                def handler(self, func, dispatch_types, args=(), kwargs=None):
                    calls.append(
                        (
                            resolution,
                            self is instance,
                            func,
                            dispatch_types,
                            args,
                            kwargs,
                        )
                    )
                    return resolution

                return types.MethodType(handler, instance)

        descriptor = StatefulModeDescriptor()

        class Mode(torch.overrides.TorchFunctionMode):
            __torch_function__ = descriptor

        tensor = torch.tensor([1.0])
        with Mode():
            self.assertEqual(torch.positive(tensor), 2)
        self.assertEqual(descriptor.lookups, 2)
        self.assertEqual(
            calls,
            [(2, True, torch.positive, (), (tensor,), None)],
        )

    def test_mode_handler_form_validation_matches_pytorch(self):
        tensor = torch.tensor([1.0])
        message = (
            "Defining your mode's `__torch_function__` as a classmethod is "
            "not supported, please make it a plain method"
        )
        calls = []

        class ClassMethodMode(torch.overrides.TorchFunctionMode):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append("classmethod")
                return object()

        class StaticMethodMode(torch.overrides.TorchFunctionMode):
            @staticmethod
            def __torch_function__(func, types, args=(), kwargs=None):
                calls.append("staticmethod")
                return object()

        class InstanceAssignedMode(torch.overrides.TorchFunctionMode):
            pass

        instance_assigned = InstanceAssignedMode()

        def instance_handler(func, types, args=(), kwargs=None):
            calls.append("instance")
            return object()

        instance_assigned.__torch_function__ = instance_handler
        for mode in (ClassMethodMode(), StaticMethodMode(), instance_assigned):
            with self.subTest(mode=type(mode).__name__):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    with mode:
                        torch.positive(tensor)
        self.assertEqual(calls, [])

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

        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
            with ClassMethodMode():
                torch.positive(Override())
        self.assertEqual(descriptor.lookups, 1)

    def test_plain_method_override_warning_is_once_only(self):
        source = r'''
import warnings

import torch_rs as torch


class ModernOverride:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return "modern"


class PlainOverride:
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return "plain"


class AcceptingMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return "mode"


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    assert torch.positive(ModernOverride()) == "modern"
    with AcceptingMode():
        assert torch.positive(PlainOverride()) == "mode"
    assert torch.positive(PlainOverride()) == "plain"
    assert torch.positive(PlainOverride()) == "plain"

assert len(caught) == 1
assert caught[0].category is UserWarning
assert str(caught[0].message).startswith(
    "Defining your `__torch_function__` as a plain method is deprecated "
)
'''
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_not_implemented_error_and_mode_cleanup_match_pytorch(self):
        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.positive'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            "  - tensor subclass <class "
            f"'{DecliningOverride.__module__}.{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.positive(DecliningOverride())

        tensor = torch.tensor([1.0])

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise RuntimeError("mode failure")

        with self.assertRaisesRegex(RuntimeError, "^mode failure$"):
            with RaisingMode():
                torch.positive(tensor)
        self.assertIs(torch.positive(tensor), tensor)

    def test_variable_function_owner_is_immutable_and_pickle_safe(self):
        function = torch.positive
        owner = function.__reduce__()[1][0]
        self.assertTrue(owner.__flags__ & (1 << 8))
        actions = (
            ("positive", lambda: setattr(owner, "positive", None)),
            ("positive", lambda: delattr(owner, "positive")),
            ("marker", lambda: setattr(owner, "marker", object())),
            ("marker", lambda: delattr(owner, "marker")),
        )
        for attribute, action in actions:
            with self.subTest(attribute=attribute):
                message = (
                    f"cannot set '{attribute}' attribute of immutable type "
                    "'torch_rs._C._VariableFunctionsClass'"
                )
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    action()
                self.assertIs(owner.positive, function)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

    def test_module_reinitialization_reuses_the_callable_owner(self):
        source = r"""
import importlib
import pickle
import sys

first = importlib.import_module("torch_rs")
first_function = first.positive
first_owner = first._C._VariableFunctionsClass
for name in tuple(sys.modules):
    if name == "torch_rs" or name.startswith("torch_rs."):
        del sys.modules[name]

second = importlib.import_module("torch_rs")
assert second.positive is first_function
assert second._C._VariableFunctionsClass is first_owner

class Override:
    calls = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append(func)
        return "override"

assert second.positive(Override()) == "override"
assert Override.calls == [second.positive]
assert pickle.loads(pickle.dumps(Override.calls[0])) is second.positive
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
