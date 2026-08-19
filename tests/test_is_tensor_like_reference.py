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
class IsTensorLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_tensor_like differentials require pinned PyTorch 2.13.0"
            )

    def lookup_observation(self, module):
        events = []

        class ClassHook:
            __torch_function__ = None

        class InstanceHook:
            pass

        class DynamicHook:
            def __getattr__(self, name):
                events.append(("getattr", name))
                if name == "__torch_function__":
                    return None
                raise AttributeError(name)

        class Descriptor:
            def __init__(self, outcome):
                self.outcome = outcome

            def __get__(self, instance, owner):
                events.append((self.outcome, instance is not None, owner.__name__))
                if self.outcome == "missing":
                    raise AttributeError("hook is hidden")
                if self.outcome == "failing":
                    raise RuntimeError("hook lookup failed")
                return None

        class ReturningHook:
            __torch_function__ = Descriptor("returning")

        class MissingHook:
            __torch_function__ = Descriptor("missing")

        class FailingHook:
            __torch_function__ = Descriptor("failing")

        instance_hook = InstanceHook()
        instance_hook.__torch_function__ = None
        values = (
            module.Tensor,
            module.tensor([1.0]),
            ClassHook(),
            instance_hook,
            DynamicHook(),
            ReturningHook(),
            MissingHook(),
            None,
            1,
            object(),
        )
        results = tuple(module.overrides.is_tensor_like(value) for value in values)
        try:
            module.overrides.is_tensor_like(FailingHook())
        except Exception as error:
            failure = (type(error).__name__, str(error))
        else:
            self.fail(f"{module.__name__}.overrides.is_tensor_like ignored failure")
        return results, events, failure

    def test_values_and_attribute_lookup_match_pytorch_2_13(self):
        self.assertEqual(
            self.lookup_observation(torch),
            self.lookup_observation(reference_torch),
        )

    def tensor_binding_observation(self, module):
        function = module.overrides.is_tensor_like
        native_tensor_type = module.Tensor
        native_tensor = module.tensor([1.0])
        baseline = (function(native_tensor_type), function(native_tensor))
        try:
            module.Tensor = int
            rebound_to_type = (
                function(native_tensor_type),
                function(native_tensor),
                function(int),
                function(1),
                function("one"),
            )

            module.Tensor = 42
            rebound_to_object = (
                function(native_tensor_type),
                function(native_tensor),
                function(int),
                function(1),
            )
        finally:
            module.Tensor = native_tensor_type
        restored = (function(native_tensor_type), function(native_tensor))
        return baseline, rebound_to_type, rebound_to_object, restored

    def test_native_tensor_class_and_rebinding_match_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_binding_observation(torch),
            self.tensor_binding_observation(reference_torch),
        )

    def subclass_observation(self, module):
        class HookBase:
            __torch_function__ = None

        class InheritedHook(HookBase):
            pass

        class MaskedHook(HookBase):
            @property
            def __torch_function__(self):
                raise AttributeError("masked")

        return (
            module.overrides.is_tensor_like(InheritedHook()),
            module.overrides.is_tensor_like(MaskedHook()),
        )

    def test_subclass_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.subclass_observation(torch),
            self.subclass_observation(reference_torch),
        )

    def mode_observation(self, module):
        calls = []
        tensor = module.tensor([1.0])

        class Hook:
            __torch_function__ = None

        class Mode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                calls.append((func, dispatch_types, args, kwargs))
                return "intercepted"

        with Mode():
            results = (
                module.overrides.is_tensor_like(tensor),
                module.overrides.is_tensor_like(Hook()),
                module.overrides.is_tensor_like(object()),
            )
        return results, calls

    def test_active_mode_independence_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_observation(torch),
            self.mode_observation(reference_torch),
        )

    def metadata_observation(self, module):
        function = module.overrides.is_tensor_like
        wildcard_namespace = {}
        exec(
            f"from {module.__name__}.overrides import *",
            wildcard_namespace,
        )
        return {
            "is_function": type(function) is types.FunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "annotations": function.__annotations__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "signature": str(inspect.signature(function)),
            "exported": "is_tensor_like" in module.overrides.__all__,
            "wildcard_identity": wildcard_namespace["is_tensor_like"] is function,
            "top_level_exported": hasattr(module, "is_tensor_like"),
            "pickle_identity": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_metadata_exports_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_observation(torch),
            self.metadata_observation(reference_torch),
        )

    def error_observation(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("invalid is_tensor_like call succeeded")

    def test_call_binding_matches_pytorch_2_13(self):
        actual = torch.overrides.is_tensor_like
        expected = reference_torch.overrides.is_tensor_like
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(1, 2), lambda: expected(1, 2)),
            (lambda: actual(input=1), lambda: expected(input=1)),
            (lambda: actual(1, inp=2), lambda: expected(1, inp=2)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(
                    self.error_observation(actual_call),
                    self.error_observation(expected_call),
                )


if __name__ == "__main__":
    unittest.main()
