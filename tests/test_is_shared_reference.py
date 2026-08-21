import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsSharedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.is_shared() differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        tracked.sum().backward()
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        channels_last = module.zeros(
            (2, 3, 4, 5), dtype=module.float32
        ).contiguous(memory_format=module.channels_last)
        return (
            module.tensor(-3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            source,
            channels_last,
            strided,
            offset,
            extreme_empty,
            leaf,
            produced,
            tracked,
            tracked.detach(),
            leaf.grad,
        )

    def contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        first = tensor.is_shared()
        second = tensor.is_shared()
        return {
            "first": first,
            "first_type": type(first).__name__,
            "second_is_false": second is False,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_supported_cpu_storage_matches_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                actual_contract = self.contract(actual)
                self.assertEqual(actual_contract, self.contract(expected))
                self.assertIs(actual_contract["first"], False)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_shared unexpectedly accepted the invalid call")

    def callable_contract(self, module):
        tensor_module = importlib.import_module(f"{module.__name__}._tensor")
        tensor = module.tensor([1.0], dtype=module.float32)
        function = inspect.getattr_static(module.Tensor, "is_shared")
        bound = tensor.is_shared
        calls = (
            lambda: tensor.is_shared(1),
            lambda: bound(1, 2),
            lambda: function(tensor, 1),
            lambda: tensor.is_shared(input=tensor),
            lambda: bound(unexpected=True),
            lambda: function(tensor, unexpected=True),
            lambda: function(),
            lambda: function(1),
            lambda: function(tensor, self=tensor),
        )
        normalized_module = function.__module__.replace("torch_rs", "torch")
        normalized_repr = re.sub(r"0x[0-9a-f]+", "0x...", repr(function))
        return {
            "function_type": type(function).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(function) is types.FunctionType,
                type(bound) is types.MethodType,
            ),
            "function_repr": normalized_repr,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": normalized_module,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_module": bound.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "bound_doc": bound.__doc__,
            "signature": str(inspect.signature(function)),
            "bound_signature": str(inspect.signature(bound)),
            "function_has_text_signature": hasattr(
                function, "__text_signature__"
            ),
            "bound_has_text_signature": hasattr(bound, "__text_signature__"),
            "annotations": function.__annotations__,
            "bound_annotations": bound.__annotations__,
            "function_dict": function.__dict__,
            "bound_dict": bound.__dict__,
            "class_dict_identity": module.Tensor.__dict__["is_shared"]
            is function,
            "base_absent": "is_shared" not in module.Tensor.__base__.__dict__,
            "class_identity": module.Tensor.is_shared is function,
            "module_identity": tensor_module.Tensor.is_shared is function,
            "module_level_absent": not hasattr(tensor_module, "is_shared"),
            "descriptor_identity": function.__get__(None, module.Tensor)
            is function,
            "bound_function_identity": bound.__func__ is function,
            "bound_receiver_identity": bound.__self__ is tensor,
            "function_result": function(tensor),
            "bound_result": bound(),
            "keyword_self_result": function(self=tensor),
            "call_errors": tuple(self.error(call) for call in calls),
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol))
                is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_python_ownership_signature_documentation_and_errors_match(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        function = inspect.getattr_static(module.Tensor, "is_shared")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            intercepted = tensor.is_shared()
        dispatched_function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.is_shared()

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor.is_shared()
        except Exception as error:
            normalized_error = str(error).replace(module.__name__, "torch")
            decline_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", normalized_error),
            )
        else:
            self.fail("a declining mode unexpectedly handled Tensor.is_shared")

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(dispatched_function).__name__,
            "function_name": dispatched_function.__name__,
            "function_qualname": dispatched_function.__qualname__,
            "function_module": dispatched_function.__module__.replace(
                "torch_rs", "torch"
            ),
            "function_identity": dispatched_function is function,
            "dispatch_types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs": kwargs,
            "forwarding_order": order,
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
            "declining_calls": len(declining.calls),
            "decline_error": decline_error,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def override_contract(self, module):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = module.Tensor.is_shared(value)
        function, dispatch_types, args, kwargs = Override.calls[0]
        return {
            "result": result is marker,
            "calls": len(Override.calls),
            "function_identity": function is module.Tensor.is_shared,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__.replace("torch_rs", "torch"),
            "types": dispatch_types == (Override,),
            "args": len(args) == 1 and args[0] is value,
            "kwargs": kwargs,
        }

    def test_operand_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.override_contract(torch),
            self.override_contract(reference_torch),
        )

    def duck_typed_receiver_contract(self, module):
        marker = object()
        events = []

        class Storage:
            def _is_shared(self):
                events.append("storage._is_shared")
                return marker

        class SpoofedTensor:
            @property
            def __class__(self):
                events.append("__class__")
                return module.Tensor

            def _typed_storage(self):
                events.append("_typed_storage")
                return Storage()

        spoofed = SpoofedTensor()
        spoofed_isinstance = isinstance(spoofed, module.Tensor)
        events.clear()
        spoofed_result = module.Tensor.is_shared(spoofed)
        spoofed_events = tuple(events)

        class ClassAccessError(Exception):
            pass

        class RaisingClass:
            @property
            def __class__(self):
                events.append("__class__")
                raise ClassAccessError("receiver class must not be read")

            def _typed_storage(self):
                events.append("_typed_storage")
                return Storage()

        raising = RaisingClass()
        try:
            isinstance(raising, module.Tensor)
        except Exception as error:
            class_probe_error = type(error).__name__, str(error)
        else:
            self.fail("the raising __class__ property was not consulted")
        events.clear()
        raising_result = module.Tensor.is_shared(raising)

        return {
            "spoofed_isinstance": spoofed_isinstance,
            "spoofed_result": spoofed_result is marker,
            "spoofed_events": spoofed_events,
            "class_probe_error": class_probe_error,
            "raising_result": raising_result is marker,
            "raising_events": tuple(events),
        }

    def test_duck_typed_receiver_fallback_matches_pytorch_2_13(self):
        self.assertEqual(
            self.duck_typed_receiver_contract(torch),
            self.duck_typed_receiver_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
