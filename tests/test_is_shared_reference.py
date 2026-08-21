import copy
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
        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)
        return leaf, tracked, (
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
            no_grad_output,
            no_grad_view,
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
        result = tensor.is_shared()
        return {
            "result": result,
            "result_type": type(result).__name__,
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

    def test_cpu_views_empties_and_gradient_storage_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                actual_contract = self.contract(actual)
                self.assertEqual(actual_contract, self.contract(expected))
                self.assertIs(actual_contract["result"], False)

        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())
        self.assertEqual(
            self.contract(actual_leaf.grad), self.contract(expected_leaf.grad)
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_shared unexpectedly accepted an invalid call")

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        function = inspect.getattr_static(module.Tensor, "is_shared")
        bound = tensor.is_shared
        calls = (
            lambda: function(),
            lambda: function(tensor, tensor),
            lambda: bound(tensor),
            lambda: function(tensor, unexpected=True),
            lambda: bound(unexpected=True),
            lambda: bound(self=tensor),
            lambda: function(1),
        )
        return {
            "function_type": type(function).__name__,
            "bound_type": type(bound).__name__,
            "repr_shape": bool(
                re.fullmatch(
                    r"<function Tensor\.is_shared at 0x[0-9a-f]+>",
                    repr(function),
                )
            ),
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__.replace("torch_rs", "torch"),
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_module": bound.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "bound_doc": bound.__doc__,
            "annotations": function.__annotations__,
            "bound_annotations": bound.__annotations__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "bound_has_text_signature": hasattr(bound, "__text_signature__"),
            "signatures": (
                str(inspect.signature(function)),
                str(inspect.signature(bound)),
            ),
            "owned_by_tensor": "is_shared" in module.Tensor.__dict__,
            "absent_from_bases": all(
                "is_shared" not in owner.__dict__
                for owner in module.Tensor.__mro__[1:]
            ),
            "module_tensor_identity": module._tensor.Tensor is module.Tensor,
            "module_function_identity": module._tensor.Tensor.is_shared is function,
            "copies_are_identical": (
                copy.copy(function) is function,
                copy.deepcopy(function) is function,
                pickle.loads(pickle.dumps(function)) is function,
            ),
            "keyword_self_result": function(self=tensor),
            "bound_result": bound(),
            "errors": tuple(self.error(call) for call in calls),
            "types_match": (
                type(function) is types.FunctionType,
                type(bound) is types.MethodType,
            ),
        }

    def test_python_ownership_signature_documentation_and_errors_match(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def spoofed_receiver_contract(self, module):
        function = inspect.getattr_static(module.Tensor, "is_shared")
        events = []

        class Storage:
            def _is_shared(self):
                events.append("_is_shared")
                return "storage-result"

        class SpoofedTensor:
            @property
            def __class__(self):
                events.append("__class__")
                return module.Tensor

            def _typed_storage(self):
                events.append("_typed_storage")
                return Storage()

        value = SpoofedTensor()
        appears_to_be_tensor = isinstance(value, module.Tensor)
        events.clear()
        result = function(value)
        return appears_to_be_tensor, result, events

    def test_spoofed_tensor_class_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.spoofed_receiver_contract(torch),
            self.spoofed_receiver_contract(reference_torch),
        )

    def override_contract(self, module):
        function = inspect.getattr_static(module.Tensor, "is_shared")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        result = function(value)
        called_function, dispatch_types, args, kwargs = Override.calls[0]
        return {
            "result": result is marker,
            "call_count": len(Override.calls),
            "function_identity": called_function is function,
            "function_type": type(called_function).__name__,
            "types": dispatch_types == (Override,),
            "args": args == (value,),
            "kwargs": kwargs,
        }

    def test_unary_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.override_contract(torch),
            self.override_contract(reference_torch),
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
        called_function, dispatch_types, args, kwargs = recording.calls[0]

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
            declining_error = (
                type(error).__name__,
                re.sub(
                    r"0x[0-9a-f]+",
                    "0x...",
                    str(error).replace("torch_rs", "torch"),
                ),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_identity": called_function is function,
            "function_type": type(called_function).__name__,
            "function_module": called_function.__module__.replace(
                "torch_rs", "torch"
            ),
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs": kwargs,
            "forwarding_order": order,
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
