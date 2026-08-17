import inspect
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorShapeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.shape differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        dense = module.tensor(
            [float(value) for value in range(60)], dtype=module.float32
        ).reshape(3, 4, 5)
        return (
            module.tensor(2.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            dense[1],
            dense.transpose(0, 2),
            module.zeros(
                (sys.maxsize, 0, sys.maxsize), dtype=module.float32
            ),
        )

    def shape_contract(self, module, tensor):
        metadata = (
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.numel(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        first = tensor.shape
        second = tensor.shape
        return {
            "canonical_type": type(first) is module.Size,
            "tuple_subtype": isinstance(first, tuple),
            "fresh": first is not second,
            "value": tuple(first),
            "repr": repr(first),
            "metadata_unchanged": metadata
            == (
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.numel(),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_scalar_empty_offset_strided_and_extreme_shapes_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.shape_contract(torch, actual),
                    self.shape_contract(reference_torch, expected),
                )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.shape unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "shape")
        tensor = module.zeros((2, 3), dtype=module.float32)
        actions = (
            lambda: setattr(tensor, "shape", (3, 2)),
            lambda: delattr(tensor, "shape"),
            lambda: descriptor.__set__(tensor, (3, 2)),
            lambda: descriptor.__delete__(tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.shape is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_type": type(descriptor.__get__(tensor, module.Tensor))
            is module.Size,
            "value": tuple(descriptor.__get__(tensor, module.Tensor)),
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_tensorbase_descriptor_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "shape")
        tensor = module.zeros((2, 3), dtype=module.float32)
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
            intercepted = tensor.shape
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.shape

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_equals_descriptor_get": function == descriptor.__get__,
            "function_self": function.__self__ is descriptor,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded_type": type(forwarded) is module.Size,
            "forwarded_value": tuple(forwarded),
        }

    def test_torch_function_mode_get_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
