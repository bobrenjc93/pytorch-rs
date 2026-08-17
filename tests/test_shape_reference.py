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
            raise AssertionError(
                "Tensor.shape differentials require pinned PyTorch 2.13.0"
            )

    def make_cases(self, module):
        base = module.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            dtype=module.float32,
        )
        return (
            module.tensor(3.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            base[1],
            base.transpose(0, 2),
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
        )

    @staticmethod
    def value_contract(module, value):
        return (
            type(value).__name__,
            type(value) is module.Size,
            tuple(value),
            repr(value),
        )

    def shape_contract(self, module, tensor):
        metadata = (
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        values = tuple(
            self.value_contract(module, tensor.shape) for _ in range(3)
        )
        size_value = self.value_contract(module, tensor.size())
        return {
            "values": values,
            "size_value": size_value,
            "metadata_unchanged": metadata
            == (
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_scalar_empty_offset_strided_and_extreme_metadata_match(self):
        for actual, expected in zip(
            self.make_cases(torch), self.make_cases(reference_torch)
        ):
            with self.subTest(shape=tuple(expected.shape)):
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
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "shape")
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
            "has_text_signature": hasattr(descriptor, "__text_signature__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.shape is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": self.value_contract(
                module, descriptor.__get__(tensor, module.Tensor)
            ),
            "receiver_error": self.error(
                lambda: descriptor.__get__(1, int)
            ),
            "assignment_errors": tuple(self.error(action) for action in actions),
        }

    def test_descriptor_ownership_documentation_and_errors_match(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "shape")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
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
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_self": function.__self__ is descriptor,
            "function_equals_descriptor_get": function == descriptor.__get__,
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded": self.value_contract(module, forwarded),
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
