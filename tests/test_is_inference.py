import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nis_inference() -> bool\n\n"
    "See :func:`torch.is_inference`\n"
)


class TensorIsInferenceTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        produced = leaf * 2.0
        tracked_view = produced.transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        return leaf, tracked_view, (
            ("ordinary leaf", source),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("ordinary strided view", strided_view),
            ("ordinary offset view", offset_view),
            ("autograd non-leaf view", tracked_view),
            ("detached autograd view", tracked_view.detach()),
            ("empty", torch.zeros((2, 0, 3))),
            ("extreme empty view", extreme_empty),
            ("no-grad output", no_grad_output),
            ("no-grad view", no_grad_view),
        )

    def test_supported_tensors_are_ordinary_without_mutation(self):
        leaf, tracked_view, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )

                result = tensor.is_inference()

                self.assertIs(type(result), bool)
                self.assertIs(result, False)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

        tracked_view.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        self.assertIs(leaf.is_inference(), False)
        self.assertIs(tracked_view.is_inference(), False)

    def test_tensorbase_descriptor_metadata_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_inference")
        bound = tensor.is_inference

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'is_inference' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "is_inference")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_inference")
        self.assertEqual(bound.__name__, "is_inference")
        self.assertEqual(bound.__qualname__, "Tensor.is_inference")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(
                    callable_object.__text_signature__, "($self, /)"
                )
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), False)

    def test_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_inference")
        bound = tensor.is_inference
        cases = (
            (
                lambda: tensor.is_inference(1),
                "TensorBase.is_inference() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.is_inference() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.is_inference() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_inference(1, 2),
                "TensorBase.is_inference() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_inference(input=tensor),
                "TensorBase.is_inference() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.is_inference() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.is_inference() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.is_inference() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'is_inference' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.is_inference() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_inference")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.is_inference()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.is_inference()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)


if __name__ == "__main__":
    unittest.main()
