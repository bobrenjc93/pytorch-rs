import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


SQRT_DOC = """
sqrt() -> Tensor

See :func:`torch.sqrt`
"""


class TensorSqrtTests(unittest.TestCase):
    def assert_tensor_bits(self, actual, expected_bits, *, shape, stride):
        self.assertEqual(tuple(actual.shape), shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), 0)
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
            np.asarray(expected_bits, dtype=np.uint32),
        )

    def test_float32_scalar_empty_offset_and_noncontiguous_tensors(self):
        base_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(base_values.tolist())
        strided = base.transpose(0, 2)

        cases = (
            ("scalar", torch.tensor(-0.0), (), ()),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1], (0, 2), (2, 1)),
            ("offset", strided[1], (3, 2), (1, 3)),
            ("noncontiguous", strided, (4, 3, 2), (1, 4, 12)),
        )
        for name, source, shape, stride in cases:
            expected = np.sqrt(np.asarray(source, dtype=np.float32))
            with self.subTest(name=name):
                self.assert_tensor_bits(
                    source.sqrt(),
                    expected.reshape(-1).view(np.uint32),
                    shape=shape,
                    stride=stride,
                )

    def test_float32_signed_zero_infinities_and_nans(self):
        input_bits = np.asarray(
            (
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x00800000,
                0x80800000,
                0x3F800000,
                0x40000000,
                0x40800000,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
                0x7F812345,
                0xFF812345,
                0x7FC12345,
                0xFFC54321,
            ),
            dtype=np.uint32,
        )
        expected_bits = np.asarray(
            (
                0x00000000,
                0x80000000,
                0x1A3504F3,
                0x7FC00000,
                0x20000000,
                0x7FC00000,
                0x3F800000,
                0x3FB504F3,
                0x40000000,
                0x5F7FFFFF,
                0x7FC00000,
                0x7F800000,
                0x7FC00000,
                0x7FC12345,
                0xFFC12345,
                0x7FC12345,
                0xFFC54321,
            ),
            dtype=np.uint32,
        )
        source = torch.tensor(memoryview(input_bits.view(np.float32)))

        self.assert_tensor_bits(
            source.sqrt(),
            expected_bits,
            shape=(len(input_bits),),
            stride=(1,),
        )

    def test_grad_recording_is_rejected_and_no_grad_is_supported(self):
        def make_cases():
            scalar = torch.tensor(4.0, requires_grad=True)
            empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]
            leaf = torch.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                requires_grad=True,
            )
            strided = leaf.transpose(0, 2)
            return (scalar, empty, strided[1], strided)

        for case, source in enumerate(make_cases()):
            with self.subTest(case=case, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^sqrt\(\): autograd recording is not supported$",
                ):
                    source.sqrt()

            with self.subTest(case=case, mode="no_grad"):
                expected = source.detach().sqrt()
                with torch.no_grad():
                    actual = source.sqrt()
                self.assert_tensor_bits(
                    actual,
                    np.asarray(expected, dtype=np.float32)
                    .reshape(-1)
                    .view(np.uint32),
                    shape=tuple(expected.shape),
                    stride=expected.stride(),
                )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([4.0])
        descriptor = inspect.getattr_static(torch.Tensor, "sqrt")
        bound = tensor.sqrt

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'sqrt' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "sqrt")
        self.assertEqual(descriptor.__qualname__, "TensorBase.sqrt")
        self.assertEqual(bound.__name__, "sqrt")
        self.assertEqual(bound.__qualname__, "Tensor.sqrt")
        self.assertEqual(descriptor.__doc__, SQRT_DOC)
        self.assertEqual(bound.__doc__, SQRT_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)

        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        cases = (
            (lambda: tensor.sqrt(1), "TensorBase.sqrt() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.sqrt() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.sqrt() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.sqrt(1, 2),
                "TensorBase.sqrt() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.sqrt(input=tensor),
                (
                    "Tensor.sqrt() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.sqrt() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.sqrt() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.sqrt() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.sqrt() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'sqrt' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.sqrt() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([4.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "sqrt")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.sqrt()
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

        plain = torch.tensor([4.0])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.sqrt()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [2.0])

    def test_inplace_dtype_and_method_out_extensions_remain_unsupported(self):
        self.assertFalse(hasattr(torch.Tensor, "sqrt_"))
        self.assertFalse(hasattr(torch, "float64"))

        tensor = torch.tensor([4.0])
        with self.assertRaisesRegex(
            TypeError,
            (
                r"^Tensor\.sqrt\(\) takes no keyword arguments$"
                if sys.version_info < (3, 11)
                else r"^TensorBase\.sqrt\(\) takes no keyword arguments$"
            ),
        ):
            tensor.sqrt(out=None)


if __name__ == "__main__":
    unittest.main()
