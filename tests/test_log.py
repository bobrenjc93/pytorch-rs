import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


LOG_DOC = """
log() -> Tensor

See :func:`torch.log`
"""


class TensorLogTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, shape, stride):
        self.assertEqual(tuple(actual.shape), shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), 0)
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32).reshape(-1),
            np.asarray(expected, dtype=np.float32).reshape(-1),
            rtol=2.0e-6,
            atol=np.nextafter(np.float32(0), np.float32(1)),
            equal_nan=True,
        )

    def assert_tensor_bits(self, actual, expected_bits):
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
        base_values = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(base_values.tolist())
        strided = base.transpose(0, 2)

        cases = (
            ("scalar", torch.tensor(1.0), (), ()),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1], (0, 2), (2, 1)),
            ("offset", strided[1], (3, 2), (1, 3)),
            ("noncontiguous", strided, (4, 3, 2), (1, 4, 12)),
        )
        for name, source, shape, stride in cases:
            with np.errstate(divide="ignore", invalid="ignore"):
                expected = np.log(np.asarray(source, dtype=np.float32))
            with self.subTest(name=name):
                self.assert_tensor_matches(
                    source.log(), expected, shape=shape, stride=stride
                )

    def test_float32_zero_domain_infinities_and_nans(self):
        input_bits = np.asarray(
            (
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x007FFFFF,
                0x807FFFFF,
                0x00800000,
                0x80800000,
                0x3F000000,
                0x3F800000,
                0x40000000,
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
                0xFF800000,
                0xFF800000,
                0xC2CE8ED0,
                0x7FC00000,
                0xC2AEAC50,
                0x7FC00000,
                0xC2AEAC50,
                0x7FC00000,
                0xBF317218,
                0x00000000,
                0x3F317218,
                0x42B17218,
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

        output = source.log()
        self.assertEqual(tuple(output.shape), (len(input_bits),))
        self.assertEqual(output.stride(), (1,))
        self.assert_tensor_bits(output, expected_bits)

    def test_grad_recording_is_rejected_and_no_grad_is_supported(self):
        def make_cases():
            scalar = torch.tensor(1.0, requires_grad=True)
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
                    r"^log\(\): autograd recording is not supported$",
                ):
                    source.log()

            with self.subTest(case=case, mode="no_grad"):
                expected = source.detach().log()
                with torch.no_grad():
                    actual = source.log()
                self.assert_tensor_bits(
                    actual,
                    np.asarray(expected, dtype=np.float32)
                    .reshape(-1)
                    .view(np.uint32),
                )
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "log")
        bound = tensor.log

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'log' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "log")
        self.assertEqual(descriptor.__qualname__, "TensorBase.log")
        self.assertEqual(bound.__name__, "log")
        self.assertEqual(bound.__qualname__, "Tensor.log")
        self.assertEqual(descriptor.__doc__, LOG_DOC)
        self.assertEqual(bound.__doc__, LOG_DOC)
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
            (lambda: tensor.log(1), "TensorBase.log() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.log() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.log() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.log(1, 2),
                "TensorBase.log() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.log(input=tensor),
                (
                    "Tensor.log() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.log() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.log() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.log() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.log() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'log' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.log() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "log")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.log()
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

        plain = torch.tensor([1.0])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.log()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.0])

    def test_top_level_inplace_dtype_and_method_out_extensions_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "log"))
        self.assertFalse(hasattr(torch.Tensor, "log_"))
        self.assertFalse(hasattr(torch, "float64"))

        tensor = torch.tensor([1.0])
        with self.assertRaisesRegex(
            TypeError,
            (
                r"^Tensor\.log\(\) takes no keyword arguments$"
                if sys.version_info < (3, 11)
                else r"^TensorBase\.log\(\) takes no keyword arguments$"
            ),
        ):
            tensor.log(out=None)


if __name__ == "__main__":
    unittest.main()
