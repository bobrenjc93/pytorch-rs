import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


RSQRT_DOC = """
rsqrt() -> Tensor

See :func:`torch.rsqrt`
"""


class TensorRsqrtTests(unittest.TestCase):
    def assert_tensor_bits(self, actual, expected_bits, *, shape, stride, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, shape)
            self.assertEqual(actual.stride(), stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected_bits, dtype=np.uint32),
            )

    def test_ieee_values_layouts_and_fresh_storage(self):
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x4080_0000,
                0xC080_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        special_output_bits = np.asarray(
            (
                0x7F80_0000,
                0xFF80_0000,
                0x64B5_04F3,
                0xFFC0_0000,
                0x5F00_0001,
                0xFFC0_0000,
                0x5F00_0000,
                0xFFC0_0000,
                0x3FDD_B3D8,
                0xFFC0_0000,
                0x3F80_0000,
                0xFFC0_0000,
                0x3F00_0000,
                0xFFC0_0000,
                0x1F80_0001,
                0xFFC0_0000,
                0x0000_0000,
                0xFFC0_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        cases = (
            ("scalar", torch.tensor(-0.0), (), (), np.asarray([0xFF80_0000])),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (0, 2),
                (2, 1),
                np.asarray([], dtype=np.uint32),
            ),
            (
                "empty singleton trailing",
                torch.zeros((0, 1)),
                (0, 1),
                (1, 1),
                np.asarray([], dtype=np.uint32),
            ),
            (
                "empty singleton middle",
                torch.zeros((0, 1, 2)),
                (0, 1, 2),
                (2, 2, 1),
                np.asarray([], dtype=np.uint32),
            ),
            (
                "empty singleton surrounding",
                torch.zeros((1, 0, 1)),
                (1, 0, 1),
                (1, 1, 1),
                np.asarray([], dtype=np.uint32),
            ),
            ("offset", strided[1], (3, 2), (1, 3), None),
            ("noncontiguous", strided, (4, 3, 2), (1, 4, 12), None),
            (
                "numerical edges",
                torch.tensor(memoryview(special_input_bits.view(np.float32))),
                (len(special_input_bits),),
                (1,),
                special_output_bits,
            ),
        )

        for case, source, shape, stride, expected_bits in cases:
            source_bits = (
                np.asarray(source, dtype=np.float32).reshape(-1).view(np.uint32).copy()
            )
            if expected_bits is None:
                with np.errstate(all="ignore"):
                    expected_bits = (
                        np.float32(1.0)
                        / np.sqrt(np.asarray(source, dtype=np.float32))
                    ).reshape(-1).view(np.uint32)

            output = source.rsqrt()
            self.assert_tensor_bits(
                output,
                expected_bits,
                shape=shape,
                stride=stride,
                case=case,
            )
            self.assertFalse(output.is_set_to(source))
            np.testing.assert_array_equal(
                np.asarray(source, dtype=np.float32).reshape(-1).view(np.uint32),
                source_bits,
            )

    def test_grad_recording_is_rejected_before_planning_and_no_grad_is_allowed(self):
        leaf = torch.tensor(
            [[-4.0, -0.0, 1.0], [4.0, 9.0, 16.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        source_bits = np.asarray(source, dtype=np.float32).view(np.uint32).copy()

        with self.assertRaisesRegex(
            RuntimeError,
            r"^rsqrt\(\): autograd recording is not supported$",
        ):
            source.rsqrt()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^rsqrt\(\): autograd recording is not supported$",
        ):
            extreme.rsqrt()

        with torch.no_grad():
            actual = source.rsqrt()
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.rsqrt()
        expected = source.detach().rsqrt()
        self.assert_tensor_bits(
            actual,
            np.asarray(expected, dtype=np.float32).view(np.uint32),
            shape=expected.shape,
            stride=expected.stride(),
            case="no_grad",
        )
        self.assertFalse(actual.is_set_to(source))
        np.testing.assert_array_equal(
            np.asarray(source, dtype=np.float32).view(np.uint32), source_bits
        )

        detached = source.detach()
        self.assertFalse(detached.rsqrt().requires_grad)

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([4.0])
        descriptor = inspect.getattr_static(torch.Tensor, "rsqrt")
        bound = tensor.rsqrt

        self.assertIs(torch.Tensor.rsqrt, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'rsqrt' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "rsqrt")
        self.assertEqual(descriptor.__qualname__, "TensorBase.rsqrt")
        self.assertEqual(bound.__name__, "rsqrt")
        self.assertEqual(bound.__qualname__, "Tensor.rsqrt")
        self.assertEqual(descriptor.__doc__, RSQRT_DOC)
        self.assertEqual(bound.__doc__, RSQRT_DOC)
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
            (lambda: tensor.rsqrt(1), "TensorBase.rsqrt() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.rsqrt() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.rsqrt() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.rsqrt(1, 2),
                "TensorBase.rsqrt() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.rsqrt(input=tensor),
                (
                    "Tensor.rsqrt() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.rsqrt() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.rsqrt() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.rsqrt() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.rsqrt() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'rsqrt' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.rsqrt() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([4.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "rsqrt")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.rsqrt()
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
                forwarded = plain.rsqrt()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.5])

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^rsqrt\(\): autograd recording is not supported$",
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tensor.rsqrt()
        self.assertEqual(order, ["upper", "lower"])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                plain.rsqrt(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_inplace_and_out_forms_remain_unsupported(self):
        tensor = torch.tensor([4.0])
        self.assertFalse(hasattr(torch, "rsqrt"))
        self.assertFalse(hasattr(torch._C, "rsqrt"))
        self.assertNotIn("rsqrt", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "rsqrt_"))
        self.assertFalse(hasattr(tensor, "rsqrt_"))
        with self.assertRaises(TypeError):
            tensor.rsqrt(out=None)


if __name__ == "__main__":
    unittest.main()
