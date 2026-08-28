import copy
import inspect
import pickle
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


EXP2_DOC = """
exp2() -> Tensor

See :func:`torch.exp2`
"""


class TensorExp2Tests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    def assert_result(
        self, output, source, expected_stride, *, case, expected_bits=None
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(output.shape, source.shape)
            self.assertEqual(output.stride(), expected_stride)
            self.assertEqual(output.storage_offset(), 0)
            self.assertFalse(output.requires_grad)
            self.assertTrue(output.is_leaf)
            self.assertIs(output.dtype, torch.float32)
            self.assertEqual(output.device, torch.device("cpu"))
            self.assertFalse(output.is_set_to(source))
            if source.numel():
                self.assertNotEqual(output.data_ptr(), source.data_ptr())

        with self.subTest(case=case, values=True):
            if expected_bits is not None:
                np.testing.assert_array_equal(
                    self.tensor_bits(output),
                    np.asarray(expected_bits, dtype=np.uint32),
                )
            else:
                with np.errstate(all="ignore"):
                    expected = np.exp2(
                        np.asarray(source, dtype=np.float32), dtype=np.float32
                    )
                np.testing.assert_allclose(
                    np.asarray(output, dtype=np.float32),
                    expected,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
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
                0x4000_0000,
                0xC000_0000,
                0x42FF_FFFF,
                0x4300_0000,
                0xC314_FFFF,
                0xC315_0000,
                0xC315_FFFF,
                0xC316_0000,
                0xC316_0001,
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
                0x3F80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x3FA1_4518,
                0x3F4B_2FF5,
                0x4000_0000,
                0x3F00_0000,
                0x4080_0000,
                0x3E80_0000,
                0x7F7F_FFA7,
                0x7F80_0000,
                0x0000_0001,
                0x0000_0001,
                0x0000_0001,
                0x0000_0000,
                0x0000_0000,
                0x7F80_0000,
                0x0000_0000,
                0x7F80_0000,
                0x0000_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        channels_last = torch.tensor(
            np.linspace(-3.0, 4.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-5.0, 6.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist()
        ).contiguous(memory_format=torch.channels_last_3d)
        return (
            ("scalar", torch.tensor(-0.0), (), None),
            (
                "empty offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 1),
                None,
            ),
            ("empty singleton trailing", torch.zeros((0, 1)), (1, 1), None),
            (
                "empty singleton middle",
                torch.zeros((0, 1, 2)),
                (2, 2, 1),
                None,
            ),
            (
                "empty singleton surrounding",
                torch.zeros((1, 0, 1)),
                (1, 1, 1),
                None,
            ),
            ("offset", strided[1], (1, 3), None),
            ("noncontiguous", strided, (1, 4, 12), None),
            ("channels last", channels_last, (60, 1, 15, 3), None),
            (
                "channels last 3d",
                channels_last_3d,
                (360, 1, 90, 18, 3),
                None,
            ),
            (
                "IEEE edges",
                torch.tensor(memoryview(special_input_bits.view(np.float32))),
                (1,),
                special_output_bits,
            ),
        )

    def test_values_layouts_offsets_empties_and_fresh_storage(self):
        for case, source, expected_stride, expected_bits in self.make_cases():
            source_bits = self.tensor_bits(source).copy()
            output = source.exp2()
            self.assert_result(
                output,
                source,
                expected_stride,
                case=case,
                expected_bits=expected_bits,
            )
            np.testing.assert_array_equal(self.tensor_bits(source), source_bits)

    def test_grad_recording_is_rejected_before_planning(self):
        leaf = torch.tensor(
            [[-2.0, -0.0], [1.0, 2.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        source_bits = self.tensor_bits(source).copy()

        with self.assertRaisesRegex(
            RuntimeError,
            r"^exp2\(\): autograd recording is not supported$",
        ):
            source.exp2()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^exp2\(\): autograd recording is not supported$",
        ):
            extreme.exp2()

        with torch.no_grad():
            output = source.exp2()
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.exp2()
        self.assert_result(output, source, (1, 2), case="no_grad")
        np.testing.assert_array_equal(self.tensor_bits(source), source_bits)

        detached = source.detach()
        detached_bits = self.tensor_bits(detached).copy()
        output = detached.exp2()
        self.assert_result(output, detached, (1, 2), case="detached")
        np.testing.assert_array_equal(self.tensor_bits(detached), detached_bits)

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([0.5])
        descriptor = inspect.getattr_static(torch.Tensor, "exp2")
        bound = tensor.exp2

        self.assertIs(torch.Tensor.exp2, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'exp2' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "exp2")
        self.assertEqual(descriptor.__qualname__, "TensorBase.exp2")
        self.assertEqual(bound.__name__, "exp2")
        self.assertEqual(bound.__qualname__, "Tensor.exp2")
        self.assertEqual(descriptor.__doc__, EXP2_DOC)
        self.assertEqual(bound.__doc__, EXP2_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(pickle.loads(pickle.dumps(descriptor, protocol)), descriptor)

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
            (lambda: tensor.exp2(1), "TensorBase.exp2() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.exp2() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.exp2() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.exp2(1, 2),
                "TensorBase.exp2() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.exp2(input=tensor),
                (
                    "Tensor.exp2() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.exp2() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.exp2() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.exp2() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.exp2() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'exp2' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.exp2() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "exp2")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.exp2()
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

        plain = torch.tensor([0.5])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.exp2()
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_allclose(forwarded.tolist(), [np.sqrt(2.0)], rtol=1.0e-6)

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^exp2\(\): autograd recording is not supported$",
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tensor.exp2()
        self.assertEqual(order, ["upper", "lower"])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                plain.exp2(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_inplace_dtype_and_device_extensions_remain_unsupported(self):
        tensor = torch.tensor([0.5])
        self.assertFalse(hasattr(torch, "exp2"))
        self.assertFalse(hasattr(torch._C, "exp2"))
        self.assertNotIn("exp2", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "exp2_"))
        self.assertFalse(hasattr(tensor, "exp2_"))
        self.assertFalse(hasattr(torch, "exp2_"))
        with self.assertRaises(TypeError):
            tensor.exp2(out=None)
        with self.assertRaises(TypeError):
            tensor.exp2(dtype=torch.float32)
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([0.5], device="cuda")


if __name__ == "__main__":
    unittest.main()
