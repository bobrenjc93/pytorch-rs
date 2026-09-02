import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


RECIPROCAL_DOC = """
reciprocal() -> Tensor

See :func:`torch.reciprocal`
"""


class TensorReciprocalTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32)
                .reshape(-1)
                .view(np.uint32),
            )

    def test_method_and_top_level_share_ieee_bits_layouts_and_fresh_storage(self):
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
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
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("empty singleton trailing", torch.zeros((0, 1))),
            ("empty singleton middle", torch.zeros((0, 1, 2))),
            ("empty singleton surrounding", torch.zeros((1, 0, 1))),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "numerical edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

        for case, source in cases:
            method_output = source.reciprocal()
            function_output = torch.reciprocal(source)
            self.assert_matches(method_output, function_output, case=case)
            self.assertFalse(method_output.is_set_to(source))
            self.assertFalse(function_output.is_set_to(source))
            self.assertFalse(method_output.is_set_to(function_output))

    def test_autograd_records_vjp_and_no_grad_is_allowed(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        weights = torch.tensor([2.0, -3.0])

        output = source.reciprocal()
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.stride(), (1,))
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(-1).view(np.uint32),
            np.asarray(
                [
                    0.0,
                    -np.inf,
                    0.0,
                    0.0,
                    np.float32(3.0 / 16.0),
                    0.0,
                ],
                dtype=np.float32,
            ).view(np.uint32),
        )

        probability = torch.tensor([-1.0], requires_grad=True).reciprocal()
        with self.assertRaisesRegex(ValueError, r"grad_fn=<ReciprocalBackward0>"):
            torch.nn.functional.dropout(
                torch.tensor([1.0]), p=probability, training=False
            )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Stride calculation overflowed",
        ):
            extreme.reciprocal()

        with torch.no_grad():
            actual = source.reciprocal()
            expected = torch.reciprocal(source)
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.reciprocal()
        self.assert_matches(actual, expected, case="no_grad")
        self.assertFalse(actual.is_set_to(source))

        detached = source.detach()
        self.assert_matches(
            detached.reciprocal(), torch.reciprocal(detached), case="detached"
        )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([4.0])
        descriptor = inspect.getattr_static(torch.Tensor, "reciprocal")
        bound = tensor.reciprocal

        self.assertIs(torch.Tensor.reciprocal, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'reciprocal' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "reciprocal")
        self.assertEqual(descriptor.__qualname__, "TensorBase.reciprocal")
        self.assertEqual(bound.__name__, "reciprocal")
        self.assertEqual(bound.__qualname__, "Tensor.reciprocal")
        self.assertEqual(descriptor.__doc__, RECIPROCAL_DOC)
        self.assertEqual(bound.__doc__, RECIPROCAL_DOC)
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
            (
                lambda: tensor.reciprocal(1),
                "TensorBase.reciprocal() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.reciprocal() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.reciprocal() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.reciprocal(1, 2),
                "TensorBase.reciprocal() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.reciprocal(input=tensor),
                (
                    "Tensor.reciprocal() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.reciprocal() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.reciprocal() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.reciprocal() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.reciprocal() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'reciprocal' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.reciprocal() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([4.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "reciprocal")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.reciprocal()
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
                forwarded = plain.reciprocal()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.25])

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked = tensor.reciprocal()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked.requires_grad)
        self.assertEqual(tracked.tolist(), [0.25])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                plain.reciprocal(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_inplace_out_dtype_and_device_forms_remain_unsupported(self):
        tensor = torch.tensor([4.0])
        self.assertFalse(hasattr(torch.Tensor, "reciprocal_"))
        self.assertFalse(hasattr(tensor, "reciprocal_"))
        self.assertFalse(hasattr(torch, "reciprocal_"))
        self.assertNotIn("reciprocal_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.reciprocal(out=None)
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([4.0], device="cuda")


if __name__ == "__main__":
    unittest.main()
