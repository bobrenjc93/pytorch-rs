import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


CEIL_DOC = """
ceil() -> Tensor

See :func:`torch.ceil`
"""


SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
        0xBF7F_FFFF,
        0xBF80_0000,
        0xBFC0_0000,
        0x3FC0_0000,
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

SPECIAL_OUTPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x3F80_0000,
        0x8000_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x8000_0000,
        0x8000_0000,
        0xBF80_0000,
        0xBF80_0000,
        0x4000_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC1_2345,
        0xFFC1_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)


def make_cases(module):
    base = module.tensor(
        np.linspace(-3.75, 3.75, 24, dtype=np.float32)
        .reshape(2, 3, 4)
        .tolist(),
        dtype=module.float32,
    )
    strided = base.transpose(0, 2)
    channels_last = module.tensor(
        np.linspace(-15.0, 15.0, 120, dtype=np.float32)
        .reshape(2, 3, 4, 5)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last)
    channels_last_3d = module.tensor(
        np.linspace(-90.0, 90.0, 720, dtype=np.float32)
        .reshape(2, 3, 4, 5, 6)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last_3d)
    return (
        ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
        (
            "empty offset",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            (2, 1),
        ),
        ("empty singleton trailing", module.zeros((0, 1)), (1, 1)),
        ("empty singleton middle", module.zeros((0, 1, 2)), (2, 2, 1)),
        ("empty singleton surrounding", module.zeros((1, 0, 1)), (1, 1, 1)),
        ("offset", strided[1], (1, 3)),
        ("noncontiguous", strided, (1, 4, 12)),
        ("channels last", channels_last, channels_last.stride()),
        ("channels last 3d", channels_last_3d, channels_last_3d.stride()),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            (1,),
        ),
    )


class TensorCeilTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        return scalar, empty, strided[1], strided

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            output = source.ceil()
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

            actual_bits = self.tensor_bits(output)
            if case == "numerical edges":
                expected_bits = SPECIAL_OUTPUT_BITS
            else:
                values = np.asarray(source, dtype=np.float32).reshape(-1)
                expected_bits = np.ceil(values).astype(np.float32).view(np.uint32)
            with self.subTest(case=case, values=True):
                np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_active_autograd_is_rejected_before_output_planning(self):
        message = r"^ceil\(\): autograd recording is not supported$"
        for case, source in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                with self.assertRaisesRegex(RuntimeError, message):
                    source.ceil()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.ceil()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.ceil()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.ceil()
            with torch.no_grad():
                actual = source.ceil()
            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertFalse(actual.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )
            with self.subTest(case=case, mode="detached"):
                self.assertFalse(expected.is_set_to(detached))
                if detached.numel():
                    self.assertNotEqual(expected.data_ptr(), detached.data_ptr())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "ceil")
        bound = tensor.ceil

        self.assertIs(torch.Tensor.ceil, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'ceil' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "ceil")
        self.assertEqual(descriptor.__qualname__, "TensorBase.ceil")
        self.assertEqual(bound.__name__, "ceil")
        self.assertEqual(bound.__qualname__, "Tensor.ceil")
        self.assertEqual(descriptor.__doc__, CEIL_DOC)
        self.assertEqual(bound.__doc__, CEIL_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.ceil(1), "TensorBase.ceil() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.ceil() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.ceil() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.ceil(1, 2),
                "TensorBase.ceil() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.ceil(input=tensor),
                (
                    "Tensor.ceil() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.ceil() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.ceil() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.ceil() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.ceil() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'ceil' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.ceil() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "ceil")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tracked.ceil()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
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
                forwarded = plain.ceil()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [2.0])

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError, r"^ceil\(\): autograd recording is not supported$"
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tracked.ceil()
        self.assertEqual(order, ["upper", "lower"])

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.ceil()
                self.assertEqual(
                    len(torch.overrides._get_current_function_mode_stack()), 1
                )
        finally:
            sys.setrecursionlimit(old_recursion_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.ceil(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertFalse(hasattr(torch, "ceil"))
        self.assertNotIn("ceil", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "ceil_"))
        self.assertFalse(hasattr(tensor, "ceil_"))
        self.assertFalse(hasattr(torch, "ceil_"))
        self.assertNotIn("ceil_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.ceil(out=None)


if __name__ == "__main__":
    unittest.main()
