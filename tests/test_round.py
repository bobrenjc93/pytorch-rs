import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

ROUND_DOC = """
round(decimals=0) -> Tensor

See :func:`torch.round`
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
        0x3F00_0001,
        0x3F7F_FFFF,
        0x3F80_0000,
        0x3FBF_FFFF,
        0x3FC0_0000,
        0x3FC0_0001,
        0x401F_FFFF,
        0x4020_0000,
        0x4020_0001,
        0xBF00_0000,
        0xBF00_0001,
        0xBFBF_FFFF,
        0xBFC0_0000,
        0xBFC0_0001,
        0xC01F_FFFF,
        0xC020_0000,
        0xC020_0001,
        0x4AFF_FFFF,
        0x4B00_0000,
        0x4B00_0001,
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
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0000,
        0x0000_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x3F80_0000,
        0x4000_0000,
        0x4000_0000,
        0x4000_0000,
        0x4000_0000,
        0x4040_0000,
        0x8000_0000,
        0xBF80_0000,
        0xBF80_0000,
        0xC000_0000,
        0xC000_0000,
        0xC000_0000,
        0xC000_0000,
        0xC040_0000,
        0x4B00_0000,
        0x4B00_0000,
        0x4B00_0001,
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


class TensorRoundTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-0.5, requires_grad=True)
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
            output = source.round()
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

            if case == "numerical edges":
                expected_bits = SPECIAL_OUTPUT_BITS
            else:
                values = np.asarray(source, dtype=np.float32).reshape(-1)
                expected_bits = np.rint(values).astype(np.float32).view(np.uint32)
            with self.subTest(case=case, values=True):
                np.testing.assert_array_equal(
                    self.tensor_bits(output), expected_bits
                )

    def test_active_autograd_is_rejected_before_output_planning(self):
        message = r"^round\(\): autograd recording is not supported$"
        for case, source in enumerate(self.make_tracked_cases()):
            with self.subTest(case=case):
                with self.assertRaisesRegex(RuntimeError, message):
                    source.round()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.round()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.round()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.round()
            with torch.no_grad():
                actual = source.round()
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

    def test_tensorbase_descriptor_metadata_and_no_argument_contract(self):
        tensor = torch.tensor([1.5])
        descriptor = inspect.getattr_static(torch.Tensor, "round")
        bound = tensor.round

        self.assertIs(torch.Tensor.round, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'round' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "round")
        self.assertEqual(descriptor.__qualname__, "TensorBase.round")
        self.assertEqual(bound.__name__, "round")
        self.assertEqual(bound.__qualname__, "Tensor.round")
        self.assertEqual(descriptor.__doc__, ROUND_DOC)
        self.assertEqual(bound.__doc__, ROUND_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)

        cases = (
            (
                lambda: tensor.round(0),
                "round() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: bound(0),
                "round() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: descriptor(tensor, 0),
                "round() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.round(0, 1),
                "round() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: tensor.round(decimals=0),
                "round() got an unexpected keyword argument 'decimals'",
            ),
            (
                lambda: bound(decimals=0),
                "round() got an unexpected keyword argument 'decimals'",
            ),
            (
                lambda: descriptor(tensor, decimals=0),
                "round() got an unexpected keyword argument 'decimals'",
            ),
            (
                lambda: tensor.round(out=None),
                "round() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.round() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'round' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.5], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "round")
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
            result = tracked.round()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
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
                forwarded = plain.round()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [2.0])

        order.clear()
        with self.assertRaisesRegex(
            RuntimeError, r"^round\(\): autograd recording is not supported$"
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tracked.round()
        self.assertEqual(order, ["upper", "lower"])

        declining = RecordingMode(NotImplemented)
        message = (
            "Multiple dispatch failed for 'torch.Tensor.round'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - mode object {declining!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with declining:
            with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                plain.round()
            self.assertEqual(
                len(torch.overrides._get_current_function_mode_stack()), 1
            )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        for call in (
            lambda: plain.round(0),
            lambda: plain.round(decimals=0),
        ):
            invalid = RecordingMode(marker)
            with self.assertRaises(TypeError):
                with invalid:
                    call()
            self.assertEqual(invalid.calls, [])

    def test_top_level_and_in_place_round_remain_unsupported(self):
        tensor = torch.tensor([1.5])
        self.assertFalse(hasattr(torch, "round"))
        self.assertNotIn("round", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "round_"))
        self.assertFalse(hasattr(tensor, "round_"))


if __name__ == "__main__":
    unittest.main()
