import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorReluTests(unittest.TestCase):
    def assert_tensor_bits_match(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    def test_no_mode_preserves_bits_layouts_and_fresh_storage(self):
        special_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x8000_0001,
                0x0000_0001,
                0xFF7F_FFFF,
                0x7F7F_FFFF,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        base = torch.tensor(memoryview(special_bits.view(np.float32))).reshape(3, 4)
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", torch.ones((2, 3, 4))[1]),
            ("strided", base.transpose(0, 1)),
            ("special values", base),
        )

        for case, source in cases:
            actual = source.relu()
            expected = torch.relu(source)
            self.assert_tensor_bits_match(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(source))
            self.assertFalse(expected.is_set_to(source))
            self.assertFalse(actual.is_set_to(expected))

    def test_autograd_name_repeated_backward_and_no_grad_are_unchanged(self):
        probability = torch.tensor([2.0], requires_grad=True).relu()
        with self.assertRaisesRegex(ValueError, r"grad_fn=<ReluBackward0>"):
            torch.nn.functional.dropout(
                torch.tensor([1.0]), p=probability, training=False
            )

        freed = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
        loss = freed.relu().sum()
        loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        accumulated = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
        accumulated.relu().sum().backward()
        first = np.asarray(accumulated.grad).copy()
        accumulated.relu().sum().backward()
        np.testing.assert_array_equal(np.asarray(accumulated.grad), first * 2.0)

        tracked = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
        with torch.no_grad():
            untracked = tracked.relu()
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertFalse(untracked.is_set_to(tracked))
        self.assertTrue(tracked.relu().requires_grad)

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([0.5])
        descriptor = inspect.getattr_static(torch.Tensor, "relu")
        bound = tensor.relu

        self.assertIs(torch.Tensor.relu, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'relu' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "relu")
        self.assertEqual(descriptor.__qualname__, "TensorBase.relu")
        self.assertEqual(bound.__name__, "relu")
        self.assertEqual(bound.__qualname__, "Tensor.relu")
        self.assertIsNone(descriptor.__doc__)
        self.assertIsNone(bound.__doc__)
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
            (lambda: tensor.relu(1), "TensorBase.relu() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.relu() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.relu() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.relu(1, 2),
                "TensorBase.relu() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.relu(input=tensor),
                (
                    "Tensor.relu() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.relu() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.relu() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.relu() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.relu() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'relu' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.relu() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_and_restore_stack(self):
        tensor = torch.tensor([-0.5, 0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "relu")
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
            result = tensor.relu()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.relu()
        bypass = RecordingMode(marker)
        with bypass:
            self.assertIs(extreme.relu(), marker)
        self.assertEqual(len(bypass.calls), 1)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.relu()
        self.assertEqual(order, ["upper", "lower"])
        forwarded.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(tensor.grad), np.asarray([0.0, 1.0], dtype=np.float32)
        )

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise ValueError("relu mode failed")

        order.clear()
        with ForwardingMode("lower"):
            with self.assertRaisesRegex(ValueError, "^relu mode failed$"):
                with RaisingMode():
                    tensor.relu()
            self.assertEqual(
                len(torch.overrides._get_current_function_mode_stack()), 1
            )
            recovered = tensor.relu()
        self.assertEqual(order, ["lower"])
        self.assertEqual(recovered.tolist(), tensor.detach().relu().tolist())
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    tensor.relu()
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
                tensor.relu(1)
        self.assertEqual(invalid.calls, [])

    def test_inplace_and_method_out_forms_remain_unsupported(self):
        tensor = torch.tensor([0.5])
        self.assertFalse(hasattr(torch.Tensor, "relu_"))
        self.assertFalse(hasattr(tensor, "relu_"))
        with self.assertRaises(TypeError):
            tensor.relu(out=None)


if __name__ == "__main__":
    unittest.main()
