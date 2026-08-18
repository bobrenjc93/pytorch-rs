import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorToTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        return (
            leaf,
            tracked,
            (
                ("scalar", torch.tensor(-3.5)),
                ("empty", torch.zeros((2, 0, 3))),
                ("contiguous", source),
                ("strided", strided),
                ("offset", offset),
                ("leaf", leaf),
                ("non-leaf", tracked),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

    def test_no_argument_call_returns_exact_receiver_for_all_supported_tensors(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.layout,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                bits = self.value_bits(tensor)

                result = tensor.to()

                self.assertIs(result, tensor)
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr(),
                        result.dtype,
                        result.device,
                        result.layout,
                        result.requires_grad,
                        result.is_leaf,
                    ),
                    metadata,
                )
                if bits is not None:
                    np.testing.assert_array_equal(self.value_bits(result), bits)

        tracked.to().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )

    def test_no_grad_preserves_existing_graph_and_no_grad_tensor_identity(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)

        with torch.no_grad():
            tracked_result = tracked.to()
            no_grad_tensor = leaf * 5.0
            no_grad_result = no_grad_tensor.to()

        self.assertIs(tracked_result, tracked)
        self.assertTrue(tracked_result.requires_grad)
        self.assertFalse(tracked_result.is_leaf)
        self.assertIs(no_grad_result, no_grad_tensor)
        self.assertFalse(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)

        tracked_result.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 3.0, dtype=np.float32)
        )

    def test_tensorbase_metadata_documentation_errors_and_pickle(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        bound = tensor.to

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'to' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "to")
        self.assertEqual(descriptor.__qualname__, "TensorBase.to")
        self.assertEqual(bound.__name__, "to")
        self.assertEqual(bound.__qualname__, "Tensor.to")
        self.assertEqual(len(descriptor.__doc__), 4025)
        self.assertTrue(descriptor.__doc__.startswith("\nto(*args, **kwargs) -> Tensor\n"))
        self.assertTrue(
            descriptor.__doc__.endswith(
                "            [ 0.3310, -0.0584]], dtype=torch.float64, "
                "device='cuda:0')\n"
            )
        )
        self.assertEqual(bound.__doc__, descriptor.__doc__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(), tensor)

        reduction = descriptor.__reduce__()
        self.assertIs(reduction[0], getattr)
        self.assertIs(reduction[1][0], descriptor.__objclass__)
        self.assertEqual(reduction[1][1], "to")
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )

        errors = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.to() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'to' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in errors:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_descriptor_pickle_remains_stable_across_package_reload(self):
        descriptor = inspect.getattr_static(torch.Tensor, "to")

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(inspect.getattr_static(torch.Tensor, "to"), descriptor)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )

    def test_torch_function_modes_match_tensorbase_dispatch(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            result = tensor.to()
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        recording = RecordingMode(marker)
        with recording:
            result = tensor.to(**{})
        self.assertIs(result, marker)
        self.assertEqual(recording.calls[0][3], {})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.to()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.to()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.to'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_conversion_and_option_forms_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        calls = (
            lambda: tensor.to(torch.float32),
            lambda: tensor.to(dtype=torch.float32),
            lambda: tensor.to(torch.device("cpu")),
            lambda: tensor.to(device=torch.device("cpu")),
            lambda: tensor.to(other),
            lambda: tensor.to(copy=False),
            lambda: tensor.to(copy=True),
            lambda: tensor.to(non_blocking=False),
            lambda: tensor.to(non_blocking=True),
            lambda: tensor.to(memory_format=None),
            lambda: tensor.to(memory_format=torch.preserve_format),
            lambda: descriptor(tensor, torch.float32),
            lambda: descriptor(tensor, copy=False),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        self.assertFalse(hasattr(torch, "to"))


if __name__ == "__main__":
    unittest.main()
