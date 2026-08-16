import inspect
import re
import sys
import types
import unittest
import warnings

import torch_rs as torch


METHOD_DOC = (
    "\nReturns true if this tensor resides in pinned memory.\n"
    "By default, the device pinned memory on will be the current "
    ":ref:`accelerator<accelerators>`.\n"
)


class TensorIsPinnedTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
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
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        channels_last = torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )
        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        return leaf, tracked, (
            ("scalar", torch.tensor(-3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("contiguous", source),
            ("channels last", channels_last),
            ("strided view", strided),
            ("offset strided view", offset),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("autograd non-leaf view", tracked),
            ("detached autograd view", tracked.detach()),
            ("no-grad output", no_grad_output),
            ("no-grad view", no_grad_view),
        )

    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def test_all_supported_storage_states_are_pageable_and_unchanged(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)

                first = tensor.is_pinned()
                second = tensor.is_pinned()

                self.assertIs(type(first), bool)
                self.assertIs(first, False)
                self.assertIs(second, False)
                self.assertEqual(self.metadata(tensor), metadata)

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        gradient_metadata = self.metadata(leaf.grad)
        self.assertIs(leaf.grad.is_pinned(), False)
        self.assertEqual(self.metadata(leaf.grad), gradient_metadata)
        self.assertIs(tracked.is_pinned(), False)

    def test_tensorbase_descriptor_metadata_and_documentation_match(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_pinned")
        bound = tensor.is_pinned

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'is_pinned' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "is_pinned")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_pinned")
        self.assertEqual(bound.__name__, "is_pinned")
        self.assertEqual(bound.__qualname__, "Tensor.is_pinned")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), False)
        self.assertIs(bound(), False)

    def test_errors_match_and_deprecated_device_argument_is_excluded(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_pinned")
        bound = tensor.is_pinned
        cases = (
            (
                lambda: tensor.is_pinned(1),
                "is_pinned() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: bound(1),
                "is_pinned() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: descriptor(tensor, 1),
                "is_pinned() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.is_pinned(1, 2),
                "is_pinned() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: tensor.is_pinned(unexpected=True),
                "is_pinned() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.is_pinned(1, unexpected=True),
                "is_pinned() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.is_pinned(device=None),
                "is_pinned() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.is_pinned() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'is_pinned' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.is_pinned() needs an argument",
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for call, message in cases:
                with self.subTest(message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
        self.assertEqual(caught, [])

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_pinned")
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
            result = tensor.is_pinned()
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
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
                forwarded = tensor.is_pinned()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.is_pinned()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.is_pinned'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaises(TypeError):
                tensor.is_pinned(device=None)
        self.assertEqual(rejected.calls, [])


if __name__ == "__main__":
    unittest.main()
