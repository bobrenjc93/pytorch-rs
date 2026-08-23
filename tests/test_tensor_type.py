import inspect
import re
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\ntype(dtype=None, non_blocking=False, **kwargs) -> str or Tensor\n"
    "Returns the type if `dtype` is not provided, else casts this object to\n"
    "the specified type.\n\n"
    "If this is already of the correct type, no copy is performed and the\n"
    "original object is returned.\n\n"
    "Args:\n"
    "    dtype (dtype or string): The desired type\n"
    "    non_blocking (bool): If ``True``, and the source is in pinned memory\n"
    "        and destination is on the GPU or vice versa, the copy is performed\n"
    "        asynchronously with respect to the host. Otherwise, the argument\n"
    "        has no effect.\n"
    "    **kwargs: For compatibility, may contain the key ``async`` in place of\n"
    "        the ``non_blocking`` argument. The ``async`` arg is deprecated.\n"
)


class TensorTypeTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        multi_output_view = produced.unbind()[1]
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
        channels_last = torch.zeros(
            (2, 3, 4, 5), dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        gradient_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        (gradient_leaf * 4.0).sum().backward()
        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertEqual(multi_output_view.output_nr, 1)
        return leaf, tracked, (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            ("empty", torch.zeros((2, 0, 3), dtype=torch.float32)),
            ("contiguous", source),
            ("channels last", channels_last),
            ("strided view", strided),
            ("offset strided view", offset),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("autograd non-leaf view", tracked),
            ("multi-output autograd view", multi_output_view),
            ("detached autograd view", tracked.detach()),
            ("accumulated gradient", gradient_leaf.grad),
            ("no-grad output", no_grad_output),
            ("no-grad view", no_grad_view),
        )

    def metadata(self, tensor):
        return (
            tensor.tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def test_every_supported_tensor_reports_cpu_float32_without_mutation(self):
        leaf, tracked, cases = self.tensor_cases()
        descriptor = inspect.getattr_static(torch.Tensor, "type")

        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                before = self.metadata(tensor)
                results = (
                    tensor.type(),
                    tensor.type(*()),
                    tensor.type(**{}),
                    descriptor(tensor),
                )

                for result in results:
                    self.assertIs(type(result), str)
                    self.assertEqual(result, "torch.FloatTensor")
                self.assertEqual(self.metadata(tensor), before)

        tracked.type()
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_tensorbase_descriptor_metadata_and_bound_unbound_calls(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type")
        bound = tensor.type

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'type' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "type")
        self.assertEqual(descriptor.__qualname__, "TensorBase.type")
        self.assertEqual(bound.__name__, "type")
        self.assertEqual(bound.__qualname__, "Tensor.type")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
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
        self.assertIs(bound.__self__, tensor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(
            descriptor.__get__(tensor, torch.Tensor)(), "torch.FloatTensor"
        )
        self.assertEqual(descriptor(tensor), "torch.FloatTensor")
        self.assertEqual(bound(), "torch.FloatTensor")

    def test_receiver_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type")
        calls = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.type() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'type' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(None),
                "descriptor 'type' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'NoneType' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.type() needs an argument",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_conversion_and_non_blocking_forms_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type")
        before = self.metadata(tensor)
        calls = (
            (
                lambda: tensor.type(torch.float32),
                "type() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.type("torch.FloatTensor"),
                "type() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.type(None, False),
                "type() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: descriptor(tensor, torch.float32),
                "type() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.type(dtype=torch.float32),
                "type() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: tensor.type(non_blocking=False),
                "type() got an unexpected keyword argument 'non_blocking'",
            ),
            (
                lambda: tensor.type(non_blocking=True),
                "type() got an unexpected keyword argument 'non_blocking'",
            ),
            (
                lambda: tensor.type(**{"async": False}),
                "type() got an unexpected keyword argument 'async'",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(self.metadata(tensor), before)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type")
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
            intercepted = tensor.type()
        self.assertIs(intercepted, marker)
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
                forwarded = tensor.type()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded, "torch.FloatTensor")

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.type()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.type'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
