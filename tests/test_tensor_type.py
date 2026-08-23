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
                    tensor.type(None),
                    tensor.type(dtype=None),
                    tensor.type(non_blocking=False),
                    tensor.type(non_blocking=True),
                    tensor.type(None, True),
                    descriptor(tensor),
                )

                for result in results:
                    self.assertIs(type(result), str)
                    self.assertEqual(result, "torch.FloatTensor")
                self.assertEqual(self.metadata(tensor), before)

        tracked.type()
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_supported_conversions_return_the_exact_tensor_without_mutation(self):
        leaf, tracked, cases = self.tensor_cases()

        for case, tensor in cases:
            calls = (
                ("float32 positional", lambda: tensor.type(torch.float32)),
                ("float alias positional", lambda: tensor.type(torch.float)),
                ("legacy string positional", lambda: tensor.type("torch.FloatTensor")),
                ("float32 keyword", lambda: tensor.type(dtype=torch.float32)),
                ("float alias keyword", lambda: tensor.type(dtype=torch.float)),
                (
                    "legacy string keyword",
                    lambda: tensor.type(dtype="torch.FloatTensor"),
                ),
                (
                    "false positional non-blocking",
                    lambda: tensor.type(torch.float32, False),
                ),
                (
                    "true positional non-blocking",
                    lambda: tensor.type(torch.float32, True),
                ),
                (
                    "false keyword non-blocking",
                    lambda: tensor.type(
                        "torch.FloatTensor", non_blocking=False
                    ),
                ),
                (
                    "true keyword non-blocking",
                    lambda: tensor.type(dtype=torch.float, non_blocking=True),
                ),
            )
            for form, call in calls:
                with self.subTest(
                    case=case,
                    form=form,
                    shape=tensor.shape,
                    stride=tensor.stride(),
                ):
                    before = self.metadata(tensor)
                    result = call()
                    self.assertIs(result, tensor)
                    self.assertEqual(self.metadata(tensor), before)

        self.assertIs(tracked.type(torch.float32), tracked)
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

    def test_argument_errors_and_unsupported_targets_do_not_mutate(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type")
        before = self.metadata(tensor)
        calls = (
            (
                lambda: tensor.type(torch.float32, False, None),
                "type() takes from 0 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: descriptor(tensor, torch.float32, False, None),
                "type() takes from 0 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.type(torch.float32, dtype=torch.float32),
                "type() got multiple values for argument 'dtype'",
            ),
            (
                lambda: tensor.type(torch.float32, False, non_blocking=True),
                "type() got multiple values for argument 'non_blocking'",
            ),
            (
                lambda: tensor.type(torch.float32, 0),
                "type(): argument 'non_blocking' (position 2) must be bool, not int",
            ),
            (
                lambda: tensor.type(dtype=torch.float32, non_blocking=1),
                "type(): argument 'non_blocking' must be bool, not int",
            ),
            (
                lambda: tensor.type(torch.float32, non_blocking=None),
                "type(): argument 'non_blocking' must be bool, not NoneType",
            ),
            (
                lambda: tensor.type(**{"async": False}),
                "type() got an unexpected keyword argument 'async'",
            ),
            (
                lambda: tensor.type(torch.float32, **{"async": True}),
                "type() got an unexpected keyword argument 'async'",
            ),
            (
                lambda: tensor.type(unknown=True),
                "type() got an unexpected keyword argument 'unknown'",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(self.metadata(tensor), before)

        unsupported_targets = (
            "torch.DoubleTensor",
            "torch.cuda.FloatTensor",
            "torch.FloatTensor ",
            torch.Tensor,
            float,
            1,
            object(),
        )
        for target in unsupported_targets:
            with self.subTest(target=target):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^type\(\): only torch\.float32 and "
                    r"'torch\.FloatTensor' are supported$",
                ):
                    tensor.type(target)
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

        original_calls = (
            ((torch.float32,), None),
            (("torch.FloatTensor", True), None),
            ((), {"dtype": torch.float, "non_blocking": False}),
            ((object(),), {"non_blocking": True}),
        )
        for positional, keywords in original_calls:
            recording = RecordingMode(marker)
            with recording:
                if keywords is None:
                    intercepted = tensor.type(*positional)
                else:
                    intercepted = tensor.type(*positional, **keywords)
            self.assertIs(intercepted, marker)
            self.assertEqual(len(recording.calls), 1)
            function, dispatch_types, args, kwargs = recording.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), len(positional) + 1)
            self.assertIs(args[0], tensor)
            for actual, expected in zip(args[1:], positional, strict=True):
                self.assertIs(actual, expected)
            if keywords is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(kwargs, keywords)
                for name, value in keywords.items():
                    self.assertIs(kwargs[name], value)

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaisesRegex(
                TypeError,
                r"^type\(\): argument 'non_blocking' must be bool, not int$",
            ):
                tensor.type(torch.float32, non_blocking=0)
            with self.assertRaisesRegex(
                TypeError,
                r"^type\(\) got an unexpected keyword argument 'async'$",
            ):
                tensor.type(torch.float32, **{"async": False})
        self.assertEqual(rejected.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.type(
                    "torch.FloatTensor", non_blocking=True
                )
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

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
