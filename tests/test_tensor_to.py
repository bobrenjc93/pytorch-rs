import inspect
import re
import types
import unittest

import torch_rs as torch


class TensorToTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        multi_output_view = tracked.unbind()[1]
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
        channels_last_3d = torch.zeros(
            (2, 3, 4, 5, 6), dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last_3d)
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
        self.assertTrue(
            channels_last_3d.is_contiguous(
                memory_format=torch.channels_last_3d
            )
        )
        self.assertEqual(multi_output_view.output_nr, 1)
        return leaf, tracked, (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            ("empty", torch.zeros((2, 0, 3), dtype=torch.float32)),
            ("contiguous", source),
            ("strided view", strided),
            ("offset strided view", offset),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
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

    def identity_calls(self):
        return (
            ("omitted", lambda tensor: tensor.to()),
            ("empty args", lambda tensor: tensor.to(*())),
            ("float32 positional", lambda tensor: tensor.to(torch.float32)),
            ("float alias positional", lambda tensor: tensor.to(torch.float)),
            ("float32 keyword", lambda tensor: tensor.to(dtype=torch.float32)),
            ("float alias keyword", lambda tensor: tensor.to(dtype=torch.float)),
            ("dtype none", lambda tensor: tensor.to(dtype=None)),
            ("device none", lambda tensor: tensor.to(device=None)),
            ("cpu string", lambda tensor: tensor.to("cpu")),
            ("cpu device", lambda tensor: tensor.to(torch.device("cpu"))),
            ("non-blocking keyword", lambda tensor: tensor.to(non_blocking=True)),
            ("copy false keyword", lambda tensor: tensor.to(copy=False)),
            ("none positional", lambda tensor: tensor.to(None)),
            ("none dtype positional", lambda tensor: tensor.to(None, None)),
            (
                "none then dtype keyword",
                lambda tensor: tensor.to(None, dtype=torch.float32),
            ),
            (
                "device dtype positional",
                lambda tensor: tensor.to("cpu", torch.float32),
            ),
            (
                "device dtype options positional",
                lambda tensor: tensor.to("cpu", torch.float32, True, False),
            ),
            (
                "keyword options",
                lambda tensor: tensor.to(
                    device="cpu",
                    dtype=torch.float32,
                    non_blocking=True,
                    copy=False,
                    memory_format=torch.preserve_format,
                ),
            ),
            (
                "preserve memory format",
                lambda tensor: tensor.to(memory_format=torch.preserve_format),
            ),
            (
                "memory format none",
                lambda tensor: tensor.to(memory_format=None),
            ),
            (
                "other tensor",
                lambda tensor: tensor.to(torch.tensor([1.0])),
            ),
            (
                "tensor keyword",
                lambda tensor: tensor.to(
                    tensor=torch.tensor([1.0]), non_blocking=True, copy=False
                ),
            ),
        )

    def test_supported_identity_forms_return_exact_receiver_without_mutation(self):
        leaf, tracked, cases = self.tensor_cases()
        descriptor = inspect.getattr_static(torch.Tensor, "to")

        for case, tensor in cases:
            calls = (
                *self.identity_calls(),
                ("descriptor", lambda tensor: descriptor(tensor)),
            )
            for form, call in calls:
                with self.subTest(
                    case=case,
                    form=form,
                    shape=tensor.shape,
                    stride=tensor.stride(),
                ):
                    before = self.metadata(tensor)
                    gradient = tensor.grad
                    result = call(tensor)

                    self.assertIs(result, tensor)
                    self.assertEqual(self.metadata(tensor), before)
                    self.assertIs(tensor.grad, gradient)

        self.assertIs(tracked.to(torch.float32), tracked)
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_contiguous_memory_format_is_identity_for_non_channel_last(self):
        tensors = (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1),
        )
        for tensor in tensors:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assertIs(
                    tensor.to(
                        dtype=torch.float32,
                        memory_format=torch.contiguous_format,
                    ),
                    tensor,
                )

    def test_matching_channels_last_memory_format_is_identity_only(self):
        cases = (
            (
                torch.zeros((2, 3, 4, 5)).contiguous(
                    memory_format=torch.channels_last
                ),
                torch.channels_last,
            ),
            (
                torch.zeros((2, 3, 4, 5, 6)).contiguous(
                    memory_format=torch.channels_last_3d
                ),
                torch.channels_last_3d,
            ),
        )
        for tensor, memory_format in cases:
            with self.subTest(memory_format=memory_format):
                self.assertIs(tensor.to(memory_format=memory_format), tensor)

        row_major = torch.zeros((2, 3, 4, 5))
        with self.assertRaisesRegex(
            NotImplementedError,
            "^to\\(\\): memory_format conversions are not supported; "
            "only no-copy identity is implemented$",
        ):
            row_major.to(memory_format=torch.channels_last)
        with self.assertRaisesRegex(
            NotImplementedError,
            "^to\\(\\): memory_format conversions are not supported; "
            "only no-copy identity is implemented$",
        ):
            cases[0][0].to(memory_format=torch.contiguous_format)

    def test_unsupported_conversions_do_not_mutate_the_source(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1)
        before = self.metadata(tensor)
        calls = (
            (
                lambda: tensor.to(copy=True),
                "^to\\(\\): copy=True requires a copy and is not supported$",
            ),
            (
                lambda: tensor.to("cpu:0"),
                "^to\\(\\): explicit indexed CPU devices require a copy "
                "and are not supported$",
            ),
            (
                lambda: tensor.to(torch.device("cpu:0")),
                "^to\\(\\): indexed CPU devices require a copy and are not "
                "supported$",
            ),
            (
                lambda: tensor.to(0),
                "^to\\(\\): device conversions are not supported; only "
                "unindexed CPU identity is implemented$",
            ),
            (
                lambda: tensor.to(True),
                "^to\\(\\): tensor target conversions are not supported; "
                "only exact native CPU float32 Tensor targets are supported$",
            ),
        )
        for call, message in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    (NotImplementedError, RuntimeError), message
                ):
                    call()
                self.assertEqual(self.metadata(tensor), before)

    def test_invalid_argument_combinations_raise_type_error(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        calls = (
            lambda: tensor.to(dtype=1),
            lambda: tensor.to(non_blocking=None),
            lambda: tensor.to(copy=0),
            lambda: tensor.to(memory_format=1),
            lambda: tensor.to(foo=1),
            lambda: tensor.to(torch.float32, dtype=torch.float32),
            lambda: tensor.to("cpu", device="cpu"),
            lambda: tensor.to(torch.float32, False, False, None),
            lambda: descriptor(tensor, "cpu", torch.float32, False, False, None),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    "^to\\(\\) (received an invalid combination|takes from 0 to 4)",
                ):
                    call()

    def test_tensorbase_descriptor_metadata_documentation_and_unbound_calls(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        bound = tensor.to

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'to' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "to")
        self.assertEqual(descriptor.__qualname__, "TensorBase.to")
        self.assertEqual(bound.__name__, "to")
        self.assertEqual(bound.__qualname__, "Tensor.to")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertIn("to(*args, **kwargs) -> Tensor", descriptor.__doc__)
        self.assertIn("Performs Tensor dtype and/or device conversion", bound.__doc__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertIs(bound.__self__, tensor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor)(), tensor)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(), tensor)

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.to() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'to' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.to() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_torch_function_modes_match_tensorbase_method_dispatch(self):
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
            result = tensor.to(torch.float32, non_blocking=True)
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, torch.float32))
        self.assertEqual(kwargs, {"non_blocking": True})

        unsupported_but_valid = RecordingMode(marker)
        with unsupported_but_valid:
            result = tensor.to(copy=True)
        self.assertIs(result, marker)
        self.assertEqual(len(unsupported_but_valid.calls), 1)

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaisesRegex(
                TypeError,
                "^to\\(\\) received an invalid combination of arguments",
            ):
                tensor.to(dtype=1)
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
                forwarded = tensor.to(
                    device="cpu",
                    dtype=torch.float32,
                    non_blocking=True,
                    copy=False,
                    memory_format=torch.contiguous_format,
                )
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

    def test_torch_function_override_arguments_receive_descriptor(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        cases = (
            ("positional", lambda override: tensor.to(override), None),
            ("tensor", lambda override: tensor.to(tensor=override), "tensor"),
            ("dtype", lambda override: tensor.to(dtype=override), "dtype"),
            ("device", lambda override: tensor.to(device=override), "device"),
            (
                "memory_format",
                lambda override: tensor.to(memory_format=override),
                "memory_format",
            ),
            ("copy", lambda override: tensor.to(copy=override), "copy"),
            (
                "non_blocking",
                lambda override: tensor.to(non_blocking=override),
                "non_blocking",
            ),
        )

        for name, call, keyword in cases:
            with self.subTest(name=name):
                override = Override()
                Override.calls.clear()

                self.assertIs(call(override), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (Override,))
                if keyword is None:
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[0], tensor)
                    self.assertIs(args[1], override)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(len(args), 1)
                    self.assertIs(args[0], tensor)
                    self.assertEqual(set(kwargs), {keyword})
                    self.assertIs(kwargs[keyword], override)

    def test_torch_function_mode_falls_through_to_argument_override(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()
        order = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append("override")
                cls.calls.append((func, types, args, kwargs))
                return marker

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append("mode")
                self.calls.append((func, types, args, kwargs))
                return NotImplemented

        mode = DecliningMode()
        override = Override()
        with mode:
            result = tensor.to(dtype=override)

        self.assertIs(result, marker)
        self.assertEqual(order, ["mode", "override"])
        self.assertEqual(len(mode.calls), 1)
        self.assertEqual(len(Override.calls), 1)
        for function, dispatch_types, args, kwargs in (
            mode.calls[0],
            Override.calls[0],
        ):
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(len(args), 1)
            self.assertIs(args[0], tensor)
            self.assertEqual(set(kwargs), {"dtype"})
            self.assertIs(kwargs["dtype"], override)

    def test_torch_function_override_arguments_notimplemented_error(self):
        tensor = torch.tensor([1.0])

        class DecliningOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return NotImplemented

        for call in (
            lambda override: tensor.to(override),
            lambda override: tensor.to(dtype=override),
        ):
            with self.subTest(call=call):
                override = DecliningOverride()
                DecliningOverride.calls.clear()
                with self.assertRaisesRegex(
                    TypeError,
                    r"^Multiple dispatch failed for 'torch\.Tensor\.to'; all "
                    r"__torch_function__ handlers returned NotImplemented:\n\n"
                    r"  - tensor subclass <class '.*DecliningOverride'>",
                ):
                    call(override)
                self.assertEqual(len(DecliningOverride.calls), 1)


if __name__ == "__main__":
    unittest.main()
