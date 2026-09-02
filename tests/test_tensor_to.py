import inspect
import re
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\nto(*args, **kwargs) -> Tensor\n\n"
    "Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are\n"
    "inferred from the arguments of ``self.to(*args, **kwargs)``.\n\n"
    ".. note::\n\n"
    "    If the ``self`` Tensor already\n"
    "    has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.\n"
    "    Otherwise, the returned tensor is a copy of ``self`` with the desired\n"
    "    :class:`torch.dtype` and :class:`torch.device`.\n\n"
    ".. note::\n\n"
    "    If ``self`` requires gradients (``requires_grad=True``) but the target\n"
    "    ``dtype`` specified is an integer type, the returned tensor will implicitly\n"
    "    set ``requires_grad=False``. This is because only tensors with\n"
    "    floating-point or complex dtypes can require gradients.\n\n"
    "Here are the ways to call ``to``:\n\n"
    ".. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor\n"
    "   :noindex:\n\n"
    "    Returns a Tensor with the specified :attr:`dtype`\n\n"
    "    Args:\n"
    "        memory_format (:class:`torch.memory_format`, optional): the desired memory format of\n"
    "        returned Tensor. Default: ``torch.preserve_format``.\n\n"
    ".. note::\n\n"
    "    According to `C++ type conversion rules <https://en.cppreference.com/w/cpp/language/implicit_conversion.html>`_,\n"
    "    converting floating point value to integer type will truncate the fractional part.\n"
    "    If the truncated value cannot fit into the target type (e.g., casting ``torch.inf`` to ``torch.long``),\n"
    "    the behavior is undefined and the result may vary across platforms.\n\n"
    ".. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor\n"
    "   :noindex:\n\n"
    "    Returns a Tensor with the specified :attr:`device` and (optional)\n"
    "    :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.\n"
    "    When :attr:`non_blocking` is set to ``True``, the function attempts to perform\n"
    "    the conversion asynchronously with respect to the host, if possible. This\n"
    "    asynchronous behavior applies to both pinned and pageable memory. However,\n"
    "    caution is advised when using this feature. For more information, refer to the\n"
    "    `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.\n"
    "    When :attr:`copy` is set, a new Tensor is created even when the Tensor\n"
    "    already matches the desired conversion.\n\n"
    "    Args:\n"
    "        memory_format (:class:`torch.memory_format`, optional): the desired memory format of\n"
    "        returned Tensor. Default: ``torch.preserve_format``.\n\n"
    ".. method:: to(other, non_blocking=False, copy=False) -> Tensor\n"
    "   :noindex:\n\n"
    "    Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as\n"
    "    the Tensor :attr:`other`.\n"
    "    When :attr:`non_blocking` is set to ``True``, the function attempts to perform\n"
    "    the conversion asynchronously with respect to the host, if possible. This\n"
    "    asynchronous behavior applies to both pinned and pageable memory. However,\n"
    "    caution is advised when using this feature. For more information, refer to the\n"
    "    `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.\n"
    "    When :attr:`copy` is set, a new Tensor is created even when the Tensor\n"
    "    already matches the desired conversion.\n\n"
    "Example::\n\n"
    "    >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu\n"
    "    >>> tensor.to(torch.float64)\n"
    "    tensor([[-0.5044,  0.0005],\n"
    "            [ 0.3310, -0.0584]], dtype=torch.float64)\n\n"
    "    >>> cuda0 = torch.device('cuda:0')\n"
    "    >>> tensor.to(cuda0)\n"
    "    tensor([[-0.5044,  0.0005],\n"
    "            [ 0.3310, -0.0584]], device='cuda:0')\n\n"
    "    >>> tensor.to(cuda0, dtype=torch.float64)\n"
    "    tensor([[-0.5044,  0.0005],\n"
    "            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')\n\n"
    "    >>> other = torch.randn((), dtype=torch.float64, device=cuda0)\n"
    "    >>> tensor.to(other, non_blocking=True)\n"
    "    tensor([[-0.5044,  0.0005],\n"
    "            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')\n"
)


class TensorToTests(unittest.TestCase):
    def test_default_equivalent_requests_return_exact_receiver(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        leaf.sum().backward()
        cases = (
            torch.tensor(-0.0),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ]
            ).transpose(0, 1)[1],
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
            leaf,
            tracked,
            leaf.grad,
        )
        other = torch.tensor([1.0])

        for case, tensor in enumerate(cases):
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                gradient = tensor.grad
                results = (
                    tensor.to(),
                    tensor.to(None),
                    tensor.to(torch.float32),
                    tensor.to(torch.float),
                    tensor.to(dtype=torch.float32),
                    tensor.to(dtype=torch.float),
                    tensor.to(dtype=None),
                    tensor.to("cpu"),
                    tensor.to(torch.device("cpu")),
                    tensor.to(device="cpu"),
                    tensor.to(device=torch.device("cpu")),
                    tensor.to(device=None),
                    tensor.to(None, torch.float32),
                    tensor.to("cpu", torch.float32, True, False),
                    tensor.to(device="cpu", dtype=torch.float32),
                    tensor.to(torch.float32, non_blocking=True, copy=False),
                    tensor.to(memory_format=None),
                    tensor.to(memory_format=torch.preserve_format),
                    tensor.to(memory_format=torch.contiguous_format),
                    tensor.to(other),
                    tensor.to(tensor=other),
                )
                for result in results:
                    self.assertIs(result, tensor)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )
                self.assertIs(tensor.grad, gradient)

    def test_existing_channel_last_format_is_identity(self):
        cases = (
            (
                torch.ones((2, 3, 4, 5)).clone(memory_format=torch.channels_last),
                torch.channels_last,
            ),
            (
                torch.ones((2, 3, 4, 5, 6)).clone(
                    memory_format=torch.channels_last_3d
                ),
                torch.channels_last_3d,
            ),
        )
        for tensor, memory_format in cases:
            with self.subTest(memory_format=memory_format):
                self.assertTrue(tensor.is_contiguous(memory_format=memory_format))
                self.assertIs(tensor.to(memory_format=memory_format), tensor)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "^torch_rs\\.Tensor\\.to only supports no-copy CPU float32 identity conversions$",
                ):
                    tensor.to(memory_format=torch.contiguous_format)

    def test_copying_or_non_identity_requests_are_rejected(self):
        tensor = torch.zeros((2, 3, 4, 5))
        cases = (
            lambda: tensor.to(copy=True),
            lambda: tensor.to(torch.float32, False, True),
            lambda: tensor.to("cpu:0"),
            lambda: tensor.to(torch.device("cpu", 0)),
            lambda: tensor.to(memory_format=torch.channels_last),
        )
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "^torch_rs\\.Tensor\\.to only supports no-copy CPU float32 identity conversions$",
                ):
                    call()

    def test_argument_binding_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: tensor.to(object()),
                r"^to\(\) received an invalid combination of arguments",
            ),
            (
                lambda: tensor.to(non_blocking=0),
                r"^to\(\) received an invalid combination of arguments",
            ),
            (
                lambda: tensor.to(copy=0),
                r"^to\(\) received an invalid combination of arguments",
            ),
            (
                lambda: tensor.to(memory_format=1),
                r"^to\(\) received an invalid combination of arguments",
            ),
            (
                lambda: tensor.to(unexpected=True),
                r"^to\(\) received an invalid combination of arguments",
            ),
            (
                lambda: tensor.to(torch.float32, "cpu"),
                r"^to\(\) received an invalid combination of arguments",
            ),
            (
                lambda: tensor.to(None, None, False, False, False),
                r"^to\(\) takes from 0 to 4 positional arguments but 5 were given$",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
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
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(descriptor(tensor, torch.float32), tensor)
        self.assertIs(descriptor(tensor, "cpu", torch.float32), tensor)

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
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_torch_function_modes_receive_descriptor_and_forward(self):
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
            intercepted = tensor.to(torch.float32, copy=True)
        self.assertIs(intercepted, marker)
        self.assertEqual(len(recording.calls), 1)
        function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], torch.float32)
        self.assertEqual(kwargs, {"copy": True})

        with recording:
            with self.assertRaisesRegex(
                TypeError,
                r"^to\(\) received an invalid combination of arguments",
            ):
                tensor.to(non_blocking=0)
        self.assertEqual(len(recording.calls), 1)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        positional = Override()
        device = Override()
        dtype = Override()
        other = Override()
        memory_format = Override()
        non_blocking = Override()
        copy = Override()
        override_cases = (
            (
                "positional",
                lambda: tensor.to(positional),
                (tensor, positional),
                None,
            ),
            (
                "device keyword",
                lambda: tensor.to(device=device),
                (tensor,),
                {"device": device},
            ),
            (
                "dtype keyword",
                lambda: tensor.to(dtype=dtype),
                (tensor,),
                {"dtype": dtype},
            ),
            (
                "tensor keyword",
                lambda: tensor.to(tensor=other),
                (tensor,),
                {"tensor": other},
            ),
            (
                "memory_format keyword",
                lambda: tensor.to(memory_format=memory_format),
                (tensor,),
                {"memory_format": memory_format},
            ),
            (
                "bool option keywords",
                lambda: tensor.to(
                    torch.float32,
                    non_blocking=non_blocking,
                    copy=copy,
                ),
                (tensor, torch.float32),
                {"non_blocking": non_blocking, "copy": copy},
            ),
        )
        for name, call, expected_args, expected_kwargs in override_cases:
            with self.subTest(override=name):
                override_calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(override_calls), 1)
                function, dispatch_types, args, kwargs = override_calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(len(args), len(expected_args))
                for actual, expected in zip(args, expected_args, strict=True):
                    self.assertIs(actual, expected)
                if expected_kwargs is None:
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(tuple(kwargs), tuple(expected_kwargs))
                    for key, expected in expected_kwargs.items():
                        self.assertIs(kwargs[key], expected)

        override_calls.clear()
        with self.assertRaisesRegex(
            TypeError,
            r"^to\(\) received an invalid combination of arguments",
        ):
            tensor.to(unexpected=Override())
        self.assertEqual(override_calls, [])

        other_tensor = torch.tensor([2.0])
        generic_duplicate_copy_cases = (
            (
                "dtype",
                lambda: tensor.to(torch.float32, False, Override(), copy=False),
            ),
            (
                "tensor",
                lambda: tensor.to(other_tensor, False, Override(), copy=False),
            ),
        )
        for name, call in generic_duplicate_copy_cases:
            with self.subTest(duplicate_copy=name):
                override_calls.clear()
                with self.assertRaisesRegex(
                    TypeError,
                    r"^to\(\) received an invalid combination of arguments",
                ):
                    call()
                self.assertEqual(override_calls, [])

        duplicate_device_copy_cases = (
            (
                "device",
                lambda: tensor.to("cpu", torch.float32, False, Override(), copy=False),
            ),
        )
        for name, call in duplicate_device_copy_cases:
            with self.subTest(duplicate_copy=name):
                override_calls.clear()
                with self.assertRaisesRegex(
                    TypeError,
                    r"^to\(\) got multiple values for argument 'copy'$",
                ):
                    call()
                self.assertEqual(override_calls, [])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.to'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            tensor.to(DecliningOverride())

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.to("cpu", torch.float32, True, False)
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


if __name__ == "__main__":
    unittest.main()
