import inspect
import re
import types
import unittest

import numpy as np
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


INVALID_TO_OVERLOAD = (
    "to\\(\\) received an invalid combination of arguments - got "
    r"\(.*\), but expected one of:\n "
    r"\* \(torch\.device device = None, torch\.dtype dtype = None, "
    r"bool non_blocking = False, bool copy = False, "
    r"\*, torch\.memory_format memory_format = None\)\n "
    r"\* \(torch\.dtype dtype, bool non_blocking = False, "
    r"bool copy = False, \*, torch\.memory_format memory_format = None\)\n "
    r"\* \(Tensor tensor, bool non_blocking = False, bool copy = False, "
    r"\*, torch\.memory_format memory_format = None\)\n"
)


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
                [0.0, -0.0, 2.0, 3.0],
                [4.0, 5.0, float("inf"), -7.0],
                [8.0, 9.0, 10.0, float("nan")],
            ],
            dtype=torch.float32,
        )
        noncontiguous = source.transpose(0, 1)
        offset = noncontiguous[1]

        self.assertFalse(noncontiguous.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("noncontiguous", noncontiguous),
            ("offset", offset),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
        )

    def identity_calls(self):
        return (
            ("no arguments", lambda tensor: tensor.to()),
            ("dtype positional", lambda tensor: tensor.to(torch.float32)),
            ("float alias positional", lambda tensor: tensor.to(torch.float)),
            ("dtype keyword", lambda tensor: tensor.to(dtype=torch.float32)),
            ("device string positional", lambda tensor: tensor.to("cpu")),
            (
                "device object positional",
                lambda tensor: tensor.to(torch.device("cpu")),
            ),
            ("device keyword", lambda tensor: tensor.to(device="cpu")),
            (
                "device dtype positional",
                lambda tensor: tensor.to("cpu", torch.float32),
            ),
            (
                "full identity keywords",
                lambda tensor: tensor.to(
                    device=torch.device("cpu"),
                    dtype=torch.float,
                    non_blocking=True,
                    copy=False,
                    memory_format=torch.preserve_format,
                ),
            ),
            (
                "explicit none defaults",
                lambda tensor: tensor.to(
                    None, None, False, False, memory_format=None
                ),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

    def test_identity_requests_return_exact_receiver(self):
        for case, tensor in self.tensor_cases():
            for form, call in self.identity_calls():
                with self.subTest(case=case, form=form):
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
                        tensor.output_nr,
                    )
                    gradient = tensor.grad
                    bits = self.value_bits(tensor)

                    result = call(tensor)

                    self.assertIs(result, tensor)
                    self.assertTrue(result.is_set_to(tensor.detach()))
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
                            result.output_nr,
                        ),
                        metadata,
                    )
                    self.assertIs(tensor.grad, gradient)
                    if bits is not None:
                        np.testing.assert_array_equal(self.value_bits(result), bits)

    def test_autograd_metadata_and_graph_are_preserved(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        graph_before = (
            source.requires_grad,
            source.is_leaf,
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            source.output_nr,
        )

        with torch.no_grad():
            result = source.to(dtype=torch.float32, device="cpu")

        self.assertIs(result, source)
        self.assertEqual(
            (
                source.requires_grad,
                source.is_leaf,
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                source.output_nr,
            ),
            graph_before,
        )
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])

    def test_descriptor_metadata_documentation_and_unbound_calls(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        bound = tensor.to

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'to' of 'torch._C.TensorBase' objects>")
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
        self.assertIsNone(bound.__module__)
        self.assertIs(bound.__self__, tensor)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(descriptor(tensor, torch.float32), tensor)
        self.assertIs(bound(dtype=torch.float), tensor)

    def test_binding_errors_match_pytorch_generated_parser_shape(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        cases = (
            (
                lambda: descriptor(),
                r"^unbound method TensorBase\.to\(\) needs an argument$",
            ),
            (
                lambda: descriptor(1),
                "^descriptor 'to' for 'torch\\._C\\.TensorBase' objects doesn't apply to a 'int' object$",
            ),
            (lambda: tensor.to("cpu", True), f"^{INVALID_TO_OVERLOAD}$"),
            (lambda: tensor.to(dtype=1), f"^{INVALID_TO_OVERLOAD}$"),
            (lambda: tensor.to(non_blocking=1), f"^{INVALID_TO_OVERLOAD}$"),
            (lambda: tensor.to(memory_format=1), f"^{INVALID_TO_OVERLOAD}$"),
            (
                lambda: tensor.to(torch.float32, dtype=torch.float32),
                f"^{INVALID_TO_OVERLOAD}$",
            ),
        )
        for call, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(TypeError, pattern):
                    call()

    def test_rejects_conversions_copy_and_unsupported_overloads(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])

        cases = (
            (lambda: tensor.to(float), "dtype conversions are not supported"),
            (lambda: tensor.to(True), "dtype conversions are not supported"),
            (lambda: tensor.to("cuda"), "device conversions are not supported"),
            (lambda: tensor.to("meta"), "device conversions are not supported"),
            (lambda: tensor.to(0), "device conversions are not supported"),
            (
                lambda: tensor.to("cpu:0"),
                "indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: tensor.to(torch.device("cpu", 0)),
                "indexed CPU devices require a copy and are not supported",
            ),
            (lambda: tensor.to(copy=True), "copy=True requires a copy"),
            (
                lambda: tensor.to(torch.float32, False, True),
                "copy=True requires a copy",
            ),
            (
                lambda: tensor.to(memory_format=torch.contiguous_format),
                "memory_format changes are not supported",
            ),
            (
                lambda: tensor.to(memory_format=torch.channels_last),
                "memory_format changes are not supported",
            ),
            (
                lambda: tensor.to(other),
                "Tensor argument overload is not supported",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(NotImplementedError, message):
                    call()

    def test_torch_function_mode_dispatches_before_native_rejections(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        cases = (
            (
                "device conversion",
                lambda: tensor.to("cuda"),
                (tensor, "cuda"),
                None,
            ),
            (
                "copy request",
                lambda: tensor.to(copy=True),
                (tensor,),
                {"copy": True},
            ),
            (
                "tensor overload",
                lambda: tensor.to(other),
                (tensor, other),
                None,
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            with self.subTest(case=case):
                mode = RecordingMode(marker)
                with mode:
                    result = call()

                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, types, args, kwargs = mode.calls[0]
                self.assertEqual(function, descriptor)
                self.assertEqual(types, ())
                self.assertEqual(len(args), len(expected_args))
                for actual, expected in zip(args, expected_args, strict=True):
                    if isinstance(expected, torch.Tensor):
                        self.assertIs(actual, expected)
                    else:
                        self.assertEqual(actual, expected)
                self.assertEqual(kwargs, expected_kwargs)

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.to'; all "
            r"__torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                tensor.to("cuda")
        self.assertEqual(len(declining.calls), 1)

    def test_torch_function_override_arguments_dispatch_before_native_rejections(self):
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
            ("tensor overload", lambda argument: tensor.to(argument), None),
            ("dtype keyword", lambda argument: tensor.to(dtype=argument), "dtype"),
            ("device keyword", lambda argument: tensor.to(device=argument), "device"),
            (
                "non blocking keyword",
                lambda argument: tensor.to(non_blocking=argument),
                "non_blocking",
            ),
            ("copy keyword", lambda argument: tensor.to(copy=argument), "copy"),
            (
                "memory format keyword",
                lambda argument: tensor.to(memory_format=argument),
                "memory_format",
            ),
        )
        for case, call, keyword in cases:
            with self.subTest(case=case):
                Override.calls = []
                argument = Override()
                self.assertIs(call(argument), marker)
                self.assertEqual(len(Override.calls), 1)
                function, types, args, kwargs = Override.calls[0]
                self.assertEqual(function, descriptor)
                self.assertEqual(types, (Override,))
                self.assertIs(args[0], tensor)
                if keyword is None:
                    self.assertIs(args[1], argument)
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(len(args), 1)
                    self.assertIs(kwargs[keyword], argument)

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


if __name__ == "__main__":
    unittest.main()
