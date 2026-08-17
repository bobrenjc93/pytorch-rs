import copy
import inspect
import pickle
import re
import types
import unittest

import torch_rs as torch


METHOD_DOC = """
to(*args, **kwargs) -> Tensor

Performs Tensor dtype and/or device conversion. A :class:`torch.dtype` and :class:`torch.device` are
inferred from the arguments of ``self.to(*args, **kwargs)``.

.. note::

    If the ``self`` Tensor already
    has the correct :class:`torch.dtype` and :class:`torch.device`, then ``self`` is returned.
    Otherwise, the returned tensor is a copy of ``self`` with the desired
    :class:`torch.dtype` and :class:`torch.device`.

.. note::

    If ``self`` requires gradients (``requires_grad=True``) but the target
    ``dtype`` specified is an integer type, the returned tensor will implicitly
    set ``requires_grad=False``. This is because only tensors with
    floating-point or complex dtypes can require gradients.

Here are the ways to call ``to``:

.. method:: to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
   :noindex:

    Returns a Tensor with the specified :attr:`dtype`

    Args:
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

.. note::

    According to `C++ type conversion rules <https://en.cppreference.com/w/cpp/language/implicit_conversion.html>`_,
    converting floating point value to integer type will truncate the fractional part.
    If the truncated value cannot fit into the target type (e.g., casting ``torch.inf`` to ``torch.long``),
    the behavior is undefined and the result may vary across platforms.

.. method:: to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format) -> Tensor
   :noindex:

    Returns a Tensor with the specified :attr:`device` and (optional)
    :attr:`dtype`. If :attr:`dtype` is ``None`` it is inferred to be ``self.dtype``.
    When :attr:`non_blocking` is set to ``True``, the function attempts to perform
    the conversion asynchronously with respect to the host, if possible. This
    asynchronous behavior applies to both pinned and pageable memory. However,
    caution is advised when using this feature. For more information, refer to the
    `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
    When :attr:`copy` is set, a new Tensor is created even when the Tensor
    already matches the desired conversion.

    Args:
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

.. method:: to(other, non_blocking=False, copy=False) -> Tensor
   :noindex:

    Returns a Tensor with same :class:`torch.dtype` and :class:`torch.device` as
    the Tensor :attr:`other`.
    When :attr:`non_blocking` is set to ``True``, the function attempts to perform
    the conversion asynchronously with respect to the host, if possible. This
    asynchronous behavior applies to both pinned and pageable memory. However,
    caution is advised when using this feature. For more information, refer to the
    `tutorial on good usage of non_blocking and pin_memory <https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html>`__.
    When :attr:`copy` is set, a new Tensor is created even when the Tensor
    already matches the desired conversion.

Example::

    >>> tensor = torch.randn(2, 2)  # Initially dtype=float32, device=cpu
    >>> tensor.to(torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64)

    >>> cuda0 = torch.device('cuda:0')
    >>> tensor.to(cuda0)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], device='cuda:0')

    >>> tensor.to(cuda0, dtype=torch.float64)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')

    >>> other = torch.randn((), dtype=torch.float64, device=cuda0)
    >>> tensor.to(other, non_blocking=True)
    tensor([[-0.5044,  0.0005],
            [ 0.3310, -0.0584]], dtype=torch.float64, device='cuda:0')
"""


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
                ("autograd leaf", leaf),
                ("autograd non-leaf", tracked),
                ("detached no-grad tensor", tracked.detach()),
            ),
        )

    def test_empty_conversion_returns_exact_receiver_without_changes(self):
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
                    tensor.grad,
                )

                for result in (tensor.to(), tensor.to(*()), tensor.to(**{})):
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
                            result.grad,
                        ),
                        metadata,
                    )

        tracked.to().sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_no_grad_call_preserves_the_same_tensor_and_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)

        with torch.no_grad():
            result = tracked.to()

        self.assertIs(result, tracked)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])

    def test_tensorbase_descriptor_documentation_and_pickling(self):
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
        self.assertIs(torch.Tensor.to, descriptor)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(), tensor)

        reducer, (owner, name) = descriptor.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertIs(owner, descriptor.__objclass__)
        self.assertEqual(name, "to")
        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )

    def test_unbound_no_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
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
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_conversion_argument_forms_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        calls = (
            lambda: tensor.to(torch.float32),
            lambda: tensor.to(dtype=torch.float32),
            lambda: tensor.to(None),
            lambda: tensor.to(dtype=None),
            lambda: tensor.to(torch.device("cpu")),
            lambda: tensor.to("cpu"),
            lambda: tensor.to(device=torch.device("cpu")),
            lambda: tensor.to(device=None),
            lambda: tensor.to(other),
            lambda: tensor.to(copy=False),
            lambda: tensor.to(copy=True),
            lambda: tensor.to(non_blocking=False),
            lambda: tensor.to(non_blocking=True),
            lambda: tensor.to(memory_format=None),
            lambda: tensor.to(memory_format=torch.preserve_format),
            lambda: tensor.to(memory_format=torch.contiguous_format),
            lambda: tensor.to(torch.float32, False),
            lambda: descriptor(tensor, torch.float32),
            lambda: descriptor(tensor, dtype=torch.float32),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_torch_function_modes_receive_and_forward_the_empty_call(self):
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

        unsupported = RecordingMode(marker)
        with unsupported:
            with self.assertRaises(TypeError):
                tensor.to(dtype=torch.float32)
        self.assertEqual(unsupported.calls, [])


if __name__ == "__main__":
    unittest.main()
