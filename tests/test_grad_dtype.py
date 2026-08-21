import inspect
import types
import unittest

import torch_rs as torch


GRAD_DTYPE_DOC = """
The allowed dtype of :attr:``grad`` for this tensor.

:attr:``grad_dtype`` can be set to a specific dtype or ``None``. By default,
``t.grad_dtype == t.dtype``. When not None, the autograd engine casts
incoming gradients to this dtype. This attribute is only accessible and
settable for leaf tensors.

.. warning::
    Use with caution. Diverging the dtypes of a tensor and its gradient may
    break downstream systems that assume they match.

Example::

    >>> x = torch.tensor([1.0, 2.0], requires_grad=True)
    >>> x.grad_dtype
    torch.float32

    >>> x.grad_dtype = torch.float16
    >>> x.grad_dtype
    torch.float16

    >>> # Allow any gradient dtype
    >>> x.grad_dtype = None
    >>> x.grad_dtype
"""


class TensorGradDtypeTests(unittest.TestCase):
    def leaf_cases(self):
        ordinary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)

        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_leaf_view = leaf.transpose(0, 1)
            no_grad_non_leaf_view = tracked.transpose(0, 1)

        tracked.sum().backward()
        return (
            ("ordinary leaf", ordinary),
            ("ordinary operation", ordinary + 1.0),
            ("ordinary view", ordinary.transpose(0, 1)),
            ("gradient leaf", leaf),
            ("empty ordinary leaf", torch.zeros((2, 0, 3))),
            (
                "empty gradient leaf",
                torch.zeros((2, 0, 3), requires_grad=True),
            ),
            ("detached operation", tracked.detach()),
            ("detached view", tracked_view.detach()),
            ("no-grad output", no_grad_output),
            ("no-grad leaf view", no_grad_leaf_view),
            ("no-grad non-leaf view", no_grad_non_leaf_view),
            ("live leaf gradient", leaf.grad),
        )

    def test_supported_leaf_states_return_the_canonical_float32_dtype(self):
        for case, tensor in self.leaf_cases():
            with self.subTest(case=case):
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

                result = tensor.grad_dtype

                self.assertTrue(tensor.is_leaf)
                self.assertIs(result, torch.float32)
                self.assertIs(result, tensor.dtype)
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

    def test_non_leaf_access_raises_the_pytorch_error_without_mutating_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        non_leaves = (
            tracked,
            tracked.transpose(0, 1),
            tracked.reshape(4),
            tracked[0],
            torch.zeros((2, 0, 3), requires_grad=True) * 2.0,
        )
        descriptor = inspect.getattr_static(torch.Tensor, "grad_dtype")

        for tensor in non_leaves:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assertFalse(tensor.is_leaf)
                for access in (
                    lambda tensor=tensor: tensor.grad_dtype,
                    lambda tensor=tensor: descriptor.__get__(
                        tensor, torch.Tensor
                    ),
                ):
                    with self.assertRaises(RuntimeError) as raised:
                        access()
                    self.assertEqual(
                        str(raised.exception),
                        "grad_dtype can only be accessed on leaf tensors.",
                    )

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertIs(leaf.grad.dtype, torch.float32)
        self.assertIs(leaf.grad_dtype, torch.float32)

    def test_gradient_accumulation_remains_float32(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        weights = torch.tensor([3.0, -4.0])

        self.assertIsNone(leaf.grad)
        self.assertIs(leaf.grad_dtype, torch.float32)
        (leaf * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [3.0, -4.0])
        self.assertIs(leaf.grad.dtype, torch.float32)
        self.assertIs(leaf.grad.grad_dtype, torch.float32)

        (leaf * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [6.0, -8.0])
        self.assertIs(leaf.grad.dtype, torch.float32)
        self.assertIs(leaf.grad_dtype, torch.float32)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        (empty * 2.0).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.numel(), 0)
        self.assertIs(empty.grad.dtype, torch.float32)
        self.assertIs(empty.grad_dtype, torch.float32)

    def test_tensorbase_descriptor_documentation_and_read_only_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "grad_dtype")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "grad_dtype")
        self.assertEqual(descriptor.__qualname__, "TensorBase.grad_dtype")
        self.assertEqual(descriptor.__doc__, GRAD_DTYPE_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'grad_dtype' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.grad_dtype, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), torch.float32)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'grad_dtype' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        mutations = (
            lambda: setattr(tensor, "grad_dtype", torch.float32),
            lambda: setattr(tensor, "grad_dtype", None),
            lambda: setattr(tensor, "grad_dtype", object()),
            lambda: delattr(tensor, "grad_dtype"),
            lambda: descriptor.__set__(tensor, torch.float32),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(AttributeError) as raised:
                    mutation()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'grad_dtype' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )
                self.assertIs(tensor.grad_dtype, torch.float32)

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        non_leaf = torch.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(torch.Tensor, "grad_dtype")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types_, args=(), kwargs=None):
                self.calls.append((func, types_, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = non_leaf.grad_dtype
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertEqual(function, descriptor.__get__)
        self.assertIs(function.__self__, descriptor)
        self.assertEqual(function.__name__, "__get__")
        self.assertEqual(function.__qualname__, "getset_descriptor.__get__")
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], non_leaf)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types_, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        leaf = torch.tensor([1.0], requires_grad=True)
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = leaf.grad_dtype
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, torch.float32)


if __name__ == "__main__":
    unittest.main()
