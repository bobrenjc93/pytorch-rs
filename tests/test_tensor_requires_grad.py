import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nrequires_grad_(requires_grad=True) -> Tensor\n\n"
    "Change if autograd should record operations on this tensor: sets this tensor's\n"
    ":attr:`requires_grad` attribute in-place. Returns this tensor.\n\n"
    ":func:`requires_grad_`'s main use case is to tell autograd to begin recording\n"
    "operations on a Tensor ``tensor``. If ``tensor`` has ``requires_grad=False``\n"
    "(because it was obtained through a DataLoader, or required preprocessing or\n"
    "initialization), ``tensor.requires_grad_()`` makes it so that autograd will\n"
    "begin to record operations on ``tensor``.\n\n"
    "Args:\n"
    "    requires_grad (bool): If autograd should record operations on this tensor.\n"
    "        Default: ``True``.\n\n"
    "Example::\n\n"
    "    >>> # Let's say we want to preprocess some saved weights and use\n"
    "    >>> # the result as new weights.\n"
    "    >>> saved_weights = [0.1, 0.2, 0.3, 0.25]\n"
    "    >>> loaded_weights = torch.tensor(saved_weights)\n"
    "    >>> weights = preprocess(loaded_weights)  # some function\n"
    "    >>> weights\n"
    "    tensor([-0.5503,  0.4926, -2.1158, -0.8303])\n\n"
    "    >>> # Now, start to record operations done to weights\n"
    "    >>> weights.requires_grad_()\n"
    "    >>> out = weights.pow(2).sum()\n"
    "    >>> out.backward()\n"
    "    >>> weights.grad\n"
    "    tensor([-1.1007,  0.9853, -4.2316, -1.6606])\n\n"
)


class TensorRequiresGradInplaceTests(unittest.TestCase):
    def test_fresh_leaf_default_true_and_repeated_toggles_keep_identity(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        metadata = (tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.data_ptr())

        self.assertIs(tensor.requires_grad_(), tensor)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertEqual(
            (tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.data_ptr()),
            metadata,
        )

        self.assertIs(tensor.requires_grad_(True), tensor)
        self.assertTrue(tensor.requires_grad)
        (tensor * 2.0).sum().backward()
        first_grad = tensor.grad
        self.assertEqual(first_grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

        self.assertIs(tensor.requires_grad_(False), tensor)
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertIs(tensor.grad, first_grad)
        self.assertEqual(tensor.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertFalse((tensor * 3.0).requires_grad)

        self.assertIs(tensor.requires_grad_(True), tensor)
        self.assertTrue(tensor.requires_grad)
        self.assertIs(tensor.grad, first_grad)
        (tensor * 3.0).sum().backward()
        self.assertIs(tensor.grad, first_grad)
        self.assertEqual(tensor.grad.tolist(), [[5.0, 5.0], [5.0, 5.0]])

    def test_existing_graph_observes_current_leaf_flag_at_backward_time(self):
        disabled = torch.tensor([1.0, 2.0], requires_grad=True)
        disabled_loss = (disabled * 2.0).sum()
        disabled.requires_grad_(False)
        disabled_loss.backward()
        self.assertIsNone(disabled.grad)

        reenabled = torch.tensor([1.0, 2.0], requires_grad=True)
        reenabled_loss = (reenabled * 2.0).sum()
        reenabled.requires_grad_(False)
        reenabled.requires_grad_(True)
        reenabled_loss.backward()
        self.assertEqual(reenabled.grad.tolist(), [2.0, 2.0])

    def test_no_grad_context_does_not_control_the_leaf_mutator(self):
        tensor = torch.tensor([1.0, 2.0])
        with torch.no_grad():
            self.assertIs(tensor.requires_grad_(), tensor)
            self.assertTrue(tensor.requires_grad)
            self.assertFalse((tensor * 2.0).requires_grad)
        self.assertTrue((tensor * 2.0).requires_grad)

    def test_views_and_non_leaves_match_supported_boundary(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        non_leaf = leaf * 2.0
        view = non_leaf.transpose(0, 1)

        self.assertIs(non_leaf.requires_grad_(True), non_leaf)
        self.assertIs(view.requires_grad_(True), view)
        self.assertTrue(non_leaf.requires_grad)
        self.assertTrue(view.requires_grad)
        self.assertFalse(non_leaf.is_leaf)
        self.assertFalse(view.is_leaf)

        for tensor in (non_leaf, view):
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                with self.assertRaises(RuntimeError) as raised:
                    tensor.requires_grad_(False)
                self.assertEqual(
                    str(raised.exception),
                    "you can only change requires_grad flags of leaf variables. "
                    "If you want to use a computed variable in a subgraph that "
                    "doesn't require differentiation use var_no_grad = var.detach().",
                )

        view.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_leaf_views_created_under_no_grad_keep_inherited_flag(self):
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            view = base.transpose(0, 1)

        self.assertTrue(view.requires_grad)
        self.assertTrue(view.is_leaf)
        self.assertIs(view.requires_grad_(False), view)
        self.assertTrue(view.requires_grad)
        (view * 3.0).sum().backward()
        self.assertIsNone(view.grad)
        self.assertIsNone(base.grad)

        self.assertIs(view.requires_grad_(True), view)
        (view * 3.0).sum().backward()
        self.assertEqual(view.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])
        self.assertIsNone(base.grad)

    def test_invalid_arguments_use_strict_bool_binding(self):
        for call, message in (
            (
                lambda: torch.tensor([1.0]).requires_grad_(None),
                "requires_grad_(): argument 'requires_grad' (position 1) "
                "must be bool, not NoneType",
            ),
            (
                lambda: torch.tensor([1.0]).requires_grad_(1),
                "requires_grad_(): argument 'requires_grad' (position 1) "
                "must be bool, not int",
            ),
            (
                lambda: torch.tensor([1.0]).requires_grad_(np.bool_(True)),
                "requires_grad_(): argument 'requires_grad' (position 1) "
                "must be bool, not numpy.bool",
            ),
            (
                lambda: torch.tensor([1.0]).requires_grad_(requires_grad=None),
                "requires_grad_(): argument 'requires_grad' must be bool, not NoneType",
            ),
            (
                lambda: torch.tensor([1.0]).requires_grad_(True, False),
                "requires_grad_() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.tensor([1.0]).requires_grad_(foo=True),
                "requires_grad_() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: torch.tensor([1.0]).requires_grad_(
                    False, requires_grad=True
                ),
                "requires_grad_() got multiple values for argument 'requires_grad'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_tensorbase_descriptor_documentation_and_dispatch(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "requires_grad_")
        bound = tensor.requires_grad_

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(repr(descriptor), "<method 'requires_grad_' of 'torch._C.TensorBase' objects>")
        self.assertEqual(descriptor.__name__, "requires_grad_")
        self.assertEqual(descriptor.__qualname__, "TensorBase.requires_grad_")
        self.assertEqual(bound.__name__, "requires_grad_")
        self.assertEqual(bound.__qualname__, "Tensor.requires_grad_")
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
        self.assertIs(torch.Tensor.requires_grad_, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(False), tensor)
        self.assertFalse(tensor.requires_grad)

    def test_torch_function_modes_receive_requires_grad_method_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "requires_grad_")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.requires_grad_(False)
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertIs(args[1], False)
        self.assertIsNone(kwargs)
        self.assertFalse(tensor.requires_grad)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.requires_grad_()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)
        self.assertTrue(tensor.requires_grad)

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                tensor.requires_grad_(1)
        self.assertEqual(invalid_mode.calls, [])


if __name__ == "__main__":
    unittest.main()
