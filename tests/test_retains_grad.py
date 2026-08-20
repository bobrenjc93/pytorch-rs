import inspect
import sys
import types
import unittest

import torch_rs as torch


PROPERTY_DOC = (
    "\nIs ``True`` if this Tensor is non-leaf and its :attr:`grad` is enabled "
    "to be\npopulated during :func:`backward`, ``False`` otherwise.\n"
)
METHOD_DOC = (
    "\nretain_grad() -> None\n\n"
    "Enables this Tensor to have their :attr:`grad` populated during\n"
    ":func:`backward`. This is a no-op for leaf tensors.\n"
)


class TensorRetainsGradTests(unittest.TestCase):
    def tensor_cases(self):
        ordinary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        detached_tracked = tracked.detach()
        detached_view = tracked_view.detach()

        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_leaf_view = leaf.transpose(0, 1)
            no_grad_non_leaf_view = tracked.transpose(0, 1)

        recorded_after_no_grad = no_grad_leaf_view + 1.0
        tracked.sum().backward()

        return (
            ("ordinary leaf", ordinary),
            ("ordinary operation", ordinary + 1.0),
            ("ordinary view", ordinary.transpose(0, 1)),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
            ("autograd non-leaf view", tracked_view),
            ("detached non-leaf", detached_tracked),
            ("detached view", detached_view),
            ("no_grad output", no_grad_output),
            ("no_grad leaf view", no_grad_leaf_view),
            ("no_grad non-leaf view", no_grad_non_leaf_view),
            ("recorded output after no_grad", recorded_after_no_grad),
            ("live leaf gradient", leaf.grad),
        )

    def test_every_supported_tensor_state_reports_false_without_side_effects(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )

                result = tensor.retains_grad

                self.assertIs(type(result), bool)
                self.assertIs(result, False)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_retain_grad_is_a_leaf_noop_preserving_gradient_accumulation(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        metadata = (
            leaf.shape,
            leaf.stride(),
            leaf.storage_offset(),
            leaf.data_ptr(),
            leaf.requires_grad,
            leaf.is_leaf,
        )

        self.assertIs(leaf.retain_grad(), None)
        self.assertIs(leaf.retain_grad(), None)
        self.assertIs(leaf.retains_grad, False)
        self.assertEqual(
            (
                leaf.shape,
                leaf.stride(),
                leaf.storage_offset(),
                leaf.data_ptr(),
                leaf.requires_grad,
                leaf.is_leaf,
            ),
            metadata,
        )

        leaf.sum().backward()
        leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertIs(leaf.retains_grad, False)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        self.assertIs(empty.retain_grad(), None)
        self.assertIs(empty.retains_grad, False)
        (empty + 2.0).sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.stride(), (3, 3, 1))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_retain_grad_accepts_leaf_views_created_under_no_grad(self):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            view = source.transpose(0, 1)

        self.assertTrue(view.requires_grad)
        self.assertTrue(view.is_leaf)
        metadata = (
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.data_ptr(),
        )
        self.assertIs(view.retain_grad(), None)
        self.assertIs(view.retains_grad, False)
        self.assertEqual(
            (
                view.shape,
                view.stride(),
                view.storage_offset(),
                view.data_ptr(),
            ),
            metadata,
        )

        (view * view).sum().backward()
        self.assertIsNone(source.grad)
        self.assertIsNone(view.grad)
        self.assertIs(view.retains_grad, False)

    def test_retain_grad_rejects_untracked_and_non_leaf_tensors(self):
        for tensor in (torch.tensor([1.0]), torch.zeros((2, 0, 3))):
            with self.subTest(shape=tensor.shape):
                with self.assertRaises(RuntimeError) as raised:
                    tensor.retain_grad()
                self.assertEqual(
                    str(raised.exception),
                    "can't retain_grad on Tensor that has requires_grad=False",
                )
                self.assertIs(tensor.retains_grad, False)

        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        non_leaf = leaf * 3.0
        with self.assertRaises(NotImplementedError) as raised:
            non_leaf.retain_grad()
        self.assertEqual(
            str(raised.exception),
            "torch_rs.Tensor.retain_grad() only supports leaf tensors",
        )
        self.assertIs(non_leaf.retains_grad, False)
        self.assertIsNone(non_leaf.grad)

        non_leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [3.0, 3.0])
        self.assertIs(non_leaf.retains_grad, False)

    def test_retain_grad_tensorbase_descriptor_and_call_errors(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "retain_grad")
        bound = tensor.retain_grad

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'retain_grad' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "retain_grad")
        self.assertEqual(descriptor.__qualname__, "TensorBase.retain_grad")
        self.assertEqual(bound.__name__, "retain_grad")
        self.assertEqual(bound.__qualname__, "Tensor.retain_grad")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(torch.Tensor.retain_grad, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor(tensor), None)
        self.assertIs(bound(), None)

        cases = (
            (
                lambda: tensor.retain_grad(1),
                "TensorBase.retain_grad() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.retain_grad() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.retain_grad(1, 2),
                "TensorBase.retain_grad() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.retain_grad(input=tensor),
                (
                    "Tensor.retain_grad() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.retain_grad() takes no keyword arguments"
                ),
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.retain_grad() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.retain_grad() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'retain_grad' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.retain_grad() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_retain_grad_torch_function_modes_dispatch_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "retain_grad")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.retain_grad()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)
        self.assertIs(tensor.retains_grad, False)

        untracked = torch.tensor([1.0])
        replacing = RecordingMode()
        with replacing:
            self.assertIs(untracked.retain_grad(), marker)
        self.assertEqual(len(replacing.calls), 1)
        self.assertIs(untracked.retains_grad, False)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.retain_grad()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, None)
        self.assertIs(tensor.retains_grad, False)

        rejected = RecordingMode()
        with rejected:
            with self.assertRaises(TypeError):
                tensor.retain_grad(unexpected=True)
        self.assertEqual(rejected.calls, [])

    def test_tensorbase_descriptor_documentation_and_receiver_behavior(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "retains_grad")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "retains_grad")
        self.assertEqual(descriptor.__qualname__, "TensorBase.retains_grad")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'retains_grad' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.retains_grad, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), False)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'retains_grad' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

    def test_property_is_read_only_with_pytorch_assignment_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "retains_grad")
        actions = (
            lambda: setattr(tensor, "retains_grad", True),
            lambda: delattr(tensor, "retains_grad"),
            lambda: descriptor.__set__(tensor, True),
            lambda: descriptor.__delete__(tensor),
        )

        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'retains_grad' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(torch.Tensor, "retains_grad")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.retains_grad
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertEqual(function, descriptor.__get__)
        self.assertIs(function.__self__, descriptor)
        self.assertEqual(function.__name__, "__get__")
        self.assertEqual(function.__qualname__, "getset_descriptor.__get__")
        self.assertEqual(dispatch_types, (torch.Tensor,))
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
                forwarded = tensor.retains_grad
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)


if __name__ == "__main__":
    unittest.main()
