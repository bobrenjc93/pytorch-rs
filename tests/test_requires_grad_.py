import inspect
import re
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
DISABLE_ERROR = (
    "requires_grad_(False) is not supported; only enabling leaf tensors is "
    "implemented"
)


class TensorRequiresGradInPlaceTests(unittest.TestCase):
    def layout_cases(self):
        base = torch.tensor([float(value) for value in range(24)]).reshape(2, 3, 4)
        source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        detached = (source * 2.0).transpose(0, 1).detach()
        return (
            ("ordinary", torch.tensor([[1.0, 2.0], [3.0, 4.0]]), None),
            ("empty", torch.zeros((2, 0, 3)), None),
            ("detached", detached, source),
            ("offset", base[1], base),
            ("offset noncontiguous", base.transpose(0, 2)[1], base),
        )

    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.tolist(),
        )

    def test_supported_leaf_layouts_are_enabled_in_place_and_accumulate(self):
        for index, (case, tensor, source) in enumerate(self.layout_cases()):
            with self.subTest(case=case):
                before = self.metadata(tensor)
                self.assertFalse(tensor.requires_grad)
                self.assertTrue(tensor.is_leaf)
                self.assertIsNone(tensor.grad)

                result = (
                    tensor.requires_grad_()
                    if index % 2 == 0
                    else tensor.requires_grad_(True)
                )

                self.assertIs(result, tensor)
                self.assertTrue(tensor.requires_grad)
                self.assertTrue(tensor.is_leaf)
                self.assertEqual(self.metadata(tensor), before)
                self.assertIs(tensor.requires_grad_(True), tensor)
                self.assertEqual(self.metadata(tensor), before)

                (tensor * torch.ones(tensor.shape)).sum().backward()
                self.assertEqual(tensor.grad.shape, tensor.shape)
                np.testing.assert_array_equal(
                    np.asarray(tensor.grad),
                    np.ones(tuple(tensor.shape), dtype=np.float32),
                )
                if source is not None:
                    self.assertIsNone(source.grad)

    def test_views_created_before_base_enablement_follow_the_live_root(self):
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        alias = base[:]
        nested = alias.transpose(0, 1)[1]
        detached = alias.detach()

        self.assertFalse(base.requires_grad)
        self.assertFalse(alias.requires_grad)
        self.assertFalse(nested.requires_grad)
        self.assertIs(base.requires_grad_(), base)

        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertTrue(nested.requires_grad)
        self.assertTrue(nested.is_leaf)
        self.assertFalse(detached.requires_grad)

        for case, view in (("alias", alias), ("nested", nested)):
            with self.subTest(case=case):
                output = view * 2.0
                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                output.sum().backward()
                self.assertIsNone(view.grad)
                self.assertIsNone(base.grad)

        self.assertIs(alias.requires_grad_(True), alias)
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        (alias * weights).sum().backward()
        np.testing.assert_array_equal(np.asarray(alias.grad), np.asarray(weights))
        self.assertIsNone(base.grad)

    def test_dense_noncontiguous_leaf_gradient_preserves_default_layout(self):
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        leaf = base.transpose(0, 1).detach()
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        second = torch.tensor([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])

        self.assertEqual(leaf.stride(), (1, 3))
        self.assertIs(leaf.requires_grad_(), leaf)
        (leaf * first).sum().backward()
        gradient = leaf.grad

        self.assertEqual(gradient.stride(), leaf.stride())
        self.assertFalse(gradient.is_contiguous())
        np.testing.assert_array_equal(np.asarray(gradient), np.asarray(first))

        (leaf * second).sum().backward()
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(leaf.grad.stride(), (1, 3))
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray(first) + np.asarray(second),
        )

    def test_existing_leaf_and_nonleaf_autograd_state_is_unchanged(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        (leaf * 2.0).sum().backward()
        cached_grad = leaf.grad
        leaf_metadata = self.metadata(leaf)

        self.assertIs(leaf.requires_grad_(), leaf)
        self.assertEqual(self.metadata(leaf), leaf_metadata)
        self.assertIs(leaf.grad, cached_grad)
        (leaf * 3.0).sum().backward()
        np.testing.assert_array_equal(np.asarray(leaf.grad), [5.0, 5.0])

        source = torch.tensor([2.0, 3.0], requires_grad=True)
        nonleaf = source * 4.0
        nonleaf_metadata = self.metadata(nonleaf)
        self.assertTrue(nonleaf.requires_grad)
        self.assertFalse(nonleaf.is_leaf)

        self.assertIs(nonleaf.requires_grad_(True), nonleaf)
        self.assertEqual(self.metadata(nonleaf), nonleaf_metadata)
        self.assertTrue(nonleaf.requires_grad)
        self.assertFalse(nonleaf.is_leaf)
        nonleaf.sum().backward()
        np.testing.assert_array_equal(np.asarray(source.grad), [4.0, 4.0])

    def test_no_grad_leaf_view_is_promoted_to_an_independent_accumulator(self):
        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = source * 2.0
        with torch.no_grad():
            view = tracked.transpose(0, 1)

        self.assertTrue(view.requires_grad)
        self.assertTrue(view.is_leaf)
        self.assertIsNone(view.grad)
        before = self.metadata(view)

        self.assertIs(view.requires_grad_(), view)
        self.assertEqual(self.metadata(view), before)
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        (view * weights).sum().backward()

        np.testing.assert_array_equal(np.asarray(view.grad), np.asarray(weights))
        self.assertIsNone(source.grad)
        self.assertIsNone(tracked.grad)

    def test_no_grad_suppresses_operations_but_not_leaf_enablement(self):
        tensor = torch.tensor([2.0, 3.0])

        with torch.no_grad():
            self.assertIs(tensor.requires_grad_(), tensor)
            suppressed = tensor * 5.0

        self.assertTrue(tensor.requires_grad)
        self.assertFalse(suppressed.requires_grad)
        self.assertTrue(suppressed.is_leaf)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            suppressed.sum().backward()

        recorded = tensor * 7.0
        self.assertTrue(recorded.requires_grad)
        self.assertFalse(recorded.is_leaf)
        recorded.sum().backward()
        np.testing.assert_array_equal(np.asarray(tensor.grad), [7.0, 7.0])

    def test_disabling_is_rejected_before_any_state_changes(self):
        plain = torch.tensor([1.0, 2.0])
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        (leaf * 2.0).sum().backward()
        cached_grad = leaf.grad
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        nonleaf = source * 3.0

        for case, tensor in (
            ("already disabled", plain),
            ("enabled leaf", leaf),
            ("enabled nonleaf", nonleaf),
        ):
            with self.subTest(case=case):
                before = (
                    self.metadata(tensor),
                    tensor.requires_grad,
                    tensor.is_leaf,
                    tensor.grad,
                )
                with self.assertRaises(RuntimeError) as raised:
                    tensor.requires_grad_(False)
                self.assertEqual(str(raised.exception), DISABLE_ERROR)
                self.assertEqual(
                    (
                        self.metadata(tensor),
                        tensor.requires_grad,
                        tensor.is_leaf,
                        tensor.grad,
                    ),
                    before,
                )

        self.assertIs(leaf.grad, cached_grad)
        nonleaf.sum().backward()
        np.testing.assert_array_equal(np.asarray(source.grad), [3.0, 3.0])

    def test_strict_boolean_binding_and_errors_match_pytorch(self):
        cases = (
            (
                lambda tensor: tensor.requires_grad_(1),
                "requires_grad_(): argument 'requires_grad' (position 1) must "
                "be bool, not int",
            ),
            (
                lambda tensor: tensor.requires_grad_(requires_grad=1),
                "requires_grad_(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda tensor: tensor.requires_grad_(None),
                "requires_grad_(): argument 'requires_grad' (position 1) must "
                "be bool, not NoneType",
            ),
            (
                lambda tensor: tensor.requires_grad_(np.bool_(True)),
                "requires_grad_(): argument 'requires_grad' (position 1) must "
                "be bool, not numpy.bool",
            ),
            (
                lambda tensor: tensor.requires_grad_(True, False),
                "requires_grad_() takes from 0 to 1 positional arguments but 2 "
                "were given",
            ),
            (
                lambda tensor: tensor.requires_grad_(foo=True),
                "requires_grad_() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda tensor: tensor.requires_grad_(True, requires_grad=True),
                "requires_grad_() got multiple values for argument 'requires_grad'",
            ),
        )
        for call, message in cases:
            tensor = torch.tensor([1.0])
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call(tensor)
                self.assertEqual(str(raised.exception), message)
                self.assertFalse(tensor.requires_grad)
                self.assertIsNone(tensor.grad)

    def test_tensorbase_ownership_documentation_and_receiver_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "requires_grad_")
        bound = tensor.requires_grad_

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'requires_grad_' of 'torch._C.TensorBase' objects>",
        )
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

        errors = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.requires_grad_() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'requires_grad_' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.requires_grad_() needs an argument",
            ),
        )
        for call, message in errors:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        descriptor = inspect.getattr_static(torch.Tensor, "requires_grad_")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        forms = (
            (lambda tensor: tensor.requires_grad_(), (), None),
            (lambda tensor: tensor.requires_grad_(True), (True,), None),
            (
                lambda tensor: tensor.requires_grad_(requires_grad=True),
                (),
                {"requires_grad": True},
            ),
        )
        for call, expected_tail, expected_kwargs in forms:
            tensor = torch.tensor([1.0])
            mode = RecordingMode()
            with mode:
                result = call(tensor)
            self.assertIs(result, marker)
            self.assertFalse(tensor.requires_grad)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertIs(args[0], tensor)
            self.assertEqual(args[1:], expected_tail)
            self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        tensor = torch.tensor([1.0])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.requires_grad_(requires_grad=True)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)
        self.assertTrue(tensor.requires_grad)

        order.clear()
        unsupported = torch.tensor([1.0])
        with self.assertRaises(RuntimeError) as raised:
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    unsupported.requires_grad_(False)
        self.assertEqual(str(raised.exception), DISABLE_ERROR)
        self.assertEqual(order, ["upper", "lower"])
        self.assertFalse(unsupported.requires_grad)

        invalid = torch.tensor([1.0])
        recording = RecordingMode()
        with recording:
            with self.assertRaises(TypeError):
                invalid.requires_grad_(1)
        self.assertEqual(recording.calls, [])
        self.assertFalse(invalid.requires_grad)

        disabled = torch.tensor([1.0])
        recording = RecordingMode()
        with recording:
            self.assertIs(disabled.requires_grad_(False), marker)
        self.assertFalse(disabled.requires_grad)

        declining = RecordingMode(NotImplemented)
        untouched = torch.tensor([1.0])
        with self.assertRaises(TypeError) as raised:
            with declining:
                untouched.requires_grad_()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.requires_grad_'; "
                r"all __torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertFalse(untouched.requires_grad)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
