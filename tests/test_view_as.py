import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nview_as(other) -> Tensor\n\n"
    "View this tensor as the same size as :attr:`other`.\n"
    "``self.view_as(other)`` is equivalent to ``self.view(other.size())``.\n"
    "\n"
    "Please see :meth:`~Tensor.view` for more information about ``view``.\n"
    "\n"
    "Args:\n"
    "    other (:class:`torch.Tensor`): The result tensor has the same size\n"
    "        as :attr:`other`.\n"
)

INCOMPATIBLE_LAYOUT = (
    "view size is not compatible with input tensor's size and stride "
    "(at least one dimension spans across two contiguous subspaces). "
    "Use .reshape(...) instead."
)


class TensorViewAsTests(unittest.TestCase):
    def assert_view_result(
        self,
        result,
        source,
        other,
        *,
        expected_shape,
        expected_stride,
        expected_offset,
    ):
        direct = source.reshape(other.shape)
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.stride(), expected_stride)
        self.assertEqual(result.storage_offset(), expected_offset)
        self.assertEqual(result.is_contiguous(), direct.is_contiguous())
        self.assertEqual(result.requires_grad, direct.requires_grad)
        self.assertEqual(result.is_leaf, direct.is_leaf)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertTrue(result.is_set_to(direct))

    def make_layout_cases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        noncontiguous = base.transpose(0, 1)
        return (
            (
                "scalar",
                torch.tensor(-0.0),
                torch.tensor(8.0),
                (),
                (),
                0,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                torch.zeros((2, 0)),
                (2, 0),
                (1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                torch.zeros((0, 1)),
                (0, 1),
                (1, 0),
                0,
            ),
            (
                "contiguous-with-strided-other",
                base,
                torch.zeros((4, 6)).transpose(0, 1),
                (6, 4),
                (4, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                torch.zeros((2, 6)),
                (2, 6),
                (6, 1),
                12,
            ),
            (
                "noncontiguous-same-shape",
                noncontiguous,
                torch.zeros((3, 2, 4)),
                (3, 2, 4),
                (4, 12, 1),
                0,
            ),
            (
                "noncontiguous-compatible-split",
                noncontiguous,
                torch.zeros((3, 2, 2, 2)),
                (3, 2, 2, 2),
                (4, 12, 2, 1),
                0,
            ),
        )

    def test_positional_and_keyword_calls_return_shared_storage_views(self):
        for (
            case,
            source,
            other,
            expected_shape,
            expected_stride,
            expected_offset,
        ) in self.make_layout_cases():
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    result = (
                        source.view_as(other=other)
                        if keyword
                        else source.view_as(other)
                    )
                    self.assert_view_result(
                        result,
                        source,
                        other,
                        expected_shape=expected_shape,
                        expected_stride=expected_stride,
                        expected_offset=expected_offset,
                    )

    def test_extreme_empty_shape_remains_a_shared_storage_view(self):
        maximum = sys.maxsize
        source = torch.zeros((0,))
        other = source.reshape((0, maximum, maximum))

        result = source.view_as(other=other)

        self.assertEqual(result.shape, (0, maximum, maximum))
        self.assertEqual(result.stride(), (1, maximum, 1))
        self.assertEqual(result.storage_offset(), 0)
        self.assertEqual(result.numel(), 0)
        self.assertEqual(result.tolist(), [])
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertTrue(result.is_set_to(source.reshape(other.shape)))

    def test_incompatible_layout_and_shape_errors_never_copy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist()).transpose(0, 1)
        other = torch.zeros((6, 4))

        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(INCOMPATIBLE_LAYOUT)}$"):
            source.view_as(other)

        reshaped = source.reshape(other.shape)
        self.assertNotEqual(reshaped.data_ptr(), source.data_ptr())
        self.assertFalse(reshaped.is_set_to(source))

        with self.assertRaisesRegex(
            RuntimeError, r"^shape '\[2, 2\]' is invalid for input of size 6$"
        ):
            torch.zeros((6,)).view_as(torch.zeros((2, 2)))

    def test_autograd_repeated_backward_and_no_grad_use_view_semantics(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        other = torch.zeros((3, 2), requires_grad=True)
        result = source.view_as(other=other)

        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.stride(), (1, 3))
        self.assertEqual(result.storage_offset(), 0)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.data_ptr(), source.data_ptr())

        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        loss = (result * weights).sum()
        loss.backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]),
        )
        self.assertIsNone(other.grad)

        repeated_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        repeated_other = torch.zeros((3, 2), requires_grad=True)
        repeated_loss = repeated_leaf.transpose(0, 1).view_as(repeated_other).sum()
        repeated_loss.backward()
        repeated_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(repeated_leaf.grad),
            np.full((2, 3), 2.0, dtype=np.float32),
        )
        self.assertIsNone(repeated_other.grad)

        no_grad_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        no_grad_other = torch.zeros((3, 2), requires_grad=True)
        with torch.no_grad():
            no_grad_result = no_grad_source.view_as(no_grad_other)

        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertEqual(no_grad_result.stride(), (1, 3))
        self.assertEqual(no_grad_result.data_ptr(), no_grad_source.data_ptr())
        self.assertIsNone(no_grad_leaf.grad)
        self.assertIsNone(no_grad_other.grad)

    def test_tensorbase_descriptor_and_documentation(self):
        tensor = torch.tensor([1.0, 2.0])
        other = torch.zeros((2, 1))
        descriptor = inspect.getattr_static(torch.Tensor, "view_as")
        bound = tensor.view_as

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "view_as")
        self.assertEqual(descriptor.__qualname__, "TensorBase.view_as")
        self.assertEqual(bound.__name__, "view_as")
        self.assertEqual(bound.__qualname__, "Tensor.view_as")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor),
            "<method 'view_as' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.view_as, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor, other).shape, (2, 1))
        self.assertEqual(descriptor(tensor, other=other).shape, (2, 1))
        self.assertTrue(hasattr(torch.Tensor, "view"))
        self.assertTrue(hasattr(tensor, "view"))

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        other = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "view_as")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        cases = (
            ("positional", lambda: tensor.view_as(other), (tensor, other), None),
            (
                "keyword",
                lambda: tensor.view_as(other=other),
                (tensor,),
                {"other": other},
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.view_as(1)
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view_as(other=other)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"other": other})
        self.assertTrue(forwarded.is_set_to(tensor.reshape(other.shape)))

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                tensor.view_as(other)
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.Tensor.view_as'; "
                "all __torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_other_torch_function_override_precedence_and_not_implemented(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        descriptor = inspect.getattr_static(torch.Tensor, "view_as")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        for case, call, expected_args, expected_kwargs in (
            (
                "positional",
                lambda value: tensor.view_as(value),
                lambda value: (tensor, value),
                lambda value: None,
            ),
            (
                "keyword",
                lambda value: tensor.view_as(other=value),
                lambda value: (tensor,),
                lambda value: {"other": value},
            ),
        ):
            value = Override()
            Override.calls.clear()
            with self.subTest(case=case):
                self.assertIs(call(value), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, expected_args(value))
                self.assertEqual(kwargs, expected_kwargs(value))

        events = []

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                events.append(("mode", func, dispatch_types, args, kwargs))
                return self.result

        value = Override()
        Override.calls.clear()
        with RecordingMode(marker):
            self.assertIs(tensor.view_as(other=value), marker)
        self.assertEqual([event[0] for event in events], ["mode"])
        self.assertEqual(Override.calls, [])
        self.assertEqual(events[0][2], (Override,))

        events.clear()
        with RecordingMode(NotImplemented):
            self.assertIs(tensor.view_as(other=value), marker)
        self.assertEqual([event[0] for event in events], ["mode"])
        self.assertEqual(len(Override.calls), 1)
        self.assertEqual(Override.calls[0][1], (Override,))

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                events.append(("forward", func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        events.clear()
        Override.calls.clear()
        with ForwardingMode():
            self.assertIs(tensor.view_as(value), marker)
        self.assertEqual([event[0] for event in events], ["forward"])
        self.assertEqual(len(Override.calls), 1)

        mutation_events = []

        class MutableOverride:
            pass

        def original_handler(func, dispatch_types, args=(), kwargs=None):
            mutation_events.append("original")
            return marker

        def replacement_handler(func, dispatch_types, args=(), kwargs=None):
            mutation_events.append("replacement")
            return marker

        MutableOverride.__torch_function__ = classmethod(original_handler)

        class MutatingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                mutation_events.append("mode")
                MutableOverride.__torch_function__ = classmethod(replacement_handler)
                return NotImplemented

        with MutatingMode():
            self.assertIs(tensor.view_as(MutableOverride()), marker)
        self.assertEqual(mutation_events, ["mode", "replacement"])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            tensor.view_as(DecliningOverride())
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.Tensor.view_as'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented",
        )

        events.clear()
        with self.assertRaises(TypeError) as raised:
            with RecordingMode(NotImplemented):
                tensor.view_as(DecliningOverride())
        self.assertIn("  - mode object ", str(raised.exception))
        self.assertIn(
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>",
            str(raised.exception),
        )

    def test_binding_and_tensor_type_error_precedence(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "view_as")
        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.view_as() needs an argument",
            ),
            (
                lambda: descriptor(self=tensor, other=other),
                "unbound method TensorBase.view_as() needs an argument",
            ),
            (
                lambda: descriptor(1, other),
                "descriptor 'view_as' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: tensor.view_as(),
                'view_as() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.view_as(other, other),
                "view_as() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.view_as(other, other=other),
                "view_as() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.view_as(foo=other),
                'view_as() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.view_as(other, extra=True),
                "view_as() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.view_as(1),
                "view_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.view_as(None),
                "view_as(): argument 'other' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.view_as([]),
                "view_as(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.view_as(np.zeros((2, 3), dtype=np.float32)),
                "view_as(): argument 'other' (position 1) must be Tensor, "
                "not numpy.ndarray",
            ),
            (
                lambda: tensor.view_as(other=1),
                "view_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.view_as(other=None),
                "view_as(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.view_as(other=[]),
                "view_as(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: tensor.view_as(**{"other": 1, "extra": True}),
                "view_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.view_as(**{"extra": True, "other": 1}),
                "view_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.view_as(1, other=other),
                "view_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.view_as(1, extra=True),
                "view_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
