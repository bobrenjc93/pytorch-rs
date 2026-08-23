import copy
import gc
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\n        unflatten(dim, sizes) -> Tensor\n\n"
    "        See :func:`torch.unflatten`.\n\n"
    "        "
)

if sys.version_info >= (3, 13):
    METHOD_DOC = inspect.cleandoc(METHOD_DOC) + "\n"


class IndexSize:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class TensorUnflattenTests(unittest.TestCase):
    def assert_view(
        self,
        source,
        result,
        *,
        shape,
        stride,
        offset,
        values,
    ):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        np.testing.assert_array_equal(np.asarray(result), values)

    def test_tuple_list_size_and_inference_delegate_to_native_view(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        for form, sizes in (
            ("tuple", (2, -1)),
            ("list", [2, -1]),
            ("Size", torch.Size((2, -1))),
        ):
            with self.subTest(form=form):
                source = torch.tensor(values.tolist())
                result = source.unflatten(2, sizes)
                self.assert_view(
                    source,
                    result,
                    shape=(2, 3, 2, 2, 5),
                    stride=(60, 20, 10, 5, 1),
                    offset=0,
                    values=values.reshape(2, 3, 2, 2, 5),
                )

        source = torch.tensor(values.tolist())
        keyword = source.unflatten(dim=-2, sizes=(2, 2))
        self.assert_view(
            source,
            keyword,
            shape=(2, 3, 2, 2, 5),
            stride=(60, 20, 10, 5, 1),
            offset=0,
            values=values.reshape(2, 3, 2, 2, 5),
        )

    def test_offset_empty_and_compatible_noncontiguous_views_preserve_layout(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        base = torch.tensor(values.tolist())

        offset = base.transpose(0, 1)[1]
        offset_result = offset.unflatten(1, (2, 2))
        self.assert_view(
            offset,
            offset_result,
            shape=(2, 2, 2, 5),
            stride=(60, 10, 5, 1),
            offset=20,
            values=values.transpose(1, 0, 2, 3)[1].reshape(2, 2, 2, 5),
        )

        noncontiguous = base.transpose(0, 1)
        noncontiguous_result = noncontiguous.unflatten(-1, (1, 5))
        self.assert_view(
            noncontiguous,
            noncontiguous_result,
            shape=(3, 2, 4, 1, 5),
            stride=(20, 60, 5, 5, 1),
            offset=0,
            values=values.transpose(1, 0, 2, 3).reshape(3, 2, 4, 1, 5),
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        empty_result = empty.unflatten(0, (0, 1))
        self.assert_view(
            empty,
            empty_result,
            shape=(0, 1, 2),
            stride=(2, 2, 1),
            offset=1,
            values=np.empty((0, 1, 2), dtype=np.float32),
        )

        unchanged_shape = empty.unflatten(0, (0,))
        self.assertTrue(unchanged_shape.is_set_to(empty))
        self.assertIsNot(unchanged_shape, empty)

    def test_views_outlive_temporary_owners(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def make_view():
            return torch.tensor(values.tolist())[1].unflatten(1, (2, 2))

        result = make_view()
        gc.collect()
        self.assertEqual(result.shape, (3, 2, 2))
        self.assertEqual(result.stride(), (4, 2, 1))
        self.assertEqual(result.storage_offset(), 12)
        np.testing.assert_array_equal(result, values[1].reshape(3, 2, 2))

    def test_view_backward_repeated_backward_and_no_grad_semantics(self):
        leaf = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 6).tolist(),
            requires_grad=True,
        )
        result = leaf.unflatten(1, (2, 3))
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(result),
            ", grad_fn=<ViewBackward0>",
        )

        weights = torch.tensor(
            np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3).tolist()
        )
        (result * weights).sum().backward()
        np.testing.assert_array_equal(
            leaf.grad,
            np.arange(1, 13, dtype=np.float32).reshape(2, 6),
        )

        repeated_leaf = torch.zeros((2, 6), requires_grad=True)
        loss = repeated_leaf.unflatten(1, (2, 3)).sum()
        loss.backward()
        loss.backward()
        np.testing.assert_array_equal(
            repeated_leaf.grad,
            np.full((2, 6), 2.0, dtype=np.float32),
        )

        no_grad_leaf = torch.zeros((2, 6), requires_grad=True)
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_result = no_grad_source.unflatten(0, (2, 3))
        self.assertEqual(no_grad_result.shape, (2, 3, 2))
        self.assertEqual(no_grad_result.stride(), (3, 1, 6))
        self.assertEqual(no_grad_result.data_ptr(), no_grad_source.data_ptr())
        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_empty_inference_and_extreme_metadata(self):
        empty = torch.zeros((2, 0, 3))
        for sizes, shape, stride in (
            ((-1,), (2, 0, 3), (3, 3, 1)),
            ((0, 1), (2, 0, 1, 3), (3, 3, 3, 1)),
            ((1, 0), (2, 1, 0, 3), (3, 3, 3, 1)),
            ((2, 0), (2, 2, 0, 3), (6, 3, 3, 1)),
            ((0, 2), (2, 0, 2, 3), (6, 6, 3, 1)),
        ):
            with self.subTest(sizes=sizes):
                result = empty.unflatten(1, sizes)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.stride(), stride)
                self.assertEqual(result.data_ptr(), empty.data_ptr())

        maximum = sys.maxsize
        extreme = torch.zeros((0,)).unflatten(
            0, (-1, maximum, maximum)
        )
        self.assertEqual(extreme.shape, (0, maximum, maximum))
        self.assertEqual(extreme.stride(), (1, maximum, 1))
        self.assertEqual(extreme.numel(), 0)

        with self.assertRaisesRegex(
            RuntimeError, "^numel: integer multiplication overflow$"
        ):
            torch.zeros((0,)).unflatten(0, (maximum, maximum, -1))

    def test_binding_integer_conversion_and_error_precedence(self):
        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            tensor.unflatten(np.int64(-2), (np.int32(1), np.uint64(3))).shape,
            (2, 1, 3, 4),
        )

        first = IndexSize(1)
        second = IndexSize(3)
        self.assertEqual(tensor.unflatten(1, (first, second)).shape, (2, 1, 3, 4))
        self.assertEqual((first.calls, second.calls), (2, 1))

        bool_tail = torch.zeros((2, 1)).unflatten(1, (1, True))
        self.assertEqual(bool_tail.shape, (2, 1, 1))

        exact_errors = (
            (
                lambda: tensor.unflatten(True, (1, 3)),
                TypeError,
                "unflatten(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: tensor.unflatten(1.0, (1, 3)),
                TypeError,
                "unflatten(): argument 'dim' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.unflatten(1, 3),
                TypeError,
                "unflatten(): argument 'sizes' (position 2) must be tuple of ints, not int",
            ),
            (
                lambda: tensor.unflatten(1, (True, 3)),
                TypeError,
                "unflatten(): argument 'sizes' (position 2) must be tuple of ints, but found element of type bool at pos 0",
            ),
            (
                lambda: tensor.unflatten(1, (1, 3.0)),
                TypeError,
                "unflatten(): argument 'sizes' failed to unpack the object at pos 2 with error \"type must be tuple of ints,but got float\"",
            ),
            (
                lambda: tensor.unflatten(3, (1, 3)),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: tensor.unflatten(-4, (1, 3)),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
            ),
            (
                lambda: tensor.unflatten(-2, (1, 2)),
                RuntimeError,
                "unflatten: Provided sizes [1, 2] don't multiply up to the size of dim 1 (3) in the input tensor",
            ),
        )
        for call, error_type, message in exact_errors:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        for sizes in ((), [], torch.Size(), None, False, {}):
            with self.subTest(empty_sizes=repr(sizes)):
                with self.assertRaisesRegex(
                    RuntimeError, "^unflatten: sizes must be non-empty$"
                ):
                    tensor.unflatten(True, sizes)

        unexpected_errors = (
            ((-1, -1), "only one dimension can be inferred"),
            ((-2,), "invalid shape dimension -2 at index 0 of shape [-2]"),
            (
                (0, -1),
                "cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous",
            ),
        )
        for sizes, message in unexpected_errors[:2]:
            with self.subTest(sizes=sizes):
                with self.assertRaises(RuntimeError) as raised:
                    tensor.unflatten(1, sizes)
                self.assertEqual(
                    str(raised.exception),
                    f"unflatten got an unexpected error:\n{message}",
                )
        with self.assertRaises(RuntimeError) as raised:
            torch.zeros((0,)).unflatten(0, unexpected_errors[2][0])
        self.assertEqual(
            str(raised.exception),
            "unflatten got an unexpected error:\n" + unexpected_errors[2][1],
        )

    def test_python_function_metadata_binding_and_unsupported_overloads(self):
        tensor = torch.zeros((2, 3))
        function = inspect.getattr_static(torch.Tensor, "unflatten")
        bound = tensor.unflatten

        self.assertIs(type(function), types.FunctionType)
        self.assertIs(type(bound), types.MethodType)
        self.assertEqual(function.__name__, "unflatten")
        self.assertEqual(function.__qualname__, "Tensor.unflatten")
        self.assertEqual(function.__module__, "torch_rs._tensor")
        self.assertEqual(bound.__module__, "torch_rs._tensor")
        self.assertEqual(function.__doc__, METHOD_DOC)
        self.assertEqual(function.__annotations__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(str(inspect.signature(function)), "(self, dim, sizes)")
        self.assertEqual(str(inspect.signature(bound)), "(dim, sizes)")
        self.assertIn("unflatten", torch.Tensor.__dict__)
        self.assertTrue(
            all("unflatten" not in owner.__dict__ for owner in torch.Tensor.__mro__[1:])
        )
        self.assertIs(torch._tensor.Tensor.unflatten, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)
        self.assertEqual(function(tensor, 1, (1, 3)).shape, (2, 1, 3))
        self.assertEqual(function(self=tensor, dim=1, sizes=(1, 3)).shape, (2, 1, 3))

        binding_errors = (
            (
                lambda: bound(),
                "Tensor.unflatten() missing 2 required positional arguments: 'dim' and 'sizes'",
            ),
            (
                lambda: bound(1),
                "Tensor.unflatten() missing 1 required positional argument: 'sizes'",
            ),
            (
                lambda: bound(1, (1, 3), 0),
                "Tensor.unflatten() takes 3 positional arguments but 4 were given",
            ),
            (
                lambda: bound(1, (1, 3), dim=1),
                "Tensor.unflatten() got multiple values for argument 'dim'",
            ),
            (
                lambda: bound(1, (1, 3), sizes=(1, 3)),
                "Tensor.unflatten() got multiple values for argument 'sizes'",
            ),
            (
                lambda: bound(1, (1, 3), unexpected=True),
                "Tensor.unflatten() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in binding_errors:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(torch, "unflatten"))
        with self.assertRaisesRegex(TypeError, "must be int, not str"):
            tensor.unflatten("features", (1, 3))
        with self.assertRaisesRegex(
            TypeError, "found element of type tuple at pos 0"
        ):
            tensor.unflatten(1, (("rows", 1), ("columns", 3)))

        symbolic = type(
            "SymInt",
            (),
            {"__module__": "torch", "__index__": lambda self: 1},
        )()
        with self.assertRaisesRegex(TypeError, "torch.SymInt"):
            tensor.unflatten(1, (symbolic, 3))

    def test_torch_function_overrides_and_modes_receive_python_function(self):
        tensor = torch.zeros((2, 3))
        function = inspect.getattr_static(torch.Tensor, "unflatten")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(function(value, 1, (1, 3)), marker)
        called_function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(called_function, function)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value, 1, (1, 3)))
        self.assertEqual(kwargs, {})

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            result = tensor.unflatten(dim=1, sizes=[1, 3])
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        mode_function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(mode_function, function)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1:], (1, [1, 3]))
        self.assertEqual(kwargs, {})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.unflatten(1, (1, 3))
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.shape, (2, 1, 3))

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                tensor.unflatten(1, (1, 3))
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^no implementation found for 'torch_rs\._tensor\.unflatten' on "
                r"types that implement __torch_function__: \[\] nor in mode "
                r"<.*RecordingMode object at 0x[0-9a-f]+>$"
            ),
        )
        self.assertEqual(len(declining.calls), 2)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
