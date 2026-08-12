import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class OverflowThenIndex:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        if self.calls == 1:
            raise OverflowError("raised by __index__")
        return 2


class ChangingIndex:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 3 if self.calls == 1 else 2


class UnpackOverflowIndex:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        if self.calls == 1:
            return 2
        raise OverflowError("raised during unpack")


class ReshapeTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, shape, stride, offset=0):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected, dtype=np.float32))

    def test_positional_keyword_scalar_and_empty_forms(self):
        source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        calls = (
            torch.reshape(source, (3, 2)),
            torch.reshape(source, shape=[3, 2]),
            torch.reshape(input=source, shape=(3, 2)),
        )
        for output in calls:
            self.assertIsNot(output, source)
            self.assert_tensor(
                output,
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                (3, 2),
                (2, 1),
            )

        scalar = torch.reshape(torch.tensor([7.0]), ())
        self.assert_tensor(scalar, 7.0, (), ())
        self.assertEqual(scalar.item(), 7.0)

        empty = torch.reshape(input=torch.zeros((0,)), shape=[2, -1, 3])
        self.assert_tensor(empty, np.empty((2, 0, 3)), (2, 0, 3), (3, 3, 1))

        indexed_shape = torch.reshape(
            source,
            (np.int64(2), IndexDimension(3)),
        )
        self.assert_tensor(indexed_shape, source.tolist(), (2, 3), (3, 1))

    def test_view_or_copy_layout_and_lifetime_match_tensor_reshape(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())

        offset_source = base[1]
        view = torch.reshape(offset_source, (2, 6))
        method_view = offset_source.reshape((2, 6))
        self.assert_tensor(view, values[1].reshape(2, 6), (2, 6), (6, 1), 12)
        self.assertEqual(view.stride(), method_view.stride())
        self.assertEqual(view.storage_offset(), method_view.storage_offset())

        non_contiguous = base.transpose(0, 1)
        copied = torch.reshape(non_contiguous, (6, 4))
        method_copy = non_contiguous.reshape((6, 4))
        expected = values.transpose(1, 0, 2).reshape(6, 4)
        self.assert_tensor(copied, expected, (6, 4), (4, 1))
        self.assertEqual(copied.stride(), method_copy.stride())
        self.assertEqual(copied.storage_offset(), method_copy.storage_offset())

        del base, offset_source, non_contiguous
        self.assert_tensor(view, values[1].reshape(2, 6), (2, 6), (6, 1), 12)
        self.assert_tensor(copied, expected, (6, 4), (4, 1))

    def test_autograd_matches_tensor_reshape_for_view_and_copy_paths(self):
        gradients = []
        for reshape in (
            lambda tensor: torch.reshape(tensor, (6,)),
            lambda tensor: tensor.reshape((6,)),
        ):
            leaf = torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            output = reshape(leaf.transpose(0, 1))
            self.assertTrue(output.requires_grad)
            weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

        expected = np.asarray([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]])
        np.testing.assert_array_equal(gradients[0], expected)
        np.testing.assert_array_equal(gradients[0], gradients[1])

        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            self.assertTrue(torch.reshape(leaf, (2, 1)).requires_grad)
            copied = torch.reshape(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
                ).transpose(0, 1),
                (6,),
            )
            self.assertFalse(copied.requires_grad)
        self.assertTrue(torch.reshape(leaf, (2, 1)).requires_grad)

    def test_signature_and_binding_errors_match_pytorch_2_13(self):
        self.assertIs(type(torch.reshape), types.BuiltinFunctionType)
        self.assertIsNone(torch.reshape.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(torch.reshape)

        tensor = torch.zeros((6,))
        cases = (
            (
                lambda: torch.reshape(),
                TypeError,
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor),
                TypeError,
                'reshape() missing 1 required positional arguments: "shape"',
            ),
            (
                lambda: torch.reshape(shape=(2, 3)),
                TypeError,
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor, 2, 3),
                TypeError,
                "reshape() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3), input=tensor),
                TypeError,
                "reshape() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3), shape=(2, 3)),
                TypeError,
                "reshape() got multiple values for argument 'shape'",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3), extra=True),
                TypeError,
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape(tensor, (2**100, 3), input=tensor),
                TypeError,
                "reshape() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reshape(tensor, (2**100, 3), shape=(2, 3)),
                TypeError,
                "reshape() got multiple values for argument 'shape'",
            ),
            (
                lambda: torch.reshape(tensor, (2**100, 3), extra=True),
                TypeError,
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape(tensor, shape=(2**100, 3), extra=True),
                TypeError,
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape([], (0,)),
                TypeError,
                "reshape(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.reshape(input=None, shape=()),
                TypeError,
                "reshape(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.reshape(tensor, 6),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, not int",
            ),
            (
                lambda: torch.reshape(tensor, shape=6),
                TypeError,
                "reshape(): argument 'shape' must be tuple of ints, not int",
            ),
            (
                lambda: torch.reshape(tensor, (2.0, 3)),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, but found element of type float at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, (2.0, 3), extra=True),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, but found element of type float at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3.0)),
                TypeError,
                "reshape(): argument 'shape' failed to unpack the object at pos 2 with error \"type must be tuple of ints,but got float\"",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3.0), input=tensor),
                TypeError,
                "reshape() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3.0), extra=True),
                TypeError,
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape(tensor, shape=(2.0, 3)),
                TypeError,
                "reshape(): argument 'shape' must be tuple of ints, not tuple",
            ),
            (
                lambda: torch.reshape(tensor, (True, 6)),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, but found element of type bool at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, ((2, 3),)),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, but found element of type tuple at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, (4, 2)),
                RuntimeError,
                "shape '[4, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.reshape(tensor, (-1, -1)),
                RuntimeError,
                "only one dimension can be inferred",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_user_index_overflow_is_not_retried_or_deferred(self):
        tensor = torch.zeros((6,))
        message = (
            "reshape(): argument 'shape' (position 2) must be tuple of ints, "
            "but found element of type OverflowThenIndex at pos 0"
        )
        for kwargs in ({}, {"input": tensor}, {"extra": True}):
            with self.subTest(keywords=tuple(kwargs)):
                dimension = OverflowThenIndex()
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    torch.reshape(tensor, (dimension, 3), **kwargs)
                self.assertEqual(dimension.calls, 1)

    def test_later_dimensions_unpack_once_after_keyword_binding(self):
        bool_output = torch.reshape(torch.zeros((2,)), (2, True))
        self.assertEqual(bool_output.shape, (2, 1))
        self.assertEqual(bool_output.stride(), (1, 1))

        dimension = ChangingIndex()
        changing_output = torch.reshape(torch.zeros((6,)), (2, dimension))
        self.assertEqual(changing_output.shape, (2, 3))
        self.assertEqual(dimension.calls, 1)

        tensor = torch.zeros((6,))
        for kwargs, message in (
            ({"input": tensor}, "reshape() got multiple values for argument 'input'"),
            ({"extra": True}, "reshape() got an unexpected keyword argument 'extra'"),
        ):
            with self.subTest(keywords=tuple(kwargs)):
                dimension = ChangingIndex()
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    torch.reshape(tensor, (2, dimension), **kwargs)
                self.assertEqual(dimension.calls, 0)

    def test_unpack_stage_protocol_failure_uses_unpack_diagnostic(self):
        tensor = torch.zeros((6,))
        message = (
            "reshape(): argument 'shape' failed to unpack the object at pos 1 "
            'with error "type must be tuple of ints,but got UnpackOverflowIndex"'
        )
        for keyword_shape in (False, True):
            with self.subTest(keyword_shape=keyword_shape):
                dimension = UnpackOverflowIndex()
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    if keyword_shape:
                        torch.reshape(input=tensor, shape=(dimension, 3))
                    else:
                        torch.reshape(tensor, (dimension, 3))
                self.assertEqual(dimension.calls, 2)


if __name__ == "__main__":
    unittest.main()
