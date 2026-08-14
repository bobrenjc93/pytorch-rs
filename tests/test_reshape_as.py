import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nreshape_as(other) -> Tensor\n\n"
    "Returns this tensor as the same shape as :attr:`other`.\n"
    "``self.reshape_as(other)`` is equivalent to ``self.reshape(other.sizes())``.\n"
    "This method returns a view if ``other.sizes()`` is compatible with the current\n"
    "shape. See :meth:`torch.Tensor.view` on when it is possible to return a view.\n"
    "\n"
    "Please see :meth:`reshape` for more information about ``reshape``.\n"
    "\n"
    "Args:\n"
    "    other (:class:`torch.Tensor`): The result tensor has the same shape\n"
    "        as :attr:`other`.\n"
)


class TensorReshapeAsTests(unittest.TestCase):
    def assert_reshape_result(
        self,
        result,
        direct,
        source,
        *,
        expected_shape,
        expected_stride,
        expected_offset,
        aliases,
    ):
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
        self.assertEqual(result.data_ptr() == source.data_ptr(), aliases)
        self.assertEqual(result.is_set_to(direct), aliases)

    def test_positional_and_keyword_calls_reuse_reshape_for_every_layout(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        cases = (
            (
                "scalar",
                torch.tensor(-0.0),
                torch.tensor(8.0),
                (),
                (),
                0,
                True,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                torch.zeros((2, 0)),
                (2, 0),
                (1, 1),
                1,
                True,
            ),
            (
                "contiguous-with-strided-other",
                base,
                torch.zeros((4, 6)).transpose(0, 1),
                (6, 4),
                (4, 1),
                0,
                True,
            ),
            (
                "contiguous-offset",
                base[1],
                torch.zeros((2, 6)),
                (2, 6),
                (6, 1),
                12,
                True,
            ),
            (
                "transposed-copy",
                base.transpose(0, 2),
                torch.zeros((6, 4)),
                (6, 4),
                (4, 1),
                0,
                False,
            ),
        )

        for (
            case,
            source,
            other,
            expected_shape,
            expected_stride,
            expected_offset,
            aliases,
        ) in cases:
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    result = (
                        source.reshape_as(other=other)
                        if keyword
                        else source.reshape_as(other)
                    )
                    direct = source.reshape(other.shape)
                    self.assert_reshape_result(
                        result,
                        direct,
                        source,
                        expected_shape=expected_shape,
                        expected_stride=expected_stride,
                        expected_offset=expected_offset,
                        aliases=aliases,
                    )

    def test_extreme_empty_shape_and_shape_mismatch_reuse_reshape(self):
        maximum = sys.maxsize
        source = torch.zeros((0,))
        other = torch.zeros((0,)).reshape((0, maximum, maximum))

        result = source.reshape_as(other=other)
        direct = source.reshape(other.shape)
        self.assertEqual(result.shape, direct.shape)
        self.assertEqual(result.stride(), direct.stride())
        self.assertEqual(result.storage_offset(), direct.storage_offset())
        self.assertTrue(result.is_set_to(direct))
        self.assertEqual(result.tolist(), [])

        incompatible = torch.zeros((2, 2))
        with self.assertRaises(RuntimeError) as reshape_as_raised:
            torch.zeros((6,)).reshape_as(incompatible)
        with self.assertRaises(RuntimeError) as reshape_raised:
            torch.zeros((6,)).reshape(incompatible.shape)
        self.assertEqual(
            str(reshape_as_raised.exception), str(reshape_raised.exception)
        )

    def test_view_and_copy_autograd_mappings_match_reshape(self):
        cases = (
            (
                "view",
                False,
                torch.zeros((3, 2), requires_grad=True),
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                True,
            ),
            (
                "copy",
                True,
                torch.zeros((6,), requires_grad=True),
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
                False,
            ),
        )
        for case, transpose, other, weights, expected_gradient, aliases in cases:
            with self.subTest(case=case):
                leaf = torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    requires_grad=True,
                )
                source = leaf.transpose(0, 1) if transpose else leaf
                result = source.reshape_as(other)
                direct = source.reshape(other.shape)

                self.assertEqual(result.shape, direct.shape)
                self.assertEqual(result.stride(), direct.stride())
                self.assertEqual(result.requires_grad, direct.requires_grad)
                self.assertEqual(result.is_leaf, direct.is_leaf)
                self.assertEqual(result.data_ptr() == source.data_ptr(), aliases)
                self.assertEqual(result.is_set_to(direct), aliases)

                (result * torch.tensor(weights)).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), np.asarray(expected_gradient)
                )
                self.assertIsNone(other.grad)

    def test_metadata_only_view_and_copy_graphs_allow_repeated_backward(self):
        for case, transpose, other in (
            ("view", False, torch.zeros((3, 2), requires_grad=True)),
            ("copy", True, torch.zeros((6,), requires_grad=True)),
        ):
            with self.subTest(case=case):
                leaf = torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    requires_grad=True,
                )
                source = leaf.transpose(0, 1) if transpose else leaf
                loss = source.reshape_as(other=other).sum()

                loss.backward()
                loss.backward()

                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), np.full((2, 3), 2.0, dtype=np.float32)
                )
                self.assertIsNone(other.grad)

    def test_no_grad_view_and_copy_states_match_reshape(self):
        for case, transpose, other, aliases, requires_grad in (
            ("view", False, torch.zeros((4,)), True, True),
            ("copy", True, torch.zeros((4,)), False, False),
        ):
            with self.subTest(case=case):
                leaf = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
                )
                source = leaf.transpose(0, 1) if transpose else leaf
                with torch.no_grad():
                    result = source.reshape_as(other)
                    direct = source.reshape(other.shape)

                self.assertEqual(result.shape, direct.shape)
                self.assertEqual(result.stride(), direct.stride())
                self.assertEqual(result.storage_offset(), direct.storage_offset())
                self.assertEqual(result.requires_grad, direct.requires_grad)
                self.assertEqual(result.is_leaf, direct.is_leaf)
                self.assertEqual(result.requires_grad, requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertEqual(result.data_ptr() == source.data_ptr(), aliases)
                self.assertEqual(result.is_set_to(direct), aliases)

    def test_tensorbase_descriptor_documentation_and_unbound_calls(self):
        tensor = torch.tensor([1.0, 2.0])
        other = torch.zeros((2, 1))
        descriptor = inspect.getattr_static(torch.Tensor, "reshape_as")
        bound = tensor.reshape_as

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "reshape_as")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor, other).shape, (2, 1))
        self.assertEqual(descriptor(tensor, other=other).shape, (2, 1))

    def test_binding_and_tensor_type_error_precedence(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "reshape_as")
        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.reshape_as() needs an argument",
            ),
            (
                lambda: descriptor(1, other),
                "descriptor 'reshape_as' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: tensor.reshape_as(),
                'reshape_as() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.reshape_as(other, other),
                "reshape_as() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.reshape_as(other, other=other),
                "reshape_as() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.reshape_as(foo=other),
                'reshape_as() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.reshape_as(other, extra=True),
                "reshape_as() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.reshape_as(1),
                "reshape_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.reshape_as(None),
                "reshape_as(): argument 'other' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.reshape_as([]),
                "reshape_as(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.reshape_as(np.zeros((2, 3), dtype=np.float32)),
                "reshape_as(): argument 'other' (position 1) must be Tensor, "
                "not numpy.ndarray",
            ),
            (
                lambda: tensor.reshape_as(other=1),
                "reshape_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.reshape_as(other=None),
                "reshape_as(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.reshape_as(other=[]),
                "reshape_as(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: tensor.reshape_as(**{"other": 1, "extra": True}),
                "reshape_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.reshape_as(**{"extra": True, "other": 1}),
                "reshape_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.reshape_as(1, other=other),
                "reshape_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.reshape_as(1, extra=True),
                "reshape_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
