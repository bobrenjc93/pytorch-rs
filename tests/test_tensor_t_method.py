import gc
import inspect
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch
from tests.signature_utils import assert_no_argument_signature


class TensorTMethodTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, *, shape, stride, offset=0):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected, dtype=np.float32)
        )

    def test_scalar_vector_and_matrix_are_unwarned_alias_views(self):
        cases = (
            (torch.tensor(3.5), 3.5, (), ()),
            (torch.tensor([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], (3,), (1,)),
            (
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
                (3, 2),
                (1, 3),
            ),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            views = [source.t() for source, _, _, _ in cases]

        self.assertEqual(caught, [])
        for (source, expected, shape, stride), view in zip(
            cases, views, strict=True
        ):
            with self.subTest(shape=source.shape):
                self.assertIsNot(view, source)
                self.assertIs(view.dtype, source.dtype)
                self.assertEqual(view.device, source.device)
                self.assertEqual(view.requires_grad, source.requires_grad)
                self.assert_tensor(view, expected, shape=shape, stride=stride)

        matrix = cases[-1][0]
        self.assertEqual(matrix.t().tolist(), matrix.T.tolist())
        self.assertEqual(matrix.t().tolist(), matrix.transpose(0, 1).tolist())

    def test_rank_guard_and_no_argument_binding_contract(self):
        for rank in (3, 4, 65):
            shape = (0,) + (1,) * (rank - 1)
            with self.subTest(rank=rank):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^t\(\) expects a tensor with <= 2 dimensions, but self is {rank}D$",
                ):
                    torch.zeros(shape).t()

        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "t")
        bound = tensor.t
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "t")
        self.assertEqual(bound.__name__, "t")
        self.assertEqual(
            descriptor.__doc__, "\nt() -> Tensor\n\nSee :func:`torch.t`\n"
        )
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor(tensor).shape, (3, 2))

        calls = (
            (lambda: tensor.t(1), "Tensor.t() takes no arguments (1 given)"),
            (
                lambda: tensor.t(dim=0),
                "Tensor.t() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method Tensor.t() needs an argument",
            ),
            (
                lambda: descriptor(tensor, 1),
                "Tensor.t() takes no arguments (1 given)",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_offsets_strides_empties_lifetime_and_double_transpose(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())
        matrix = source.transpose(0, 2)[1]
        view = matrix.t()
        self.assert_tensor(
            view,
            values[:, :, 1],
            shape=(2, 3),
            stride=(12, 4),
            offset=1,
        )
        restored = view.t()
        self.assertIsNot(restored, matrix)
        self.assert_tensor(
            restored,
            values[:, :, 1].T,
            shape=(3, 2),
            stride=(4, 12),
            offset=1,
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        empty_t = empty.t()
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(empty_t.shape, (2, 0))
        self.assertEqual(empty_t.stride(), (3, 3))
        self.assertEqual(empty_t.storage_offset(), 1)
        self.assertEqual(empty_t.numel(), 0)
        self.assertEqual(empty_t.tolist(), [[], []])

        def view_after_source_drops():
            temporary = torch.tensor(values.tolist())
            return temporary.transpose(0, 2)[2].t()

        surviving = view_after_source_drops()
        gc.collect()
        self.assert_tensor(
            surviving,
            values[:, :, 2],
            shape=(2, 3),
            stride=(12, 4),
            offset=2,
        )

    def test_t_records_the_inverse_autograd_transform(self):
        leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        transformed = leaf.transpose(0, 2)[1].t()
        weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertTrue(transformed.requires_grad)
        self.assertFalse(transformed.is_leaf)
        (transformed * weights).sum().backward()

        expected = np.zeros((2, 3, 4), dtype=np.float32)
        expected[:, :, 1] = np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

        scalar = torch.tensor(2.0, requires_grad=True)
        (scalar.t() * 5.0).backward()
        self.assertEqual(scalar.grad.item(), 5.0)

        vector = torch.tensor([2.0, 3.0], requires_grad=True)
        (vector.t() * torch.tensor([7.0, 11.0])).sum().backward()
        np.testing.assert_array_equal(np.asarray(vector.grad), [7.0, 11.0])

        with torch.no_grad():
            no_grad_view = leaf[0].t()
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)


if __name__ == "__main__":
    unittest.main()
