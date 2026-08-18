import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


UNSUPPORTED = "atleast_3d() only supports a single Tensor input"


class Atleast3dTests(unittest.TestCase):
    def make_base(self, *, requires_grad=False):
        return torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=requires_grad,
        )

    def test_rank_three_and_higher_tensors_are_returned_exactly(self):
        base = self.make_base()
        cases = (
            base,
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2),
            torch.zeros((1, 2, 0, 3)),
        )
        for source in cases:
            with self.subTest(shape=source.shape, stride=source.stride()):
                self.assertIs(torch.atleast_3d(source), source)

    def test_matrices_become_trailing_singleton_shared_storage_views(self):
        base = self.make_base()
        cases = (
            base[1],
            base[1].transpose(0, 1),
            base.transpose(0, 2)[2],
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            torch.zeros((2, 0, 3))[1].transpose(0, 1),
        )
        for source in cases:
            with self.subTest(
                shape=source.shape,
                stride=source.stride(),
                offset=source.storage_offset(),
            ):
                result = torch.atleast_3d(source)
                repeated = torch.atleast_3d(source)
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, (*source.shape, 1))
                self.assertEqual(result.stride(), (*source.stride(), 1))
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(repeated))
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(
                    np.asarray(result), np.expand_dims(np.asarray(source), -1)
                )

    def test_vectors_gain_leading_and_trailing_singleton_dimensions(self):
        base = self.make_base()
        empty_strided = (
            torch.zeros((2, 0, 3)).transpose(0, 2)[1].transpose(0, 1)[1]
        )
        cases = (
            base[1, 2],
            base.transpose(0, 2)[3].transpose(0, 1)[1],
            torch.zeros((0,)),
            empty_strided,
        )
        for source in cases:
            with self.subTest(
                shape=source.shape,
                stride=source.stride(),
                offset=source.storage_offset(),
            ):
                result = torch.atleast_3d(source)
                repeated = torch.atleast_3d(source)
                source_stride = source.stride()[0]
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, (1, source.shape[0], 1))
                self.assertEqual(
                    result.stride(),
                    (source.shape[0] * source_stride, source_stride, 1),
                )
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(repeated))
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(
                    np.asarray(result), np.asarray(source).reshape(1, -1, 1)
                )

    def test_scalars_become_one_by_one_by_one_shared_storage_views(self):
        base = self.make_base()
        for source in (torch.tensor(-0.0), base.transpose(0, 2)[3, 2, 1]):
            with self.subTest(offset=source.storage_offset()):
                result = torch.atleast_3d(source)
                direct = source.reshape((1, 1, 1))
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, (1, 1, 1))
                self.assertEqual(result.stride(), (1, 1, 1))
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(direct))
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))

    def test_autograd_repeated_backward_empty_views_and_no_grad(self):
        scalar_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        scalar_result = torch.atleast_3d(scalar_leaf[1])
        self.assertFalse(scalar_result.is_leaf)
        self.assertEqual(scalar_result.output_nr, 0)
        scalar_loss = scalar_result.sum()
        scalar_loss.backward()
        scalar_loss.backward()
        self.assertEqual(scalar_leaf.grad.tolist(), [0.0, 2.0, 0.0])

        vector_leaf = self.make_base(requires_grad=True)
        vector_source = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
        vector_result = torch.atleast_3d(vector_source)
        self.assertFalse(vector_result.is_leaf)
        self.assertEqual(vector_result.output_nr, 0)
        self.assertEqual(vector_result.shape, (1, 3, 1))
        self.assertEqual(vector_result.stride(), (12, 4, 1))
        vector_loss = vector_result.sum()
        vector_loss.backward()
        vector_loss.backward()
        expected_vector_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_vector_grad[1, :, 3] = 2.0
        np.testing.assert_array_equal(
            np.asarray(vector_leaf.grad), expected_vector_grad
        )

        matrix_leaf = self.make_base(requires_grad=True)
        matrix_source = matrix_leaf[1].transpose(0, 1)
        matrix_result = torch.atleast_3d(matrix_source)
        self.assertFalse(matrix_result.is_leaf)
        self.assertEqual(matrix_result.output_nr, 0)
        self.assertEqual(matrix_result.shape, (4, 3, 1))
        self.assertEqual(matrix_result.stride(), (1, 4, 1))
        matrix_loss = matrix_result.sum()
        matrix_loss.backward()
        matrix_loss.backward()
        expected_matrix_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_matrix_grad[1] = 2.0
        np.testing.assert_array_equal(
            np.asarray(matrix_leaf.grad), expected_matrix_grad
        )

        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty_source = empty_leaf.transpose(0, 2)[1].transpose(0, 1)[1]
        empty_result = torch.atleast_3d(empty_source)
        self.assertEqual(empty_result.shape, (1, 0, 1))
        self.assertEqual(empty_result.stride(), (0, 3, 1))
        empty_result.sum().backward()
        self.assertEqual(empty_leaf.grad.shape, (2, 0, 3))
        self.assertEqual(empty_leaf.grad.tolist(), [[], []])

        scalar_source = torch.tensor(3.0, requires_grad=True)
        vector_source = torch.tensor([1.0, 2.0], requires_grad=True)
        matrix_source = torch.tensor([[1.0, 2.0]], requires_grad=True)
        rank_three_leaf = torch.zeros((1, 2, 3), requires_grad=True)
        rank_three_source = rank_three_leaf * 2.0
        with torch.no_grad():
            scalar_no_grad = torch.atleast_3d(scalar_source)
            vector_no_grad = torch.atleast_3d(vector_source)
            matrix_no_grad = torch.atleast_3d(matrix_source)
            rank_three_no_grad = torch.atleast_3d(rank_three_source)

        for result, source in (
            (scalar_no_grad, scalar_source),
            (vector_no_grad, vector_source),
            (matrix_no_grad, matrix_source),
        ):
            self.assertTrue(result.requires_grad)
            self.assertTrue(result.is_leaf)
            self.assertEqual(result.output_nr, 0)
            self.assertEqual(result.data_ptr(), source.data_ptr())
            (result * result).sum().backward()
            self.assertIsNone(source.grad)
            self.assertIsNone(result.grad)

        self.assertIs(rank_three_no_grad, rank_three_source)
        rank_three_no_grad.sum().backward()
        self.assertEqual(rank_three_leaf.grad.tolist(), [[[2.0] * 3] * 2])

    def test_modes_and_overrides_receive_the_public_function(self):
        source = torch.tensor([[1.0, 2.0]])
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.atleast_3d(value), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.atleast_3d)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {})

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.atleast_3d(source)
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.atleast_3d for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == (source,) for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(result.shape, (1, 2, 1))
        self.assertEqual(result.stride(), (2, 1, 1))
        self.assertEqual(result.data_ptr(), source.data_ptr())

    def test_function_metadata_exports_and_pickle(self):
        function = torch.atleast_3d
        self.assertIs(function, torch.functional.atleast_3d)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "atleast_3d")
        self.assertEqual(function.__qualname__, "atleast_3d")
        self.assertEqual(function.__module__, "torch_rs.functional")
        self.assertEqual(str(inspect.signature(function)), "(*tensors)")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(torch.__all__.count("atleast_3d"), 1)
        self.assertEqual(torch.functional.__all__.count("atleast_3d"), 1)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["atleast_3d"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_single_input_errors_and_unsupported_forms(self):
        invalid_message = (
            "atleast_3d() received an invalid combination of arguments - got "
            "(NoneType), but expected one of:\n * (Tensor input)\n      didn't "
            "match because some of the arguments have invalid types: "
            "(!NoneType!)\n * (tuple of Tensors tensors)\n      didn't match "
            "because some of the arguments have invalid types: (!NoneType!)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_message)}$"):
            torch.atleast_3d(None)
        with self.assertRaisesRegex(
            TypeError,
            "^atleast_3d\\(\\) got an unexpected keyword argument 'input'$",
        ):
            torch.atleast_3d(input=torch.tensor(1.0))

        source = torch.tensor(1.0)
        unsupported_calls = (
            lambda: torch.atleast_3d(),
            lambda: torch.atleast_3d(source, source),
            lambda: torch.atleast_3d((source,)),
            lambda: torch.atleast_3d([source]),
        )
        for call in unsupported_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                call()


if __name__ == "__main__":
    unittest.main()
