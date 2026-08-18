import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


UNSUPPORTED = "atleast_2d() only supports a single Tensor input"


class Atleast2dTests(unittest.TestCase):
    def make_base(self):
        return torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )

    def test_rank_two_and_higher_tensors_are_returned_exactly(self):
        base = self.make_base()
        cases = (
            base[1],
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        for source in cases:
            with self.subTest(shape=source.shape, stride=source.stride()):
                result = torch.atleast_2d(source)
                self.assertIs(result, source)

    def test_vectors_become_leading_singleton_shared_storage_views(self):
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
                result = torch.atleast_2d(source)
                repeated = torch.atleast_2d(source)
                source_stride = source.stride()[0]
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, (1, source.shape[0]))
                self.assertEqual(
                    result.stride(),
                    (source.shape[0] * source_stride, source_stride),
                )
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(repeated))
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(
                    np.asarray(result), np.asarray(source).reshape(1, -1)
                )

    def test_scalars_become_one_by_one_shared_storage_reshape_views(self):
        base = self.make_base()
        for source in (torch.tensor(-0.0), base.transpose(0, 2)[3, 2, 1]):
            with self.subTest(offset=source.storage_offset()):
                result = torch.atleast_2d(source)
                direct = source.reshape((1, 1))
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, (1, 1))
                self.assertEqual(result.stride(), (1, 1))
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(direct))
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))

    def test_autograd_repeated_backward_and_no_grad(self):
        scalar_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        scalar_result = torch.atleast_2d(scalar_leaf[1])
        self.assertFalse(scalar_result.is_leaf)
        self.assertEqual(scalar_result.output_nr, 0)
        scalar_loss = scalar_result.sum()
        scalar_loss.backward()
        scalar_loss.backward()
        self.assertEqual(scalar_leaf.grad.tolist(), [0.0, 2.0, 0.0])

        vector_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        vector_source = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
        vector_result = torch.atleast_2d(vector_source)
        self.assertFalse(vector_result.is_leaf)
        self.assertEqual(vector_result.output_nr, 0)
        self.assertEqual(vector_result.shape, (1, 3))
        self.assertEqual(vector_result.stride(), (12, 4))
        vector_loss = vector_result.sum()
        vector_loss.backward()
        vector_loss.backward()
        expected_vector_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_vector_grad[1, :, 3] = 2.0
        np.testing.assert_array_equal(
            np.asarray(vector_leaf.grad), expected_vector_grad
        )

        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty_source = empty_leaf.transpose(0, 2)[1].transpose(0, 1)[1]
        empty_result = torch.atleast_2d(empty_source)
        self.assertEqual(empty_result.shape, (1, 0))
        self.assertEqual(empty_result.stride(), (0, 3))
        empty_result.sum().backward()
        self.assertEqual(empty_leaf.grad.shape, (2, 0, 3))
        self.assertEqual(empty_leaf.grad.tolist(), [[], []])

        scalar_source = torch.tensor(3.0, requires_grad=True)
        vector_source = torch.tensor([1.0, 2.0], requires_grad=True)
        matrix_leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        matrix_source = matrix_leaf * 2.0
        with torch.no_grad():
            scalar_no_grad = torch.atleast_2d(scalar_source)
            vector_no_grad = torch.atleast_2d(vector_source)
            matrix_no_grad = torch.atleast_2d(matrix_source)

        for result, source in (
            (scalar_no_grad, scalar_source),
            (vector_no_grad, vector_source),
        ):
            self.assertTrue(result.requires_grad)
            self.assertTrue(result.is_leaf)
            self.assertEqual(result.output_nr, 0)
            self.assertEqual(result.data_ptr(), source.data_ptr())
            (result * result).sum().backward()
            self.assertIsNone(source.grad)
            self.assertIsNone(result.grad)

        self.assertIs(matrix_no_grad, matrix_source)
        self.assertTrue(matrix_no_grad.requires_grad)
        self.assertFalse(matrix_no_grad.is_leaf)
        matrix_no_grad.sum().backward()
        self.assertEqual(matrix_leaf.grad.tolist(), [[2.0, 2.0]])

    def test_modes_and_overrides_receive_the_public_function(self):
        source = torch.tensor([1.0, 2.0])
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.atleast_2d(value), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.atleast_2d)
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
                result = torch.atleast_2d(source)
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.atleast_2d for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == (source,) for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result.stride(), (2, 1))
        self.assertEqual(result.data_ptr(), source.data_ptr())

    def test_function_metadata_exports_and_pickle(self):
        function = torch.atleast_2d
        self.assertIs(function, torch.functional.atleast_2d)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "atleast_2d")
        self.assertEqual(function.__qualname__, "atleast_2d")
        self.assertEqual(function.__module__, "torch_rs.functional")
        self.assertEqual(str(inspect.signature(function)), "(*tensors)")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(torch.__all__.count("atleast_2d"), 1)
        self.assertEqual(torch.functional.__all__.count("atleast_2d"), 1)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["atleast_2d"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_single_input_errors_and_unsupported_forms(self):
        invalid_message = (
            "atleast_2d() received an invalid combination of arguments - got "
            "(NoneType), but expected one of:\n * (Tensor input)\n      didn't "
            "match because some of the arguments have invalid types: "
            "(!NoneType!)\n * (tuple of Tensors tensors)\n      didn't match "
            "because some of the arguments have invalid types: (!NoneType!)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_message)}$"):
            torch.atleast_2d(None)
        with self.assertRaisesRegex(
            TypeError,
            "^atleast_2d\\(\\) got an unexpected keyword argument 'input'$",
        ):
            torch.atleast_2d(input=torch.tensor(1.0))

        source = torch.tensor(1.0)
        unsupported_calls = (
            lambda: torch.atleast_2d(),
            lambda: torch.atleast_2d(source, source),
            lambda: torch.atleast_2d((source,)),
            lambda: torch.atleast_2d([source]),
        )
        for call in unsupported_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                call()


if __name__ == "__main__":
    unittest.main()
