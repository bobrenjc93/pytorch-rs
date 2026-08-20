import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


UNSUPPORTED = "atleast_3d() only supports a single Tensor input"
UNSUPPORTED_SEQUENCE = (
    "atleast_3d() sequence inputs only support an exact tuple or list of "
    "exact Tensors"
)


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

    def test_tuple_and_list_sequences_use_native_views(self):
        base = self.make_base()
        empty_strided = (
            torch.zeros((2, 0, 3)).transpose(0, 2)[1].transpose(0, 1)[1]
        )
        sources = (
            torch.tensor(-0.0),
            base.transpose(0, 2)[3, 2, 1],
            base[1, 2],
            base.transpose(0, 2)[3].transpose(0, 1)[1],
            torch.zeros((0,)),
            empty_strided,
            base[1],
            base[1].transpose(0, 1),
            base.transpose(0, 2)[2],
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            torch.zeros((2, 0, 3))[1].transpose(0, 1),
            base.transpose(0, 2),
            torch.zeros((1, 2, 0, 3)),
        )
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                result = torch.atleast_3d(sequence_type(sources))
                self.assertIs(type(result), tuple)
                self.assertEqual(len(result), len(sources))

                for source, item in zip(sources, result, strict=True):
                    direct = torch.atleast_3d(source)
                    if len(source.shape) >= 3:
                        self.assertIs(item, source)
                    else:
                        self.assertIsNot(item, source)
                        self.assertTrue(item.is_set_to(direct))
                    self.assertEqual(item.shape, direct.shape)
                    self.assertEqual(item.stride(), direct.stride())
                    self.assertEqual(
                        item.storage_offset(), direct.storage_offset()
                    )
                    self.assertEqual(item.data_ptr(), source.data_ptr())
                    self.assertIs(item.dtype, source.dtype)
                    self.assertEqual(item.device, source.device)
                    self.assertEqual(item.layout, source.layout)
                    np.testing.assert_array_equal(
                        np.asarray(item), np.asarray(direct)
                    )

        for empty in ((), []):
            with self.subTest(empty_type=type(empty).__name__):
                result = torch.atleast_3d(empty)
                self.assertIs(type(result), tuple)
                self.assertEqual(result, ())

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

    def test_sequence_autograd_repeated_backward_empty_views_and_no_grad(self):
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                scalar_leaf = torch.tensor(
                    [1.0, 2.0, 3.0], requires_grad=True
                )
                scalar = scalar_leaf[1]
                vector_leaf = self.make_base(requires_grad=True)
                vector = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
                matrix_leaf = self.make_base(requires_grad=True)
                matrix = matrix_leaf[1].transpose(0, 1)
                empty_vector_leaf = torch.zeros(
                    (2, 0, 3), requires_grad=True
                )
                empty_vector = (
                    empty_vector_leaf.transpose(0, 2)[1]
                    .transpose(0, 1)[1]
                )
                empty_matrix_leaf = torch.zeros(
                    (2, 0, 3), requires_grad=True
                )
                empty_matrix = empty_matrix_leaf.transpose(0, 2)[1]
                rank_three_leaf = torch.zeros(
                    (1, 2, 3), requires_grad=True
                )
                rank_three = rank_three_leaf * 2.0

                sources = (
                    scalar,
                    vector,
                    matrix,
                    empty_vector,
                    empty_matrix,
                    rank_three,
                )
                results = torch.atleast_3d(sequence_type(sources))
                self.assertIs(type(results), tuple)
                self.assertEqual(len(results), len(sources))
                for source, result in zip(sources, results, strict=True):
                    direct = torch.atleast_3d(source)
                    self.assertEqual(result.shape, direct.shape)
                    self.assertEqual(result.stride(), direct.stride())
                    self.assertEqual(
                        result.storage_offset(), direct.storage_offset()
                    )
                    self.assertEqual(result.data_ptr(), source.data_ptr())
                    if len(source.shape) >= 3:
                        self.assertIs(result, source)
                    else:
                        self.assertTrue(result.is_set_to(direct))

                (
                    scalar_result,
                    vector_result,
                    matrix_result,
                    empty_vector_result,
                    empty_matrix_result,
                    rank_three_result,
                ) = results
                self.assertFalse(scalar_result.is_leaf)
                self.assertFalse(vector_result.is_leaf)
                self.assertFalse(matrix_result.is_leaf)
                self.assertIs(rank_three_result, rank_three)

                scalar_loss = scalar_result.sum()
                scalar_loss.backward()
                scalar_loss.backward()
                self.assertEqual(scalar_leaf.grad.tolist(), [0.0, 2.0, 0.0])

                vector_loss = vector_result.sum()
                vector_loss.backward()
                vector_loss.backward()
                expected_vector_grad = np.zeros((2, 3, 4), dtype=np.float32)
                expected_vector_grad[1, :, 3] = 2.0
                np.testing.assert_array_equal(
                    np.asarray(vector_leaf.grad), expected_vector_grad
                )

                matrix_loss = matrix_result.sum()
                matrix_loss.backward()
                matrix_loss.backward()
                expected_matrix_grad = np.zeros((2, 3, 4), dtype=np.float32)
                expected_matrix_grad[1] = 2.0
                np.testing.assert_array_equal(
                    np.asarray(matrix_leaf.grad), expected_matrix_grad
                )

                empty_vector_result.sum().backward()
                self.assertEqual(empty_vector_leaf.grad.shape, (2, 0, 3))
                self.assertEqual(empty_vector_leaf.grad.tolist(), [[], []])
                empty_matrix_result.sum().backward()
                self.assertEqual(empty_matrix_leaf.grad.shape, (2, 0, 3))
                self.assertEqual(empty_matrix_leaf.grad.tolist(), [[], []])

                rank_three_result.sum().backward()
                self.assertEqual(
                    rank_three_leaf.grad.tolist(), [[[2.0] * 3] * 2]
                )

                no_grad_scalar = torch.tensor(3.0, requires_grad=True)
                no_grad_vector = torch.tensor(
                    [1.0, 2.0], requires_grad=True
                )
                no_grad_matrix = torch.tensor(
                    [[1.0, 2.0]], requires_grad=True
                )
                no_grad_rank_three_leaf = torch.zeros(
                    (1, 2, 3), requires_grad=True
                )
                no_grad_rank_three = no_grad_rank_three_leaf * 2.0
                with torch.no_grad():
                    no_grad_results = torch.atleast_3d(
                        sequence_type(
                            (
                                no_grad_scalar,
                                no_grad_vector,
                                no_grad_matrix,
                                no_grad_rank_three,
                            )
                        )
                    )

                self.assertIs(type(no_grad_results), tuple)
                for result, source in zip(
                    no_grad_results[:3],
                    (no_grad_scalar, no_grad_vector, no_grad_matrix),
                    strict=True,
                ):
                    self.assertTrue(result.requires_grad)
                    self.assertTrue(result.is_leaf)
                    self.assertEqual(result.output_nr, 0)
                    self.assertEqual(result.data_ptr(), source.data_ptr())
                    (result * result).sum().backward()
                    self.assertIsNone(source.grad)
                    self.assertIsNone(result.grad)

                self.assertIs(no_grad_results[3], no_grad_rank_three)
                no_grad_results[3].sum().backward()
                self.assertEqual(
                    no_grad_rank_three_leaf.grad.tolist(),
                    [[[2.0] * 3] * 2],
                )

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

    def test_inner_sequence_override_dispatch_is_explicitly_unsupported(self):
        source = torch.tensor([1.0, 2.0])

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        value = Override()
        for sequence in ((source, value), [source, value]):
            with self.subTest(sequence_type=type(sequence).__name__):
                with self.assertRaisesRegex(
                    TypeError, f"^{re.escape(UNSUPPORTED_SEQUENCE)}$"
                ):
                    torch.atleast_3d(sequence)
        self.assertEqual(Override.calls, [])

    def test_outer_sequence_overrides_and_modes_precede_the_fast_path(self):
        source = torch.tensor([1.0, 2.0])
        marker = object()

        class TupleOverride(tuple):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        class ListOverride(list):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for sequence in (TupleOverride((source,)), ListOverride([source])):
            override_type = type(sequence)
            with self.subTest(override_type=override_type.__name__):
                self.assertIs(torch.atleast_3d(sequence), marker)
                function, dispatch_types, args, kwargs = override_type.calls[0]
                self.assertIs(function, torch.atleast_3d)
                self.assertEqual(dispatch_types, (override_type,))
                self.assertEqual(args, (sequence,))
                self.assertEqual(kwargs, {})

        class SpoofedSequence:
            calls = []

            @property
            def __class__(self):
                return tuple

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        spoofed = SpoofedSequence()
        self.assertTrue(isinstance(spoofed, tuple))
        self.assertIs(torch.atleast_3d(spoofed), marker)
        function, dispatch_types, args, kwargs = SpoofedSequence.calls[0]
        self.assertIs(function, torch.atleast_3d)
        self.assertEqual(dispatch_types, (SpoofedSequence,))
        self.assertEqual(args, (spoofed,))
        self.assertEqual(kwargs, {})

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        sequence = (source,)
        mode = RecordingMode(marker)
        with mode:
            result = torch.atleast_3d(sequence)
        self.assertIs(result, marker)
        self.assertEqual(
            mode.calls,
            [(torch.atleast_3d, (), (sequence,), {})],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            result = torch.atleast_3d(sequence)
        self.assertEqual(calls, [(torch.atleast_3d, (), (sequence,), {})])
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 2, 1))
        self.assertEqual(result[0].data_ptr(), source.data_ptr())

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
            lambda: torch.atleast_3d(source, source),
        )
        for call in unsupported_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                call()

        mixed_sequences = (
            (source, None),
            [source, 1],
            ((source,),),
        )
        for sequence in mixed_sequences:
            with self.subTest(sequence=sequence), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED_SEQUENCE)}$"
            ):
                torch.atleast_3d(sequence)


if __name__ == "__main__":
    unittest.main()
