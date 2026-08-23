import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


UNSUPPORTED = "atleast_2d() only supports a single Tensor input"
UNSUPPORTED_SEQUENCE = (
    "atleast_2d() sequence inputs only support an exact tuple or list of "
    "exact Tensors"
)


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
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                result = torch.atleast_2d(sequence_type(sources))
                self.assertIs(type(result), tuple)
                self.assertEqual(len(result), len(sources))

                for source, item in zip(sources, result, strict=True):
                    direct = torch.atleast_2d(source)
                    if len(source.shape) >= 2:
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
                result = torch.atleast_2d(empty)
                self.assertIs(type(result), tuple)
                self.assertEqual(result, ())

    def test_variadic_tensors_use_native_views_in_order(self):
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
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        result = torch.atleast_2d(*sources)
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), len(sources))

        for source, item in zip(sources, result, strict=True):
            direct = torch.atleast_2d(source)
            if len(source.shape) >= 2:
                self.assertIs(item, source)
            else:
                self.assertIsNot(item, source)
                self.assertTrue(item.is_set_to(direct))
            self.assertEqual(item.shape, direct.shape)
            self.assertEqual(item.stride(), direct.stride())
            self.assertEqual(item.storage_offset(), direct.storage_offset())
            self.assertEqual(item.data_ptr(), source.data_ptr())
            self.assertIs(item.dtype, source.dtype)
            self.assertEqual(item.device, source.device)
            self.assertEqual(item.layout, source.layout)
            np.testing.assert_array_equal(np.asarray(item), np.asarray(direct))

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

    def test_sequence_autograd_repeated_backward_and_no_grad(self):
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                scalar_leaf = torch.tensor(
                    [1.0, 2.0, 3.0], requires_grad=True
                )
                scalar = scalar_leaf[1]
                vector_leaf = torch.tensor(
                    np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
                    requires_grad=True,
                )
                vector = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
                empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
                empty = empty_leaf.transpose(0, 2)[1].transpose(0, 1)[1]
                matrix_leaf = torch.tensor(
                    [[1.0, 2.0]], requires_grad=True
                )
                matrix = matrix_leaf * 2.0

                results = torch.atleast_2d(
                    sequence_type((scalar, vector, empty, matrix))
                )
                self.assertIs(type(results), tuple)
                scalar_result, vector_result, empty_result, matrix_result = results
                self.assertEqual(scalar_result.shape, (1, 1))
                self.assertEqual(scalar_result.stride(), (1, 1))
                self.assertFalse(scalar_result.is_leaf)
                self.assertEqual(scalar_result.data_ptr(), scalar.data_ptr())
                self.assertEqual(vector_result.shape, (1, 3))
                self.assertEqual(vector_result.stride(), (12, 4))
                self.assertFalse(vector_result.is_leaf)
                self.assertEqual(vector_result.data_ptr(), vector.data_ptr())
                self.assertEqual(empty_result.shape, (1, 0))
                self.assertEqual(empty_result.stride(), (0, 3))
                self.assertEqual(empty_result.data_ptr(), empty.data_ptr())
                self.assertIs(matrix_result, matrix)

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

                empty_result.sum().backward()
                self.assertEqual(empty_leaf.grad.shape, (2, 0, 3))
                self.assertEqual(empty_leaf.grad.tolist(), [[], []])

                matrix_result.sum().backward()
                self.assertEqual(matrix_leaf.grad.tolist(), [[2.0, 2.0]])

                no_grad_scalar = torch.tensor(3.0, requires_grad=True)
                no_grad_vector = torch.tensor(
                    [1.0, 2.0], requires_grad=True
                )
                no_grad_matrix_leaf = torch.tensor(
                    [[1.0, 2.0]], requires_grad=True
                )
                no_grad_matrix = no_grad_matrix_leaf * 2.0
                with torch.no_grad():
                    scalar_result, vector_result, matrix_result = torch.atleast_2d(
                        sequence_type(
                            (no_grad_scalar, no_grad_vector, no_grad_matrix)
                        )
                    )

                for result, source in (
                    (scalar_result, no_grad_scalar),
                    (vector_result, no_grad_vector),
                ):
                    self.assertTrue(result.requires_grad)
                    self.assertTrue(result.is_leaf)
                    self.assertEqual(result.output_nr, 0)
                    self.assertEqual(result.data_ptr(), source.data_ptr())
                    (result * result).sum().backward()
                    self.assertIsNone(source.grad)
                    self.assertIsNone(result.grad)

                self.assertIs(matrix_result, no_grad_matrix)
                self.assertTrue(matrix_result.requires_grad)
                self.assertFalse(matrix_result.is_leaf)
                matrix_result.sum().backward()
                self.assertEqual(
                    no_grad_matrix_leaf.grad.tolist(), [[2.0, 2.0]]
                )

    def test_variadic_autograd_repeated_backward_and_no_grad(self):
        scalar_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        scalar = scalar_leaf[1]
        vector_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        vector = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty = empty_leaf.transpose(0, 2)[1].transpose(0, 1)[1]
        matrix_leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        matrix = matrix_leaf * 2.0

        results = torch.atleast_2d(scalar, vector, empty, matrix)
        self.assertIs(type(results), tuple)
        scalar_result, vector_result, empty_result, matrix_result = results
        self.assertEqual(scalar_result.shape, (1, 1))
        self.assertEqual(scalar_result.stride(), (1, 1))
        self.assertFalse(scalar_result.is_leaf)
        self.assertEqual(scalar_result.data_ptr(), scalar.data_ptr())
        self.assertEqual(vector_result.shape, (1, 3))
        self.assertEqual(vector_result.stride(), (12, 4))
        self.assertFalse(vector_result.is_leaf)
        self.assertEqual(vector_result.data_ptr(), vector.data_ptr())
        self.assertEqual(empty_result.shape, (1, 0))
        self.assertEqual(empty_result.stride(), (0, 3))
        self.assertEqual(empty_result.data_ptr(), empty.data_ptr())
        self.assertIs(matrix_result, matrix)

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

        empty_result.sum().backward()
        self.assertEqual(empty_leaf.grad.shape, (2, 0, 3))
        self.assertEqual(empty_leaf.grad.tolist(), [[], []])

        matrix_result.sum().backward()
        self.assertEqual(matrix_leaf.grad.tolist(), [[2.0, 2.0]])

        no_grad_scalar_leaf = torch.tensor(
            [1.0, 2.0, 3.0], requires_grad=True
        )
        no_grad_scalar = no_grad_scalar_leaf[1]
        no_grad_vector_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        no_grad_vector = no_grad_vector_leaf.transpose(0, 2)[3].transpose(
            0, 1
        )[1]
        no_grad_empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        no_grad_empty = no_grad_empty_leaf.transpose(0, 2)[1].transpose(
            0, 1
        )[1]
        no_grad_matrix_leaf = torch.tensor(
            [[1.0, 2.0]], requires_grad=True
        )
        no_grad_matrix = no_grad_matrix_leaf * 2.0
        with torch.no_grad():
            (
                scalar_result,
                vector_result,
                empty_result,
                matrix_result,
            ) = torch.atleast_2d(
                no_grad_scalar,
                no_grad_vector,
                no_grad_empty,
                no_grad_matrix,
            )

        for result, source, leaf in (
            (scalar_result, no_grad_scalar, no_grad_scalar_leaf),
            (vector_result, no_grad_vector, no_grad_vector_leaf),
            (empty_result, no_grad_empty, no_grad_empty_leaf),
        ):
            self.assertTrue(result.requires_grad)
            self.assertTrue(result.is_leaf)
            self.assertEqual(result.output_nr, 0)
            self.assertEqual(result.storage_offset(), source.storage_offset())
            self.assertEqual(result.data_ptr(), source.data_ptr())
            (result * result).sum().backward()
            self.assertIsNone(leaf.grad)
            self.assertIsNone(result.grad)

        self.assertEqual(scalar_result.shape, (1, 1))
        self.assertEqual(scalar_result.stride(), (1, 1))
        self.assertEqual(vector_result.shape, (1, 3))
        self.assertEqual(vector_result.stride(), (12, 4))
        self.assertEqual(empty_result.shape, (1, 0))
        self.assertEqual(empty_result.stride(), (0, 3))

        self.assertIs(matrix_result, no_grad_matrix)
        self.assertTrue(matrix_result.requires_grad)
        self.assertFalse(matrix_result.is_leaf)
        matrix_result.sum().backward()
        self.assertEqual(no_grad_matrix_leaf.grad.tolist(), [[2.0, 2.0]])

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
                    torch.atleast_2d(sequence)
        self.assertEqual(Override.calls, [])

    def test_variadic_operand_overrides_follow_subclass_precedence(self):
        source = torch.tensor([1.0, 2.0])
        marker = object()
        events = []

        class BaseOverride:
            label = "base"
            result = NotImplemented

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((cls.label, func, types, args, kwargs))
                return cls.result

        class SubOverride(BaseOverride):
            label = "sub"

        class AcceptingOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("accepting", func, types, args, kwargs))
                return marker

        operands = (
            BaseOverride(),
            source,
            AcceptingOverride(),
            SubOverride(),
            BaseOverride(),
        )
        self.assertIs(torch.atleast_2d(*operands), marker)
        self.assertEqual([event[0] for event in events], ["sub", "base", "accepting"])
        self.assertTrue(all(event[1] is torch.atleast_2d for event in events))
        self.assertTrue(
            all(
                event[2]
                == (SubOverride, BaseOverride, torch.Tensor, AcceptingOverride)
                for event in events
            )
        )
        self.assertTrue(
            all(
                len(event[3]) == len(operands)
                and all(
                    argument is operand
                    for argument, operand in zip(
                        event[3], operands, strict=True
                    )
                )
                for event in events
            )
        )
        self.assertTrue(all(event[4] == {} for event in events))
        self.assertEqual(len({id(event[4]) for event in events}), 1)

    def test_variadic_operand_overrides_follow_modes_and_restore_the_stack(self):
        source = torch.tensor([1.0, 2.0])
        override_marker = object()
        mode_marker = object()
        events = []

        def stack():
            return tuple(torch.overrides._get_current_function_mode_stack())

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("override", func, types, args, kwargs, stack()))
                return override_marker

        value = Override()
        operands = (source, value)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label, result):
                self.label = label
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (self.label, func, types, args, kwargs, stack())
                )
                return self.result

        accepting = RecordingMode("accepting-mode", mode_marker)
        with accepting:
            self.assertIs(torch.atleast_2d(*operands), mode_marker)
            self.assertEqual(stack(), (accepting,))
        self.assertEqual([event[0] for event in events], ["accepting-mode"])

        events.clear()
        declining = RecordingMode("declining-mode", NotImplemented)
        with declining:
            self.assertIs(torch.atleast_2d(*operands), override_marker)
            self.assertEqual(stack(), (declining,))
        self.assertEqual(
            [event[0] for event in events],
            ["declining-mode", "override"],
        )
        self.assertEqual(events[0][5], ())
        self.assertEqual(events[1][5], (declining,))
        self.assertIs(events[0][4], events[1][4])

        events.clear()

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (self.label, func, types, args, kwargs, stack())
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                self.assertIs(torch.atleast_2d(*operands), override_marker)
                self.assertEqual(stack(), (lower, upper))
        self.assertEqual(
            [event[0] for event in events],
            ["upper", "lower", "override"],
        )
        self.assertEqual(events[0][5], (lower,))
        self.assertEqual(events[1][5], ())
        self.assertEqual(events[2][5], ())
        self.assertTrue(all(event[1] is torch.atleast_2d for event in events))
        self.assertTrue(
            all(event[2] == (torch.Tensor, Override) for event in events)
        )
        self.assertTrue(all(event[3] == operands for event in events))
        self.assertTrue(all(event[4] == {} for event in events))
        self.assertEqual(stack(), ())

    def test_variadic_operand_override_failures_restore_the_stack(self):
        source = torch.tensor([1.0, 2.0])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^no implementation found for "
            "'torch_rs\\.functional\\.atleast_2d' on types that implement "
            "__torch_function__:",
        ):
            torch.atleast_2d(source, DecliningOverride())

        expected_error = ValueError("operand failed")

        class RaisingOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise expected_error

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        mode = DecliningMode()
        with mode:
            with self.assertRaisesRegex(
                TypeError,
                "^no implementation found for "
                "'torch_rs\\.functional\\.atleast_2d' on types that "
                "implement __torch_function__:",
            ):
                torch.atleast_2d(source, DecliningOverride())
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )

        with mode:
            with self.assertRaises(ValueError) as raised:
                torch.atleast_2d(source, RaisingOverride())
            self.assertIs(raised.exception, expected_error)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_variadic_exact_tensors_dispatch_through_nested_modes(self):
        scalar = torch.tensor(1.0)
        vector = torch.tensor([2.0, 3.0])
        sources = (scalar, vector)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            result = torch.atleast_2d(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [accepting],
            )
        self.assertIs(result, marker)
        self.assertEqual(
            accepting.calls,
            [(torch.atleast_2d, (torch.Tensor,), sources, {})],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        self.label,
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                results = torch.atleast_2d(*sources)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )

        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.atleast_2d for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == sources for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(calls[0][5], (lower,))
        self.assertEqual(calls[1][5], ())
        self.assertEqual(
            torch.overrides._get_current_function_mode_stack(), []
        )
        self.assertEqual(tuple(result.shape for result in results), ((1, 1), (1, 2)))
        self.assertTrue(
            all(
                result.data_ptr() == source.data_ptr()
                for result, source in zip(results, sources, strict=True)
            )
        )

    def test_variadic_declining_and_raising_modes_restore_the_stack(self):
        sources = (torch.tensor(1.0), torch.tensor([2.0, 3.0]))

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return NotImplemented

        declining = DecliningMode()
        with declining:
            with self.assertRaisesRegex(
                TypeError,
                "^no implementation found for "
                "'torch_rs\\.functional\\.atleast_2d' on types that implement "
                "__torch_function__: \\[\\] nor in mode ",
            ):
                torch.atleast_2d(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [declining],
            )
        self.assertEqual(
            declining.calls,
            [
                (torch.atleast_2d, (torch.Tensor,), sources, {}),
                (torch.atleast_2d, (), sources, {}),
            ],
        )

        expected_error = ValueError("mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        raising = RaisingMode()
        with raising:
            with self.assertRaises(ValueError) as raised:
                torch.atleast_2d(*sources)
            self.assertIs(raised.exception, expected_error)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [raising],
            )
        self.assertEqual(
            torch.overrides._get_current_function_mode_stack(), []
        )

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
                self.assertIs(torch.atleast_2d(sequence), marker)
                function, dispatch_types, args, kwargs = override_type.calls[0]
                self.assertIs(function, torch.atleast_2d)
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
        self.assertIs(torch.atleast_2d(spoofed), marker)
        function, dispatch_types, args, kwargs = SpoofedSequence.calls[0]
        self.assertIs(function, torch.atleast_2d)
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
            result = torch.atleast_2d(sequence)
        self.assertIs(result, marker)
        self.assertEqual(
            mode.calls,
            [(torch.atleast_2d, (), (sequence,), {})],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            result = torch.atleast_2d(sequence)
        self.assertEqual(calls, [(torch.atleast_2d, (), (sequence,), {})])
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 2))
        self.assertEqual(result[0].data_ptr(), source.data_ptr())

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
            lambda: torch.atleast_2d(source, None),
            lambda: torch.atleast_2d(None, source),
            lambda: torch.atleast_2d(source, source, 1),
        )
        for call in unsupported_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                call()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode, self.assertRaisesRegex(
            TypeError, f"^{re.escape(UNSUPPORTED)}$"
        ):
            torch.atleast_2d(source, None)
        self.assertEqual(mode.calls, [])

        mixed_sequences = (
            (source, None),
            [source, 1],
            ((source,),),
        )
        for sequence in mixed_sequences:
            with self.subTest(sequence=sequence), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED_SEQUENCE)}$"
            ):
                torch.atleast_2d(sequence)


if __name__ == "__main__":
    unittest.main()
