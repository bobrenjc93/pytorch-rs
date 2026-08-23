import gc
import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorFullSliceIndexTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("vector", torch.tensor([-0.0, 1.0])),
            ("matrix", torch.tensor([[-0.0, 1.0], [2.0, 3.0]])),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("offset-noncontiguous", base.transpose(0, 3)[1]),
        )

    def assert_metadata_alias(self, source, alias):
        self.assertIsNot(alias, source)
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertTrue(alias.is_set_to(source))
        self.assertEqual(alias.data_ptr(), source.data_ptr())
        self.assertIs(alias.dtype, source.dtype)
        self.assertEqual(alias.device, source.device)
        self.assertEqual(alias.tolist(), source.tolist())

    def test_full_slice_returns_a_distinct_exact_metadata_alias(self):
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[:]
                self.assert_metadata_alias(source, alias)

        signed_zero = torch.tensor([-0.0])[:]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000)

    def test_singleton_full_slice_tuple_returns_a_distinct_exact_metadata_alias(self):
        index = (slice(None),)
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[index]
                self.assert_metadata_alias(source, alias)

        signed_zero = torch.tensor([-0.0])[index]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000)

    def test_double_full_slice_tuple_returns_a_distinct_exact_metadata_alias(self):
        for case, source in self.layout_cases():
            if len(source.shape) < 2:
                continue
            with self.subTest(case=case):
                alias = source[:, :]
                self.assert_metadata_alias(source, alias)

        signed_zero = torch.tensor([[-0.0]])[:, :]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000)

    def test_scalar_full_slice_raises_the_exact_pytorch_error(self):
        with self.assertRaises(IndexError) as raised:
            torch.tensor(-0.0)[:]
        self.assertEqual(
            str(raised.exception),
            "slice() cannot be applied to a 0-dim tensor.",
        )

    def test_scalar_singleton_full_slice_tuple_raises_too_many_indices(self):
        with self.assertRaises(IndexError) as raised:
            torch.tensor(-0.0)[(slice(None),)]
        self.assertEqual(
            str(raised.exception),
            "too many indices for tensor of dimension 0",
        )

    def test_double_full_slice_tuple_rejects_lower_rank_tensors(self):
        for dimensions, source in (
            (0, torch.tensor(-0.0)),
            (1, torch.tensor([-0.0, 1.0])),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(IndexError) as raised:
                    source[:, :]
                self.assertEqual(
                    str(raised.exception),
                    f"too many indices for tensor of dimension {dimensions}",
                )

    def assert_autograd_gradient_node_and_no_grad_leaf_status(
        self, index, node_name, diagnostic_rank=1
    ):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        alias = leaf[index]
        self.assert_metadata_alias(leaf, alias)
        self.assertTrue(alias.requires_grad)
        self.assertFalse(alias.is_leaf)
        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
        (alias * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), weights.tolist())

        diagnostic_values = [2.0] if diagnostic_rank == 1 else [[2.0]]
        diagnostic_display = "[2.]" if diagnostic_rank == 1 else "[[2.]]"
        diagnostic_leaf = torch.tensor(diagnostic_values, requires_grad=True)
        with self.assertRaises(ValueError) as raised:
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[index], training=False
            )
        self.assertEqual(
            str(raised.exception),
            "dropout probability has to be between 0 and 1, but got "
            f"tensor({diagnostic_display}, grad_fn=<{node_name}>)",
        )

        no_grad_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_alias = no_grad_source[index]
        self.assert_metadata_alias(no_grad_source, no_grad_alias)
        self.assertTrue(no_grad_alias.requires_grad)
        self.assertTrue(no_grad_alias.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_slice_autograd_gradient_node_and_no_grad_leaf_status(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            slice(None), "SliceBackward0"
        )

    def test_singleton_full_slice_tuple_uses_alias_autograd_semantics(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            (slice(None),), "AliasBackward0"
        )

    def test_double_full_slice_tuple_uses_alias_autograd_semantics(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            (slice(None), slice(None)), "AliasBackward0", diagnostic_rank=2
        )

    def test_empty_tuple_uses_alias_autograd_semantics(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            (), "AliasBackward0"
        )

    def assert_storage_and_autograd_survive_source_lifetime(self, index):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[index]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 2))
        self.assertEqual(surviving.stride(), (4, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(np.asarray(surviving), values[:, :, 1].T)

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[index]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights).T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_slice_storage_and_autograd_survive_source_lifetime(self):
        self.assert_storage_and_autograd_survive_source_lifetime(slice(None))

    def test_singleton_full_slice_tuple_survives_source_lifetime(self):
        self.assert_storage_and_autograd_survive_source_lifetime((slice(None),))

    def test_double_full_slice_tuple_survives_source_lifetime(self):
        self.assert_storage_and_autograd_survive_source_lifetime(
            (slice(None), slice(None))
        )

    def assert_dispatches_through_tensorbase_mode_before_indexing(self, index):
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )

        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        self.assertIs(type(function), types.WrapperDescriptorType)
        self.assertIs(function, descriptor)
        self.assertEqual(function.__qualname__, "TensorBase.__getitem__")
        self.assertEqual(function.__objclass__.__name__, "TensorBase")
        self.assertEqual(function.__objclass__.__module__, "torch._C")
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual(handler_stack, ())
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        scalar = torch.tensor(1.0)
        mode.calls.clear()
        with mode:
            scalar_result = scalar[index]
        self.assertIs(scalar_result, marker)
        self.assertEqual(len(mode.calls), 1)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_metadata_alias(source, forwarded)

    def test_full_slice_dispatches_through_tensorbase_mode_before_indexing(self):
        self.assert_dispatches_through_tensorbase_mode_before_indexing(slice(None))

    def test_singleton_full_slice_tuple_dispatches_with_original_index(self):
        self.assert_dispatches_through_tensorbase_mode_before_indexing(
            (slice(None),)
        )

    def test_double_full_slice_tuple_dispatches_with_original_index(self):
        self.assert_dispatches_through_tensorbase_mode_before_indexing(
            (slice(None), slice(None))
        )

    def test_tuple_subclasses_are_normalized_through_overridden_iteration(self):
        source = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

        class IntegerRemapTuple(tuple):
            def __iter__(self):
                return iter((0,))

        selected = source[IntegerRemapTuple((slice(None),))]
        self.assertEqual(selected.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(selected.shape, (3,))
        self.assertEqual(selected.stride(), (1,))
        self.assertEqual(selected.storage_offset(), 0)

        class FullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None),))

        alias = source[FullSliceRemapTuple((0,))]
        self.assert_metadata_alias(source, alias)

        class DoubleFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None), slice(None)))

        double_alias = source[DoubleFullSliceRemapTuple((0,))]
        self.assert_metadata_alias(source, double_alias)

        class EmptyRemapTuple(tuple):
            def __iter__(self):
                return iter(())

        empty_index = EmptyRemapTuple((0,))
        empty_alias = source[empty_index]
        self.assert_metadata_alias(source, empty_alias)
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            empty_index, "AliasBackward0"
        )

        class IterationErrorTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration exploded")

        with self.assertRaisesRegex(RuntimeError, "^tuple iteration exploded$"):
            source[IterationErrorTuple((slice(None),))]

    def test_existing_indices_and_unsupported_slice_forms_are_unchanged(self):
        tensor = torch.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ]
        )
        indexed = tensor[-1, 0]
        self.assertEqual(indexed.tolist(), [4.0, 5.0])
        self.assertEqual(indexed.stride(), (1,))
        self.assertEqual(indexed.storage_offset(), 4)
        self.assert_metadata_alias(tensor, tensor[...])

        unsupported = (
            slice(1, None),
            slice(None, -1),
            slice(None, None, 1),
            slice(None, None, 2),
            (slice(1, None),),
            (slice(None, -1),),
            (slice(None, None, 1),),
            (slice(None, None, 2),),
            (slice(1, None), slice(None)),
            (slice(None), slice(1, None)),
            (slice(None, None, 1), slice(None)),
            (slice(None), slice(None, None, 1)),
            (slice(None, None, 2), slice(None)),
            (slice(None), slice(None, None, 2)),
            (slice(None), 0),
            (0, slice(None)),
            (slice(None), Ellipsis),
            (slice(None), slice(None), slice(None)),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]


if __name__ == "__main__":
    unittest.main()
