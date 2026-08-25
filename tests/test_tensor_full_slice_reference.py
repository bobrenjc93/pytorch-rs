import gc
import inspect
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFullSliceIndexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "full-slice indexing differentials require pinned PyTorch 2.13.0"
            )

    def layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            module.tensor([-0.0, 1.0], dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            base[1],
            base.transpose(0, 3)[1],
        )

    def double_full_slice_layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            module.tensor(
                [[-0.0, 1.0], [2.0, 3.0]], dtype=module.float32
            ),
            module.zeros((2, 0, 3), dtype=module.float32),
            base[1],
            base.transpose(0, 3)[1],
        )

    def leading_integer_full_slice_layout_cases(self, module):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            (
                module.tensor(
                    [[1.0, 2.0], [-0.0, 4.0]], dtype=module.float32
                ),
                -1,
            ),
            (module.zeros((2, 0, 3), dtype=module.float32), 1),
            (base[1], -1),
            (base.transpose(0, 3), -1),
            (base.transpose(0, 3)[1], -1),
        )

    def higher_rank_full_slice_layout_cases(self, module, rank):
        shape = tuple(range(2, rank + 2))
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        base_shape = (2, *shape)
        base_values = np.arange(np.prod(base_shape), dtype=np.float32).reshape(
            base_shape
        )
        base = module.tensor(base_values.tolist(), dtype=module.float32)
        empty_shape = (shape[0], 0, *shape[2:])
        return (
            module.tensor(values.tolist(), dtype=module.float32),
            module.zeros(empty_shape, dtype=module.float32),
            base[1],
            base.transpose(0, rank)[1],
        )

    def full_slice_ellipsis_indices(self, slice_count):
        full_slices = (slice(None),) * slice_count
        return tuple(
            (*full_slices[:position], Ellipsis, *full_slices[position:])
            for position in range(slice_count + 1)
        )

    def full_slice_ellipsis_layout_cases(self, module, slice_count):
        if slice_count == 1:
            return self.layout_cases(module)
        if slice_count == 2:
            return self.double_full_slice_layout_cases(module)
        return self.higher_rank_full_slice_layout_cases(module, slice_count)

    def alias_contract(self, source, index):
        alias = source[index]
        values = np.asarray(alias.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": alias is not source,
            "shape": tuple(alias.shape),
            "stride": alias.stride(),
            "storage_offset": alias.storage_offset(),
            "same_logical_view": alias.is_set_to(source),
            "same_data_pointer": alias.data_ptr() == source.data_ptr(),
            "dtype": str(alias.dtype),
            "device": str(alias.device),
            "requires_grad": alias.requires_grad,
            "is_leaf": alias.is_leaf,
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def leading_integer_full_slice_contract(self, source, index, expected_index):
        expected = source[expected_index]
        selected = source[index, :]
        values = np.asarray(selected.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": selected is not expected,
            "shape": tuple(selected.shape),
            "stride": selected.stride(),
            "storage_offset": selected.storage_offset(),
            "is_contiguous": selected.is_contiguous(),
            "same_logical_view": selected.is_set_to(expected),
            "same_data_pointer": selected.data_ptr() == expected.data_ptr(),
            "dtype": str(selected.dtype),
            "device": str(selected.device),
            "requires_grad": selected.requires_grad,
            "is_leaf": selected.is_leaf,
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def assert_layout_values_aliasing_dtype_and_device_match_pytorch_2_13(
        self, index
    ):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.alias_contract(actual, index),
                    self.alias_contract(expected, index),
                )

    def test_layout_values_aliasing_dtype_and_device_match_pytorch_2_13(self):
        self.assert_layout_values_aliasing_dtype_and_device_match_pytorch_2_13(
            slice(None)
        )

    def test_singleton_tuple_layout_aliases_match_pytorch_2_13(self):
        self.assert_layout_values_aliasing_dtype_and_device_match_pytorch_2_13(
            (slice(None),)
        )

    def test_double_full_slice_tuple_layout_aliases_match_pytorch_2_13(self):
        index = (slice(None), slice(None))
        actual_cases = self.double_full_slice_layout_cases(torch)
        expected_cases = self.double_full_slice_layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.alias_contract(actual, index),
                    self.alias_contract(expected, index),
                )

    def test_leading_integer_full_slice_layouts_match_pytorch_2_13(self):
        actual_cases = self.leading_integer_full_slice_layout_cases(torch)
        expected_cases = self.leading_integer_full_slice_layout_cases(reference_torch)
        for case, ((actual, index), (expected, expected_index)) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(index, expected_index)
                self.assertEqual(
                    self.leading_integer_full_slice_contract(actual, index, index),
                    self.leading_integer_full_slice_contract(
                        expected, expected_index, expected_index
                    ),
                )

    def integer_protocol_contract(self, module):
        class IntSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        source = module.tensor(
            [[0.0, 1.0], [2.0, 3.0]], dtype=module.float32
        )
        index_value = IndexValue(-1)
        outcomes = tuple(
            self.leading_integer_full_slice_contract(source, index, -1)
            for index in (IntSubclass(-1), np.int64(-1), index_value)
        )
        return outcomes, index_value.calls

    def test_leading_integer_full_slice_index_protocol_matches_pytorch_2_13(self):
        self.assertEqual(
            self.integer_protocol_contract(torch),
            self.integer_protocol_contract(reference_torch),
        )

    def test_three_or_more_full_slice_tuple_layout_aliases_match_pytorch_2_13(
        self,
    ):
        for count in (3, 4):
            index = (slice(None),) * count
            actual_cases = self.higher_rank_full_slice_layout_cases(torch, count)
            expected_cases = self.higher_rank_full_slice_layout_cases(
                reference_torch, count
            )
            for case, (actual, expected) in enumerate(
                zip(actual_cases, expected_cases, strict=True)
            ):
                with self.subTest(count=count, case=case):
                    self.assertEqual(
                        self.alias_contract(actual, index),
                        self.alias_contract(expected, index),
                    )

    def test_full_slice_ellipsis_tuple_layout_aliases_match_pytorch_2_13(self):
        for slice_count in range(1, 5):
            actual_cases = self.full_slice_ellipsis_layout_cases(
                torch, slice_count
            )
            expected_cases = self.full_slice_ellipsis_layout_cases(
                reference_torch, slice_count
            )
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                for case, (actual, expected) in enumerate(
                    zip(actual_cases, expected_cases, strict=True)
                ):
                    with self.subTest(
                        slice_count=slice_count,
                        position=position,
                        case=case,
                    ):
                        self.assertEqual(
                            self.alias_contract(actual, index),
                            self.alias_contract(expected, index),
                        )

    def scalar_error_contract(self, module, index):
        try:
            module.tensor(-0.0, dtype=module.float32)[index]
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("scalar full slice unexpectedly succeeded")

    def test_scalar_error_matches_pytorch_2_13(self):
        self.assertEqual(
            self.scalar_error_contract(torch, slice(None)),
            self.scalar_error_contract(reference_torch, slice(None)),
        )

    def test_singleton_tuple_scalar_error_matches_pytorch_2_13(self):
        index = (slice(None),)
        self.assertEqual(
            self.scalar_error_contract(torch, index),
            self.scalar_error_contract(reference_torch, index),
        )

    def lower_rank_error_contract(self, module, shape, index):
        try:
            module.zeros(shape, dtype=module.float32)[index]
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("lower-rank full-slice tuple unexpectedly succeeded")

    def test_double_full_slice_lower_rank_errors_match_pytorch_2_13(self):
        index = (slice(None), slice(None))
        for shape in ((), (2,)):
            with self.subTest(shape=shape):
                self.assertEqual(
                    self.lower_rank_error_contract(torch, shape, index),
                    self.lower_rank_error_contract(reference_torch, shape, index),
                )

    def leading_integer_error_contract(self, module, shape, index):
        try:
            module.zeros(shape, dtype=module.float32)[index, :]
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("leading-integer full-slice indexing unexpectedly succeeded")

    def lower_rank_index_protocol_error_contract(self, module, shape):
        class IndexValue:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return 0

        index = IndexValue()
        outcome = self.leading_integer_error_contract(module, shape, index)
        return outcome, index.calls

    def test_leading_integer_full_slice_errors_match_pytorch_2_13(self):
        for shape in ((), (2,)):
            with self.subTest(shape=shape):
                self.assertEqual(
                    self.lower_rank_index_protocol_error_contract(torch, shape),
                    self.lower_rank_index_protocol_error_contract(
                        reference_torch, shape
                    ),
                )

        for index in (2, -3):
            with self.subTest(index=index):
                self.assertEqual(
                    self.leading_integer_error_contract(torch, (2, 3), index),
                    self.leading_integer_error_contract(
                        reference_torch, (2, 3), index
                    ),
                )

    def test_three_or_more_full_slice_lower_rank_errors_match_pytorch_2_13(self):
        for count in (3, 4):
            index = (slice(None),) * count
            for dimensions in range(count):
                shape = (2,) * dimensions
                with self.subTest(count=count, dimensions=dimensions):
                    self.assertEqual(
                        self.lower_rank_error_contract(torch, shape, index),
                        self.lower_rank_error_contract(
                            reference_torch, shape, index
                        ),
                    )

    def test_full_slice_ellipsis_rank_errors_match_pytorch_2_13(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                for dimensions in range(slice_count):
                    shape = (2,) * dimensions
                    with self.subTest(
                        slice_count=slice_count,
                        position=position,
                        dimensions=dimensions,
                    ):
                        self.assertEqual(
                            self.lower_rank_error_contract(torch, shape, index),
                            self.lower_rank_error_contract(
                                reference_torch, shape, index
                            ),
                        )

    def tuple_subclass_contract(self, module):
        source = module.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=module.float32
        )

        class IntegerRemapTuple(tuple):
            def __iter__(self):
                return iter((0,))

        selected = source[IntegerRemapTuple((slice(None),))]

        class LeadingIntegerFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((-1, slice(None)))

        leading_integer_selected = source[
            LeadingIntegerFullSliceRemapTuple((slice(None),))
        ]
        leading_integer_direct = source[-1]

        class FullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None),))

        alias = source[FullSliceRemapTuple((0,))]

        class DoubleFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None), slice(None)))

        double_alias = source[DoubleFullSliceRemapTuple((0,))]

        higher_rank_source = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )

        class TripleFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None),) * 3)

        triple_alias = higher_rank_source[TripleFullSliceRemapTuple((0,))]

        class FullSliceEllipsisRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None), Ellipsis, slice(None)))

        mixed_alias = source[FullSliceEllipsisRemapTuple((0,))]

        class EmptyRemapTuple(tuple):
            def __iter__(self):
                return iter(())

        empty_index = EmptyRemapTuple((0,))
        empty_alias = source[empty_index]

        class IterationErrorTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration exploded")

        try:
            source[IterationErrorTuple((slice(None),))]
        except Exception as error:
            iteration_error = type(error).__name__, str(error)
        else:
            self.fail("tuple subclass iteration error was suppressed")

        return {
            "selected_values": selected.tolist(),
            "selected_shape": tuple(selected.shape),
            "selected_stride": selected.stride(),
            "selected_offset": selected.storage_offset(),
            "leading_integer_values": leading_integer_selected.tolist(),
            "leading_integer_shape": tuple(leading_integer_selected.shape),
            "leading_integer_stride": leading_integer_selected.stride(),
            "leading_integer_offset": leading_integer_selected.storage_offset(),
            "leading_integer_same_logical_view": (
                leading_integer_selected.is_set_to(leading_integer_direct)
            ),
            "leading_integer_same_data_pointer": (
                leading_integer_selected.data_ptr()
                == leading_integer_direct.data_ptr()
            ),
            "alias_values": alias.tolist(),
            "alias_shape": tuple(alias.shape),
            "alias_stride": alias.stride(),
            "alias_offset": alias.storage_offset(),
            "alias_same_logical_view": alias.is_set_to(source),
            "alias_same_data_pointer": alias.data_ptr() == source.data_ptr(),
            "double_alias_values": double_alias.tolist(),
            "double_alias_shape": tuple(double_alias.shape),
            "double_alias_stride": double_alias.stride(),
            "double_alias_offset": double_alias.storage_offset(),
            "double_alias_same_logical_view": double_alias.is_set_to(source),
            "double_alias_same_data_pointer": (
                double_alias.data_ptr() == source.data_ptr()
            ),
            "triple_alias_values": triple_alias.tolist(),
            "triple_alias_shape": tuple(triple_alias.shape),
            "triple_alias_stride": triple_alias.stride(),
            "triple_alias_offset": triple_alias.storage_offset(),
            "triple_alias_same_logical_view": triple_alias.is_set_to(
                higher_rank_source
            ),
            "triple_alias_same_data_pointer": (
                triple_alias.data_ptr() == higher_rank_source.data_ptr()
            ),
            "mixed_alias_values": mixed_alias.tolist(),
            "mixed_alias_shape": tuple(mixed_alias.shape),
            "mixed_alias_stride": mixed_alias.stride(),
            "mixed_alias_offset": mixed_alias.storage_offset(),
            "mixed_alias_same_logical_view": mixed_alias.is_set_to(source),
            "mixed_alias_same_data_pointer": (
                mixed_alias.data_ptr() == source.data_ptr()
            ),
            "empty_alias_values": empty_alias.tolist(),
            "empty_alias_shape": tuple(empty_alias.shape),
            "empty_alias_stride": empty_alias.stride(),
            "empty_alias_offset": empty_alias.storage_offset(),
            "empty_alias_same_logical_view": empty_alias.is_set_to(source),
            "empty_alias_same_data_pointer": (
                empty_alias.data_ptr() == source.data_ptr()
            ),
            "empty_alias_node": self.node_diagnostic(module, empty_index),
            "iteration_error": iteration_error,
        }

    def test_tuple_subclass_iteration_matches_pytorch_2_13(self):
        self.assertEqual(
            self.tuple_subclass_contract(torch),
            self.tuple_subclass_contract(reference_torch),
        )

    def autograd_contract(self, module, index):
        index_rank = len(index) if isinstance(index, tuple) else 1
        rank = max(index_rank, 2)
        shape = (2,) * rank
        values = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(
            shape
        )
        leaf = module.tensor(
            values.tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 2.0).transpose(0, 1)
        alias = source[index]
        metadata = (
            alias is not source,
            alias.is_set_to(source),
            alias.requires_grad,
            alias.is_leaf,
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
        )
        weight_values = np.arange(10, 10 + np.prod(shape), dtype=np.float32).reshape(
            shape
        )
        weights = module.tensor(weight_values.tolist(), dtype=module.float32)
        (alias * weights).sum().backward()
        return metadata, np.asarray(leaf.grad).copy()

    def node_diagnostic(self, module, index, rank=1):
        values = np.full((1,) * rank, 2.0, dtype=np.float32).tolist()
        leaf = module.tensor(values, dtype=module.float32, requires_grad=True)
        try:
            module.nn.functional.dropout(None, p=leaf[index], training=False)
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def no_grad_contract(self, module, index):
        index_rank = len(index) if isinstance(index, tuple) else 1
        rank = max(index_rank, 2)
        shape = (2,) * rank
        values = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(
            shape
        )
        leaf = module.tensor(
            values.tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        with module.no_grad():
            alias = source[index]
        return (
            alias is not source,
            alias.is_set_to(source),
            alias.requires_grad,
            alias.is_leaf,
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
            leaf.grad,
        )

    def assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
        self, index, diagnostic_rank=1
    ):
        actual_metadata, actual_gradient = self.autograd_contract(torch, index)
        expected_metadata, expected_gradient = self.autograd_contract(
            reference_torch, index
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)
        self.assertEqual(
            self.node_diagnostic(torch, index, diagnostic_rank),
            self.node_diagnostic(reference_torch, index, diagnostic_rank),
        )
        self.assertEqual(
            self.no_grad_contract(torch, index),
            self.no_grad_contract(reference_torch, index),
        )

    def test_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(self):
        self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
            slice(None)
        )

    def test_singleton_tuple_autograd_matches_pytorch_2_13(self):
        self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
            (slice(None),)
        )

    def test_double_full_slice_tuple_autograd_matches_pytorch_2_13(self):
        self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
            (slice(None), slice(None)), diagnostic_rank=2
        )

    def leading_integer_autograd_contract(self, module):
        values = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        source = (leaf * 2.0).transpose(0, 1)
        selected = source[-1, :]
        direct = source[-1]
        metadata = (
            selected.is_set_to(direct),
            selected.data_ptr() == direct.data_ptr(),
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
        )
        weights = module.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 4).tolist(),
            dtype=module.float32,
        )
        (selected * weights).sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            untracked = no_grad_source[1, :]

        empty = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
        empty[1, :].sum().backward()
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "node": self.node_diagnostic(module, (0, slice(None)), rank=2),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
                untracked.is_set_to(no_grad_source[1]),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_leading_integer_full_slice_autograd_matches_pytorch_2_13(self):
        self.assertEqual(
            self.leading_integer_autograd_contract(torch),
            self.leading_integer_autograd_contract(reference_torch),
        )

    def test_three_or_more_full_slice_tuple_autograd_matches_pytorch_2_13(self):
        for count in (3, 4):
            with self.subTest(count=count):
                self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
                    (slice(None),) * count, diagnostic_rank=count
                )

    def test_full_slice_ellipsis_tuple_autograd_matches_pytorch_2_13(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                with self.subTest(slice_count=slice_count, position=position):
                    self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
                        index, diagnostic_rank=slice_count
                    )

    def test_empty_tuple_autograd_matches_pytorch_2_13(self):
        self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(())

    def lifetime_contract(self, module, index, source_rank=2):
        input_shape = tuple(range(2, source_rank + 3))
        values = np.arange(np.prod(input_shape), dtype=np.float32).reshape(
            input_shape
        )
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_alias():
            source = (leaf * 2.0).transpose(0, source_rank)[1]
            return source[index]

        alias = make_alias()
        gc.collect()
        metadata = (
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
            alias.tolist(),
            alias.requires_grad,
            alias.is_leaf,
        )
        weight_values = np.arange(
            1, np.prod(alias.shape) + 1, dtype=np.float32
        ).reshape(alias.shape)
        weights = module.tensor(weight_values.tolist(), dtype=module.float32)
        (alias * weights).sum().backward()
        return metadata, np.asarray(leaf.grad).copy()

    def assert_source_lifetime_matches_pytorch_2_13(self, index, source_rank=2):
        actual_metadata, actual_gradient = self.lifetime_contract(
            torch, index, source_rank
        )
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch, index, source_rank
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def test_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13(slice(None))

    def test_singleton_tuple_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13((slice(None),))

    def test_double_full_slice_tuple_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13(
            (slice(None), slice(None))
        )

    def leading_integer_lifetime_contract(self, module):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_alias():
            source = (leaf * 2.0).transpose(0, 3)[1]
            return source[-1, :]

        alias = make_alias()
        gc.collect()
        metadata = (
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
            alias.tolist(),
            alias.requires_grad,
            alias.is_leaf,
        )
        weights = module.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(4, 2).tolist(),
            dtype=module.float32,
        )
        (alias * weights).sum().backward()
        return metadata, leaf.grad.tolist()

    def test_leading_integer_full_slice_lifetime_matches_pytorch_2_13(self):
        self.assertEqual(
            self.leading_integer_lifetime_contract(torch),
            self.leading_integer_lifetime_contract(reference_torch),
        )

    def test_three_or_more_full_slice_tuple_source_lifetime_matches_pytorch_2_13(
        self,
    ):
        for count in (3, 4):
            with self.subTest(count=count):
                self.assert_source_lifetime_matches_pytorch_2_13(
                    (slice(None),) * count, source_rank=count
                )

    def test_full_slice_ellipsis_tuple_source_lifetime_matches_pytorch_2_13(
        self,
    ):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                with self.subTest(slice_count=slice_count, position=position):
                    self.assert_source_lifetime_matches_pytorch_2_13(
                        index, source_rank=max(slice_count, 2)
                    )

    def mode_dispatch_contract(self, module, index):
        index_rank = len(index) if isinstance(index, tuple) else 1
        shape = (2,) * max(index_rank, 2)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        source = module.tensor(values.tolist(), dtype=module.float32)
        scalar = module.tensor(1.0, dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
            context_depth = len(
                module.overrides._get_current_function_mode_stack()
            )
        function, dispatch_types, args, kwargs, handler_depth = mode.calls[0]
        descriptor = inspect.getattr_static(module.Tensor, "__getitem__")

        mode.calls.clear()
        with mode:
            scalar_result = scalar[index]
        scalar_argument_preserved = mode.calls[0][2][1] is index

        events = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label, replacement=None):
                self.label = label
                self.replacement = replacement

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                events.append(
                    (
                        self.label,
                        func.__qualname__,
                        dispatch_types == (),
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                if self.replacement is not None:
                    return self.replacement
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower", marker)
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = source[index]

        return {
            "replacement": result is marker,
            "call_count": len(mode.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "owner_name": function.__objclass__.__name__,
            "owner_module": function.__objclass__.__module__,
            "descriptor_identity": function is descriptor,
            "dispatch_types_empty": dispatch_types == (),
            "argument_count": len(args),
            "receiver_identity": args[0] is source,
            "index_identity": args[1] is index,
            "kwargs_none": kwargs is None,
            "handler_depth": handler_depth,
            "context_depth": context_depth,
            "scalar_replaced": scalar_result is marker,
            "scalar_argument_preserved": scalar_argument_preserved,
            "forwarded_replacement": forwarded is marker,
            "forwarding_events": tuple(events),
            "final_stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def assert_tensorbase_mode_dispatch_matches_pytorch_2_13(self, index):
        self.assertEqual(
            self.mode_dispatch_contract(torch, index),
            self.mode_dispatch_contract(reference_torch, index),
        )

    def test_tensorbase_mode_dispatch_matches_pytorch_2_13(self):
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(slice(None))

    def test_singleton_tuple_mode_dispatch_matches_pytorch_2_13(self):
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(
            (slice(None),)
        )

    def test_double_full_slice_tuple_mode_dispatch_matches_pytorch_2_13(self):
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(
            (slice(None), slice(None))
        )

    def test_leading_integer_full_slice_mode_dispatch_matches_pytorch_2_13(self):
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(
            (0, slice(None))
        )

    def test_three_or_more_full_slice_tuple_mode_dispatch_matches_pytorch_2_13(
        self,
    ):
        for count in (3, 4):
            with self.subTest(count=count):
                self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(
                    (slice(None),) * count
                )

    def test_full_slice_ellipsis_tuple_mode_dispatch_matches_pytorch_2_13(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                with self.subTest(slice_count=slice_count, position=position):
                    self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(
                        index
                    )


if __name__ == "__main__":
    unittest.main()
