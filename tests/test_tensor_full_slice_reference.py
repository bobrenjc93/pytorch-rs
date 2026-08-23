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
            module.tensor(
                [[-0.0, 1.0], [2.0, 3.0]], dtype=module.float32
            ),
            module.zeros((2, 0, 3), dtype=module.float32),
            base[1],
            base.transpose(0, 3)[1],
        )

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

    def assert_layout_values_aliasing_dtype_and_device_match_pytorch_2_13(
        self, index, minimum_rank=1
    ):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            if len(actual.shape) < minimum_rank:
                continue
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

    def test_double_tuple_layout_aliases_match_pytorch_2_13(self):
        self.assert_layout_values_aliasing_dtype_and_device_match_pytorch_2_13(
            (slice(None), slice(None)), minimum_rank=2
        )

    def index_error_contract(self, module, value, index):
        try:
            module.tensor(value, dtype=module.float32)[index]
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("full-slice indexing unexpectedly succeeded")

    def test_scalar_error_matches_pytorch_2_13(self):
        self.assertEqual(
            self.index_error_contract(torch, -0.0, slice(None)),
            self.index_error_contract(reference_torch, -0.0, slice(None)),
        )

    def test_singleton_tuple_scalar_error_matches_pytorch_2_13(self):
        index = (slice(None),)
        self.assertEqual(
            self.index_error_contract(torch, -0.0, index),
            self.index_error_contract(reference_torch, -0.0, index),
        )

    def test_double_tuple_lower_rank_errors_match_pytorch_2_13(self):
        index = (slice(None), slice(None))
        for value, rank in ((-0.0, 0), ([-0.0], 1)):
            with self.subTest(rank=rank):
                self.assertEqual(
                    self.index_error_contract(torch, value, index),
                    self.index_error_contract(reference_torch, value, index),
                )

    def tuple_subclass_contract(self, module):
        source = module.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=module.float32
        )

        class IntegerRemapTuple(tuple):
            def __iter__(self):
                return iter((0,))

        selected = source[IntegerRemapTuple((slice(None),))]

        class FullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None),))

        alias = source[FullSliceRemapTuple((0,))]

        class DoubleFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None), slice(None)))

        double_alias = source[DoubleFullSliceRemapTuple((0,))]

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
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
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
        weights = module.tensor(
            [[10.0, 20.0], [30.0, 40.0]], dtype=module.float32
        )
        (alias * weights).sum().backward()
        return metadata, np.asarray(leaf.grad).copy()

    def node_diagnostic(self, module, index, diagnostic_rank=1):
        values = [2.0] if diagnostic_rank == 1 else [[2.0]]
        leaf = module.tensor(values, dtype=module.float32, requires_grad=True)
        try:
            module.nn.functional.dropout(None, p=leaf[index], training=False)
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def no_grad_contract(self, module, index):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        source = leaf.transpose(0, 2)[1]
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

    def test_double_tuple_autograd_matches_pytorch_2_13(self):
        self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(
            (slice(None), slice(None)), diagnostic_rank=2
        )

    def empty_autograd_contract(self, module, index):
        leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        alias = leaf[index]
        tracked_metadata = (
            alias is not leaf,
            alias.is_set_to(leaf),
            alias.requires_grad,
            alias.is_leaf,
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
        )
        alias.sum().backward()
        gradient = (
            tuple(leaf.grad.shape),
            leaf.grad.stride(),
            leaf.grad.numel(),
            np.asarray(leaf.grad, dtype=np.float32).copy(),
        )

        no_grad_leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        with module.no_grad():
            no_grad_alias = no_grad_leaf[index]
        no_grad_metadata = (
            no_grad_alias is not no_grad_leaf,
            no_grad_alias.is_set_to(no_grad_leaf),
            no_grad_alias.requires_grad,
            no_grad_alias.is_leaf,
            tuple(no_grad_alias.shape),
            no_grad_alias.stride(),
            no_grad_alias.storage_offset(),
            no_grad_leaf.grad,
        )
        return tracked_metadata, gradient, no_grad_metadata

    def test_double_tuple_empty_autograd_matches_pytorch_2_13(self):
        index = (slice(None), slice(None))
        actual_metadata, actual_gradient, actual_no_grad = (
            self.empty_autograd_contract(torch, index)
        )
        expected_metadata, expected_gradient, expected_no_grad = (
            self.empty_autograd_contract(reference_torch, index)
        )
        self.assertEqual(actual_metadata, expected_metadata)
        self.assertEqual(actual_gradient[:3], expected_gradient[:3])
        np.testing.assert_array_equal(actual_gradient[3], expected_gradient[3])
        self.assertEqual(actual_no_grad, expected_no_grad)

    def test_empty_tuple_autograd_matches_pytorch_2_13(self):
        self.assert_autograd_node_gradient_and_no_grad_status_match_pytorch_2_13(())

    def lifetime_contract(self, module, index):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_alias():
            source = (leaf * 2.0).transpose(0, 2)[1]
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
        weights = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=module.float32
        )
        (alias * weights).sum().backward()
        return metadata, np.asarray(leaf.grad).copy()

    def assert_source_lifetime_matches_pytorch_2_13(self, index):
        actual_metadata, actual_gradient = self.lifetime_contract(torch, index)
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch, index
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def test_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13(slice(None))

    def test_singleton_tuple_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13((slice(None),))

    def test_double_tuple_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13(
            (slice(None), slice(None))
        )

    def mode_dispatch_contract(self, module, index):
        source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
        )
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

    def test_double_tuple_mode_dispatch_matches_pytorch_2_13(self):
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(
            (slice(None), slice(None))
        )


if __name__ == "__main__":
    unittest.main()
