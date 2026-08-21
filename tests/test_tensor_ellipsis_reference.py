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
class TensorEllipsisIndexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Ellipsis indexing differentials require pinned PyTorch 2.13.0"
            )

    def layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            module.tensor(-0.0, dtype=module.float32),
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

    def assert_layout_aliases_match_pytorch_2_13(self, index):
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

    def test_bare_ellipsis_layout_aliases_match_pytorch_2_13(self):
        self.assert_layout_aliases_match_pytorch_2_13(Ellipsis)

    def test_singleton_tuple_layout_aliases_match_pytorch_2_13(self):
        self.assert_layout_aliases_match_pytorch_2_13((Ellipsis,))

    def trailing_layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            (base, (1, Ellipsis), (1,)),
            (base, (-1, 1, Ellipsis), (-1, 1)),
            (base, (1, -1, -2, -3, Ellipsis), (1, -1, -2, -3)),
            (module.zeros((2, 0, 3), dtype=module.float32), (1, Ellipsis), (1,)),
            (base.transpose(0, 3), (1, -1, Ellipsis), (1, -1)),
            (base[1], (1, Ellipsis), (1,)),
        )

    def trailing_view_contract(self, source, index, integer_only):
        indexed = source[index]
        expected = source[integer_only]
        values = np.asarray(indexed.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": indexed is not expected,
            "shape": tuple(indexed.shape),
            "stride": indexed.stride(),
            "storage_offset": indexed.storage_offset(),
            "same_integer_view": indexed.is_set_to(expected),
            "same_integer_pointer": indexed.data_ptr() == expected.data_ptr(),
            "dtype": str(indexed.dtype),
            "device": str(indexed.device),
            "requires_grad": indexed.requires_grad,
            "is_leaf": indexed.is_leaf,
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def test_trailing_ellipsis_views_match_pytorch_2_13(self):
        actual_cases = self.trailing_layout_cases(torch)
        expected_cases = self.trailing_layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.trailing_view_contract(actual[0], actual[1], actual[2]),
                    self.trailing_view_contract(
                        expected[0], expected[1], expected[2]
                    ),
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

    def alias_node_diagnostic(self, module, index):
        leaf = module.tensor([2.0], dtype=module.float32, requires_grad=True)
        try:
            module.nn.functional.dropout(None, p=leaf[index], training=False)
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def no_grad_contract(self, module, index):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
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

    def assert_autograd_and_no_grad_status_match_pytorch_2_13(self, index):
        actual_metadata, actual_gradient = self.autograd_contract(torch, index)
        expected_metadata, expected_gradient = self.autograd_contract(
            reference_torch, index
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)
        self.assertEqual(
            self.alias_node_diagnostic(torch, index),
            self.alias_node_diagnostic(reference_torch, index),
        )
        self.assertEqual(
            self.no_grad_contract(torch, index),
            self.no_grad_contract(reference_torch, index),
        )

    def test_bare_ellipsis_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assert_autograd_and_no_grad_status_match_pytorch_2_13(Ellipsis)

    def test_singleton_tuple_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assert_autograd_and_no_grad_status_match_pytorch_2_13((Ellipsis,))

    def trailing_autograd_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        source = (leaf * 2.0).transpose(0, 2)
        indexed = source[1, 1, Ellipsis]
        integer_only = source[1, 1]
        metadata = (
            indexed is not integer_only,
            indexed.is_set_to(integer_only),
            indexed.data_ptr() == integer_only.data_ptr(),
            indexed.requires_grad,
            indexed.is_leaf,
            tuple(indexed.shape),
            indexed.stride(),
            indexed.storage_offset(),
            indexed.tolist(),
        )
        weights = module.tensor([3.0, 5.0], dtype=module.float32)
        (indexed * weights).sum().backward()
        return metadata, np.asarray(leaf.grad).copy()

    def trailing_no_grad_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        source = leaf.transpose(0, 2)
        with module.no_grad():
            indexed = source[-1, 0, Ellipsis]
            integer_only = source[-1, 0]
        return (
            indexed is not integer_only,
            indexed.is_set_to(integer_only),
            indexed.data_ptr() == integer_only.data_ptr(),
            indexed.requires_grad,
            indexed.is_leaf,
            tuple(indexed.shape),
            indexed.stride(),
            indexed.storage_offset(),
            indexed.tolist(),
            leaf.grad,
        )

    def trailing_node_diagnostic(self, module):
        leaf = module.tensor(
            [[2.0]], dtype=module.float32, requires_grad=True
        )
        try:
            module.nn.functional.dropout(
                None, p=leaf[0, Ellipsis], training=False
            )
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def test_trailing_ellipsis_autograd_and_no_grad_match_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.trailing_autograd_contract(torch)
        expected_metadata, expected_gradient = self.trailing_autograd_contract(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)
        self.assertEqual(
            self.trailing_no_grad_contract(torch),
            self.trailing_no_grad_contract(reference_torch),
        )
        self.assertEqual(
            self.trailing_node_diagnostic(torch),
            self.trailing_node_diagnostic(reference_torch),
        )

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

    def test_bare_ellipsis_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13(Ellipsis)

    def test_singleton_tuple_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13((Ellipsis,))

    def trailing_lifetime_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_view():
            source = (leaf * 2.0).transpose(0, 2)
            return source[1, 1, Ellipsis]

        indexed = make_view()
        gc.collect()
        metadata = (
            tuple(indexed.shape),
            indexed.stride(),
            indexed.storage_offset(),
            indexed.tolist(),
            indexed.requires_grad,
            indexed.is_leaf,
        )
        weights = module.tensor([3.0, 5.0], dtype=module.float32)
        (indexed * weights).sum().backward()
        return metadata, np.asarray(leaf.grad).copy()

    def test_trailing_ellipsis_source_lifetime_matches_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.trailing_lifetime_contract(torch)
        expected_metadata, expected_gradient = self.trailing_lifetime_contract(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def mode_dispatch_contract(self, module, index):
        source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
        )
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

        class IndexBomb:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                raise AssertionError("index parsing must be deferred")

        bomb = IndexBomb()
        mode.calls.clear()
        with mode:
            bomb_result = source[bomb]
        bomb_argument_preserved = mode.calls[0][2][1] is bomb

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
            "bomb_replaced": bomb_result is marker,
            "bomb_calls": bomb.calls,
            "bomb_argument_preserved": bomb_argument_preserved,
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

    def test_bare_ellipsis_tensorbase_mode_dispatch_matches_pytorch_2_13(self):
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(Ellipsis)

    def test_singleton_tuple_mode_dispatch_matches_pytorch_2_13(self):
        index = (Ellipsis,)
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(index)

    def test_trailing_ellipsis_mode_dispatch_matches_pytorch_2_13(self):
        index = (1, Ellipsis)
        self.assert_tensorbase_mode_dispatch_matches_pytorch_2_13(index)

    def index_rule_contract(self, module):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        def outcome(source, index):
            try:
                result = source[index]
            except Exception as error:
                return ("error", type(error).__name__, str(error))
            return (
                "value",
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
                result.tolist(),
            )

        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        first = IndexValue(-1)
        second = IndexValue(0)
        success = outcome(tensor, (first, second, Ellipsis))
        success_calls = (first.calls, second.calls)

        first = IndexValue(2)
        later = IndexValue(0)
        bounds = outcome(tensor, (first, later, Ellipsis))
        bounds_calls = (first.calls, later.calls)

        too_many_indices = tuple(IndexValue(0) for _ in range(4))
        too_many = outcome(tensor, (*too_many_indices, Ellipsis))
        too_many_calls = tuple(index.calls for index in too_many_indices)

        scalar_index = IndexValue(0)
        scalar = outcome(
            module.tensor(1.0, dtype=module.float32),
            (scalar_index, Ellipsis),
        )

        return {
            "success": success,
            "success_calls": success_calls,
            "bounds": bounds,
            "bounds_calls": bounds_calls,
            "too_many": too_many,
            "too_many_calls": too_many_calls,
            "scalar": scalar,
            "scalar_calls": scalar_index.calls,
            "empty_bounds": outcome(
                module.zeros((2, 0, 3), dtype=module.float32),
                (1, 0, Ellipsis),
            ),
            "overflow": outcome(tensor, (1 << 100, Ellipsis)),
        }

    def test_trailing_ellipsis_conversion_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.index_rule_contract(torch),
            self.index_rule_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
