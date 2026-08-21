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

    def ellipsis_indices(self):
        return (("bare", Ellipsis), ("singleton-tuple", (Ellipsis,)))

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

    def test_scalar_empty_offset_and_noncontiguous_aliases_match_pytorch_2_13(self):
        for syntax, index in self.ellipsis_indices():
            actual_cases = self.layout_cases(torch)
            expected_cases = self.layout_cases(reference_torch)
            for case, (actual, expected) in enumerate(
                zip(actual_cases, expected_cases, strict=True)
            ):
                with self.subTest(syntax=syntax, case=case):
                    self.assertEqual(
                        self.alias_contract(actual, index),
                        self.alias_contract(expected, index),
                    )

    def tuple_subclass_contract(self, module):
        class RedirectingTuple(tuple):
            def __new__(cls):
                return super().__new__(cls, (Ellipsis,))

            def __init__(self):
                self.iter_calls = 0

            def __iter__(self):
                self.iter_calls += 1
                return iter((0,))

        source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
        )
        index = RedirectingTuple()
        result = source[index]
        expected_view = source[0]
        return {
            "stored_ellipsis": tuple.__getitem__(index, 0) is Ellipsis,
            "iter_calls": index.iter_calls,
            "values": result.tolist(),
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "same_logical_view": result.is_set_to(expected_view),
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

    def test_alias_autograd_and_no_grad_status_match_pytorch_2_13(self):
        for syntax, index in self.ellipsis_indices():
            with self.subTest(syntax=syntax):
                actual_metadata, actual_gradient = self.autograd_contract(
                    torch, index
                )
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

    def test_source_lifetime_matches_pytorch_2_13(self):
        for syntax, index in self.ellipsis_indices():
            with self.subTest(syntax=syntax):
                actual_metadata, actual_gradient = self.lifetime_contract(
                    torch, index
                )
                expected_metadata, expected_gradient = self.lifetime_contract(
                    reference_torch, index
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

    def test_tensorbase_mode_dispatch_matches_pytorch_2_13(self):
        for syntax, index in self.ellipsis_indices():
            with self.subTest(syntax=syntax):
                self.assertEqual(
                    self.mode_dispatch_contract(torch, index),
                    self.mode_dispatch_contract(reference_torch, index),
                )


if __name__ == "__main__":
    unittest.main()
