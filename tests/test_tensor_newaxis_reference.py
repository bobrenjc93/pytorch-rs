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
class TensorNewAxisIndexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "new-axis indexing differentials require pinned PyTorch 2.13.0"
            )

    def namespace_contract(self, module):
        namespace = {}
        exec(f"from {module.__name__} import *", namespace)
        return {
            "is_none": module.newaxis is None,
            "all_count": module.__all__.count("newaxis"),
            "native_has_newaxis": hasattr(module._C, "newaxis"),
            "wildcard_is_none": namespace["newaxis"] is None,
        }

    def test_newaxis_namespace_matches_pytorch_2_13(self):
        self.assertEqual(
            self.namespace_contract(torch),
            self.namespace_contract(reference_torch),
        )

    def layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32)),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def leading_unsqueeze_contract(self, source, index):
        result = source[index]
        values = np.asarray(result.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": result is not source,
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "shared_data_pointer": result.data_ptr() == source.data_ptr(),
            "same_logical_view": result.is_set_to(source),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "values": result.tolist(),
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def test_layout_value_and_aliasing_contracts_match_pytorch_2_13(self):
        for spelling in ("none", "newaxis"):
            actual_cases = self.layout_cases(torch)
            expected_cases = self.layout_cases(reference_torch)
            for (actual_case, actual), (expected_case, expected) in zip(
                actual_cases, expected_cases, strict=True
            ):
                self.assertEqual(actual_case, expected_case)
                with self.subTest(spelling=spelling, case=actual_case):
                    actual_index = None if spelling == "none" else torch.newaxis
                    expected_index = (
                        None
                        if spelling == "none"
                        else reference_torch.newaxis
                    )
                    self.assertEqual(
                        self.leading_unsqueeze_contract(actual, actual_index),
                        self.leading_unsqueeze_contract(expected, expected_index),
                    )

    def make_autograd_case(self, module, case):
        if case == "scalar":
            leaf = module.tensor(
                -2.0, dtype=module.float32, requires_grad=True
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def autograd_contract(self, module, case):
        leaf, source = self.make_autograd_case(module, case)
        result = source[module.newaxis]
        metadata = (
            result is not source,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            result.requires_grad,
            result.is_leaf,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
        )
        weights = module.ones(tuple(result.shape), dtype=module.float32)
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def no_grad_contract(self, module, case):
        leaf, source = self.make_autograd_case(module, case)
        with module.no_grad():
            result = source[None]
        return (
            result is not source,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            result.requires_grad,
            result.is_leaf,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            leaf.grad,
        )

    def unsqueeze_node_diagnostic(self, module):
        leaf = module.tensor(
            [2.0], dtype=module.float32, requires_grad=True
        )
        try:
            module.nn.functional.dropout(
                None, p=leaf[module.newaxis], training=False
            )
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def test_autograd_and_no_grad_match_pytorch_2_13_for_every_layout(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            with self.subTest(case=case, mode="autograd"):
                actual_metadata, actual_gradient = self.autograd_contract(
                    torch, case
                )
                expected_metadata, expected_gradient = self.autograd_contract(
                    reference_torch, case
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(actual_gradient, expected_gradient)

            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(
                    self.no_grad_contract(torch, case),
                    self.no_grad_contract(reference_torch, case),
                )

        self.assertEqual(
            self.unsqueeze_node_diagnostic(torch),
            self.unsqueeze_node_diagnostic(reference_torch),
        )

    def lifetime_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[module.newaxis]

        result = make_view()
        gc.collect()
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.tolist(),
            result.requires_grad,
            result.is_leaf,
        )
        weights = module.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def test_source_lifetime_matches_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.lifetime_contract(torch)
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def mode_contract(self, module):
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

        descriptor = inspect.getattr_static(module.Tensor, "__getitem__")
        mode = RecordingMode()
        records = []
        forwarded_layouts = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        for case, source in self.layout_cases(module):
            mode.calls.clear()
            with mode:
                result = source[module.newaxis]
                context_depth = len(
                    module.overrides._get_current_function_mode_stack()
                )
            function, dispatch_types, args, kwargs, handler_depth = mode.calls[0]
            records.append(
                (
                    case,
                    result is marker,
                    len(mode.calls),
                    type(function).__name__,
                    function.__name__,
                    function.__qualname__,
                    function.__objclass__.__name__,
                    function.__objclass__.__module__,
                    function is descriptor,
                    dispatch_types == (),
                    len(args),
                    args[0] is source,
                    args[1] is module.newaxis,
                    kwargs is None,
                    handler_depth,
                    context_depth,
                )
            )

            with ForwardingMode():
                forwarded = source[None]
            forwarded_layouts.append(
                (
                    case,
                    tuple(forwarded.shape),
                    forwarded.stride(),
                    forwarded.storage_offset(),
                    forwarded.data_ptr() == source.data_ptr(),
                    forwarded.is_set_to(source),
                )
            )

        events = []

        class NestedMode(module.overrides.TorchFunctionMode):
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

        source = self.layout_cases(module)[0][1]
        lower = NestedMode("lower", marker)
        upper = NestedMode("upper")
        with lower:
            with upper:
                nested_result = source[module.newaxis]

        return {
            "records": tuple(records),
            "forwarded_layouts": tuple(forwarded_layouts),
            "nested_replacement": nested_result is marker,
            "nested_events": tuple(events),
            "final_stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )


if __name__ == "__main__":
    unittest.main()
