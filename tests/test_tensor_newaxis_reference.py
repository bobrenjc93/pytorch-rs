import gc
import inspect
import sys
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
        contiguous_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32)),
            (
                "contiguous",
                module.tensor(contiguous_values.tolist(), dtype=module.float32),
            ),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def unsqueeze_contract(self, source, index):
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
                        self.unsqueeze_contract(actual, actual_index),
                        self.unsqueeze_contract(expected, expected_index),
                    )

    def test_exact_trailing_layout_value_and_aliasing_match_pytorch_2_13(self):
        for spelling in ("none", "newaxis"):
            actual_cases = self.layout_cases(torch)
            expected_cases = self.layout_cases(reference_torch)
            for (actual_case, actual), (expected_case, expected) in zip(
                actual_cases, expected_cases, strict=True
            ):
                self.assertEqual(actual_case, expected_case)
                with self.subTest(spelling=spelling, case=actual_case):
                    actual_newaxis = (
                        None if spelling == "none" else torch.newaxis
                    )
                    expected_newaxis = (
                        None
                        if spelling == "none"
                        else reference_torch.newaxis
                    )
                    self.assertEqual(
                        self.unsqueeze_contract(
                            actual, (Ellipsis, actual_newaxis)
                        ),
                        self.unsqueeze_contract(
                            expected, (Ellipsis, expected_newaxis)
                        ),
                    )

    def extreme_empty_contract(
        self, module, leading_dimension, trailing_dimension
    ):
        source = module.zeros((0,), dtype=module.float32).reshape(
            (leading_dimension, 0, trailing_dimension)
        )
        try:
            result = source[module.newaxis]
        except Exception as error:
            message = str(error)
            non_concrete = (
                "SymIntArrayRef expected to contain only concrete integers"
            )
            if non_concrete in message:
                message = non_concrete
            return ("error", type(error).__name__, message)
        return (
            "result",
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source),
            str(result.dtype),
            str(result.device),
        )

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "signed 64-bit stride wrapping requires a 64-bit Python build",
    )
    def test_extreme_empty_stride_boundaries_match_pytorch_2_13(self):
        cases = (
            ((1 << 62) - 1, 2),
            (1 << 62, 2),
            ((1 << 62) + 1, 2),
            ((1 << 62) - 1, 3),
            (1 << 62, 3),
            (sys.maxsize, 2),
            (sys.maxsize, 3),
        )
        for leading_dimension, trailing_dimension in cases:
            with self.subTest(
                leading_dimension=leading_dimension,
                trailing_dimension=trailing_dimension,
            ):
                self.assertEqual(
                    self.extreme_empty_contract(
                        torch, leading_dimension, trailing_dimension
                    ),
                    self.extreme_empty_contract(
                        reference_torch,
                        leading_dimension,
                        trailing_dimension,
                    ),
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
        if case == "contiguous":
            values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            leaf = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
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

    def autograd_contract(self, module, case, trailing=False):
        leaf, source = self.make_autograd_case(module, case)
        index = (Ellipsis, module.newaxis) if trailing else module.newaxis
        result = source[index]
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

    def no_grad_contract(self, module, case, trailing=False):
        leaf, source = self.make_autograd_case(module, case)
        index = (Ellipsis, None) if trailing else None
        with module.no_grad():
            result = source[index]
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

    def unsqueeze_node_diagnostic(self, module, trailing=False):
        leaf = module.tensor(
            [2.0], dtype=module.float32, requires_grad=True
        )
        index = (Ellipsis, module.newaxis) if trailing else module.newaxis
        try:
            module.nn.functional.dropout(
                None, p=leaf[index], training=False
            )
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def assert_autograd_and_no_grad_match_pytorch_2_13(self, trailing):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            with self.subTest(case=case, mode="autograd"):
                actual_metadata, actual_gradient = self.autograd_contract(
                    torch, case, trailing
                )
                expected_metadata, expected_gradient = self.autograd_contract(
                    reference_torch, case, trailing
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(actual_gradient, expected_gradient)

            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(
                    self.no_grad_contract(torch, case, trailing),
                    self.no_grad_contract(reference_torch, case, trailing),
                )

        self.assertEqual(
            self.unsqueeze_node_diagnostic(torch, trailing),
            self.unsqueeze_node_diagnostic(reference_torch, trailing),
        )

    def test_autograd_and_no_grad_match_pytorch_2_13_for_every_layout(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13(False)

    def test_trailing_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13(True)

    def public_unsqueeze_contract(self, module, source, dim, form="function"):
        if form == "method":
            result = source.unsqueeze(dim)
        elif form == "method keyword":
            result = source.unsqueeze(dim=dim)
        elif form == "function keyword":
            result = module.unsqueeze(input=source, dim=dim)
        elif form == "function alias x":
            result = module.unsqueeze(x=source, dim=dim)
        elif form == "function alias a":
            result = module.unsqueeze(a=source, dim=dim)
        elif form == "function alias x1":
            result = module.unsqueeze(x1=source, dim=dim)
        else:
            result = module.unsqueeze(source, dim)
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

    def public_unsqueeze_autograd_contract(self, module, case, dim, method=False):
        leaf, source = self.make_autograd_case(module, case)
        result = source.unsqueeze(dim) if method else module.unsqueeze(source, dim)
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
        result.sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def public_unsqueeze_no_grad_contract(self, module, case, dim, method=False):
        leaf, source = self.make_autograd_case(module, case)
        with module.no_grad():
            result = source.unsqueeze(dim) if method else module.unsqueeze(source, dim)
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

    def public_unsqueeze_dimensions(self, source):
        rank = source.dim()
        dimensions = []
        for axis in range(rank + 1):
            dimensions.append((f"axis {axis} positive", axis))
            dimensions.append((f"axis {axis} negative", axis - (rank + 1)))
        return tuple(dimensions)

    def test_public_unsqueeze_layouts_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for (actual_case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(actual_case, expected_case)
            dimensions = self.public_unsqueeze_dimensions(actual)
            for dimension_case, dim in dimensions:
                for form in (
                    "function",
                    "function keyword",
                    "function alias x",
                    "function alias a",
                    "function alias x1",
                    "method",
                    "method keyword",
                ):
                    with self.subTest(
                        case=actual_case, dimension=dimension_case, form=form
                    ):
                        self.assertEqual(
                            self.public_unsqueeze_contract(
                                torch, actual, dim, form=form
                            ),
                            self.public_unsqueeze_contract(
                                reference_torch, expected, dim, form=form
                            ),
                        )

    def test_public_unsqueeze_autograd_no_grad_and_full_sum_match_pytorch_2_13(self):
        for case in ("scalar", "empty", "contiguous", "offset", "noncontiguous"):
            _, dimension_source = self.make_autograd_case(torch, case)
            for dimension_case, dim in self.public_unsqueeze_dimensions(
                dimension_source
            ):
                for method in (False, True):
                    with self.subTest(
                        case=case,
                        dimension=dimension_case,
                        method=method,
                        mode="autograd",
                    ):
                        actual_metadata, actual_gradient = (
                            self.public_unsqueeze_autograd_contract(
                                torch, case, dim, method
                            )
                        )
                        expected_metadata, expected_gradient = (
                            self.public_unsqueeze_autograd_contract(
                                reference_torch, case, dim, method
                            )
                        )
                        self.assertEqual(actual_metadata, expected_metadata)
                        np.testing.assert_array_equal(actual_gradient, expected_gradient)

                    with self.subTest(
                        case=case,
                        dimension=dimension_case,
                        method=method,
                        mode="no_grad",
                    ):
                        self.assertEqual(
                            self.public_unsqueeze_no_grad_contract(
                                torch, case, dim, method
                            ),
                            self.public_unsqueeze_no_grad_contract(
                                reference_torch, case, dim, method
                            ),
                        )

    def lifetime_contract(self, module, trailing=False):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )

        def make_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            index = (Ellipsis, module.newaxis) if trailing else module.newaxis
            return source[index]

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
        values = (
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]
            if trailing
            else [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]
        )
        weights = module.tensor(values, dtype=module.float32)
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def test_source_lifetime_matches_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.lifetime_contract(torch)
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def test_trailing_source_lifetime_matches_pytorch_2_13(self):
        actual_metadata, actual_gradient = self.lifetime_contract(torch, True)
        expected_metadata, expected_gradient = self.lifetime_contract(
            reference_torch, True
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)

    def mode_contract(self, module, trailing=False):
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
            index = (
                (Ellipsis, module.newaxis) if trailing else module.newaxis
            )
            mode.calls.clear()
            with mode:
                result = source[index]
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
                    args[1] is index,
                    kwargs is None,
                    handler_depth,
                    context_depth,
                )
            )

            forwarding_index = (Ellipsis, None) if trailing else None
            with ForwardingMode():
                forwarded = source[forwarding_index]
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
        nested_index = (
            (Ellipsis, module.newaxis) if trailing else module.newaxis
        )
        with lower:
            with upper:
                nested_result = source[nested_index]

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

    def test_trailing_torch_function_mode_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch, True),
            self.mode_contract(reference_torch, True),
        )


if __name__ == "__main__":
    unittest.main()
