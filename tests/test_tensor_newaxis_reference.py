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
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32)),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def newaxis_index(self, module, form, spelling):
        newaxis = None if spelling == "none" else module.newaxis
        if form == "bare":
            return newaxis
        if form == "tuple":
            return (newaxis,)
        if form == "trailing":
            return (Ellipsis, newaxis)
        raise AssertionError(f"unknown new-axis index form: {form}")

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
                    self.assertEqual(
                        self.unsqueeze_contract(
                            actual,
                            self.newaxis_index(torch, "bare", spelling),
                        ),
                        self.unsqueeze_contract(
                            expected,
                            self.newaxis_index(
                                reference_torch, "bare", spelling
                            ),
                        ),
                    )

    def test_exact_leading_singleton_tuple_matches_pytorch_2_13(self):
        for spelling in ("none", "newaxis"):
            actual_cases = self.layout_cases(torch)
            expected_cases = self.layout_cases(reference_torch)
            for (actual_case, actual), (expected_case, expected) in zip(
                actual_cases, expected_cases, strict=True
            ):
                self.assertEqual(actual_case, expected_case)
                with self.subTest(spelling=spelling, case=actual_case):
                    self.assertEqual(
                        self.unsqueeze_contract(
                            actual,
                            self.newaxis_index(torch, "tuple", spelling),
                        ),
                        self.unsqueeze_contract(
                            expected,
                            self.newaxis_index(
                                reference_torch, "tuple", spelling
                            ),
                        ),
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
                    self.assertEqual(
                        self.unsqueeze_contract(
                            actual,
                            self.newaxis_index(torch, "trailing", spelling),
                        ),
                        self.unsqueeze_contract(
                            expected,
                            self.newaxis_index(
                                reference_torch, "trailing", spelling
                            ),
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

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def autograd_contract(self, module, case, form):
        leaf, source = self.make_autograd_case(module, case)
        index = self.newaxis_index(module, form, "newaxis")
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

    def no_grad_contract(self, module, case, form):
        leaf, source = self.make_autograd_case(module, case)
        index = self.newaxis_index(module, form, "none")
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

    def unsqueeze_node_diagnostic(self, module, form):
        leaf = module.tensor(
            [2.0], dtype=module.float32, requires_grad=True
        )
        index = self.newaxis_index(module, form, "newaxis")
        try:
            module.nn.functional.dropout(
                None, p=leaf[index], training=False
            )
        except ValueError as error:
            return str(error)
        self.fail("dropout unexpectedly accepted an out-of-range tensor probability")

    def assert_autograd_and_no_grad_match_pytorch_2_13(self, form):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            with self.subTest(case=case, mode="autograd"):
                actual_metadata, actual_gradient = self.autograd_contract(
                    torch, case, form
                )
                expected_metadata, expected_gradient = self.autograd_contract(
                    reference_torch, case, form
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(actual_gradient, expected_gradient)

            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(
                    self.no_grad_contract(torch, case, form),
                    self.no_grad_contract(reference_torch, case, form),
                )

        self.assertEqual(
            self.unsqueeze_node_diagnostic(torch, form),
            self.unsqueeze_node_diagnostic(reference_torch, form),
        )

    def test_autograd_and_no_grad_match_pytorch_2_13_for_every_layout(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13("bare")

    def test_exact_leading_tuple_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13("tuple")

    def test_trailing_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assert_autograd_and_no_grad_match_pytorch_2_13("trailing")

    def lifetime_contract(self, module, case, form):
        if case == "scalar":
            leaf = module.tensor(
                -2.0, dtype=module.float32, requires_grad=True
            )
        elif case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
        else:
            values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
            leaf = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
            )

        def make_view():
            source = leaf * 2.0
            if case == "offset":
                source = source[1]
            elif case == "noncontiguous":
                source = source.transpose(0, 3)[1]
            index = self.newaxis_index(module, form, "newaxis")
            return source[index], source.data_ptr()

        result, source_data_ptr = make_view()
        gc.collect()
        metadata = (
            result.data_ptr() == source_data_ptr,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.tolist(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
        )
        weights = module.ones(tuple(result.shape), dtype=module.float32)
        (result * weights).sum().backward()
        return metadata, np.asarray(leaf.grad, dtype=np.float32).copy()

    def assert_source_lifetime_matches_pytorch_2_13(self, form):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            with self.subTest(case=case):
                actual_metadata, actual_gradient = self.lifetime_contract(
                    torch, case, form
                )
                expected_metadata, expected_gradient = self.lifetime_contract(
                    reference_torch, case, form
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(
                    actual_gradient, expected_gradient
                )

    def test_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13("bare")

    def test_exact_leading_tuple_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13("tuple")

    def test_trailing_source_lifetime_matches_pytorch_2_13(self):
        self.assert_source_lifetime_matches_pytorch_2_13("trailing")

    def mode_contract(self, module, form):
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
            index = self.newaxis_index(module, form, "newaxis")
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

            forwarding_index = self.newaxis_index(module, form, "none")
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
        nested_index = self.newaxis_index(module, form, "newaxis")
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
            self.mode_contract(torch, "bare"),
            self.mode_contract(reference_torch, "bare"),
        )

    def test_exact_leading_tuple_torch_function_mode_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch, "tuple"),
            self.mode_contract(reference_torch, "tuple"),
        )

    def test_trailing_torch_function_mode_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch, "trailing"),
            self.mode_contract(reference_torch, "trailing"),
        )


if __name__ == "__main__":
    unittest.main()
