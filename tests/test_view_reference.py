import inspect
import operator
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class StatefulIndexDimension:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __index__(self):
        value = self.values[self.calls]
        self.calls += 1
        return value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorViewReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("view differentials require pinned PyTorch 2.13.0")

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        noncontiguous = base.transpose(0, 1)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (2, 0),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1),
            ),
            ("contiguous", base, (6, 4)),
            ("contiguous-offset", base[1], (2, 6)),
            ("noncontiguous-same-shape", noncontiguous, (3, 2, 4)),
            (
                "noncontiguous-compatible-split",
                noncontiguous,
                (3, 2, 2, 2),
            ),
        )

    def shape_argument(self, module, form, shape):
        if form == "tuple" or form == "keyword":
            return tuple(shape)
        if form == "list":
            return list(shape)
        if form == "Size":
            return module.Size(shape)
        raise AssertionError(form)

    def view_observation(self, module, source, shape, form):
        argument = self.shape_argument(module, form, shape)
        result = (
            source.view(size=argument)
            if form == "keyword"
            else source.view(argument)
        )
        direct = source.reshape(tuple(result.shape))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def test_shapes_strides_offsets_aliasing_and_values_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, shape = actual_case
            expected_name, expected_source, expected_shape = expected_case
            self.assertEqual((case, shape), (expected_name, expected_shape))
            for form in ("tuple", "list", "Size", "keyword"):
                with self.subTest(case=case, form=form):
                    actual = self.view_observation(torch, actual_source, shape, form)
                    expected = self.view_observation(
                        reference_torch, expected_source, shape, form
                    )
                    self.assertEqual(actual[:-1], expected[:-1])
                    np.testing.assert_array_equal(actual[-1], expected[-1])

    def single_view_observation(self, module, source, dimension):
        result = source.view(dimension)
        direct = source.reshape(tuple(result.shape))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def single_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32), 1),
            ("inferred", base, -1),
            ("offset", base[1], IntSubclass(12)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                np.int64(-1),
            ),
            (
                "compatible-noncontiguous",
                module.tensor(
                    np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
                    dtype=module.float32,
                ).transpose(0, 1)[0],
                IndexDimension(2),
            ),
        )

    def test_single_integer_shapes_and_views_match_pytorch_2_13(self):
        actual_cases = self.single_view_cases(torch)
        expected_cases = self.single_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimension = actual_case
            expected_name, expected_source, expected_dimension = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.single_view_observation(
                    torch, actual_source, actual_dimension
                )
                expected = self.single_view_observation(
                    reference_torch, expected_source, expected_dimension
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

    def two_view_observation(self, module, source, dim0, dim1):
        result = source.view(dim0, dim1)
        direct = source.reshape(tuple(result.shape))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def two_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = module.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32), 1, 1),
            ("inferred-contiguous", base, IntSubclass(6), np.int64(-1)),
            ("offset", base[1], IndexDimension(2), np.uint32(6)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                2,
                0,
            ),
            ("noncontiguous-same-shape", noncontiguous, 3, 2),
        )

    def test_two_positional_dimensions_match_pytorch_2_13(self):
        actual_cases = self.two_view_cases(torch)
        expected_cases = self.two_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dim0, actual_dim1 = actual_case
            expected_name, expected_source, expected_dim0, expected_dim1 = (
                expected_case
            )
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.two_view_observation(
                    torch, actual_source, actual_dim0, actual_dim1
                )
                expected = self.two_view_observation(
                    reference_torch,
                    expected_source,
                    expected_dim0,
                    expected_dim1,
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 1, 2))
            second = StatefulIndexDimension((3,))
            result = module.zeros((6,), dtype=module.float32).view(first, second)
            outcomes.append(
                (tuple(result.shape), result.stride(), first.calls, second.calls)
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def test_inference_extreme_empty_and_view_errors_match_pytorch_2_13(self):
        actual_source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=torch.float32,
        )
        expected_source = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=reference_torch.float32,
        )
        for form in ("tuple", "list", "Size", "keyword"):
            with self.subTest(kind="inferred", form=form):
                actual = self.view_observation(
                    torch, actual_source, (2, -1, 2), form
                )
                expected = self.view_observation(
                    reference_torch, expected_source, (2, -1, 2), form
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        maximum = sys.maxsize
        actual_empty = torch.zeros((0,), dtype=torch.float32)
        expected_empty = reference_torch.zeros(
            (0,), dtype=reference_torch.float32
        )
        for form in ("tuple", "list", "Size", "keyword"):
            with self.subTest(kind="extreme-empty", form=form):
                actual_argument = self.shape_argument(
                    torch, form, (0, maximum, maximum)
                )
                expected_argument = self.shape_argument(
                    reference_torch, form, (0, maximum, maximum)
                )
                actual_result = (
                    actual_empty.view(size=actual_argument)
                    if form == "keyword"
                    else actual_empty.view(actual_argument)
                )
                expected_result = (
                    expected_empty.view(size=expected_argument)
                    if form == "keyword"
                    else expected_empty.view(expected_argument)
                )
                self.assertEqual(
                    (
                        tuple(actual_result.shape),
                        actual_result.stride(),
                        actual_result.storage_offset(),
                        actual_result.numel(),
                        actual_result.data_ptr() == actual_empty.data_ptr(),
                        actual_result.tolist(),
                    ),
                    (
                        tuple(expected_result.shape),
                        expected_result.stride(),
                        expected_result.storage_offset(),
                        expected_result.numel(),
                        expected_result.data_ptr() == expected_empty.data_ptr(),
                        expected_result.tolist(),
                    ),
                )

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_noncontiguous = torch.tensor(
            values.tolist(), dtype=torch.float32
        ).transpose(0, 1)
        expected_noncontiguous = reference_torch.tensor(
            values.tolist(), dtype=reference_torch.float32
        ).transpose(0, 1)
        error_cases = (
            (
                lambda: actual_noncontiguous.view((6, 4)),
                lambda: expected_noncontiguous.view((6, 4)),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view((2, 2)),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view((2, 2)),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view((-1, -1)),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view((-1, -1)),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view((2, -2)),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view((2, -2)),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view((0, -1)),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view((0, -1)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        single_error_cases = (
            (
                lambda: actual_noncontiguous.view(-1),
                lambda: expected_noncontiguous.view(-1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(5),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(5),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(1),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(single_error_cases):
            with self.subTest(single_error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        two_dimension_error_cases = (
            (
                lambda: actual_noncontiguous.view(6, 4),
                lambda: expected_noncontiguous.view(6, 4),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(2, 2),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(2, 2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(-1, -1),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(-1, -1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(2, -2),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(2, -2),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(0, -1),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(0, -1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(
            two_dimension_error_cases
        ):
            with self.subTest(two_dimension_error_case=case):
                self.assert_error_matches(actual_call, expected_call)

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        result = source.view(3, -1)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def repeated_backward_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        loss = leaf.transpose(0, 1).view([3, 2]).sum()
        loss.backward()
        loss.backward()
        return self.tensor_array(leaf.grad, module).copy()

    def single_autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        result = leaf.view(-1)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == leaf.data_ptr(),
        )
        weights = module.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=module.float32
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def no_grad_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        with module.no_grad():
            result = source.view(3, 2)
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
            leaf.grad,
        )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        actual_metadata, actual_grad = self.autograd_outcome(torch)
        expected_metadata, expected_grad = self.autograd_outcome(reference_torch)
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_grad, expected_grad)
        np.testing.assert_array_equal(
            self.repeated_backward_outcome(torch),
            self.repeated_backward_outcome(reference_torch),
        )
        actual_single_metadata, actual_single_grad = self.single_autograd_outcome(
            torch
        )
        expected_single_metadata, expected_single_grad = self.single_autograd_outcome(
            reference_torch
        )
        self.assertEqual(actual_single_metadata, expected_single_metadata)
        np.testing.assert_array_equal(actual_single_grad, expected_single_grad)
        self.assertEqual(
            self.no_grad_outcome(torch), self.no_grad_outcome(reference_torch)
        )

    def descriptor_contract(self, module):
        tensor = module.tensor([1.0, 2.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "view")
        bound = tensor.view
        contract = []
        for callable_object, expected_type in (
            (descriptor, types.MethodDescriptorType),
            (bound, types.BuiltinMethodType),
        ):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                signature_error = type(error).__name__
            else:
                signature_error = None
            contract.append(
                (
                    type(callable_object) is expected_type,
                    callable_object.__name__,
                    callable_object.__qualname__,
                    callable_object.__doc__,
                    callable_object.__text_signature__,
                    getattr(callable_object, "__module__", "missing"),
                    signature_error,
                )
            )
        return (
            tuple(contract),
            descriptor.__objclass__.__name__,
            descriptor.__objclass__.__module__,
            repr(descriptor),
            descriptor is module.Tensor.view,
            descriptor.__get__(None, module.Tensor) is descriptor,
            tuple(descriptor(tensor, (2, 1)).shape),
            tuple(descriptor(tensor, -1).shape),
            tuple(descriptor(tensor, 2, 1).shape),
            tuple(descriptor(tensor, size=[2, 1]).shape),
        )

    def test_tensorbase_descriptor_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=module.float32
        )
        descriptor = inspect.getattr_static(module.Tensor, "view")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        def normalize(value):
            if value is tensor:
                return "self"
            if isinstance(value, list):
                return "list", tuple(value)
            if isinstance(value, tuple):
                return type(value).__name__, tuple(value)
            return value

        def normalize_call(call):
            function, dispatch_types, args, kwargs = call
            return (
                function is descriptor,
                function.__qualname__,
                dispatch_types,
                tuple(normalize(argument) for argument in args),
                {key: normalize(value) for key, value in kwargs.items()}
                if kwargs is not None
                else None,
            )

        records = []
        calls = (
            lambda: tensor.view((2, 3)),
            lambda: tensor.view([2, 3]),
            lambda: tensor.view(module.Size((2, 3))),
            lambda: tensor.view(-1),
            lambda: tensor.view(2, 3),
            lambda: tensor.view(size=(2, 3)),
        )
        for call in calls:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            records.append((result is marker, tuple(map(normalize_call, mode.calls))))

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.view(2, 3.0)

        invalid = RecordingMode(marker)
        try:
            with invalid:
                tensor.view(range(2))
        except Exception as error:
            invalid_error = type(error).__name__
        else:
            self.fail(f"{module.__name__} accepted a range shape")

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(size=[2, 3])
        sequence_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                integer_forwarded = tensor.view(-1)
        integer_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                two_dimension_forwarded = tensor.view(2, 3)
        two_dimension_order = tuple(order)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor.view(2, 3)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "deferred": (
                deferred_result is marker,
                tuple(map(normalize_call, deferred.calls)),
            ),
            "invalid": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in sequence_order
            ),
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "integer_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in integer_order
            ),
            "integer_forwarded": (
                tuple(integer_forwarded.shape),
                integer_forwarded.stride(),
                integer_forwarded.storage_offset(),
                integer_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "two_dimension_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in two_dimension_order
            ),
            "two_dimension_forwarded": (
                tuple(two_dimension_forwarded.shape),
                two_dimension_forwarded.stride(),
                two_dimension_forwarded.storage_offset(),
                two_dimension_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )

    def test_sequence_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        shape_factories = (
            lambda module: (IntSubclass(2), np.int64(3)),
            lambda module: [IndexDimension(2), np.uint32(3)],
            lambda module: module.Size((2, 3)),
            lambda module: (1, True, 6),
        )
        for factory in shape_factories:
            actual_result = actual.view(factory(torch))
            expected_result = expected.view(factory(reference_torch))
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        self.assert_error_matches(
            lambda: actual.view((2, 3.0)),
            lambda: expected.view((2, 3.0)),
        )
        with self.assertRaises(TypeError) as actual_overflow:
            actual.view((2**63, 1))
        with self.assertRaises(TypeError) as expected_overflow:
            expected.view((2**63, 1))
        for error in (actual_overflow.exception, expected_overflow.exception):
            self.assertIn("failed to unpack the object at pos 1", str(error))
            self.assertIn("Overflow when unpacking long long", str(error))

    def test_single_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        for dimension in (IntSubclass(6), np.int64(6), IndexDimension(6)):
            with self.subTest(dimension_type=type(dimension).__name__):
                actual_result = actual.view(dimension)
                expected_result = expected.view(dimension)
                self.assertEqual(
                    (
                        tuple(actual_result.shape),
                        actual_result.stride(),
                        actual_result.data_ptr() == actual.data_ptr(),
                    ),
                    (
                        tuple(expected_result.shape),
                        expected_result.stride(),
                        expected_result.data_ptr() == expected.data_ptr(),
                    ),
                )

        with self.assertRaises(TypeError) as actual_overflow:
            actual.view(2**63)
        with self.assertRaises(TypeError) as expected_overflow:
            expected.view(2**63)
        for error in (actual_overflow.exception, expected_overflow.exception):
            self.assertIn("failed to unpack the object at pos 1", str(error))
            self.assertIn("Overflow when unpacking long long", str(error))

        outcomes = []
        for module in (torch, reference_torch):
            dimension = StatefulIndexDimension((6, 1, 6))
            result = module.zeros((6,), dtype=module.float32).view(dimension)
            outcomes.append((tuple(result.shape), result.stride(), dimension.calls))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_two_positional_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        dimension_factories = (
            lambda: (IntSubclass(2), np.int64(3)),
            lambda: (IndexDimension(2), np.uint32(3)),
        )
        for factory in dimension_factories:
            actual_dimensions = factory()
            expected_dimensions = factory()
            actual_result = actual.view(*actual_dimensions)
            expected_result = expected.view(*expected_dimensions)
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        self.assert_error_matches(
            lambda: actual.view(2, 3.0),
            lambda: expected.view(2, 3.0),
        )
        for dimensions, position in (((2**63, 1), 1), ((1, 2**63), 2)):
            with self.assertRaises(TypeError) as actual_overflow:
                actual.view(*dimensions)
            with self.assertRaises(TypeError) as expected_overflow:
                expected.view(*dimensions)
            for error in (actual_overflow.exception, expected_overflow.exception):
                self.assertIn(
                    f"failed to unpack the object at pos {position}", str(error)
                )
                self.assertIn("Overflow when unpacking long long", str(error))

    def test_operator_index_poisoning_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        original_index = operator.index
        try:
            operator.index = lambda value: {2: 1, 3: 6}.get(value, value)

            actual_result = actual.view((2, 3))
            expected_result = expected.view((2, 3))
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )
            actual_positional = actual.view(2, 3)
            expected_positional = expected.view(2, 3)
            self.assertEqual(
                (
                    tuple(actual_positional.shape),
                    actual_positional.stride(),
                    actual_positional.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_positional.shape),
                    expected_positional.stride(),
                    expected_positional.data_ptr() == expected.data_ptr(),
                ),
            )
            self.assert_error_matches(
                lambda: actual.view((2, 3.0)),
                lambda: expected.view((2, 3.0)),
            )
            actual_flattened = actual.view(-1)
            expected_flattened = expected.view(-1)
            self.assertEqual(
                (
                    tuple(actual_flattened.shape),
                    actual_flattened.stride(),
                    actual_flattened.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_flattened.shape),
                    expected_flattened.stride(),
                    expected_flattened.data_ptr() == expected.data_ptr(),
                ),
            )
        finally:
            operator.index = original_index

    def test_deliberately_unsupported_overloads_remain_outside_the_binding(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        with self.assertRaises(TypeError):
            actual.view(1, 2, 3)
        self.assertEqual(expected.view(1, 2, 3).numel(), 6)

        with self.assertRaises(TypeError):
            actual.view(torch.float32)
        self.assertEqual(expected.view(reference_torch.float32).numel(), 6)

        with self.assertRaises(TypeError):
            actual.view(size=-1)
        with self.assertRaises(TypeError):
            expected.view(size=-1)
        with self.assertRaises(TypeError):
            actual.view(True)
        with self.assertRaises(TypeError):
            expected.view(True)


if __name__ == "__main__":
    unittest.main()
