import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DotReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.dot differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_scalar_matches(
        self, actual, expected, *, case, exact_bits=False, rtol=0.0, atol=0.0
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

        actual_values = np.asarray(actual).reshape(-1)
        expected_values = expected.detach().cpu().numpy().reshape(-1)
        if exact_bits:
            np.testing.assert_array_equal(
                actual_values.view(np.uint32), expected_values.view(np.uint32)
            )
            return
        with self.subTest(case=case, classifications=True):
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
            non_nan = ~np.isnan(expected_values)
            np.testing.assert_array_equal(
                np.signbit(actual_values[non_nan]),
                np.signbit(expected_values[non_nan]),
            )
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    def layout_cases(self, module):
        return (
            (
                "contiguous",
                module.tensor([1.0, -2.0, 3.0], dtype=module.float32),
                module.tensor([4.0, 5.0, -6.0], dtype=module.float32),
            ),
            (
                "offset contiguous",
                module.tensor(
                    [[9.0, 8.0, 7.0], [1.0, -2.0, 3.0]],
                    dtype=module.float32,
                )[1],
                module.tensor(
                    [[6.0, 5.0, 4.0], [4.0, 5.0, -6.0]],
                    dtype=module.float32,
                )[1],
            ),
            (
                "noncontiguous",
                module.tensor(
                    [[1.0, 4.0], [-2.0, 5.0], [3.0, -6.0]],
                    dtype=module.float32,
                ).transpose(0, 1)[0],
                module.tensor(
                    [[7.0, 2.0], [11.0, 3.0], [13.0, 5.0]],
                    dtype=module.float32,
                ).transpose(0, 1)[1],
            ),
            (
                "empty",
                module.zeros((2, 0), dtype=module.float32)[1],
                module.ones((0,), dtype=module.float32),
            ),
        )

    def test_rank_one_values_layouts_and_argument_forms_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            calls = (
                (
                    "top positional",
                    lambda: torch.dot(actual_left, actual_right),
                    lambda: reference_torch.dot(expected_left, expected_right),
                ),
                (
                    "top keywords",
                    lambda: torch.dot(input=actual_left, tensor=actual_right),
                    lambda: reference_torch.dot(
                        input=expected_left, tensor=expected_right
                    ),
                ),
                (
                    "top input alias",
                    lambda: torch.dot(x=actual_left, tensor=actual_right),
                    lambda: reference_torch.dot(x=expected_left, tensor=expected_right),
                ),
                (
                    "method positional",
                    lambda: actual_left.dot(actual_right),
                    lambda: expected_left.dot(expected_right),
                ),
                (
                    "method keyword",
                    lambda: actual_left.dot(tensor=actual_right),
                    lambda: expected_left.dot(tensor=expected_right),
                ),
            )
            for form, actual_call, expected_call in calls:
                self.assert_scalar_matches(
                    actual_call(),
                    expected_call(),
                    case=(case, form),
                    atol=np.finfo(np.float32).eps,
                )

    def test_finite_accumulation_uses_composed_multiply_sum_and_stays_close_to_pytorch_2_13(
        self,
    ):
        left_values = [0.1257302165, -0.1321048588, 0.6404226422, 0.1049001142]
        right_values = [-0.5356693864, 0.3615950644, 1.3040000200, 0.9470809698]
        actual_left = torch.tensor(left_values)
        actual_right = torch.tensor(right_values)
        expected_left = reference_torch.tensor(
            left_values, dtype=reference_torch.float32
        )
        expected_right = reference_torch.tensor(
            right_values, dtype=reference_torch.float32
        )

        actual = torch.dot(actual_left, actual_right)
        composed = torch.mul(actual_left, actual_right).sum()
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            np.asarray(composed).reshape(-1).view(np.uint32),
        )
        self.assert_scalar_matches(
            actual,
            reference_torch.dot(expected_left, expected_right),
            case="finite accumulation",
            atol=np.finfo(np.float32).eps,
        )

    def test_signed_zero_and_nonfinite_values_match_pytorch_2_13(self):
        signed_zero_cases = (
            ("single negative zero", [-0.0], [1.0]),
            ("mixed signed zeros", [-0.0, 0.0], [1.0, 1.0]),
            ("negative zero products", [0.0, -0.0], [-1.0, 1.0]),
        )
        for case, left, right in signed_zero_cases:
            self.assert_scalar_matches(
                torch.dot(torch.tensor(left), torch.tensor(right)),
                reference_torch.dot(reference_torch.tensor(left), reference_torch.tensor(right)),
                case=case,
                exact_bits=True,
            )

        nonfinite_cases = (
            ("inf times zero", [float("inf"), 1.0], [0.0, 2.0]),
            ("opposite infinities", [float("inf"), float("-inf")], [1.0, 1.0]),
            ("quiet nan", [float("nan"), 1.0], [1.0, 2.0]),
        )
        for case, left, right in nonfinite_cases:
            self.assert_scalar_matches(
                torch.dot(torch.tensor(left), torch.tensor(right)),
                reference_torch.dot(reference_torch.tensor(left), reference_torch.tensor(right)),
                case=case,
            )

    def test_rank_and_shape_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.dot(torch.zeros((2,)), torch.zeros((3,))),
                lambda: reference_torch.dot(
                    reference_torch.zeros((2,)), reference_torch.zeros((3,))
                ),
            ),
            (
                lambda: torch.dot(torch.tensor(1.0), torch.zeros((1,))),
                lambda: reference_torch.dot(
                    reference_torch.tensor(1.0), reference_torch.zeros((1,))
                ),
            ),
            (
                lambda: torch.dot(torch.zeros((1,)), torch.tensor(1.0)),
                lambda: reference_torch.dot(
                    reference_torch.zeros((1,)), reference_torch.tensor(1.0)
                ),
            ),
            (
                lambda: torch.dot(torch.zeros((1, 1)), torch.zeros((1,))),
                lambda: reference_torch.dot(
                    reference_torch.zeros((1, 1)), reference_torch.zeros((1,))
                ),
            ),
            (
                lambda: torch.zeros((1,)).dot(torch.zeros((1, 1))),
                lambda: reference_torch.zeros((1,)).dot(
                    reference_torch.zeros((1, 1))
                ),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_error_matches(actual_call, expected_call)

    def test_no_grad_and_first_order_backward_match_pytorch_2_13(self):
        actual_left = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        actual_right = torch.tensor([4.0, 5.0, -6.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, -2.0, 3.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [4.0, 5.0, -6.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_output = torch.dot(actual_left, actual_right)
        expected_output = reference_torch.dot(expected_left, expected_right)
        self.assert_scalar_matches(actual_output, expected_output, case="tracked output")
        actual_output.backward()
        expected_output.backward()
        np.testing.assert_array_equal(np.asarray(actual_left.grad), expected_left.grad.numpy())
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor(
            [2.0, -3.0], dtype=reference_torch.float32, requires_grad=True
        )
        torch.dot(actual_shared, actual_shared).backward()
        reference_torch.dot(expected_shared, expected_shared).backward()
        np.testing.assert_array_equal(
            np.asarray(actual_shared.grad), expected_shared.grad.numpy()
        )

        actual_base = torch.tensor(
            [[1.0, 4.0], [-2.0, 5.0], [3.0, -6.0]], requires_grad=True
        )
        actual_weight = torch.tensor(
            [[7.0, 2.0], [11.0, 3.0], [13.0, 5.0]], requires_grad=True
        )
        expected_base = reference_torch.tensor(
            [[1.0, 4.0], [-2.0, 5.0], [3.0, -6.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_weight = reference_torch.tensor(
            [[7.0, 2.0], [11.0, 3.0], [13.0, 5.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        torch.dot(
            actual_base.transpose(0, 1)[0], actual_weight.transpose(0, 1)[1]
        ).backward()
        reference_torch.dot(
            expected_base.transpose(0, 1)[0],
            expected_weight.transpose(0, 1)[1],
        ).backward()
        np.testing.assert_array_equal(np.asarray(actual_base.grad), expected_base.grad.numpy())
        np.testing.assert_array_equal(
            np.asarray(actual_weight.grad), expected_weight.grad.numpy()
        )

        actual_empty = torch.zeros((0,), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (0,), dtype=reference_torch.float32, requires_grad=True
        )
        torch.dot(actual_empty, actual_empty).backward()
        reference_torch.dot(expected_empty, expected_empty).backward()
        np.testing.assert_array_equal(np.asarray(actual_empty.grad), expected_empty.grad.numpy())

        actual_no_grad = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_no_grad = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_untracked = torch.dot(actual_no_grad, actual_no_grad)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.dot(expected_no_grad, expected_no_grad)
        self.assert_scalar_matches(
            actual_untracked, expected_untracked, case="no_grad output"
        )
        self.assertFalse(actual_untracked.requires_grad)
        self.assertTrue(torch.dot(actual_no_grad, actual_no_grad).requires_grad)
        self.assertTrue(reference_torch.dot(expected_no_grad, expected_no_grad).requires_grad)

    def dispatch_observation(self, module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        function = module.dot
        descriptor = inspect.getattr_static(module.Tensor, "dot")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        calls = (
            (lambda: function(left, right), function, 2, None),
            (lambda: function(input=left, tensor=right), function, 0, ("input", "tensor")),
            (lambda: function(x=left, tensor=right), function, 0, ("x", "tensor")),
            (lambda: left.dot(right), descriptor, 2, None),
            (lambda: left.dot(tensor=right), descriptor, 1, ("tensor",)),
        )
        for call, expected_function, expected_arg_count, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is expected_function,
                    dispatch_types == (),
                    len(args) == expected_arg_count,
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == expected_keywords,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, expected_function, expected_arg_count, expected_keywords in (
            (lambda value: function(value, right), function, 2, None),
            (lambda value: function(left, value), function, 2, None),
            (
                lambda value: function(input=left, tensor=value),
                function,
                0,
                ("input", "tensor"),
            ),
            (lambda value: function(left, right, out=value), function, 2, ("out",)),
            (lambda value: left.dot(value), descriptor, 2, None),
            (lambda value: left.dot(tensor=value), descriptor, 1, ("tensor",)),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is expected_function,
                    dispatch_types == (Override,),
                    len(args) == expected_arg_count,
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == expected_keywords,
                )
            )

        order = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("left", tuple(item.__name__ for item in types)))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("right", tuple(item.__name__ for item in types)))
                return marker

        both_result = function(LeftOverride(), RightOverride())

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append(("override", func is function))
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(input=left, tensor=FallbackOverride())

        method_fallback_events = []

        class MethodFallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                method_fallback_events.append(("override", func is descriptor))
                return marker

        method_declining_mode = RecordingMode(NotImplemented)
        with method_declining_mode:
            method_fallback_result = left.dot(tensor=MethodFallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, other=right),
            lambda: function(left, right, dtype=module.float32),
            lambda: left.dot([]),
            lambda: left.dot(other=right),
            lambda: left.dot(right, out=None),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            both_result is marker,
            order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            method_fallback_result is marker,
            len(method_declining_mode.calls),
            method_fallback_events,
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        function = module.dot
        owner = function.__reduce__()[1][0]
        descriptor = inspect.getattr_static(module.Tensor, "dot")
        tensor = module.tensor([1.0])
        bound = tensor.dot
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            function_signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            function_signature_error = None
        try:
            inspect.signature(descriptor)
        except Exception as error:
            descriptor_signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            descriptor_signature_error = None
        return {
            "function_type": type(function).__name__,
            "function_is_builtin": type(function) is types.BuiltinFunctionType,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__,
            "function_owner_name": owner.__name__,
            "function_owner_qualname": owner.__qualname__,
            "function_owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.dot is function,
            "function_doc": function.__doc__,
            "function_text_signature": function.__text_signature__,
            "function_signature_error": function_signature_error,
            "all_count": module.__all__.count("dot"),
            "wildcard_identity": wildcard_namespace["dot"] is function,
            "function_copy_identity": copy.copy(function) is function,
            "function_deepcopy_identity": copy.deepcopy(function) is function,
            "function_pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "descriptor_type": type(descriptor).__name__,
            "descriptor_is_method_descriptor": type(descriptor)
            is types.MethodDescriptorType,
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "descriptor_doc": descriptor.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "descriptor_signature_error": descriptor_signature_error,
            "descriptor_objclass_name": descriptor.__objclass__.__name__,
            "descriptor_objclass_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "descriptor_copy_identity": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy_identity": copy.deepcopy(descriptor) is descriptor,
            "descriptor_pickle_identity": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol=protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "bound_type": type(bound).__name__,
            "bound_is_builtin_method": type(bound) is types.BuiltinMethodType,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_module": bound.__module__,
            "bound_doc": bound.__doc__,
            "bound_text_signature": bound.__text_signature__,
            "bound_copy_identity": copy.copy(bound) is bound,
            "bound_deepcopy_identity": copy.deepcopy(bound) is bound,
        }

    def test_callable_import_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.dot(), lambda: reference_torch.dot()),
            (lambda: torch.dot(actual), lambda: reference_torch.dot(expected)),
            (lambda: torch.dot([], actual), lambda: reference_torch.dot([], expected)),
            (lambda: torch.dot(actual, []), lambda: reference_torch.dot(expected, [])),
            (
                lambda: torch.dot(actual, actual, actual),
                lambda: reference_torch.dot(expected, expected, expected),
            ),
            (
                lambda: torch.dot(input=actual, other=actual),
                lambda: reference_torch.dot(input=expected, other=expected),
            ),
            (
                lambda: torch.dot(actual, other=actual),
                lambda: reference_torch.dot(expected, other=expected),
            ),
            (
                lambda: torch.dot(actual, x2=actual),
                lambda: reference_torch.dot(expected, x2=expected),
            ),
            (
                lambda: torch.dot(actual, actual, input=actual),
                lambda: reference_torch.dot(expected, expected, input=expected),
            ),
            (
                lambda: torch.dot(actual, actual, tensor=actual),
                lambda: reference_torch.dot(expected, expected, tensor=expected),
            ),
            (
                lambda: torch.dot(actual, actual, dtype=torch.float32),
                lambda: reference_torch.dot(
                    expected, expected, dtype=reference_torch.float32
                ),
            ),
            (lambda: actual.dot(), lambda: expected.dot()),
            (lambda: actual.dot([]), lambda: expected.dot([])),
            (lambda: actual.dot(other=actual), lambda: expected.dot(other=expected)),
            (lambda: actual.dot(x2=actual), lambda: expected.dot(x2=expected)),
            (lambda: actual.dot(actual, actual), lambda: expected.dot(expected, expected)),
            (
                lambda: actual.dot(actual, tensor=actual),
                lambda: expected.dot(expected, tensor=expected),
            ),
            (
                lambda: actual.dot(actual, out=None),
                lambda: expected.dot(expected, out=None),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_error_matches(actual_call, expected_call)

    def test_explicit_unsupported_boundaries_are_not_expanded(self):
        actual_out = torch.tensor(99.0)
        with self.assertRaisesRegex(
            RuntimeError, r"^dot\(\): the 'out' argument is not supported$"
        ):
            torch.dot(torch.zeros((1,)), torch.ones((1,)), out=actual_out)
        self.assertEqual(actual_out.item(), 99.0)

        expected_out = reference_torch.tensor(99.0, dtype=reference_torch.float32)
        self.assertIs(
            reference_torch.dot(
                reference_torch.zeros((1,)),
                reference_torch.ones((1,)),
                out=expected_out,
            ),
            expected_out,
        )
        self.assertEqual(expected_out.item(), 0.0)

        for name in ("vdot", "inner", "outer"):
            self.assertFalse(hasattr(torch, name))
        with self.assertRaisesRegex(RuntimeError, "rank-2 tensors"):
            torch.matmul(torch.ones((2,)), torch.ones((2,)))


if __name__ == "__main__":
    unittest.main()
