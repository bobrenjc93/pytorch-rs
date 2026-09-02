import copy
import importlib
import inspect
import pickle
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
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.dot differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_scalar_matches(self, actual, expected, actual_left, actual_right, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertFalse(actual.is_set_to(actual_left))
            self.assertFalse(actual.is_set_to(actual_right))

        actual_value = np.asarray(actual).reshape(())
        expected_value = expected.detach().cpu().numpy().reshape(())
        with self.subTest(case=case, classification=True):
            self.assertEqual(np.isnan(actual_value), np.isnan(expected_value))
            self.assertEqual(np.signbit(actual_value), np.signbit(expected_value))
        if not np.isnan(expected_value):
            with self.subTest(case=case, bits=True):
                self.assertEqual(
                    actual_value.view(np.uint32).item(),
                    expected_value.view(np.uint32).item(),
                )

    @staticmethod
    def make_cases(module):
        dense = module.tensor(
            np.arange(24, dtype=np.float32).reshape(4, 2, 3).tolist(),
            dtype=module.float32,
        )
        offset_left = dense[2][1]
        offset_right = module.tensor(
            [[100.0, 200.0, 300.0], [4.0, -5.0, 6.0]], dtype=module.float32
        )[1]

        strided_base = module.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]],
            dtype=module.float32,
        )
        noncontiguous_left = strided_base.transpose(0, 1)[0]
        noncontiguous_right = module.tensor(
            [[-1.0, 9.0], [2.0, 8.0], [-3.0, 7.0], [4.0, 6.0]],
            dtype=module.float32,
        ).transpose(0, 1)[0]

        empty_left = module.zeros((0, 5), dtype=module.float32).transpose(0, 1)[2]
        empty_right = module.ones((5, 0), dtype=module.float32)[3]

        return (
            (
                "contiguous",
                module.tensor([1.0, -2.0, 3.5], dtype=module.float32),
                module.tensor([4.0, 5.0, -6.0], dtype=module.float32),
            ),
            ("offset", offset_left, offset_right),
            ("noncontiguous", noncontiguous_left, noncontiguous_right),
            ("empty", empty_left, empty_right),
            (
                "signed zero",
                module.tensor([-0.0, 0.0, -0.0, 0.0], dtype=module.float32),
                module.tensor([1.0, 1.0, 1.0, 1.0], dtype=module.float32),
            ),
            (
                "nan",
                module.tensor([1.0, float("nan"), 2.0], dtype=module.float32),
                module.tensor([3.0, 4.0, 5.0], dtype=module.float32),
            ),
            (
                "inf",
                module.tensor([float("inf"), 1.0], dtype=module.float32),
                module.tensor([2.0, 3.0], dtype=module.float32),
            ),
            (
                "inf times zero",
                module.tensor([float("inf"), 1.0], dtype=module.float32),
                module.tensor([0.0, 3.0], dtype=module.float32),
            ),
        )

    @staticmethod
    def call_dot(module, left, right, form):
        if form == "method positional":
            return left.dot(right)
        if form == "method keyword":
            return left.dot(tensor=right)
        if form == "function positional":
            return module.dot(left, right)
        if form == "function keywords":
            return module.dot(input=left, tensor=right)
        if form == "function x alias":
            return module.dot(x=left, tensor=right)
        if form == "function a alias":
            return module.dot(a=left, tensor=right)
        if form == "function x1 alias":
            return module.dot(x1=left, tensor=right)
        if form == "function out none":
            return module.dot(left, right, out=None)
        raise AssertionError(f"unknown dot form: {form}")

    def test_values_layouts_edge_cases_and_output_metadata_match_pytorch_2_13(self):
        forms = (
            "method positional",
            "method keyword",
            "function positional",
            "function keywords",
            "function x alias",
            "function a alias",
            "function x1 alias",
            "function out none",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(actual_left.shape, tuple(expected_left.shape))
            self.assertEqual(actual_left.stride(), expected_left.stride())
            self.assertEqual(actual_left.storage_offset(), expected_left.storage_offset())
            self.assertEqual(actual_right.shape, tuple(expected_right.shape))
            self.assertEqual(actual_right.stride(), expected_right.stride())
            self.assertEqual(actual_right.storage_offset(), expected_right.storage_offset())
            for form in forms:
                self.assert_scalar_matches(
                    self.call_dot(torch, actual_left, actual_right, form),
                    self.call_dot(reference_torch, expected_left, expected_right, form),
                    actual_left,
                    actual_right,
                    case=(case, form),
                )

    def test_rounding_regression_matches_pytorch_2_13(self):
        left_values = [
            -1611.425048828125,
            -1832.085693359375,
            458.4940490722656,
            -483.6356201171875,
        ]
        right_values = [
            -1716.787109375,
            1609.2052001953125,
            1125.843994140625,
            -1272.5712890625,
        ]
        actual_left = torch.tensor(left_values, dtype=torch.float32)
        actual_right = torch.tensor(right_values, dtype=torch.float32)
        expected_left = reference_torch.tensor(left_values, dtype=reference_torch.float32)
        expected_right = reference_torch.tensor(right_values, dtype=reference_torch.float32)
        expected = reference_torch.dot(expected_left, expected_right)

        self.assertEqual(expected.detach().numpy().view(np.uint32).item(), 0x4967EA58)
        self.assert_scalar_matches(
            torch.dot(actual_left, actual_right),
            expected,
            actual_left,
            actual_right,
            case="float32 rounding regression",
        )

    def test_shape_rank_and_binding_errors_match_pytorch_2_13(self):
        actual_vector = torch.ones((2,))
        expected_vector = reference_torch.ones((2,), dtype=reference_torch.float32)
        actual_other = torch.ones((2,))
        expected_other = reference_torch.ones((2,), dtype=reference_torch.float32)

        cases = (
            (lambda: torch.dot(), lambda: reference_torch.dot()),
            (lambda: torch.dot(actual_vector), lambda: reference_torch.dot(expected_vector)),
            (
                lambda: torch.dot(actual_vector, actual_other, actual_other),
                lambda: reference_torch.dot(expected_vector, expected_other, expected_other),
            ),
            (
                lambda: torch.dot([], actual_other),
                lambda: reference_torch.dot([], expected_other),
            ),
            (
                lambda: torch.dot(actual_vector, []),
                lambda: reference_torch.dot(expected_vector, []),
            ),
            (
                lambda: torch.dot(actual_vector, actual_other, foo=True),
                lambda: reference_torch.dot(expected_vector, expected_other, foo=True),
            ),
            (
                lambda: torch.dot(actual_vector, actual_other, input=actual_vector),
                lambda: reference_torch.dot(
                    expected_vector, expected_other, input=expected_vector
                ),
            ),
            (lambda: actual_vector.dot(), lambda: expected_vector.dot()),
            (
                lambda: actual_vector.dot(actual_other, actual_other),
                lambda: expected_vector.dot(expected_other, expected_other),
            ),
            (
                lambda: actual_vector.dot(input=actual_other),
                lambda: expected_vector.dot(input=expected_other),
            ),
            (
                lambda: actual_vector.dot(actual_other, tensor=actual_other),
                lambda: expected_vector.dot(expected_other, tensor=expected_other),
            ),
            (
                lambda: actual_vector.dot(actual_other, out=actual_other),
                lambda: expected_vector.dot(expected_other, out=expected_other),
            ),
            (
                lambda: actual_vector.dot([]),
                lambda: expected_vector.dot([]),
            ),
            (
                lambda: torch.dot(torch.tensor(1.0), actual_vector),
                lambda: reference_torch.dot(
                    reference_torch.tensor(1.0, dtype=reference_torch.float32),
                    expected_vector,
                ),
            ),
            (
                lambda: torch.dot(torch.ones((2, 1)), actual_vector),
                lambda: reference_torch.dot(
                    reference_torch.ones((2, 1), dtype=reference_torch.float32),
                    expected_vector,
                ),
            ),
            (
                lambda: torch.dot(actual_vector, torch.ones((3,))),
                lambda: reference_torch.dot(
                    expected_vector,
                    reference_torch.ones((3,), dtype=reference_torch.float32),
                ),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_error_matches(actual_call, expected_call)

    def test_no_grad_and_first_order_backward_match_pytorch_2_13(self):
        left_values = [[1.0, 99.0], [-2.0, 88.0], [3.0, 77.0], [4.0, 66.0]]
        right_values = [[5.0, 55.0], [6.0, 44.0], [-7.0, 33.0], [8.0, 22.0]]
        actual_left_leaf = torch.tensor(left_values, requires_grad=True)
        actual_right_leaf = torch.tensor(right_values, requires_grad=True)
        expected_left_leaf = reference_torch.tensor(
            left_values, dtype=reference_torch.float32, requires_grad=True
        )
        expected_right_leaf = reference_torch.tensor(
            right_values, dtype=reference_torch.float32, requires_grad=True
        )

        actual_left = actual_left_leaf.transpose(0, 1)[0]
        actual_right = actual_right_leaf.transpose(0, 1)[0]
        expected_left = expected_left_leaf.transpose(0, 1)[0]
        expected_right = expected_right_leaf.transpose(0, 1)[0]
        actual_loss = torch.dot(actual_left, actual_right)
        expected_loss = reference_torch.dot(expected_left, expected_right)
        self.assert_scalar_matches(
            actual_loss, expected_loss, actual_left, actual_right, case="tracked"
        )
        actual_loss.backward()
        expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left_leaf.grad), expected_left_leaf.grad.detach().cpu().numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right_leaf.grad), expected_right_leaf.grad.detach().cpu().numpy()
        )

        actual_empty_left = torch.zeros((0,), requires_grad=True)
        actual_empty_right = torch.zeros((0,), requires_grad=True)
        expected_empty_left = reference_torch.zeros(
            (0,), dtype=reference_torch.float32, requires_grad=True
        )
        expected_empty_right = reference_torch.zeros(
            (0,), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_loss = actual_empty_left.dot(actual_empty_right)
        expected_empty_loss = expected_empty_left.dot(expected_empty_right)
        self.assert_scalar_matches(
            actual_empty_loss,
            expected_empty_loss,
            actual_empty_left,
            actual_empty_right,
            case="empty tracked",
        )
        actual_empty_loss.backward()
        expected_empty_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty_left.grad), expected_empty_left.grad.cpu().numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_empty_right.grad), expected_empty_right.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = torch.dot(actual_left, actual_right)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.dot(expected_left, expected_right)
        self.assert_scalar_matches(
            actual_untracked,
            expected_untracked,
            actual_left,
            actual_right,
            case="no_grad",
        )

    def callable_observation(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        function = module.dot
        descriptor = inspect.getattr_static(module.Tensor, "dot")
        bound = tensor.dot
        imported = __import__(module.__name__, fromlist=["dot"]).dot
        package = importlib.import_module(module.__name__)

        observations = [
            type(function).__name__,
            function.__name__,
            function.__qualname__,
            function.__module__,
            function.__doc__,
            function.__text_signature__,
            module.__all__.count("dot"),
            imported is function,
            package.dot is function,
            copy.copy(function) is function,
            copy.deepcopy(function) is function,
            type(descriptor),
            descriptor.__name__,
            descriptor.__qualname__,
            descriptor.__objclass__.__name__,
            descriptor.__objclass__.__module__,
            hasattr(descriptor, "__module__"),
            descriptor.__doc__,
            descriptor.__text_signature__,
            copy.copy(descriptor) is descriptor,
            copy.deepcopy(descriptor) is descriptor,
            type(bound),
            bound.__name__,
            bound.__qualname__,
            bound.__module__,
            bound.__doc__,
            bound.__text_signature__,
            copy.copy(bound) is bound,
            copy.deepcopy(bound) is bound,
        ]
        for callable_object in (function, descriptor, bound):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                observations.append(type(error).__name__)
            else:
                observations.append("accepted")
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            observations.append(pickle.loads(pickle.dumps(function, protocol)) is function)
            observations.append(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
            )
        return observations

    def test_callable_import_copy_pickle_and_metadata_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_observation(torch),
            self.callable_observation(reference_torch),
        )

    def dispatch_observation(self, module):
        left = module.tensor([1.0, 2.0], dtype=module.float32)
        right = module.tensor([3.0, 4.0], dtype=module.float32)
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
        for call, expected_function, expected_arg_count, expected_keywords in (
            (lambda: left.dot(right), descriptor, 2, None),
            (lambda: left.dot(tensor=right), descriptor, 1, ("tensor",)),
            (lambda: module.dot(left, right), module.dot, 2, None),
            (lambda: module.dot(input=left, tensor=right), module.dot, 0, ("input", "tensor")),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            function, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    function is expected_function,
                    dispatch_types == (),
                    len(args) == expected_arg_count,
                    kwargs is None,
                    kwargs is not None
                    and tuple(kwargs) == expected_keywords
                    and all(
                        any(kwargs[key] is candidate for candidate in (left, right))
                        for key in expected_keywords
                    ),
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, expected_function, expected_types in (
            (lambda value: left.dot(value), descriptor, (Override,)),
            (lambda value: module.dot(value, right), module.dot, (Override,)),
            (lambda value: module.dot(left, value), module.dot, (Override,)),
            (lambda value: module.dot(left, right, out=value), module.dot, (Override,)),
        ):
            Override.calls.clear()
            value = Override()
            result = call(value)
            function, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    function is expected_function,
                    dispatch_types == expected_types,
                    bool(args) or bool(kwargs),
                )
            )

        events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(
                    (
                        func is module.dot,
                        types == (FallbackOverride,),
                        len(args),
                        kwargs is not None and tuple(kwargs) == ("input", "tensor"),
                    )
                )
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = module.dot(input=left, tensor=FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: module.dot([], right),
            lambda: module.dot(left, []),
            lambda: left.dot([]),
            lambda: module.dot(left, right, unexpected=True),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )
            else:
                invalid_observations.append(None)

        return (
            mode_observations,
            override_observations,
            fallback_result is marker,
            declining_mode.calls[0][0] is module.dot,
            declining_mode.calls[0][1] == (FallbackOverride,),
            events,
            invalid_observations,
        )

    def test_dispatch_boundaries_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_concrete_out_and_related_expansion_are_documented_boundaries(self):
        actual_left = torch.tensor([1.0, 2.0])
        actual_right = torch.tensor([3.0, 4.0])
        expected_left = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32
        )
        expected_right = reference_torch.tensor(
            [3.0, 4.0], dtype=reference_torch.float32
        )
        actual_out = torch.tensor(-9.0)
        expected_out = reference_torch.tensor(-9.0, dtype=reference_torch.float32)

        with self.assertRaisesRegex(
            RuntimeError, r"^dot\(\): the 'out' argument is not supported$"
        ):
            torch.dot(actual_left, actual_right, out=actual_out)
        self.assertEqual(actual_out.item(), -9.0)
        self.assertIs(reference_torch.dot(expected_left, expected_right, out=expected_out), expected_out)
        self.assertEqual(expected_out.item(), 11.0)

        self.assertFalse(hasattr(torch, "vdot"))
        self.assertTrue(hasattr(reference_torch, "vdot"))
        self.assertFalse(hasattr(torch, "inner"))
        self.assertTrue(hasattr(reference_torch, "inner"))
        self.assertFalse(hasattr(torch, "outer"))
        self.assertTrue(hasattr(reference_torch, "outer"))
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))

        with self.assertRaisesRegex(RuntimeError, "requires two rank-2 tensors"):
            torch.matmul(actual_left, actual_right)
        self.assertEqual(tuple(reference_torch.matmul(expected_left, expected_right).shape), ())


if __name__ == "__main__":
    unittest.main()
