import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorMatmulReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.matmul differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

        actual_values = np.asarray(actual)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, classifications=True):
            np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(expected_values))
            non_nan = ~np.isnan(expected_values)
            np.testing.assert_array_equal(
                np.signbit(actual_values[non_nan]), np.signbit(expected_values[non_nan])
            )
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def layout_cases(self, module):
        offset_left = module.tensor(
            np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist(),
            dtype=module.float32,
        )[1]
        offset_right = module.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist(),
            dtype=module.float32,
        )[1]
        strided_left = module.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
        ).transpose(0, 1)
        strided_right = module.tensor(
            [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]], dtype=module.float32
        ).transpose(0, 1)
        return (
            (
                "contiguous",
                module.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
                ),
                module.tensor(
                    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                    dtype=module.float32,
                ),
            ),
            ("offset contiguous", offset_left, offset_right),
            ("strided", strided_left, strided_right),
            (
                "offset empty rows",
                module.zeros((2, 0, 2), dtype=module.float32).transpose(0, 2)[1],
                module.ones((2, 4), dtype=module.float32),
            ),
            (
                "empty inner",
                module.ones((2, 0), dtype=module.float32),
                module.zeros((0, 3), dtype=module.float32),
            ),
            (
                "non-finite",
                module.tensor(
                    [
                        [float("inf"), 1.0],
                        [float("-inf"), -1.0],
                        [float("nan"), 2.0],
                    ],
                    dtype=module.float32,
                ),
                module.tensor([[1.0, -1.0], [0.5, 1.0]], dtype=module.float32),
            ),
        )

    def test_rank_two_positional_keyword_results_and_layouts_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            self.assert_matches(
                actual_left.matmul(actual_right),
                expected_left.matmul(expected_right),
                case=(case, "positional"),
            )
            self.assert_matches(
                actual_left.matmul(other=actual_right),
                expected_left.matmul(other=expected_right),
                case=(case, "keyword"),
            )
            self.assert_matches(
                actual_left.matmul(x2=actual_right),
                expected_left.matmul(x2=expected_right),
                case=(case, "x2 alias"),
            )

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([[1.0]])
        right = module.tensor([[2.0]])
        descriptor = inspect.getattr_static(module.Tensor, "matmul")
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for keyword in (None, "other", "x2"):
            mode = RecordingMode()
            with mode:
                result = (
                    left.matmul(right)
                    if keyword is None
                    else left.matmul(**{keyword: right})
                )
            function, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    function is descriptor,
                    dispatch_types == (),
                    args[0] is left,
                    len(args),
                    len(args) == 2 and args[1] is right,
                    kwargs is None,
                    kwargs is not None
                    and tuple(kwargs) == (keyword,)
                    and kwargs[keyword] is right,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for keyword in (None, "other", "x2"):
            value = Override()
            Override.calls.clear()
            result = (
                left.matmul(value)
                if keyword is None
                else left.matmul(**{keyword: value})
            )
            function, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    function is descriptor,
                    dispatch_types == (Override,),
                    args[0] is left,
                    len(args),
                    len(args) == 2 and args[1] is value,
                    kwargs is None,
                    kwargs is not None
                    and tuple(kwargs) == (keyword,)
                    and kwargs[keyword] is value,
                )
            )

        events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "override",
                        func is descriptor,
                        types == (FallbackOverride,),
                        args[0] is left,
                        len(args) == 1,
                        tuple(kwargs) == ("x2",),
                        isinstance(kwargs["x2"], FallbackOverride),
                    )
                )
                return marker

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "mode",
                        func is descriptor,
                        types == (FallbackOverride,),
                        args[0] is left,
                        len(args) == 1,
                        tuple(kwargs) == ("x2",),
                        isinstance(kwargs["x2"], FallbackOverride),
                    )
                )
                return NotImplemented

        with DecliningMode():
            fallback_result = left.matmul(x2=FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: left.matmul([]),
            lambda: left.matmul(x2=[]),
            lambda: left.matmul(x2=right, wat=right),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (
                        type(error).__name__,
                        str(error),
                        len(invalid_mode.calls),
                    )
                )
            else:
                invalid_observations.append(None)

        return (
            mode_observations,
            override_observations,
            fallback_result is marker,
            events,
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def test_rank_two_shape_errors_match_pytorch_2_13(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        ):
            actual_left = torch.zeros(left_shape)
            actual_right = torch.zeros(right_shape)
            expected_left = reference_torch.zeros(left_shape)
            expected_right = reference_torch.zeros(right_shape)
            for actual_call, expected_call in (
                (
                    lambda: actual_left.matmul(actual_right),
                    lambda: expected_left.matmul(expected_right),
                ),
                (
                    lambda: actual_left.matmul(other=actual_right),
                    lambda: expected_left.matmul(other=expected_right),
                ),
                (
                    lambda: actual_left.matmul(x2=actual_right),
                    lambda: expected_left.matmul(x2=expected_right),
                ),
            ):
                with self.subTest(left=left_shape, right=right_shape):
                    self.assert_error_matches(actual_call, expected_call)

    def test_descriptor_metadata_and_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([[1.0]])
        expected = reference_torch.tensor([[1.0]], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "matmul")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "matmul")

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual.matmul, expected.matmul, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                actual_callable.__text_signature__, expected_callable.__text_signature__
            )
            with self.assertRaises(ValueError):
                inspect.signature(actual_callable)
            with self.assertRaises(ValueError):
                inspect.signature(expected_callable)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(actual_descriptor.__qualname__, expected_descriptor.__qualname__)
        self.assertEqual(actual.matmul.__qualname__, expected.matmul.__qualname__)
        self.assertEqual(actual.matmul.__module__, expected.matmul.__module__)
        self.assertEqual(
            hasattr(actual_descriptor, "__module__"),
            hasattr(expected_descriptor, "__module__"),
        )
        self.assert_matches(
            actual_descriptor(actual, other=actual),
            expected_descriptor(expected, other=expected),
            case="unbound keyword",
        )
        self.assert_matches(
            actual_descriptor(actual, x2=actual),
            expected_descriptor(expected, x2=expected),
            case="unbound x2 alias",
        )

        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(1, actual),
                lambda: expected_descriptor(1, expected),
            ),
            (lambda: actual.matmul(), lambda: expected.matmul()),
            (
                lambda: actual.matmul(actual, actual),
                lambda: expected.matmul(expected, expected),
            ),
            (
                lambda: actual.matmul(actual, other=actual),
                lambda: expected.matmul(expected, other=expected),
            ),
            (
                lambda: actual.matmul(actual, out=actual),
                lambda: expected.matmul(expected, out=expected),
            ),
            (lambda: actual.matmul(wat=actual), lambda: expected.matmul(wat=expected)),
            (lambda: actual.matmul([]), lambda: expected.matmul([])),
            (lambda: actual.matmul(other=None), lambda: expected.matmul(other=None)),
            (
                lambda: actual.matmul([], out=actual),
                lambda: expected.matmul([], out=expected),
            ),
            (lambda: actual.matmul(x2=[]), lambda: expected.matmul(x2=[])),
            (
                lambda: actual.matmul(actual, x2=actual),
                lambda: expected.matmul(expected, x2=expected),
            ),
            (
                lambda: actual.matmul(x2=actual, wat=actual),
                lambda: expected.matmul(x2=expected, wat=expected),
            ),
            (
                lambda: actual.matmul(**{"wat": actual, "x2": actual}),
                lambda: expected.matmul(**{"wat": expected, "x2": expected}),
            ),
            (
                lambda: actual.matmul(x2=actual, other=[]),
                lambda: expected.matmul(x2=expected, other=[]),
            ),
            (
                lambda: actual.matmul(x2=[], other=actual),
                lambda: expected.matmul(x2=[], other=expected),
            ),
            (
                lambda: actual.matmul(
                    actual, **{"x2": actual, "other": actual}
                ),
                lambda: expected.matmul(
                    expected, **{"x2": expected, "other": expected}
                ),
            ),
            (
                lambda: actual.matmul(
                    actual, **{"other": actual, "x2": actual}
                ),
                lambda: expected.matmul(
                    expected, **{"other": expected, "x2": expected}
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
