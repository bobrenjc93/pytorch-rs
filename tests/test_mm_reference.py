import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None

if __package__:
    from .test_mm import mm_layout_cases
else:
    from test_mm import mm_layout_cases


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorMmReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.mm differentials require pinned PyTorch 2.13.0")

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

        actual_values = np.asarray(actual.detach().cpu())
        expected_values = expected.detach().cpu().numpy()
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
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def test_rank_two_results_and_layouts_match_pytorch_2_13(self):
        actual_cases = mm_layout_cases(torch)
        expected_cases = mm_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            self.assert_matches(
                actual_left.mm(actual_right),
                expected_left.mm(expected_right),
                case=(case, "positional"),
            )
            self.assert_matches(
                actual_left.mm(mat2=actual_right),
                expected_left.mm(mat2=expected_right),
                case=(case, "mat2 keyword"),
            )

    def autograd_observation(self, module):
        def values(tensor):
            return np.asarray(tensor.detach().cpu()).tolist()

        left = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        right = module.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        output = left.mm(mat2=right)
        forward = (
            tuple(output.shape),
            output.stride(),
            output.storage_offset(),
            output.requires_grad,
            output.is_leaf,
            values(output),
        )
        (output * module.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
        gradients = (values(left.grad), values(right.grad))

        left_base = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        right_base = module.tensor(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
        )
        strided = left_base.transpose(0, 1).mm(right_base.transpose(0, 1))
        (strided * module.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
        strided_snapshot = (
            tuple(strided.shape),
            strided.stride(),
            values(strided),
            values(left_base.grad),
            values(right_base.grad),
        )

        offset_left_base = module.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist(),
            requires_grad=True,
        )
        offset_right_base = module.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 2, 3).tolist(),
            requires_grad=True,
        )
        offset_left = offset_left_base[1].transpose(0, 1)
        offset_right = offset_right_base[1].transpose(0, 1)
        offset_output = offset_left.mm(offset_right)
        (offset_output * module.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()
        offset_snapshot = (
            offset_left.storage_offset(),
            offset_left.stride(),
            offset_right.storage_offset(),
            offset_right.stride(),
            values(offset_output),
            values(offset_left_base.grad),
            values(offset_right_base.grad),
        )

        empty_left = module.zeros((2, 0), requires_grad=True)
        empty_right = module.zeros((0, 3), requires_grad=True)
        empty_output = empty_left.mm(empty_right)
        empty_output.sum().backward()
        empty_snapshot = (
            tuple(empty_output.shape),
            empty_output.stride(),
            empty_output.requires_grad,
            tuple(empty_left.grad.shape),
            empty_left.grad.stride(),
            tuple(empty_right.grad.shape),
            empty_right.grad.stride(),
        )

        no_rows = module.zeros((0, 2), requires_grad=True)
        populated_right = module.ones((2, 3), requires_grad=True)
        no_rows.mm(populated_right).sum().backward()
        no_rows_snapshot = (
            tuple(no_rows.grad.shape),
            no_rows.grad.stride(),
            values(populated_right.grad),
        )

        no_grad_left = module.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = module.tensor([[3.0], [4.0]], requires_grad=True)
        with module.no_grad():
            untracked = no_grad_left.mm(mat2=no_grad_right)
        no_grad_snapshot = (
            tuple(untracked.shape),
            untracked.stride(),
            untracked.requires_grad,
            untracked.is_leaf,
            values(untracked),
            no_grad_left.grad is None,
            no_grad_right.grad is None,
        )

        repeated_left = module.tensor([[1.0]], requires_grad=True)
        repeated_right = module.tensor([[2.0]], requires_grad=True)
        repeated_loss = repeated_left.mm(repeated_right).sum()
        repeated_loss.backward()
        try:
            repeated_loss.backward()
        except Exception as error:
            repeated_error = (type(error).__name__, str(error))
        else:
            repeated_error = None

        return (
            forward,
            gradients,
            strided_snapshot,
            offset_snapshot,
            empty_snapshot,
            no_rows_snapshot,
            no_grad_snapshot,
            repeated_error,
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_observation(torch),
            self.autograd_observation(reference_torch),
        )

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([[1.0]])
        right = module.tensor([[2.0]])
        descriptor = inspect.getattr_static(module.Tensor, "mm")
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for keyword in (None, "mat2"):
            mode = RecordingMode()
            with mode:
                result = (
                    left.mm(right)
                    if keyword is None
                    else left.mm(**{keyword: right})
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

        for keyword in (None, "mat2"):
            value = Override()
            Override.calls.clear()
            result = (
                left.mm(value)
                if keyword is None
                else left.mm(**{keyword: value})
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
                        tuple(kwargs) == ("mat2",),
                        isinstance(kwargs["mat2"], FallbackOverride),
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
                        tuple(kwargs) == ("mat2",),
                        isinstance(kwargs["mat2"], FallbackOverride),
                    )
                )
                return NotImplemented

        with DecliningMode():
            fallback_result = left.mm(mat2=FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: left.mm([]),
            lambda: left.mm(mat2=[]),
            lambda: left.mm(mat2=right, wat=right),
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
            events,
            invalid_observations,
        )

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def test_descriptor_binding_and_errors_match_pytorch_2_13(self):
        actual = torch.tensor([[1.0]])
        expected = reference_torch.tensor([[1.0]], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "mm")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "mm")

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual.mm, expected.mm, types.BuiltinMethodType),
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
        self.assertEqual(actual.mm.__qualname__, expected.mm.__qualname__)
        self.assertEqual(actual.mm.__module__, expected.mm.__module__)
        self.assertEqual(
            hasattr(actual_descriptor, "__module__"),
            hasattr(expected_descriptor, "__module__"),
        )
        self.assert_matches(
            actual_descriptor(actual, mat2=actual),
            expected_descriptor(expected, mat2=expected),
            case="unbound mat2 keyword",
        )

        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(1, actual),
                lambda: expected_descriptor(1, expected),
            ),
            (lambda: actual.mm(), lambda: expected.mm()),
            (lambda: actual.mm(actual, actual), lambda: expected.mm(expected, expected)),
            (
                lambda: actual.mm(actual, mat2=actual),
                lambda: expected.mm(expected, mat2=expected),
            ),
            (
                lambda: actual.mm(actual, out=actual),
                lambda: expected.mm(expected, out=expected),
            ),
            (lambda: actual.mm(other=actual), lambda: expected.mm(other=expected)),
            (lambda: actual.mm(x2=actual), lambda: expected.mm(x2=expected)),
            (lambda: actual.mm([]), lambda: expected.mm([])),
            (lambda: actual.mm(mat2=None), lambda: expected.mm(mat2=None)),
            (
                lambda: actual.mm([], out=actual),
                lambda: expected.mm([], out=expected),
            ),
            (
                lambda: actual.mm(mat2=[], wat=actual),
                lambda: expected.mm(mat2=[], wat=expected),
            ),
            (
                lambda: actual.mm(mat2=actual, wat=actual),
                lambda: expected.mm(mat2=expected, wat=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_rank_and_shape_errors_match_pytorch_2_13(self):
        shape_cases = (
            ((), (1, 1)),
            ((2,), (2, 2)),
            ((1, 2, 2), (2, 2)),
            ((2, 2), ()),
            ((2, 2), (2,)),
            ((2, 2), (1, 2, 2)),
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        )
        for left_shape, right_shape in shape_cases:
            actual_left = torch.ones(left_shape)
            actual_right = torch.ones(right_shape)
            expected_left = reference_torch.ones(left_shape)
            expected_right = reference_torch.ones(right_shape)
            with self.subTest(left=left_shape, right=right_shape):
                self.assert_error_matches(
                    lambda: actual_left.mm(actual_right),
                    lambda: expected_left.mm(expected_right),
                )


if __name__ == "__main__":
    unittest.main()
