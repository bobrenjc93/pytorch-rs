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
class TensorSubReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.sub differentials require pinned PyTorch 2.13.0")

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
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_calls_broadcast_layouts_empties_and_ieee_edges_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[10.0], [20.0], [30.0]])
        expected_right = reference_torch.tensor([[10.0], [20.0], [30.0]])

        calls = (
            (
                "tensor positional",
                actual_left.sub(actual_right),
                expected_left.sub(expected_right),
            ),
            (
                "tensor keyword",
                actual_left.sub(other=actual_right),
                expected_left.sub(other=expected_right),
            ),
            (
                "x2 alias",
                actual_left.sub(x2=actual_right),
                expected_left.sub(x2=expected_right),
            ),
            (
                "integer alpha",
                actual_left.sub(actual_right, alpha=1),
                expected_left.sub(expected_right, alpha=1),
            ),
            (
                "floating alpha",
                actual_left.sub(other=actual_right, alpha=1.0),
                expected_left.sub(other=expected_right, alpha=1.0),
            ),
            (
                "numpy alpha",
                actual_left.sub(actual_right, alpha=np.int64(1)),
                expected_left.sub(expected_right, alpha=np.int64(1)),
            ),
            (
                "numpy boolean alpha",
                actual_left.sub(actual_right, alpha=np.bool_(True)),
                expected_left.sub(expected_right, alpha=np.bool_(True)),
            ),
            (
                "tensor scalar alpha",
                actual_left.sub(actual_right, alpha=torch.tensor(1.0)),
                expected_left.sub(
                    expected_right, alpha=reference_torch.tensor(1.0)
                ),
            ),
            (
                "offset scalar",
                actual_left[1].sub(other=np.float32(-0.0)),
                expected_left[1].sub(other=np.float32(-0.0)),
            ),
        )
        for case, actual, expected in calls:
            self.assert_matches(actual, expected, case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        actual_broadcast = torch.ones((1, 1, 2))
        expected_broadcast = reference_torch.ones((1, 1, 2))
        self.assert_matches(
            actual_empty.sub(other=actual_broadcast),
            expected_empty.sub(other=expected_broadcast),
            case="strided broadcast empty",
        )

        left_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        right_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x3F80_0000,
                0xBF80_0000,
            ),
            dtype=np.uint32,
        )
        values_left = memoryview(left_bits.view(np.float32))
        values_right = memoryview(right_bits.view(np.float32))
        self.assert_matches(
            torch.tensor(values_left).sub(torch.tensor(values_right)),
            reference_torch.tensor(values_left).sub(
                reference_torch.tensor(values_right)
            ),
            case="tensor IEEE edges",
        )

        actual_bad = torch.zeros((2, 3))
        expected_bad = reference_torch.zeros((2, 3))
        actual_other = torch.ones((4,))
        expected_other = reference_torch.ones((4,))
        self.assert_error_matches(
            lambda: actual_bad.sub(actual_other),
            lambda: expected_bad.sub(expected_other),
        )

    def test_autograd_shared_operands_empties_and_no_grad_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            left = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            right = module.tensor([[10.0, 20.0]], requires_grad=True)
            weights = module.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            output = left.transpose(0, 1).sub(other=right, alpha=1)
            (output * weights).sum().backward()

            shared = module.tensor([2.0, -3.0], requires_grad=True)
            shared_loss = shared.sub(shared).sum()
            shared_loss.backward()
            shared_loss.backward()

            empty = module.zeros((2, 0, 3), requires_grad=True)
            broadcast = module.ones((1, 1, 3), requires_grad=True)
            empty_output = empty.sub(other=broadcast)
            empty_output.sum().backward()

            tracked = module.tensor([[1.0, 2.0]], requires_grad=True)
            with module.no_grad():
                untracked_tensor = tracked.transpose(0, 1).sub(
                    other=module.tensor([[3.0, 4.0]])
                )
                untracked_scalar = tracked.sub(2.0, alpha=1)

            outcomes.append(
                (
                    tuple(output.shape),
                    output.stride(),
                    output.requires_grad,
                    output.is_leaf,
                    np.asarray(left.grad).copy(),
                    np.asarray(right.grad).copy(),
                    np.asarray(shared.grad).copy(),
                    tuple(empty_output.shape),
                    empty_output.stride(),
                    empty_output.requires_grad,
                    tuple(empty.grad.shape),
                    empty.grad.numel(),
                    np.asarray(broadcast.grad).copy(),
                    untracked_tensor.requires_grad,
                    untracked_scalar.requires_grad,
                    tracked.sub(2.0).requires_grad,
                )
            )

        for actual, expected in zip(outcomes[0], outcomes[1], strict=True):
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            else:
                self.assertEqual(actual, expected)

    def test_descriptor_modes_overrides_and_supported_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "sub")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "sub")

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual.sub, expected.sub, types.BuiltinMethodType),
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

        self.assertEqual(actual_descriptor.__qualname__, expected_descriptor.__qualname__)
        self.assertEqual(actual.sub.__qualname__, expected.sub.__qualname__)
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(
            hasattr(actual_descriptor, "__module__"),
            hasattr(expected_descriptor, "__module__"),
        )

        self.assert_matches(
            actual_descriptor(actual, other=actual, alpha=1),
            expected_descriptor(expected, other=expected, alpha=1),
            case="unbound descriptor call",
        )

        cases = (
            (lambda: actual.sub(), lambda: expected.sub()),
            (
                lambda: actual.sub(actual, actual),
                lambda: expected.sub(expected, expected),
            ),
            (
                lambda: actual.sub(actual, other=actual),
                lambda: expected.sub(expected, other=expected),
            ),
            (
                lambda: actual.sub(actual, out=actual),
                lambda: expected.sub(expected, out=expected),
            ),
            (lambda: actual.sub([]), lambda: expected.sub([])),
            (lambda: actual.sub(other=None), lambda: expected.sub(other=None)),
            (
                lambda: actual.sub(actual, alpha=None),
                lambda: expected.sub(expected, alpha=None),
            ),
            (
                lambda: actual.sub(actual, alpha=True),
                lambda: expected.sub(expected, alpha=True),
            ),
            (
                lambda: actual.sub(np.uint64(2**63)),
                lambda: expected.sub(np.uint64(2**63)),
            ),
            (lambda: actual.sub(2**64), lambda: expected.sub(2**64)),
            (
                lambda: actual.sub(-(2**63) - 1),
                lambda: expected.sub(-(2**63) - 1),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(1, actual),
                lambda: expected_descriptor(1, expected),
            ),
            (
                lambda: actual_descriptor(self=actual, other=actual),
                lambda: expected_descriptor(self=expected, other=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertEqual(
            self.dispatch_observations(torch),
            self.dispatch_observations(reference_torch),
        )

    def dispatch_observations(self, module):
        tensor = module.tensor([1.0])
        other = module.tensor([2.0])
        descriptor = inspect.getattr_static(module.Tensor, "sub")
        marker = object()
        observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        for form in ("positional", "other", "x2", "alpha", "nonunit"):
            mode = RecordingMode()
            with mode:
                if form == "positional":
                    result = tensor.sub(other)
                elif form == "other":
                    result = tensor.sub(other=other)
                elif form == "x2":
                    result = tensor.sub(x2=other)
                elif form == "alpha":
                    result = tensor.sub(other, alpha=1)
                else:
                    result = tensor.sub(other, alpha=2)
            function, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    function is descriptor,
                    tuple(item.__name__ for item in dispatch_types),
                    args[0] is tensor,
                    len(args),
                    len(args) == 2 and args[1] is other,
                    None if kwargs is None else tuple(kwargs),
                    kwargs is not None and kwargs.get("other", other) is other,
                    kwargs is not None and kwargs.get("x2", other) is other,
                    kwargs is not None and kwargs.get("alpha", 1),
                )
            )

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        for location in ("other", "alpha"):
            value = Override()
            Override.calls.clear()
            result = (
                tensor.sub(value)
                if location == "other"
                else tensor.sub(other, alpha=value)
            )
            function, dispatch_types, args, kwargs = Override.calls[0]
            observations.append(
                (
                    location,
                    result is marker,
                    function is descriptor,
                    tuple(item.__name__ for item in dispatch_types),
                    args[0] is tensor,
                    len(args),
                    None if kwargs is None else tuple(kwargs),
                )
            )

        class NumericAlpha(float):
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        NumericAlpha.calls.clear()
        numeric = tensor.sub(other, alpha=NumericAlpha(1.0))
        observations.append(
            (
                NumericAlpha.calls == [],
                tuple(numeric.shape),
                numeric.stride(),
                np.asarray(numeric).tolist(),
            )
        )

        invalid = RecordingMode()
        try:
            with invalid:
                tensor.sub([])
        except Exception as error:
            observations.append((type(error).__name__, str(error), len(invalid.calls)))

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.sub(other=other, alpha=1)
        observations.append(
            (order, tuple(forwarded.shape), forwarded.stride(), np.asarray(forwarded).tolist())
        )
        return observations

    def test_nonunit_alpha_and_related_aliases_remain_deliberately_unsupported(self):
        tensor = torch.tensor([1.0])
        for alpha in (0, -1, 2, -0.0, np.float32(1.5), float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    RuntimeError, r"^Tensor\.sub only supports alpha=1$"
                ):
                    tensor.sub(tensor, alpha=alpha)

        self.assertFalse(hasattr(torch.Tensor, "sub_"))
        self.assertFalse(hasattr(torch.Tensor, "subtract"))
        self.assertFalse(hasattr(torch, "sub"))
        self.assertNotIn("sub", torch.__all__)
        self.assertTrue(hasattr(reference_torch.Tensor, "sub_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "subtract"))
        self.assertTrue(hasattr(reference_torch, "sub"))


if __name__ == "__main__":
    unittest.main()
