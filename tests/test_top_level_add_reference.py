import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelAddReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.add differentials require pinned PyTorch 2.13.0"
            )

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

    def test_tensor_tensor_values_layouts_and_ieee_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected_right = reference_torch.tensor([[2.0], [-0.0], [float("inf")]])

        calls = (
            (
                "positional tensors",
                lambda: torch.add(actual_left, actual_right),
                lambda: reference_torch.add(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.add(input=actual_left, other=actual_right),
                lambda: reference_torch.add(
                    input=expected_left, other=expected_right
                ),
            ),
            (
                "x aliases",
                lambda: torch.add(x=actual_left, x2=actual_right),
                lambda: reference_torch.add(x=expected_left, x2=expected_right),
            ),
            (
                "x1 aliases",
                lambda: torch.add(x1=actual_left, x2=actual_right),
                lambda: reference_torch.add(x1=expected_left, x2=expected_right),
            ),
            (
                "integer alpha one",
                lambda: torch.add(actual_left, actual_right, alpha=1),
                lambda: reference_torch.add(expected_left, expected_right, alpha=1),
            ),
            (
                "float alpha one",
                lambda: torch.add(actual_left, actual_right, alpha=np.float32(1.0)),
                lambda: reference_torch.add(
                    expected_left, expected_right, alpha=np.float32(1.0)
                ),
            ),
            (
                "out none",
                lambda: torch.add(actual_left, actual_right, out=None),
                lambda: reference_torch.add(expected_left, expected_right, out=None),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_offset = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ).transpose(0, 1)[1]
        expected_offset = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ).transpose(0, 1)[1]
        self.assert_matches(
            torch.add(actual_offset, torch.tensor([10.0, 20.0])),
            reference_torch.add(
                expected_offset, reference_torch.tensor([10.0, 20.0])
            ),
            case="offset noncontiguous",
        )

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.add(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.add(
                expected_empty, reference_torch.ones((1, 1, 2))
            ),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.add(torch.tensor(values), torch.zeros((5,))),
            reference_torch.add(
                reference_torch.tensor(values), reference_torch.zeros((5,))
            ),
            case="signed zero and non-finites",
        )

    def test_autograd_empties_shared_operands_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )

        actual_output = torch.add(
            actual_left.transpose(0, 1), actual_right.transpose(0, 1)
        )
        expected_output = reference_torch.add(
            expected_left.transpose(0, 1), expected_right.transpose(0, 1)
        )
        self.assert_matches(actual_output, expected_output, case="tracked views")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(actual_shared, actual_shared).sum().backward()
        reference_torch.add(expected_shared, expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(actual_empty, torch.ones((1, 1, 3))).sum().backward()
        reference_torch.add(
            expected_empty, reference_torch.ones((1, 1, 3))
        ).sum().backward()
        self.assert_matches(
            actual_empty.grad, expected_empty.grad, case="empty gradient"
        )

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.add(
                actual_no_grad.transpose(0, 1), torch.tensor([[3.0, 4.0]])
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.add(
                expected_no_grad.transpose(0, 1),
                reference_torch.tensor([[3.0, 4.0]]),
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(
            torch.add(actual_no_grad, torch.tensor([[3.0], [4.0]])).requires_grad
        )
        self.assertTrue(
            reference_torch.add(
                expected_no_grad, reference_torch.tensor([[3.0], [4.0]])
            ).requires_grad
        )

    @staticmethod
    def dispatch_observation(module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        destination = module.tensor([0.0])
        function = module.add
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (
                lambda: function(input=left, other=right, alpha=2),
                ("input", "other", "alpha"),
            ),
            (lambda: function(left, right, out=destination), ("out",)),
            (lambda: function(x1=left, x2=right), ("x1", "x2")),
        )
        for call, keyword_names in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value, right), None),
            (lambda value: function(left, value), None),
            (lambda value: function(input=left, other=value, alpha=2), "other"),
            (lambda value: function(left, right, alpha=value), "alpha"),
            (lambda value: function(left, right, out=value), "out"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
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

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(("base", tuple(item.__name__ for item in types)))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), DerivedOverride())

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(input=left, other=FallbackOverride(), alpha=2)

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, alpha=[]),
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
            subclass_result is marker,
            subclass_order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
