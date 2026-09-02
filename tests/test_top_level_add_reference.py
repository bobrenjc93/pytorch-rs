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
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.add differentials require pinned PyTorch 2.13.0")

    def assert_matches(self, actual, expected, *, case):
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
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def assert_values_match(self, actual, expected, *, case):
        with self.subTest(case=case, shape=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def make_cases(module):
        broadcast_left = module.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]],
            dtype=module.float32,
        ).transpose(0, 2)
        broadcast_right = module.tensor([[2.0], [3.0], [4.0]], dtype=module.float32)

        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        zero_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x0000_0000, 0x8000_0000, 0x0000_0000),
            dtype=np.uint32,
        )

        return (
            ("broadcast", broadcast_left, broadcast_right),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                module.ones((1, 1, 2), dtype=module.float32),
            ),
            ("offset noncontiguous", strided[1], module.ones((1, 2), dtype=module.float32)),
            (
                "signed zero nan infinity",
                module.tensor(memoryview(special_bits.view(np.float32))),
                module.tensor(memoryview(zero_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_add(module, left, right, form):
        if form == "positional":
            return module.add(left, right)
        if form == "keywords":
            return module.add(input=left, other=right)
        if form == "x aliases":
            return module.add(x=left, x2=right)
        if form == "x1 aliases":
            return module.add(x1=left, x2=right)
        if form == "mixed keyword":
            return module.add(left, other=right)
        if form == "out none":
            return module.add(left, right, out=None)
        if form == "alpha int one":
            return module.add(left, right, alpha=1)
        if form == "alpha float one":
            return module.add(left, right, alpha=1.0)
        if form == "alpha numpy int one":
            return module.add(left, right, alpha=np.int64(1))
        if form == "alpha numpy float one":
            return module.add(left, right, alpha=np.float32(1.0))
        raise AssertionError(f"unknown torch.add form: {form}")

    @staticmethod
    def make_autograd_case(module, case):
        if case == "empty":
            left = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
            right = module.ones((1, 1, 3), dtype=module.float32, requires_grad=True)
            return left, right, left, right

        left_leaf = module.tensor(
            [[2.0, 3.0]], dtype=module.float32, requires_grad=True
        )
        right_leaf = module.tensor(
            [[5.0], [7.0], [11.0]], dtype=module.float32, requires_grad=True
        )
        if case == "broadcast":
            return left_leaf, right_leaf, left_leaf, right_leaf
        if case == "noncontiguous":
            return (
                left_leaf,
                right_leaf,
                left_leaf.transpose(0, 1),
                right_leaf.transpose(0, 1),
            )
        raise AssertionError(f"unknown torch.add autograd case: {case}")

    def test_tensor_tensor_values_layouts_and_edges_match_pytorch_2_13(self):
        forms = (
            "positional",
            "keywords",
            "x aliases",
            "x1 aliases",
            "mixed keyword",
            "out none",
            "alpha int one",
            "alpha float one",
            "alpha numpy int one",
            "alpha numpy float one",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_matches(
                    self.call_add(torch, actual_left, actual_right, form),
                    self.call_add(reference_torch, expected_left, expected_right, form),
                    case=(case, form),
                )

    def test_first_order_backward_through_full_sum_and_no_grad_match_pytorch_2_13(self):
        for case in ("broadcast", "noncontiguous", "empty"):
            actual_left_leaf, actual_right_leaf, actual_left, actual_right = (
                self.make_autograd_case(torch, case)
            )
            expected_left_leaf, expected_right_leaf, expected_left, expected_right = (
                self.make_autograd_case(reference_torch, case)
            )

            actual_output = torch.add(actual_left, actual_right, out=None)
            expected_output = reference_torch.add(expected_left, expected_right, out=None)
            self.assert_matches(actual_output, expected_output, case=(case, "forward"))

            actual_output.sum().backward()
            expected_output.sum().backward()
            self.assert_values_match(
                actual_left_leaf.grad,
                expected_left_leaf.grad,
                case=(case, "left gradient"),
            )
            self.assert_values_match(
                actual_right_leaf.grad,
                expected_right_leaf.grad,
                case=(case, "right gradient"),
            )

        actual_left = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)
        actual_right = torch.tensor([[3.0], [4.0]], dtype=torch.float32, requires_grad=True)
        expected_left = reference_torch.tensor(
            [[1.0, 2.0]], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [[3.0], [4.0]], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual = torch.add(actual_left.transpose(0, 1), actual_right.transpose(0, 1))
        with reference_torch.no_grad():
            expected = reference_torch.add(
                expected_left.transpose(0, 1), expected_right.transpose(0, 1)
            )
        self.assert_matches(actual, expected, case="no_grad")

    def dispatch_observation(self, module):
        native = module.tensor([2.0], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        marker = object()
        mode_calls = []
        override_calls = []
        order = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                mode_calls.append((func.__name__, types, args, kwargs))
                return marker

        with RecordingMode():
            mode_result = module.add(input=native, other=native, out=destination)

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append(("left", func.__name__, types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append(("right", func.__name__, types, args, kwargs))
                return marker

        override_result = module.add(LeftOverride(), RightOverride())

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("derived", tuple(item.__name__ for item in types)))
                return marker

        subclass_result = module.add(BaseOverride(), DerivedOverride())

        return {
            "mode_result": mode_result is marker,
            "mode_calls": (
                len(mode_calls),
                mode_calls[0][0],
                tuple(item.__name__ for item in mode_calls[0][1]),
                len(mode_calls[0][2]),
                tuple(mode_calls[0][3]),
                mode_calls[0][3]["input"] is native,
                mode_calls[0][3]["other"] is native,
                mode_calls[0][3]["out"] is destination,
            ),
            "override_result": override_result is marker,
            "override_calls": tuple(
                (label, name, tuple(item.__name__ for item in types), len(args), kwargs)
                for label, name, types, args, kwargs in override_calls
            ),
            "subclass_result": subclass_result is marker,
            "subclass_order": tuple(order),
        }

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        actual = self.dispatch_observation(torch)
        expected = self.dispatch_observation(reference_torch)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
