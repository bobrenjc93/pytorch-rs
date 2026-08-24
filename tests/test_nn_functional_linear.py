import importlib
import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalLinearTests(unittest.TestCase):
    @staticmethod
    def linear_composition(input, weight):
        transposed_weight = weight.transpose(0, 1)
        if len(input.shape) == 1:
            return input[None].matmul(transposed_weight).squeeze(0)
        if len(input.shape) in (3, 4):
            output_shape = (*input.shape[:-1], weight.shape[0])
            return (
                input.flatten(0, len(input.shape) - 2)
                .matmul(transposed_weight)
                .reshape(output_shape)
            )
        return input.matmul(transposed_weight)

    def assert_matches_matmul(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def layout_cases(self):
        contiguous_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
        )
        contiguous_weight = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(4, 3).tolist()
        )
        strided_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(3, 2).tolist()
        ).transpose(0, 1)
        strided_weight = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist()
        ).transpose(0, 1)
        offset_input = torch.tensor(
            np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist()
        )[1]
        offset_weight = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
        )[1]
        offset_strided_input = torch.tensor(
            np.arange(18, dtype=np.float32).reshape(3, 3, 2).tolist()
        )[1].transpose(0, 1)
        offset_strided_weight = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )[1].transpose(0, 1)
        empty_offset_input = torch.zeros((2, 0, 2)).transpose(0, 2)[1]

        return (
            ("contiguous", contiguous_input, contiguous_weight),
            ("strided", strided_input, strided_weight),
            ("offset", offset_input, offset_weight),
            (
                "offset-strided",
                offset_strided_input,
                offset_strided_weight,
            ),
            ("zero rows", torch.zeros((0, 3)), contiguous_weight),
            ("zero inner", torch.zeros((2, 0)), torch.zeros((4, 0))),
            ("zero outputs", contiguous_input, torch.zeros((0, 3))),
            ("offset zero rows", empty_offset_input, torch.ones((4, 2))),
            ("all zero", torch.zeros((0, 0)), torch.zeros((0, 0))),
        )

    def vector_cases(self):
        contiguous_input = torch.tensor([1.0, -2.0, 3.0])
        contiguous_weight = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(4, 3).tolist()
        )
        strided_source = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist()
        ).transpose(0, 1)
        offset_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
        )[1]
        strided_weight = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist()
        ).transpose(0, 1)

        return (
            ("contiguous vector", contiguous_input, contiguous_weight),
            ("strided vector", strided_source[0], contiguous_weight),
            ("offset vector", offset_input, contiguous_weight),
            ("offset-strided vector", strided_source[1], strided_weight),
            ("zero features", torch.zeros((0,)), torch.zeros((4, 0))),
            ("zero outputs", contiguous_input, torch.zeros((0, 3))),
            ("all zero", torch.zeros((0,)), torch.zeros((0, 0))),
        )

    def rank_three_cases(self):
        contiguous_input = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        contiguous_weight = torch.tensor(
            np.arange(20, dtype=np.float32).reshape(5, 4).tolist()
        )
        strided_input = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
        ).transpose(1, 2)
        strided_weight = torch.tensor(
            np.arange(20, dtype=np.float32).reshape(4, 5).tolist()
        ).transpose(0, 1)
        offset_input = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist()
        )[1]
        offset_weight = torch.tensor(
            np.arange(40, dtype=np.float32).reshape(2, 5, 4).tolist()
        )[1]
        offset_strided_input = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist()
        )[1].transpose(1, 2)
        offset_strided_weight = torch.tensor(
            np.arange(40, dtype=np.float32).reshape(2, 4, 5).tolist()
        )[1].transpose(0, 1)
        empty_offset_input = torch.zeros((2, 2, 0, 4)).transpose(1, 2)[1]

        return (
            ("contiguous batch", contiguous_input, contiguous_weight),
            ("strided batch", strided_input, strided_weight),
            ("offset batch", offset_input, offset_weight),
            (
                "offset-strided batch",
                offset_strided_input,
                offset_strided_weight,
            ),
            ("zero batch", torch.zeros((0, 3, 4)), contiguous_weight),
            ("zero sequence", torch.zeros((2, 0, 4)), contiguous_weight),
            ("zero inner", torch.zeros((2, 3, 0)), torch.zeros((5, 0))),
            ("zero outputs", contiguous_input, torch.zeros((0, 4))),
            ("offset zero batch", empty_offset_input, torch.ones((5, 4))),
            ("all zero", torch.zeros((0, 0, 0)), torch.zeros((0, 0))),
        )

    def rank_four_cases(self):
        contiguous_input = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )
        contiguous_weight = torch.tensor(
            np.arange(30, dtype=np.float32).reshape(6, 5).tolist()
        )
        permuted_input = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(3, 4, 2, 5).tolist()
        ).permute(2, 0, 1, 3)
        strided_input = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 5, 4).tolist()
        ).transpose(2, 3)
        strided_weight = torch.tensor(
            np.arange(30, dtype=np.float32).reshape(5, 6).tolist()
        ).transpose(0, 1)
        offset_input = torch.tensor(
            np.arange(240, dtype=np.float32).reshape(2, 2, 3, 4, 5).tolist()
        )[1]
        offset_weight = torch.tensor(
            np.arange(60, dtype=np.float32).reshape(2, 6, 5).tolist()
        )[1]
        offset_strided_input = torch.tensor(
            np.arange(240, dtype=np.float32).reshape(2, 2, 3, 5, 4).tolist()
        )[1].transpose(2, 3)
        offset_strided_weight = torch.tensor(
            np.arange(60, dtype=np.float32).reshape(2, 5, 6).tolist()
        )[1].transpose(0, 1)
        empty_offset_input = torch.zeros((2, 2, 3, 0, 5)).transpose(2, 3)[1]

        return (
            ("contiguous rank-four", contiguous_input, contiguous_weight),
            ("permuted rank-four", permuted_input, contiguous_weight),
            ("strided rank-four", strided_input, strided_weight),
            ("offset rank-four", offset_input, offset_weight),
            (
                "offset-strided rank-four",
                offset_strided_input,
                offset_strided_weight,
            ),
            ("zero outer", torch.zeros((0, 3, 4, 5)), contiguous_weight),
            ("zero batch", torch.zeros((2, 0, 4, 5)), contiguous_weight),
            ("zero sequence", torch.zeros((2, 3, 0, 5)), contiguous_weight),
            ("zero inner", torch.zeros((2, 3, 4, 0)), torch.zeros((6, 0))),
            ("zero outputs", contiguous_input, torch.zeros((0, 5))),
            ("offset zero batch", empty_offset_input, torch.ones((6, 5))),
            ("all zero", torch.zeros((0, 0, 0, 0)), torch.zeros((0, 0))),
        )

    def test_import_signature_documentation_and_exports(self):
        imported = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn.functional import linear

        self.assertIs(imported, functional)
        self.assertIs(linear, functional.linear)
        self.assertIs(type(linear), types.FunctionType)
        self.assertEqual(linear.__name__, "linear")
        self.assertEqual(linear.__qualname__, "linear")
        self.assertEqual(linear.__module__, "torch_rs.nn.functional")
        self.assertEqual(linear.__defaults__, (None,))
        self.assertIsNone(linear.__kwdefaults__)
        self.assertFalse(hasattr(linear, "__text_signature__"))
        self.assertTrue(linear.__doc__.startswith("\nlinear(input, weight, bias=None)"))
        normalized_doc = " ".join(linear.__doc__.split())
        for documented_limit in (
            "bias-free rank-1, rank-2, rank-3, or rank-4 transformation",
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "``bias`` must be ``None``",
            "fresh, independent row-major tensor",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)
        for unsupported_claim in (
            "sparse layout",
            "TensorFloat32",
            "additional dimensions",
            "Bias: :math:",
        ):
            self.assertNotIn(unsupported_claim, normalized_doc)
        signature = inspect.signature(linear)
        self.assertEqual(tuple(signature.parameters), ("input", "weight", "bias"))
        self.assertIs(signature.parameters["input"].annotation, torch.Tensor)
        self.assertIs(signature.parameters["weight"].annotation, torch.Tensor)
        self.assertIsNone(signature.parameters["bias"].default)
        self.assertIs(signature.return_annotation, torch.Tensor)
        self.assertFalse(hasattr(torch, "_nn_functional_linear"))

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["linear"], linear)

    def test_rank_two_layouts_reuse_transpose_and_matmul(self):
        for case, input, weight in self.layout_cases():
            expected = self.linear_composition(input, weight)
            calls = (
                ("positional", lambda: functional.linear(input, weight)),
                (
                    "explicit none",
                    lambda: functional.linear(input, weight, None),
                ),
                (
                    "keywords",
                    lambda: functional.linear(input=input, weight=weight, bias=None),
                ),
            )
            for form, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, form))

    def test_rank_one_layouts_reuse_unsqueeze_matmul_and_squeeze(self):
        for case, input, weight in self.vector_cases():
            expected = self.linear_composition(input, weight)
            calls = (
                ("positional", lambda: functional.linear(input, weight)),
                (
                    "explicit none",
                    lambda: functional.linear(input, weight, None),
                ),
                (
                    "keywords",
                    lambda: functional.linear(input=input, weight=weight, bias=None),
                ),
            )
            for form, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, form))

    def test_rank_three_layouts_reuse_flatten_transpose_matmul_and_reshape(self):
        for case, input, weight in self.rank_three_cases():
            expected = self.linear_composition(input, weight)
            calls = (
                ("positional", lambda: functional.linear(input, weight)),
                (
                    "explicit none",
                    lambda: functional.linear(input, weight, None),
                ),
                (
                    "keywords",
                    lambda: functional.linear(input=input, weight=weight, bias=None),
                ),
            )
            for form, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, form))

    def test_rank_four_layouts_reuse_flatten_transpose_matmul_and_reshape(self):
        for case, input, weight in self.rank_four_cases():
            expected = self.linear_composition(input, weight)
            calls = (
                ("positional", lambda: functional.linear(input, weight)),
                (
                    "explicit none",
                    lambda: functional.linear(input, weight, None),
                ),
                (
                    "keywords",
                    lambda: functional.linear(input=input, weight=weight, bias=None),
                ),
            )
            for form, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, form))

    def test_rank_four_finite_overflow_matches_pytorch_kernel_classification(self):
        maximum = np.finfo(np.float32).max
        weight_values = np.zeros((16, 2), dtype=np.float32)
        weight_values[0] = (maximum, -maximum)
        weight = torch.tensor(weight_values.tolist())

        folded_input = torch.tensor(
            np.full((1, 1, 4, 2), maximum, dtype=np.float32).tolist()
        )
        folded_values = np.asarray(functional.linear(folded_input, weight))
        self.assertTrue(np.isposinf(folded_values[..., 0]).all())
        self.assertFalse(np.isnan(folded_values).any())
        np.testing.assert_array_equal(folded_values[..., 1:], 0.0)

        batched_input = torch.tensor(
            np.full((2, 2, 1, 2), maximum, dtype=np.float32).tolist()
        ).transpose(0, 1)
        batched_values = np.asarray(functional.linear(batched_input, weight))
        self.assertTrue(np.isnan(batched_values[..., 0]).all())
        self.assertFalse(np.isinf(batched_values[..., 1:]).any())
        np.testing.assert_array_equal(batched_values[..., 1:], 0.0)

    def test_every_call_returns_fresh_storage_including_empty_outputs(self):
        cases = tuple(
            (f"matrix {case}", input, weight)
            for case, input, weight in self.layout_cases()
        ) + tuple(
            (f"vector {case}", input, weight)
            for case, input, weight in self.vector_cases()
        ) + tuple(
            (f"rank-three {case}", input, weight)
            for case, input, weight in self.rank_three_cases()
        ) + tuple(
            (f"rank-four {case}", input, weight)
            for case, input, weight in self.rank_four_cases()
        )
        for case, input, weight in cases:
            first = functional.linear(input, weight)
            second = functional.linear(input, weight)
            with self.subTest(case=case):
                self.assertIsNot(first, second)
                self.assertFalse(first.is_set_to(second))
                self.assertFalse(first.is_set_to(input))
                self.assertFalse(first.is_set_to(weight))
                if first.numel() != 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_incompatible_inner_dimensions_reuse_matmul_error(self):
        cases = (
            (
                torch.zeros((2, 3)),
                torch.zeros((4, 5)),
                "mat1 and mat2 shapes cannot be multiplied (2x3 and 5x4)",
            ),
            (
                torch.zeros((3,)),
                torch.zeros((4, 5)),
                "mat1 and mat2 shapes cannot be multiplied (1x3 and 5x4)",
            ),
            (
                torch.zeros((2, 3, 4)),
                torch.zeros((5, 6)),
                "mat1 and mat2 shapes cannot be multiplied (6x4 and 6x5)",
            ),
            (
                torch.zeros((3, 2, 4)).transpose(0, 1),
                torch.zeros((5, 6)),
                "Expected size for first two dimensions of batch2 tensor to be: "
                "[2, 4] but got: [2, 6].",
            ),
            (
                torch.zeros((2, 3, 4)).permute(1, 2, 0),
                torch.zeros((5, 4)),
                "mat1 and mat2 shapes cannot be multiplied (12x2 and 4x5)",
            ),
            (
                torch.zeros((2, 3, 4, 5)),
                torch.zeros((6, 7)),
                "mat1 and mat2 shapes cannot be multiplied (24x5 and 7x6)",
            ),
            (
                torch.zeros((3, 2, 4, 5)).transpose(0, 1),
                torch.zeros((6, 7)),
                "Expected size for first two dimensions of batch2 tensor to be: "
                "[6, 5] but got: [6, 7].",
            ),
            (
                torch.zeros((2, 3, 4, 5)).permute(1, 2, 3, 0),
                torch.zeros((6, 7)),
                "mat1 and mat2 shapes cannot be multiplied (60x2 and 7x6)",
            ),
            (
                torch.zeros((0, 3, 4, 5)),
                torch.zeros((6, 7)),
                "mat1 and mat2 shapes cannot be multiplied (0x5 and 7x6)",
            ),
        )
        for input, weight, message in cases:
            with self.subTest(input=input.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    functional.linear(input, weight)

    def test_requires_grad_operands_need_no_grad(self):
        for rank, input_values in (
            (1, [1.0, 2.0, 3.0]),
            (2, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            (
                3,
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                ],
            ),
            (
                4,
                np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3).tolist(),
            ),
        ):
            for input_requires_grad, weight_requires_grad in (
                (True, False),
                (False, True),
                (True, True),
            ):
                input = torch.tensor(
                    input_values,
                    requires_grad=input_requires_grad,
                )
                weight = torch.tensor(
                    [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                    requires_grad=weight_requires_grad,
                )
                with self.subTest(
                    rank=rank,
                    input_requires_grad=input_requires_grad,
                    weight_requires_grad=weight_requires_grad,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^linear\(\): autograd recording is not supported$",
                    ):
                        functional.linear(input, weight)

                    with torch.no_grad():
                        output = functional.linear(input, weight)
                        expected = self.linear_composition(input, weight)
                    self.assert_matches_matmul(output, expected, case="no_grad")
                    self.assertFalse(output.requires_grad)
                    self.assertTrue(output.is_leaf)
                    self.assertIsNone(input.grad)
                    self.assertIsNone(weight.grad)

        noncontiguous_input = torch.zeros((3, 2, 4)).transpose(0, 1)
        incompatible_weight = torch.zeros((5, 6), requires_grad=True)
        with torch.no_grad():
            with self.assertRaisesRegex(
                RuntimeError,
                r"^mat1 and mat2 shapes cannot be multiplied \(6x4 and 6x5\)$",
            ):
                functional.linear(noncontiguous_input, incompatible_weight)

        noncontiguous_rank_four = torch.zeros((3, 2, 4, 5)).transpose(0, 1)
        incompatible_rank_four_weight = torch.zeros((6, 7), requires_grad=True)
        with torch.no_grad():
            with self.assertRaisesRegex(
                RuntimeError,
                r"^mat1 and mat2 shapes cannot be multiplied \(24x5 and 7x6\)$",
            ):
                functional.linear(
                    noncontiguous_rank_four,
                    incompatible_rank_four_weight,
                )

    def test_unsupported_features_are_rejected_before_native_composition(self):
        matrix = torch.ones((2, 2), requires_grad=True)
        vector = torch.ones((2,), requires_grad=True)
        scalar = torch.tensor(1.0)
        rank_three = torch.ones((1, 2, 2))
        rank_five = torch.ones((1, 1, 1, 2, 2))

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.linear only supports bias=None$",
        ):
            functional.linear(vector, matrix, bias=torch.ones((2,)))

        for input, weight in (
            (scalar, matrix),
            (rank_five, matrix),
            (matrix, vector),
            (vector, rank_three),
        ):
            with self.subTest(input=input.shape, weight=weight.shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports "
                    r"rank-1, rank-2, rank-3, or rank-4 input and rank-2 "
                    r"weight tensors$",
                ):
                    functional.linear(input, weight)

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "linear() only supports exact native Tensor input and weight operands"
        )
        for input, weight in ((Override(), matrix), (vector, Override())):
            with self.subTest(input=type(input), weight=type(weight)):
                with self.assertRaisesRegex(
                    TypeError, f"^{re.escape(exact_tensor_error)}$"
                ):
                    functional.linear(input, weight)
        self.assertEqual(Override.calls, 0)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        mode = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^linear\(\) does not support an active TorchFunctionMode$",
        ):
            with mode:
                functional.linear(vector, matrix, bias=matrix)
        self.assertEqual(mode.calls, 0)


if __name__ == "__main__":
    unittest.main()
