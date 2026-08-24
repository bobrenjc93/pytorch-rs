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
        if len(input.shape) == 3:
            output_shape = (*input.shape[:-1], weight.shape[0])
            return (
                input.flatten(0, 1)
                .matmul(transposed_weight)
                .reshape(output_shape)
            )
        return input.matmul(transposed_weight)

    def assert_matches_composition(self, actual, expected, *, case):
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
        bias_values = np.asarray([0.5, -1.25, 2.0, -3.5], dtype=np.float32)
        contiguous_bias = torch.tensor(bias_values.tolist())
        strided_bias = torch.tensor(
            np.stack(
                (bias_values, np.full(4, 99.0, dtype=np.float32)), axis=1
            ).tolist()
        ).transpose(0, 1)[0]
        offset_bias = torch.tensor(
            np.stack((np.full(4, 99.0, dtype=np.float32), bias_values)).tolist()
        )[1]
        offset_strided_bias = torch.tensor(
            np.arange(16, dtype=np.float32).reshape(2, 4, 2).tolist()
        )[1].transpose(0, 1)[0]
        empty_bias = torch.zeros((0,))

        return (
            (
                "contiguous vector",
                contiguous_input,
                contiguous_weight,
                contiguous_bias,
            ),
            (
                "strided vector",
                strided_source[0],
                contiguous_weight,
                strided_bias,
            ),
            ("offset vector", offset_input, contiguous_weight, offset_bias),
            (
                "offset-strided vector",
                strided_source[1],
                strided_weight,
                offset_strided_bias,
            ),
            (
                "zero features",
                torch.zeros((0,)),
                torch.zeros((4, 0)),
                strided_bias,
            ),
            (
                "zero outputs",
                contiguous_input,
                torch.zeros((0, 3)),
                empty_bias,
            ),
            (
                "all zero",
                torch.zeros((0,)),
                torch.zeros((0, 0)),
                empty_bias,
            ),
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
            "rank-1, rank-2, or rank-3 transformation",
            "optional bias for rank-1 input",
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "``bias`` must be ``None``",
            "exact rank-1 tensor",
            "``(out_features,)``",
            "For rank-2 and rank-3 input, ``bias`` must be ``None``",
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
            "bias-free rank-1",
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
                self.assert_matches_composition(call(), expected, case=(case, form))

    def test_rank_one_layouts_reuse_unsqueeze_matmul_and_squeeze(self):
        for case, input, weight, _ in self.vector_cases():
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
                self.assert_matches_composition(call(), expected, case=(case, form))

    def test_rank_one_bias_reuses_vector_composition_and_same_shape_add(self):
        for case, input, weight, bias in self.vector_cases():
            expected = self.linear_composition(input, weight) + bias
            calls = (
                ("positional", lambda: functional.linear(input, weight, bias)),
                (
                    "bias keyword",
                    lambda: functional.linear(input, weight, bias=bias),
                ),
                (
                    "keywords",
                    lambda: functional.linear(
                        input=input,
                        weight=weight,
                        bias=bias,
                    ),
                ),
            )
            for form, call in calls:
                actual = call()
                self.assert_matches_composition(
                    actual,
                    expected,
                    case=(case, form),
                )
                repeat = call()
                with self.subTest(case=(case, form), storage=True):
                    self.assertIsNot(actual, repeat)
                    self.assertFalse(actual.is_set_to(repeat))
                    self.assertFalse(actual.is_set_to(input))
                    self.assertFalse(actual.is_set_to(weight))
                    self.assertFalse(actual.is_set_to(bias))
                    if actual.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), repeat.data_ptr())

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
                self.assert_matches_composition(call(), expected, case=(case, form))

    def test_every_call_returns_fresh_storage_including_empty_outputs(self):
        cases = tuple(
            (f"matrix {case}", input, weight)
            for case, input, weight in self.layout_cases()
        ) + tuple(
            (f"vector {case}", input, weight)
            for case, input, weight, _ in self.vector_cases()
        ) + tuple(
            (f"rank-three {case}", input, weight)
            for case, input, weight in self.rank_three_cases()
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
        )
        for input, weight, message in cases:
            with self.subTest(input=input.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    functional.linear(input, weight)

        vector = torch.zeros((3,))
        incompatible_weight = torch.zeros((4, 5))
        message = "mat1 and mat2 shapes cannot be multiplied (1x3 and 5x4)"
        for bias in (torch.zeros((4,)), torch.zeros((5,))):
            with self.subTest(vector_bias=bias.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    functional.linear(vector, incompatible_weight, bias)

    def test_vector_bias_length_mismatch_uses_pytorch_expansion_error(self):
        input = torch.zeros((3,))
        weight = torch.zeros((4, 3))
        for bias_features in (0, 3, 5):
            message = (
                "The expanded size of the tensor (4) must match the existing size "
                f"({bias_features}) at non-singleton dimension 1.  "
                "Target sizes: [1, 4].  "
                f"Tensor sizes: [{bias_features}]"
            )
            with self.subTest(bias_features=bias_features):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    functional.linear(input, weight, torch.zeros((bias_features,)))

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
                    self.assert_matches_composition(output, expected, case="no_grad")
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

    def test_vector_bias_requires_grad_operands_need_no_grad(self):
        for input_requires_grad, weight_requires_grad, bias_requires_grad in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ):
            input = torch.tensor(
                [1.0, 2.0, 3.0],
                requires_grad=input_requires_grad,
            )
            weight = torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                requires_grad=weight_requires_grad,
            )
            bias = torch.tensor(
                [0.5, -1.5],
                requires_grad=bias_requires_grad,
            )
            with self.subTest(
                input_requires_grad=input_requires_grad,
                weight_requires_grad=weight_requires_grad,
                bias_requires_grad=bias_requires_grad,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^linear\(\): autograd recording is not supported$",
                ):
                    functional.linear(input, weight, bias)

                with torch.no_grad():
                    output = functional.linear(input, weight, bias)
                    expected = self.linear_composition(input, weight) + bias
                self.assert_matches_composition(
                    output,
                    expected,
                    case="bias no_grad",
                )
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(weight.grad)
                self.assertIsNone(bias.grad)

    def test_unsupported_features_are_rejected_before_native_composition(self):
        matrix = torch.ones((2, 2), requires_grad=True)
        plain_matrix = torch.ones((2, 2))
        vector = torch.ones((2,), requires_grad=True)
        scalar = torch.tensor(1.0)
        rank_three = torch.ones((1, 2, 2))
        rank_four = torch.ones((1, 1, 2, 2))

        for input in (plain_matrix, rank_three):
            with self.subTest(biased_input=input.shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports bias "
                    r"for rank-1 input$",
                ):
                    functional.linear(input, plain_matrix, torch.ones((2,)))

        for shape in ((), (1, 2), (1, 1, 2)):
            with self.subTest(bias_shape=shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports a "
                    r"rank-1 bias tensor$",
                ):
                    functional.linear(vector, plain_matrix, torch.ones(shape))

        for input, weight in (
            (scalar, matrix),
            (rank_four, matrix),
            (matrix, vector),
            (vector, rank_three),
        ):
            with self.subTest(input=input.shape, weight=weight.shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports "
                    r"rank-1, rank-2, or rank-3 input and rank-2 weight tensors$",
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

        exact_bias_error = (
            "linear() only supports an exact native Tensor bias or bias=None"
        )
        for bias in (Override(), 1.0, [1.0, 2.0]):
            with self.subTest(bias=type(bias)):
                with self.assertRaisesRegex(
                    TypeError,
                    f"^{re.escape(exact_bias_error)}$",
                ):
                    functional.linear(vector, plain_matrix, bias)
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
                functional.linear(vector, plain_matrix, bias=torch.ones((2,)))
        self.assertEqual(mode.calls, 0)


if __name__ == "__main__":
    unittest.main()
