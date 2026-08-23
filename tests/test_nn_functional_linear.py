import importlib
import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalLinearTests(unittest.TestCase):
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
        bias_values = np.asarray([0.5, -1.25, 2.0, -3.5], dtype=np.float32)
        contiguous_bias = torch.tensor(bias_values.tolist())
        strided_bias = torch.tensor(
            np.stack((bias_values, np.full(4, 99.0, dtype=np.float32)), axis=1).tolist()
        ).transpose(0, 1)[0]
        offset_bias = torch.tensor(
            np.stack((np.full(4, 99.0, dtype=np.float32), bias_values)).tolist()
        )[1]
        offset_strided_bias = torch.tensor(
            np.arange(16, dtype=np.float32).reshape(2, 4, 2).tolist()
        )[1].transpose(0, 1)[0]
        singleton_bias = torch.tensor([[0.5, 99.0]]).transpose(0, 1)[0]
        empty_bias = torch.zeros((0,))

        return (
            ("contiguous", contiguous_input, contiguous_weight, contiguous_bias),
            ("strided", strided_input, strided_weight, strided_bias),
            ("offset", offset_input, offset_weight, offset_bias),
            (
                "offset-strided",
                offset_strided_input,
                offset_strided_weight,
                offset_strided_bias,
            ),
            ("zero rows", torch.zeros((0, 3)), contiguous_weight, contiguous_bias),
            (
                "zero rows singleton output",
                torch.zeros((0, 3)),
                torch.ones((1, 3)),
                singleton_bias,
            ),
            (
                "zero inner",
                torch.zeros((2, 0)),
                torch.zeros((4, 0)),
                strided_bias,
            ),
            ("zero outputs", contiguous_input, torch.zeros((0, 3)), empty_bias),
            ("offset zero rows", empty_offset_input, torch.ones((4, 2)), offset_bias),
            (
                "all zero",
                torch.zeros((0, 0)),
                torch.zeros((0, 0)),
                empty_bias,
            ),
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
            "rank-2 transformation",
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "``bias`` may be ``None``",
            "exact rank-1 tensor",
            "``(out_features,)``",
            "fresh, independent row-major tensor",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording for gradient-requiring operands",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)
        for unsupported_claim in (
            "sparse layout",
            "TensorFloat32",
            "additional dimensions",
            "Bias: :math:",
            "bias-free",
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
        for case, input, weight, _ in self.layout_cases():
            expected = input.matmul(weight.transpose(0, 1))
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

    def test_rank_one_bias_reuses_broadcast_add_with_linear_layouts(self):
        for case, input, weight, bias in self.layout_cases():
            expected = input.matmul(weight.transpose(0, 1)) + bias
            if expected.shape == (0, 1):
                expected = expected.squeeze(1)[..., None]
            calls = (
                ("positional", lambda: functional.linear(input, weight, bias)),
                (
                    "bias keyword",
                    lambda: functional.linear(input, weight, bias=bias),
                ),
                (
                    "keywords",
                    lambda: functional.linear(input=input, weight=weight, bias=bias),
                ),
            )
            for form, call in calls:
                self.assert_matches_composition(call(), expected, case=(case, form))

    def test_every_call_returns_fresh_storage_including_empty_outputs(self):
        for case, input, weight, bias in self.layout_cases():
            for bias_case, selected_bias in (("none", None), ("rank one", bias)):
                first = functional.linear(input, weight, selected_bias)
                second = functional.linear(input, weight, selected_bias)
                with self.subTest(case=case, bias=bias_case):
                    self.assertIsNot(first, second)
                    self.assertFalse(first.is_set_to(second))
                    self.assertFalse(first.is_set_to(input))
                    self.assertFalse(first.is_set_to(weight))
                    if selected_bias is not None:
                        self.assertFalse(first.is_set_to(selected_bias))
                    if first.numel() != 0:
                        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_incompatible_inner_dimensions_reuse_matmul_error(self):
        input = torch.zeros((2, 3))
        weight = torch.zeros((4, 5))
        message = "mat1 and mat2 shapes cannot be multiplied (2x3 and 5x4)"
        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
            functional.linear(input, weight)
        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
            functional.linear(input, weight, torch.zeros((4,)))

    def test_bias_length_mismatch_uses_pytorch_expansion_error(self):
        weight = torch.zeros((4, 3))
        bias = torch.zeros((5,))
        for rows in (0, 2):
            message = (
                "The expanded size of the tensor (4) must match the existing size "
                "(5) at non-singleton dimension 1.  "
                f"Target sizes: [{rows}, 4].  Tensor sizes: [5]"
            )
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    functional.linear(torch.zeros((rows, 3)), weight, bias)

    def test_requires_grad_operands_need_no_grad(self):
        cases = ((True, False), (False, True), (True, True))
        for input_requires_grad, weight_requires_grad in cases:
            input = torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                requires_grad=input_requires_grad,
            )
            weight = torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                requires_grad=weight_requires_grad,
            )
            with self.subTest(
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
                    expected = input.matmul(weight.transpose(0, 1))
                self.assert_matches_composition(output, expected, case="no_grad")
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(weight.grad)

    def test_bias_requires_grad_operands_need_no_grad(self):
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
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
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
                    expected = input.matmul(weight.transpose(0, 1)) + bias
                self.assert_matches_composition(output, expected, case="bias no_grad")
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(weight.grad)
                self.assertIsNone(bias.grad)

    def test_unsupported_features_are_rejected_before_native_composition(self):
        matrix = torch.ones((2, 2), requires_grad=True)
        plain_matrix = torch.ones((2, 2))
        vector = torch.ones((2,), requires_grad=True)

        for input, weight in ((vector, matrix), (matrix, vector)):
            with self.subTest(input=input.shape, weight=weight.shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports "
                    r"rank-2 input and weight tensors$",
                ):
                    functional.linear(input, weight)

        for shape in ((), (1, 2), (1, 1, 2)):
            with self.subTest(bias_shape=shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports a "
                    r"rank-1 bias tensor$",
                ):
                    functional.linear(plain_matrix, plain_matrix, torch.ones(shape))

        broadcast_only_bias = torch.ones((1,))
        message = (
            "The expanded size of the tensor (2) must match the existing size "
            "(1) at non-singleton dimension 1.  "
            "Target sizes: [2, 2].  Tensor sizes: [1]"
        )
        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
            functional.linear(plain_matrix, plain_matrix, broadcast_only_bias)

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "linear() only supports exact native Tensor input and weight operands"
        )
        for input, weight in ((Override(), matrix), (matrix, Override())):
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
                    TypeError, f"^{re.escape(exact_bias_error)}$"
                ):
                    functional.linear(plain_matrix, plain_matrix, bias)
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
                functional.linear(plain_matrix, plain_matrix, bias=vector)
        self.assertEqual(mode.calls, 0)


if __name__ == "__main__":
    unittest.main()
