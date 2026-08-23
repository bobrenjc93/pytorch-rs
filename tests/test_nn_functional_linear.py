import importlib
import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalLinearTests(unittest.TestCase):
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
            "bias-free rank-2 transformation",
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
                self.assert_matches_matmul(call(), expected, case=(case, form))

    def test_every_call_returns_fresh_storage_including_empty_outputs(self):
        for case, input, weight in self.layout_cases():
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
        input = torch.zeros((2, 3))
        weight = torch.zeros((4, 5))
        message = "mat1 and mat2 shapes cannot be multiplied (2x3 and 5x4)"
        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
            functional.linear(input, weight)

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
                self.assert_matches_matmul(output, expected, case="no_grad")
                self.assertFalse(output.requires_grad)
                self.assertTrue(output.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(weight.grad)

    def test_unsupported_features_are_rejected_before_native_composition(self):
        matrix = torch.ones((2, 2), requires_grad=True)
        vector = torch.ones((2,), requires_grad=True)

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.linear only supports bias=None$",
        ):
            functional.linear(vector, matrix, bias=torch.ones((2,)))

        for input, weight in ((vector, matrix), (matrix, vector)):
            with self.subTest(input=input.shape, weight=weight.shape):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.functional\.linear only supports "
                    r"rank-2 input and weight tensors$",
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
        for input, weight in ((Override(), matrix), (matrix, Override())):
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
                functional.linear(matrix, matrix, bias=matrix)
        self.assertEqual(mode.calls, 0)


if __name__ == "__main__":
    unittest.main()
