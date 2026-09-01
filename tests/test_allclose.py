import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_INPUT_ERROR = (
    "allclose(): only exact native CPU float32 Tensor inputs with identical "
    "shapes are supported; broadcasting is not supported"
)
MODE_ERROR = "allclose(): __torch_function__ modes are not supported"


class TensorAllCloseTests(unittest.TestCase):
    def assert_allclose_result(self, left, right, expected, **kwargs):
        method_result = left.allclose(right, **kwargs)
        function_result = torch.allclose(left, right, **kwargs)
        self.assertIs(type(method_result), bool)
        self.assertIs(type(function_result), bool)
        self.assertIs(method_result, expected)
        self.assertIs(function_result, expected)

    def test_scalar_empty_layouts_signed_zero_infinity_and_nan(self):
        contiguous_offset = torch.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0]]
        )[1]
        strided = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).transpose(
            0, 1
        )
        offset = torch.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]]
        ).transpose(0, 1)[1]
        strided_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        nan_pair = (torch.tensor([float("nan")]), torch.tensor([float("nan")]))

        cases = (
            (torch.tensor(1.0), torch.tensor(1.0), True, {}),
            (torch.zeros((2, 0, 3)), torch.ones((2, 0, 3)), True, {}),
            (contiguous_offset, torch.tensor([1.0, 2.0, 3.0]), True, {}),
            (contiguous_offset, torch.tensor([1.0, 2.0, 3.1]), False, {}),
            (strided, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), True, {}),
            (strided, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.1]]), False, {}),
            (offset, torch.tensor([20.0, 3.0, 4.0]), True, {}),
            (torch.tensor([0.0, -0.0]), torch.tensor([-0.0, 0.0]), True, {}),
            (
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                True,
                {},
            ),
            (torch.tensor([float("inf")]), torch.tensor([-float("inf")]), False, {}),
            (*nan_pair, False, {}),
            (*nan_pair, True, {"equal_nan": True}),
        )
        for left, right, expected, kwargs in cases:
            with self.subTest(
                left_shape=left.shape,
                left_stride=left.stride(),
                right_shape=right.shape,
                right_stride=right.stride(),
                kwargs=kwargs,
            ):
                self.assert_allclose_result(left, right, expected, **kwargs)

        self.assertEqual(strided_empty.stride(), (1, 3, 3))
        self.assert_allclose_result(
            strided_empty,
            torch.zeros((3, 0, 2)),
            True,
        )

    def test_finite_tolerance_boundaries(self):
        cases = (
            (
                torch.tensor([1.000009]),
                torch.tensor([1.0]),
                True,
                {"rtol": 1.0e-5, "atol": 0.0},
            ),
            (
                torch.tensor([1.00002]),
                torch.tensor([1.0]),
                False,
                {"rtol": 1.0e-5, "atol": 0.0},
            ),
            (
                torch.tensor([1.0e-8]),
                torch.tensor([0.0]),
                True,
                {"rtol": 0.0, "atol": 1.0e-8},
            ),
            (
                torch.tensor([1.1e-8]),
                torch.tensor([0.0]),
                False,
                {"rtol": 0.0, "atol": 1.0e-8},
            ),
            (
                torch.tensor([2.0]),
                torch.tensor([1.0]),
                True,
                {"rtol": True, "atol": False},
            ),
            (
                torch.tensor([2.0]),
                torch.tensor([1.0]),
                True,
                {"rtol": np.float32(1.0), "atol": np.int64(0)},
            ),
        )
        for left, right, expected, kwargs in cases:
            with self.subTest(kwargs=kwargs, left=left.tolist(), right=right.tolist()):
                self.assert_allclose_result(left, right, expected, **kwargs)

    def test_input_nonmutation_and_grad_state_preservation(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        left = leaf.transpose(0, 1)
        right = torch.tensor([[1.0, 3.0], [2.0, 4.0001]])
        left_before = left.clone()
        right_before = right.clone()

        self.assertIs(torch.allclose(left, right), False)
        self.assertIs(left.allclose(right, atol=0.001, rtol=0.0), True)
        self.assertIs(torch.equal(left, left_before), True)
        self.assertIs(torch.equal(right, right_before), True)
        self.assertIsNone(leaf.grad)
        self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            self.assertIs(torch.allclose(left, right, atol=0.001, rtol=0.0), True)
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(torch.is_grad_enabled())
        self.assertIsNone(leaf.grad)

    def test_callable_metadata_and_keyword_forms(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "allclose")
        bound = tensor.allclose

        self.assertIs(type(torch.allclose), types.BuiltinFunctionType)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertTrue(callable(torch.allclose))
        self.assertTrue(callable(descriptor))
        self.assertTrue(callable(bound))
        self.assertEqual(torch.allclose.__name__, "allclose")
        self.assertEqual(descriptor.__name__, "allclose")
        self.assertEqual(bound.__name__, "allclose")
        self.assertIsNone(torch.allclose.__text_signature__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        for callable_object in (torch.allclose, descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(torch.allclose(input=tensor, other=tensor), True)
        self.assertIs(torch.allclose(x=tensor, x2=tensor), True)
        self.assertIs(tensor.allclose(other=tensor), True)
        self.assertIs(tensor.allclose(x2=tensor), True)
        self.assertIs(descriptor(tensor, tensor), True)
        self.assertIn("allclose", torch.__all__)

    def test_binding_tolerance_and_equal_nan_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.allclose(),
                TypeError,
                'allclose() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.allclose(tensor),
                TypeError,
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.allclose(tensor, tensor, tensor, tensor, tensor, tensor),
                TypeError,
                "allclose() takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: tensor.allclose(),
                TypeError,
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.allclose(tensor, 1.0, 0.0, False, None),
                TypeError,
                "allclose() takes from 1 to 4 positional arguments but 5 were given",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=None),
                TypeError,
                "allclose(): argument 'rtol' must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, tensor, "1"),
                TypeError,
                "allclose(): argument 'rtol' (position 3) must be float, not str",
            ),
            (
                lambda: tensor.allclose(tensor, 1.0e-5, "1"),
                TypeError,
                "allclose(): argument 'atol' (position 3) must be float, not str",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol=-1.0e-8),
                RuntimeError,
                "atol must be greater than or equal to zero, but got -1e-08",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=float("nan")),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got nan",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=-float("inf")),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got -inf",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=1),
                TypeError,
                "allclose(): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: tensor.allclose(tensor, 1.0e-5, 1.0e-8, np.bool_(True)),
                TypeError,
                "allclose(): argument 'equal_nan' (position 4) must be bool, not numpy.bool",
            ),
            (
                lambda: torch.allclose(tensor, tensor, extra=True),
                TypeError,
                "allclose() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.allclose(tensor, other=tensor),
                TypeError,
                "allclose() got multiple values for argument 'other'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_unsupported_boundaries_are_explicit(self):
        tensor = torch.tensor([1.0])
        broadcast_cases = (
            lambda: torch.allclose(torch.zeros((2, 1)), torch.zeros((2, 3))),
            lambda: torch.allclose(torch.tensor(1.0), torch.tensor([1.0])),
            lambda: torch.allclose(torch.zeros((0,)), torch.zeros((1, 0))),
            lambda: tensor.allclose(torch.zeros((1, 1))),
        )
        for call in broadcast_cases:
            with self.assertRaisesRegex(
                NotImplementedError, f"^{re.escape(SUPPORTED_INPUT_ERROR)}$"
            ):
                call()

        non_tensor_cases = (
            (
                lambda: torch.allclose(None, tensor),
                "allclose(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, 1),
                "allclose(): argument 'other' (position 2) must be Tensor, not int",
            ),
            (
                lambda: tensor.allclose(1),
                "allclose(): argument 'other' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in non_tensor_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        override_cases = (
            lambda: torch.allclose(Override(), tensor),
            lambda: torch.allclose(tensor, Override()),
            lambda: tensor.allclose(Override()),
        )
        for call in override_cases:
            with self.assertRaisesRegex(
                NotImplementedError, f"^{re.escape(SUPPORTED_INPUT_ERROR)}$"
            ):
                call()

        class RejectingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise AssertionError("allclose should reject modes before dispatch")

        with RejectingMode():
            with self.assertRaisesRegex(NotImplementedError, f"^{re.escape(MODE_ERROR)}$"):
                torch.allclose(tensor, tensor)
            with self.assertRaisesRegex(NotImplementedError, f"^{re.escape(MODE_ERROR)}$"):
                tensor.allclose(tensor)

    @unittest.skipIf(reference_torch is None, "install the reference dependency group")
    def test_foreign_dtype_and_device_tensors_are_not_native_inputs(self):
        tensor = torch.tensor([1.0])
        foreign_tensors = (
            reference_torch.tensor([1.0], dtype=reference_torch.float64),
            reference_torch.empty((1,), device="meta"),
        )
        for foreign in foreign_tensors:
            with self.subTest(foreign=str(foreign)):
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{re.escape(SUPPORTED_INPUT_ERROR)}$"
                ):
                    torch.allclose(tensor, foreign)


if __name__ == "__main__":
    unittest.main()
