import inspect
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


class TensorAllCloseTests(unittest.TestCase):
    def assert_allclose_result(self, left, right, expected, **kwargs):
        method_result = left.allclose(right, **kwargs)
        function_result = torch.allclose(left, right, **kwargs)
        self.assertIs(type(method_result), bool)
        self.assertIs(type(function_result), bool)
        self.assertIs(method_result, expected)
        self.assertIs(function_result, expected)

    def test_default_custom_and_float32_tolerance_behavior(self):
        cases = (
            (torch.tensor([1.0, 1.000001]), torch.tensor([1.0, 1.0]), {}, True),
            (torch.tensor([1.0, 1.00002]), torch.tensor([1.0, 1.0]), {}, False),
            (
                torch.tensor([1.0, 1.1000001]),
                torch.tensor([1.0, 1.0]),
                {"rtol": 0.11, "atol": 0.0},
                True,
            ),
            (
                torch.tensor([1.0, 1.1000001]),
                torch.tensor([1.0, 1.0]),
                {"rtol": 0.09, "atol": 0.0},
                False,
            ),
            (
                torch.tensor([np.float32(1.0000001e-08).item()]),
                torch.tensor([np.float32(6.024795e-16).item()]),
                {},
                True,
            ),
        )
        for left, right, kwargs, expected in cases:
            with self.subTest(left=left.tolist(), right=right.tolist(), kwargs=kwargs):
                self.assert_allclose_result(left, right, expected, **kwargs)

    def test_special_values_empty_views_and_rank_zero_broadcast(self):
        strided = torch.tensor([[1.0, 4.0], [2.0, 5.000001], [3.0, 6.0]]).transpose(
            0, 1
        )
        offset = torch.tensor(
            [[10.0, 20.0], [1.0, 2.000001], [3.0, 4.0]]
        ).transpose(0, 1)[1]
        cases = (
            (torch.tensor([0.0, -0.0]), torch.tensor([-0.0, 0.0]), {}, True),
            (
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                {},
                True,
            ),
            (
                torch.tensor([float("inf")]),
                torch.tensor([1.0]),
                {"rtol": float("inf"), "atol": float("inf")},
                False,
            ),
            (
                torch.tensor([float("nan")]),
                torch.tensor([float("nan")]),
                {},
                False,
            ),
            (
                torch.tensor([float("nan")]),
                torch.tensor([float("nan")]),
                {"equal_nan": True},
                True,
            ),
            (torch.zeros((2, 0, 3)), torch.ones((2, 0, 3)), {}, True),
            (
                strided,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                {},
                True,
            ),
            (offset, torch.tensor([20.0, 2.0, 4.0]), {}, True),
            (torch.tensor(1.0), torch.tensor([1.0, 1.0]), {}, True),
            (torch.tensor([1.0, 1.0]), torch.tensor(1.0), {}, True),
            (torch.tensor(1.0), torch.tensor([1.0, 1.0001]), {}, False),
            (torch.zeros((0,)), torch.tensor(1.0), {}, True),
        )
        for left, right, kwargs, expected in cases:
            with self.subTest(
                left_shape=left.shape,
                left_stride=left.stride(),
                right_shape=right.shape,
                right_stride=right.stride(),
                kwargs=kwargs,
            ):
                self.assert_allclose_result(left, right, expected, **kwargs)

    def test_aliases_callable_metadata_and_no_isclose_bool_tensor_surface(self):
        left = torch.tensor([1.0])
        right = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "allclose")
        bound = left.allclose

        self.assertIs(torch.allclose(input=left, other=right), True)
        self.assertIs(torch.allclose(x=left, x2=right), True)
        self.assertIs(torch.allclose(a=left, other=right), True)
        self.assertIs(torch.allclose(x1=left, x2=right), True)
        self.assertIs(left.allclose(x2=right), True)

        self.assertIs(type(torch.allclose), types.BuiltinFunctionType)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(torch.allclose.__name__, "allclose")
        self.assertEqual(descriptor.__name__, "allclose")
        self.assertEqual(bound.__name__, "allclose")
        self.assertIsNone(torch.allclose.__text_signature__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertIn("allclose", torch.__all__)
        for callable_object in (torch.allclose, descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertFalse(hasattr(torch, "isclose"))

    def test_binding_tolerance_and_unsupported_errors(self):
        tensor = torch.tensor([1.0])
        binding_cases = (
            (
                lambda: torch.allclose(),
                TypeError,
                'allclose\\(\\) missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.allclose(tensor),
                TypeError,
                'allclose\\(\\) missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.allclose(None, tensor),
                TypeError,
                "allclose\\(\\): argument 'input' \\(position 1\\) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, 1),
                TypeError,
                "allclose\\(\\): argument 'other' \\(position 2\\) must be Tensor, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, tensor, tensor, tensor, tensor),
                TypeError,
                "allclose\\(\\) takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=None),
                TypeError,
                "allclose\\(\\): argument 'rtol' must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol="1"),
                TypeError,
                "allclose\\(\\): argument 'atol' must be float, not str",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=1),
                TypeError,
                "allclose\\(\\): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=-1e-5, equal_nan=1),
                TypeError,
                "allclose\\(\\): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=-1e-5),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got -1e-05",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol=float("nan")),
                RuntimeError,
                "atol must be greater than or equal to zero, but got nan",
            ),
            (
                lambda: torch.allclose(tensor, tensor, extra=True),
                TypeError,
                "allclose\\(\\) got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, other=tensor),
                TypeError,
                "allclose\\(\\) got multiple values for argument 'other'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, x=tensor),
                TypeError,
                "allclose\\(\\) got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.allclose(input=tensor, other=tensor, x2=tensor),
                TypeError,
                "allclose\\(\\) got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: tensor.allclose(),
                TypeError,
                'allclose\\(\\) missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.allclose(tensor, 1, 2, False, 3),
                TypeError,
                "allclose\\(\\) takes from 1 to 4 positional arguments but 5 were given",
            ),
            (
                lambda: tensor.allclose(1),
                TypeError,
                "allclose\\(\\): argument 'other' \\(position 1\\) must be Tensor, not int",
            ),
            (
                lambda: tensor.allclose(input=tensor),
                TypeError,
                'allclose\\(\\) missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.allclose(tensor, other=tensor),
                TypeError,
                "allclose\\(\\) got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.allclose(tensor, x2=tensor),
                TypeError,
                "allclose\\(\\) got an unexpected keyword argument 'x2'",
            ),
        )
        for call, exception_type, message in binding_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception_type, f"^{message}$"):
                    call()

        with self.assertRaises(OverflowError):
            torch.allclose(tensor, tensor, rtol=2**2000)

        self.assertIs(torch.allclose(tensor, tensor, rtol=True, atol=np.uint64(0)), True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIs(torch.allclose(tensor, tensor, rtol=np.complex64(1 + 2j)), True)

        with self.assertRaisesRegex(
            NotImplementedError,
            "^allclose\\(\\): only exact native CPU float32 Tensor inputs with identical shapes or rank-0 scalar broadcasting are supported$",
        ):
            torch.allclose(torch.ones((2, 1)), torch.ones((2, 3)))
        with self.assertRaisesRegex(
            RuntimeError,
            "^The size of tensor a \\(2\\) must match the size of tensor b \\(3\\) at non-singleton dimension 0$",
        ):
            torch.allclose(torch.zeros((2,)), torch.zeros((3,)))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return True

        with RecordingMode(), self.assertRaisesRegex(
            NotImplementedError,
            "^allclose\\(\\): __torch_function__ modes are not supported$",
        ):
            torch.allclose(tensor, tensor)


if __name__ == "__main__":
    unittest.main()
