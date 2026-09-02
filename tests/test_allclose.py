import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TorchAllcloseTests(unittest.TestCase):
    def assert_allclose_result(self, left, right, expected, **kwargs):
        result = torch.allclose(left, right, **kwargs)
        self.assertIs(type(result), bool)
        self.assertIs(result, expected)

    @staticmethod
    def bits_tensor(bits):
        values = np.asarray(bits, dtype=np.uint32).view(np.float32)
        return torch.tensor(memoryview(values), dtype=torch.float32)

    def test_equal_broadcasted_and_empty_shapes(self):
        cases = (
            (
                "same-shape exact",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                {},
                True,
            ),
            (
                "same-shape outside-default-tolerance",
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 2.0002]),
                {},
                False,
            ),
            (
                "rank-zero broadcasts",
                torch.tensor(3.0),
                torch.tensor([3.0, 3.0]),
                {},
                True,
            ),
            (
                "unequal broadcasted shapes",
                torch.tensor([[1.0], [1.0]]),
                torch.tensor([1.0, 1.0]),
                {},
                True,
            ),
            (
                "unequal broadcasted shapes mismatch",
                torch.tensor([[1.0], [1.0]]),
                torch.tensor([1.0, 2.0]),
                {},
                False,
            ),
            ("empty vector broadcasts", torch.zeros((0,)), torch.ones((1, 0)), {}, True),
            (
                "empty middle dimension broadcasts",
                torch.zeros((1, 0, 2)),
                torch.ones((3, 0, 2)),
                {},
                True,
            ),
        )
        for name, left, right, kwargs, expected in cases:
            with self.subTest(name=name):
                self.assert_allclose_result(left, right, expected, **kwargs)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^The size of tensor a \(3\) must match the size of tensor b \(4\) "
            r"at non-singleton dimension 2$",
        ):
            torch.allclose(torch.zeros((2, 0, 3)), torch.zeros((2, 0, 4)))

    def test_offset_and_noncontiguous_inputs(self):
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

        self.assertTrue(contiguous_offset.is_contiguous())
        self.assertEqual(contiguous_offset.storage_offset(), 3)
        self.assertFalse(strided.is_contiguous())
        self.assertEqual(strided.stride(), (1, 2))
        self.assertEqual(offset.storage_offset(), 1)
        self.assertEqual(offset.stride(), (2,))

        self.assert_allclose_result(
            contiguous_offset,
            torch.tensor([1.0, 2.000001, 3.0]),
            True,
            rtol=0.0,
            atol=2.0e-6,
        )
        self.assert_allclose_result(strided, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), True)
        self.assert_allclose_result(
            strided,
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0002]]),
            False,
        )
        self.assert_allclose_result(offset, torch.tensor([20.0, 3.0, 4.00003]), True)
        self.assert_allclose_result(offset, torch.tensor([20.0, 3.0, 4.0001]), False)
        self.assert_allclose_result(strided_empty[1], torch.zeros((0, 2)), True)

    def test_signed_zero_infinities_nans_and_tolerance_boundaries(self):
        self.assert_allclose_result(
            torch.tensor([0.0, -0.0]), torch.tensor([-0.0, 0.0]), True
        )
        self.assert_allclose_result(
            torch.tensor([float("inf"), -float("inf")]),
            torch.tensor([float("inf"), -float("inf")]),
            True,
        )
        self.assert_allclose_result(
            torch.tensor([float("inf")]), torch.tensor([-float("inf")]), False
        )
        self.assert_allclose_result(
            torch.tensor([float("inf")]),
            torch.tensor([0.0]),
            False,
            atol=float("inf"),
        )
        self.assert_allclose_result(
            torch.tensor([1.0]), torch.tensor([2.0]), True, atol=float("inf")
        )
        self.assert_allclose_result(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            False,
            rtol=float("inf"),
            atol=0.0,
        )

        left = torch.tensor([float("nan"), 1.0])
        right = torch.tensor([float("nan"), 1.0])
        self.assert_allclose_result(left, right, False)
        self.assert_allclose_result(left, right, False, equal_nan=False)
        self.assert_allclose_result(left, right, True, equal_nan=True)
        self.assert_allclose_result(
            torch.tensor([float("nan")]),
            torch.tensor([1.0]),
            False,
            equal_nan=True,
        )

        self.assert_allclose_result(
            torch.tensor([0.0]),
            self.bits_tensor([0x0000_0001]),
            True,
            rtol=0.0,
            atol=1.0e-45,
        )
        self.assert_allclose_result(
            torch.tensor([0.0]),
            self.bits_tensor([0x0000_0001]),
            False,
            rtol=0.0,
            atol=0.0,
        )
        next_after_one = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        next_step = float(np.float32(next_after_one - np.float32(1.0)))
        self.assert_allclose_result(
            torch.tensor([1.0]),
            torch.tensor([next_after_one]),
            True,
            rtol=0.0,
            atol=next_step,
        )
        self.assert_allclose_result(
            torch.tensor([1.0]),
            torch.tensor([next_after_one]),
            False,
            rtol=0.0,
            atol=next_step / 2.0,
        )

    def test_keyword_forms_callable_metadata_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        self.assertIs(type(torch.allclose), types.BuiltinFunctionType)
        self.assertTrue(callable(torch.allclose))
        self.assertEqual(torch.allclose.__name__, "allclose")
        self.assertIsNone(torch.allclose.__text_signature__)
        self.assertEqual(
            torch.allclose.__doc__.splitlines()[1],
            "allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, "
            "atol: float = 1e-08, equal_nan: bool = False) -> bool",
        )
        with self.assertRaises(ValueError):
            inspect.signature(torch.allclose)
        self.assertIs(copy.copy(torch.allclose), torch.allclose)
        self.assertIs(copy.deepcopy(torch.allclose), torch.allclose)
        self.assertIs(pickle.loads(pickle.dumps(torch.allclose)), torch.allclose)
        self.assertIn("allclose", torch.__all__)

        self.assertIs(torch.allclose(input=tensor, other=tensor), True)
        self.assertIs(torch.allclose(x=tensor, other=tensor), True)
        self.assertIs(torch.allclose(a=tensor, x2=tensor, rtol=1, atol=True), True)
        self.assertIs(torch.allclose(x1=tensor, x2=tensor, equal_nan=False), True)
        self.assertIs(torch.allclose(tensor, tensor, True, 0, False), True)

        self.assertFalse(hasattr(torch, "isclose"))
        self.assertFalse(hasattr(torch.Tensor, "allclose"))
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")
        with self.assertRaisesRegex(
            TypeError,
            r"^type 'torch_rs\.Tensor' is not an acceptable base type$",
        ):
            type("TensorSubclass", (torch.Tensor,), {})

    def test_type_and_binding_errors(self):
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
                lambda: torch.allclose(None, tensor),
                TypeError,
                "allclose(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, None),
                TypeError,
                "allclose(): argument 'other' (position 2) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(torch.float32, tensor),
                TypeError,
                "allclose(): argument 'input' (position 1) must be Tensor, not torch.dtype",
            ),
            (
                lambda: torch.allclose(tensor, torch.device("cpu")),
                TypeError,
                "allclose(): argument 'other' (position 2) must be Tensor, not torch.device",
            ),
            (
                lambda: torch.allclose(tensor, tensor, tensor, tensor, tensor, tensor),
                TypeError,
                "allclose() takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=object()),
                TypeError,
                "allclose(): argument 'rtol' must be float, not object",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 0.0, object()),
                TypeError,
                "allclose(): argument 'atol' (position 4) must be float, not object",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=1),
                TypeError,
                "allclose(): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 0.0, 0.0, 1),
                TypeError,
                "allclose(): argument 'equal_nan' (position 5) must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, out=None),
                TypeError,
                "allclose() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 0.0, rtol=0.0),
                TypeError,
                "allclose() got multiple values for argument 'rtol'",
            ),
            (
                lambda: torch.allclose(input=tensor, x=tensor, other=tensor),
                TypeError,
                "allclose() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=float("nan")),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got nan",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=-1.0e-5),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got -1e-05",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
