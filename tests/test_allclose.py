import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TorchAllcloseTests(unittest.TestCase):
    def assert_allclose_result(self, left, right, expected, **kwargs):
        left_state = (
            repr(left.tolist()),
            left.shape,
            left.stride(),
            left.storage_offset(),
            left.data_ptr(),
            left.requires_grad,
            left.is_leaf,
        )
        right_state = (
            repr(right.tolist()),
            right.shape,
            right.stride(),
            right.storage_offset(),
            right.data_ptr(),
            right.requires_grad,
            right.is_leaf,
        )

        result = torch.allclose(left, right, **kwargs)

        self.assertIs(type(result), bool)
        self.assertIs(result, expected)
        self.assertEqual(
            (
                repr(left.tolist()),
                left.shape,
                left.stride(),
                left.storage_offset(),
                left.data_ptr(),
                left.requires_grad,
                left.is_leaf,
            ),
            left_state,
        )
        self.assertEqual(
            (
                repr(right.tolist()),
                right.shape,
                right.stride(),
                right.storage_offset(),
                right.data_ptr(),
                right.requires_grad,
                right.is_leaf,
            ),
            right_state,
        )

    def test_values_layouts_and_special_float_values(self):
        contiguous = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        close = torch.tensor([[1.0, 2.00001], [3.0, 4.0]])
        far = torch.tensor([[1.0, 2.0001], [3.0, 4.0]])

        noncontiguous = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).transpose(0, 1)
        offset = torch.tensor([[99.0, 98.0, 97.0], [1.0, 2.0, 3.0]])[1]
        strided_offset = torch.tensor(
            [[99.0, 1.0], [98.0, 2.0], [97.0, 3.0]]
        ).transpose(0, 1)[1]

        cases = (
            ("scalar", torch.tensor(1.0), torch.tensor(1.0), {}, True),
            ("empty", torch.zeros((2, 0, 3)), torch.ones((2, 0, 3)), {}, True),
            (
                "noncontiguous",
                noncontiguous,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                {},
                True,
            ),
            ("offset", offset, torch.tensor([1.0, 2.0, 3.0]), {}, True),
            (
                "strided offset",
                strided_offset,
                torch.tensor([1.0, 2.0, 3.0]),
                {},
                True,
            ),
            ("default tolerance true", contiguous, close, {}, True),
            ("default tolerance false", contiguous, far, {}, False),
            ("explicit rtol", contiguous, far, {"rtol": 6.0e-5}, True),
            (
                "explicit atol positional",
                contiguous,
                far,
                {"rtol": 0.0, "atol": 2.0e-4},
                True,
            ),
            (
                "signed zero",
                torch.tensor([0.0, -0.0]),
                torch.tensor([-0.0, 0.0]),
                {},
                True,
            ),
            (
                "same infinities",
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                {},
                True,
            ),
            (
                "opposite infinities",
                torch.tensor([float("inf")]),
                torch.tensor([-float("inf")]),
                {},
                False,
            ),
            (
                "finite and infinity",
                torch.tensor([1.0]),
                torch.tensor([float("inf")]),
                {"rtol": float("inf"), "atol": 0.0},
                False,
            ),
            (
                "nan default",
                torch.tensor([float("nan")]),
                torch.tensor([float("nan")]),
                {},
                False,
            ),
            (
                "nan equal",
                torch.tensor([float("nan")]),
                torch.tensor([float("nan")]),
                {"equal_nan": True},
                True,
            ),
        )
        for case, left, right, kwargs, expected in cases:
            with self.subTest(case=case):
                self.assert_allclose_result(left, right, expected, **kwargs)

    def test_tolerance_binding(self):
        left = torch.tensor([3.0])
        right = torch.tensor([2.0])

        self.assertIs(torch.allclose(input=left, other=right, rtol=True, atol=False), True)
        self.assertIs(torch.allclose(x=left, x2=right, rtol=True, atol=0), True)
        self.assertIs(torch.allclose(left, right, 0, 1.0, False), True)
        self.assertIs(
            torch.allclose(left, right, rtol=np.float32(0.5), atol=np.float64(0.0)),
            True,
        )

        error_cases = (
            (lambda: torch.allclose(left, right, rtol=-1.0), RuntimeError, "rtol"),
            (lambda: torch.allclose(left, right, atol=-1.0), RuntimeError, "atol"),
            (lambda: torch.allclose(left, right, rtol=float("nan")), RuntimeError, "rtol"),
            (lambda: torch.allclose(left, right, atol=float("nan")), RuntimeError, "atol"),
            (
                lambda: torch.allclose(left, right, rtol=None),
                TypeError,
                "argument 'rtol' must be float",
            ),
            (
                lambda: torch.allclose(left, right, atol="0"),
                TypeError,
                "argument 'atol' must be float",
            ),
            (
                lambda: torch.allclose(left, right, equal_nan=1),
                TypeError,
                "argument 'equal_nan' must be bool",
            ),
            (
                lambda: torch.allclose(left, right, 0, 0, 1),
                TypeError,
                "argument 'equal_nan' (position 5) must be bool",
            ),
        )
        for call, error_type, message in error_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, re.escape(message)):
                    call()

    def test_shape_mismatch_and_unsupported_boundaries(self):
        tensor = torch.tensor([1.0])
        cases = (
            (lambda: torch.allclose(tensor, torch.tensor([[1.0]])), RuntimeError, ""),
            (lambda: torch.allclose(tensor, 1.0), TypeError, "must be Tensor, not float"),
            (lambda: torch.allclose(input=1, other=tensor), TypeError, "must be Tensor, not int"),
            (
                lambda: torch.allclose(tensor, tensor, out=None),
                TypeError,
                "unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, dtype=torch.float32),
                TypeError,
                "unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, device=torch.device("cpu")),
                TypeError,
                "unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, object()),
                TypeError,
                "argument 'rtol' (position 3) must be float",
            ),
            (
                lambda: torch.allclose(tensor, tensor, input=tensor),
                TypeError,
                "got multiple values for argument 'input'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, other=tensor),
                TypeError,
                "got multiple values for argument 'other'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                context = (
                    self.assertRaises(error_type)
                    if not message
                    else self.assertRaisesRegex(error_type, re.escape(message))
                )
                with context:
                    call()

        self.assertFalse(hasattr(torch, "isclose"))
        self.assertFalse(hasattr(torch.Tensor, "allclose"))
        with self.assertRaises(AttributeError):
            tensor.allclose(tensor)

    def test_does_not_create_autograd_edges(self):
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        tracked = leaf * 2.0
        expected = torch.tensor([2.0, 4.0])

        before = (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            tracked.requires_grad,
            tracked.is_leaf,
        )
        self.assertIs(torch.allclose(tracked, expected), True)
        self.assertEqual(
            (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                tracked.requires_grad,
                tracked.is_leaf,
            ),
            before,
        )

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [2.0, 2.0])

    def test_callable_import_wildcard_copy_pickle_and_reload(self):
        from torch_rs import allclose as imported

        function = torch.allclose
        self.assertIs(imported, function)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "allclose")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.allclose")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = torch._C._VariableFunctionsClass
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner.allclose, function)
        self.assertEqual(torch.__all__.count("allclose"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["allclose"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol=protocol)), function)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.allclose, function)


if __name__ == "__main__":
    unittest.main()
