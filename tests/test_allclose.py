import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC_FIRST_LINE = (
    "allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, "
    "atol: float = 1e-08, equal_nan: bool = False) -> bool"
)
METHOD_DOC_FIRST_LINE = (
    "allclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor"
)


class AllCloseTests(unittest.TestCase):
    def assert_allclose_result(self, left, right, expected, **kwargs):
        function_result = torch.allclose(left, right, **kwargs)
        method_result = left.allclose(right, **kwargs)
        self.assertIs(type(function_result), bool)
        self.assertIs(type(method_result), bool)
        self.assertIs(function_result, expected)
        self.assertIs(method_result, expected)

    def tensor_snapshot(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
            np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32).tolist(),
        )

    def test_scalar_empty_same_shape_broadcast_and_strided_inputs(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        offset = dense[1]
        noncontiguous = dense.transpose(0, 2)[1]

        cases = (
            ("scalar", torch.tensor(1.0), torch.tensor(1.0), {}, True),
            (
                "same shape",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                {},
                True,
            ),
            (
                "same shape mismatch",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.1]]),
                {},
                False,
            ),
            (
                "broadcast true",
                torch.tensor([[1.0], [1.0]]),
                torch.tensor([1.0, 1.0]),
                {},
                True,
            ),
            (
                "broadcast false",
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([1.0, 2.0]),
                {},
                False,
            ),
            (
                "empty broadcast",
                torch.zeros((2, 0, 3)),
                torch.ones((1, 0, 1)),
                {},
                True,
            ),
            (
                "empty rank broadcast",
                torch.zeros((0,)),
                torch.zeros((1, 0)),
                {},
                True,
            ),
            (
                "offset",
                offset,
                torch.tensor(
                    np.arange(12, 24, dtype=np.float32).reshape(3, 4).tolist()
                ),
                {},
                True,
            ),
            (
                "noncontiguous",
                noncontiguous,
                torch.tensor([[1.0, 13.0], [5.0, 17.0], [9.0, 21.0]]),
                {},
                True,
            ),
        )
        for case, left, right, kwargs, expected in cases:
            with self.subTest(case=case):
                self.assert_allclose_result(left, right, expected, **kwargs)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^The size of tensor a \(2\) must match the size of tensor b \(3\) "
            r"at non-singleton dimension 0$",
        ):
            torch.allclose(torch.zeros((2, 0, 3)), torch.zeros((3, 0, 3)))

    def test_signed_zero_nan_infinity_and_tolerance_edges(self):
        cases = (
            (
                "signed zero",
                torch.tensor([0.0, -0.0]),
                torch.tensor([-0.0, 0.0]),
                {},
                True,
            ),
            (
                "nan false",
                torch.tensor([float("nan")]),
                torch.tensor([float("nan")]),
                {},
                False,
            ),
            (
                "nan true",
                torch.tensor([float("nan")]),
                torch.tensor([float("nan")]),
                {"equal_nan": True},
                True,
            ),
            (
                "nan mixed",
                torch.tensor([float("nan")]),
                torch.tensor([1.0]),
                {"equal_nan": True},
                False,
            ),
            (
                "infinity same",
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                {},
                True,
            ),
            (
                "infinity different",
                torch.tensor([float("inf")]),
                torch.tensor([-float("inf")]),
                {},
                False,
            ),
            (
                "infinity finite infinite tolerance",
                torch.tensor([float("inf")]),
                torch.tensor([3.4e38]),
                {"rtol": float("inf")},
                False,
            ),
            (
                "default tolerance true",
                torch.tensor([1.0]),
                torch.tensor([1.0 + 1.0e-6]),
                {},
                True,
            ),
            (
                "default tolerance false",
                torch.tensor([1.0]),
                torch.tensor([1.0 + 1.0e-4]),
                {},
                False,
            ),
            (
                "custom tolerance",
                torch.tensor([1.0]),
                torch.tensor([1.11]),
                {"rtol": 0.2, "atol": 0},
                True,
            ),
        )
        for case, left, right, kwargs, expected in cases:
            with self.subTest(case=case):
                self.assert_allclose_result(left, right, expected, **kwargs)

    def test_allclose_does_not_mutate_inputs_or_grad_metadata(self):
        left_leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        left = left_leaf.transpose(0, 2)[1]
        right = torch.tensor([[2.0, 14.0], [6.0, 18.0], [10.0, 22.0]])
        before = (self.tensor_snapshot(left), self.tensor_snapshot(right))

        self.assertTrue(torch.allclose(left, right))
        self.assertEqual(
            (self.tensor_snapshot(left), self.tensor_snapshot(right)),
            before,
        )
        self.assertIsNone(left_leaf.grad)
        self.assertTrue(left.requires_grad)

    def test_keyword_aliases_and_positional_tolerances(self):
        tensor = torch.tensor([1.0])

        self.assertTrue(torch.allclose(input=tensor, other=tensor))
        self.assertTrue(torch.allclose(x=tensor, x2=tensor))
        self.assertTrue(torch.allclose(a=tensor, other=tensor))
        self.assertTrue(torch.allclose(input=tensor, x2=tensor))
        self.assertTrue(torch.allclose(tensor, tensor, 0, 0, False))
        self.assertTrue(tensor.allclose(other=tensor))
        self.assertTrue(tensor.allclose(x2=tensor))

    def test_import_wildcard_copy_pickle_reload_and_method_metadata(self):
        function = torch.allclose
        descriptor = inspect.getattr_static(torch.Tensor, "allclose")
        bound = torch.tensor([1.0]).allclose
        owner = function.__reduce__()[1][0]

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(function.__name__, "allclose")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.allclose")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(descriptor.__name__, "allclose")
        self.assertEqual(bound.__name__, "allclose")
        self.assertIsNone(function.__text_signature__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            next(line for line in function.__doc__.splitlines() if line),
            FUNCTION_DOC_FIRST_LINE,
        )
        self.assertEqual(
            next(line for line in descriptor.__doc__.splitlines() if line),
            METHOD_DOC_FIRST_LINE,
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)

        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.allclose, function)
        self.assertIs(torch._C.allclose, function)
        self.assertEqual(torch.__all__.count("allclose"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["allclose"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.allclose, function)
        self.assertFalse(hasattr(torch, "isclose"))
        self.assertFalse(hasattr(torch, "bool"))

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
                lambda: torch.allclose(None),
                TypeError,
                "allclose(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(input=1),
                TypeError,
                "allclose(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, tensor),
                TypeError,
                "allclose(): argument 'rtol' (position 3) must be float, not Tensor",
            ),
            (
                lambda: torch.allclose(tensor, tensor, tensor, extra=True),
                TypeError,
                "allclose(): argument 'rtol' (position 3) must be float, not Tensor",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 1.0e-5, 1.0e-8, False, None),
                TypeError,
                "allclose() takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: torch.allclose(tensor, 1),
                TypeError,
                "allclose(): argument 'other' (position 2) must be Tensor, not int",
            ),
            (
                lambda: torch.allclose(input=tensor, other=[]),
                TypeError,
                "allclose(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: torch.allclose(foo=tensor, other=tensor),
                TypeError,
                'allclose() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.allclose(input=tensor, b=tensor),
                TypeError,
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.allclose(tensor, tensor, extra=True),
                TypeError,
                "allclose() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, other=tensor),
                TypeError,
                "allclose() got multiple values for argument 'other'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, out=None),
                TypeError,
                "allclose() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, dtype=torch.float32),
                TypeError,
                "allclose() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, device="cpu"),
                TypeError,
                "allclose() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol="x"),
                TypeError,
                "allclose(): argument 'rtol' must be float, not str",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol="x"),
                TypeError,
                "allclose(): argument 'atol' must be float, not str",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=1),
                TypeError,
                "allclose(): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=-1.0e-5),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got -1e-05",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol=float("nan")),
                RuntimeError,
                "atol must be greater than or equal to zero, but got nan",
            ),
            (
                lambda: tensor.allclose(),
                TypeError,
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.allclose(tensor, tensor),
                TypeError,
                "allclose(): argument 'rtol' (position 2) must be float, not Tensor",
            ),
            (
                lambda: tensor.allclose(tensor, tensor, tensor, tensor, tensor),
                TypeError,
                "allclose() takes from 1 to 4 positional arguments but 5 were given",
            ),
            (
                lambda: tensor.allclose(other=tensor, out=None),
                TypeError,
                "allclose() got an unexpected keyword argument 'out'",
            ),
        )
        for call, exception_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception_type, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
