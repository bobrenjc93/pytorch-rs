import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nallclose(input: Tensor, other: Tensor, rtol: float = 1e-05, "
    "atol: float = 1e-08, equal_nan: bool = False) -> bool\n\n"
    "This function checks if :attr:`input` and :attr:`other` satisfy the condition:\n\n"
    ".. math::\n"
    "    \\lvert \\text{input}_i - \\text{other}_i \\rvert \\leq "
    "\\texttt{atol} + \\texttt{rtol} \\times \\lvert \\text{other}_i \\rvert\n\n"
    "elementwise, for all elements of :attr:`input` and :attr:`other`. "
    "The behaviour of this function is analogous to\n"
    "`numpy.allclose <https://numpy.org/doc/stable/reference/generated/numpy.allclose.html>`_\n\n"
    "Args:\n"
    "    input (Tensor): first tensor to compare\n"
    "    other (Tensor): second tensor to compare\n"
    "    atol (float, optional): absolute tolerance. Default: 1e-08\n"
    "    rtol (float, optional): relative tolerance. Default: 1e-05\n"
    "    equal_nan (bool, optional): if ``True``, then two ``NaN`` s will be considered equal. Default: ``False``\n\n"
    "Example::\n\n"
    "    >>> torch.allclose(torch.tensor([10000., 1e-07]), torch.tensor([10000.1, 1e-08]))\n"
    "    False\n"
    "    >>> torch.allclose(torch.tensor([10000., 1e-08]), torch.tensor([10000.1, 1e-09]))\n"
    "    True\n"
    "    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]))\n"
    "    False\n"
    "    >>> torch.allclose(torch.tensor([1.0, float('nan')]), torch.tensor([1.0, float('nan')]), equal_nan=True)\n"
    "    True\n"
)


class AllCloseTests(unittest.TestCase):
    def assert_python_bool(self, value, expected):
        self.assertIs(type(value), bool)
        self.assertIs(value, expected)

    def tensor_bits(self, tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32).tolist()

    def tensor_snapshot(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
            self.tensor_bits(tensor),
        )

    def test_scalar_empty_same_shape_and_broadcast_results(self):
        self.assert_python_bool(torch.allclose(torch.tensor(1.0), torch.tensor(1.0)), True)
        self.assert_python_bool(torch.allclose(torch.tensor(1.0), torch.tensor(1.1)), False)
        self.assert_python_bool(
            torch.allclose(torch.zeros((2, 0, 3)), torch.ones((1, 0, 1))),
            True,
        )

        left = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.00001, 2.99998]])
        right = torch.tensor([[1.0, 2.0, 3.0]])
        self.assert_python_bool(torch.allclose(left, right, rtol=1e-5, atol=1e-8), True)
        self.assert_python_bool(torch.allclose(left, right, rtol=1e-6, atol=1e-8), False)

    def test_offset_noncontiguous_and_no_mutation(self):
        base = torch.tensor(
            [
                [[99.0, 98.0, 97.0], [96.0, 95.0, 94.0]],
                [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            ]
        )
        left = base[1].transpose(0, 1)
        right = torch.tensor([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
        left_before = self.tensor_snapshot(left)
        right_before = self.tensor_snapshot(right)

        self.assertFalse(left.is_contiguous())
        self.assertNotEqual(left.storage_offset(), 0)
        self.assert_python_bool(torch.allclose(left, right), True)
        self.assertEqual(self.tensor_snapshot(left), left_before)
        self.assertEqual(self.tensor_snapshot(right), right_before)

    def test_signed_zero_nan_infinity_and_equal_nan(self):
        self.assert_python_bool(
            torch.allclose(
                torch.tensor([0.0, -0.0]),
                torch.tensor([-0.0, 0.0]),
                rtol=0.0,
                atol=0.0,
            ),
            True,
        )
        self.assert_python_bool(
            torch.allclose(
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
            ),
            True,
        )
        self.assert_python_bool(
            torch.allclose(torch.tensor([float("inf")]), torch.tensor([-float("inf")])),
            False,
        )
        self.assert_python_bool(
            torch.allclose(
                torch.tensor([1.0]), torch.tensor([float("inf")]), rtol=float("inf")
            ),
            False,
        )
        extreme = torch.tensor([3.4028235e38])
        opposite_extreme = torch.tensor([-1.0e38])
        self.assert_python_bool(
            torch.allclose(extreme, opposite_extreme, rtol=float("inf"), atol=0.0),
            False,
        )
        self.assert_python_bool(
            torch.allclose(extreme, opposite_extreme, rtol=1.0e38, atol=0.0),
            False,
        )
        self.assert_python_bool(
            torch.allclose(extreme, opposite_extreme, rtol=0.0, atol=float("inf")),
            False,
        )

        nan_left = torch.tensor([1.0, float("nan")])
        nan_right = torch.tensor([1.0, float("nan")])
        self.assert_python_bool(torch.allclose(nan_left, nan_right), False)
        self.assert_python_bool(
            torch.allclose(nan_left, nan_right, equal_nan=True),
            True,
        )

    def test_tolerance_argument_forms(self):
        left = torch.tensor([1.0, 2.0])
        close = torch.tensor([1.00002, 2.00004])
        far = torch.tensor([1.0002, 2.0004])

        self.assert_python_bool(torch.allclose(left, close), False)
        self.assert_python_bool(torch.allclose(left, close, 3e-5, 0.0), True)
        self.assert_python_bool(torch.allclose(left, far, rtol=3e-5, atol=0.0), False)
        self.assert_python_bool(
            torch.allclose(left, close, rtol=torch.tensor(3e-5), atol=torch.tensor(0.0)),
            True,
        )
        self.assert_python_bool(torch.allclose(left, far, rtol=True, atol=False), True)

        for keyword in ("rtol", "atol"):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{keyword} must be greater than or equal to zero, but got -1$",
                ):
                    torch.allclose(left, close, **{keyword: -1.0})
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{keyword} must be greater than or equal to zero, but got nan$",
                ):
                    torch.allclose(left, close, **{keyword: float("nan")})

    def test_callable_metadata_documentation_pickling_exports_and_reload(self):
        function = torch.allclose
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "allclose")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.allclose")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method allclose of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.allclose, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("allclose"), 1)
        self.assertFalse(hasattr(torch, "isclose"))
        self.assertNotIn("isclose", torch.__all__)
        self.assertFalse(hasattr(torch, "bool"))
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["allclose"], function)

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.allclose, function)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([1.0])
        right = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.allclose(input=left, other=right, rtol=1e-5), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.allclose)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other", "rtol"))
        self.assertIs(kwargs["input"], left)
        self.assertIs(kwargs["other"], right)
        self.assertEqual(kwargs["rtol"], 1e-5)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.allclose(input=left, other=right, rtol=1e-5)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_python_bool(result, True)

    def test_binding_and_type_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.allclose(),
                'allclose() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.allclose(tensor),
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.allclose(None, tensor),
                "allclose(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.allclose(input=1, other=tensor),
                "allclose(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.allclose(tensor, 1.0),
                "allclose(): argument 'other' (position 2) must be Tensor, not float",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 1, 2, False, 0),
                "allclose() takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: torch.allclose(tensor, tensor, out=None),
                "allclose() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, extra=True),
                "allclose() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.allclose(a=tensor, other=tensor, extra=True),
                "allclose() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.allclose(input=tensor, x2=tensor, other=tensor),
                "allclose() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.allclose(a=tensor, input=tensor, other=tensor),
                "allclose() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, other=tensor),
                "allclose() got multiple values for argument 'other'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, 1e-5, rtol=1e-5),
                "allclose() got multiple values for argument 'rtol'",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=None),
                "allclose(): argument 'rtol' must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(tensor, tensor, atol="1"),
                "allclose(): argument 'atol' must be float, not str",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=1),
                "allclose(): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(tensor, tensor, equal_nan=np.bool_(True)),
                "allclose(): argument 'equal_nan' must be bool, not numpy.bool",
            ),
            (
                lambda: torch.allclose(tensor, tensor, rtol=torch.tensor([1e-5])),
                "allclose(): argument 'rtol' must be float, not Tensor",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            RuntimeError,
            r"^The size of tensor a \(3\) must match the size of tensor b \(4\) at non-singleton dimension 1$",
        ):
            torch.allclose(torch.zeros((2, 3)), torch.zeros((4,)))

    def test_supported_keyword_aliases_match_pytorch_generation(self):
        left = torch.tensor([1.0])
        right = torch.tensor([1.0])
        for call in (
            lambda: torch.allclose(a=left, other=right),
            lambda: torch.allclose(x=left, other=right),
            lambda: torch.allclose(x1=left, x2=right),
            lambda: torch.allclose(input=left, x2=right),
        ):
            with self.subTest(call=call):
                self.assert_python_bool(call(), True)


if __name__ == "__main__":
    unittest.main()
