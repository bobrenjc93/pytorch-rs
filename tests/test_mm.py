import copy
import importlib
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist(),
        dtype=module.float32,
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist(),
        dtype=module.float32,
    )[1]
    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]], dtype=module.float32
    ).transpose(0, 1)

    return (
        (
            "square",
            module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32),
            module.tensor([[-5.0, 6.0], [7.0, -8.0]], dtype=module.float32),
        ),
        (
            "rectangular",
            module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
            ),
            module.tensor(
                [
                    [7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0],
                ],
                dtype=module.float32,
            ),
        ),
        (
            "empty rows",
            module.zeros((0, 3), dtype=module.float32),
            module.ones((3, 4), dtype=module.float32),
        ),
        (
            "empty columns",
            module.ones((2, 3), dtype=module.float32),
            module.zeros((3, 0), dtype=module.float32),
        ),
        (
            "empty inner",
            module.ones((2, 0), dtype=module.float32),
            module.zeros((0, 3), dtype=module.float32),
        ),
        ("offset", offset_left, offset_right),
        ("noncontiguous", strided_left, strided_right),
        (
            "signed zero",
            module.tensor([[-0.0], [0.0]], dtype=module.float32),
            module.tensor([[1.0, -1.0]], dtype=module.float32),
        ),
        (
            "nan inf",
            module.tensor(
                [
                    [float("inf"), 1.0],
                    [float("-inf"), -1.0],
                    [float("nan"), 2.0],
                ],
                dtype=module.float32,
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]], dtype=module.float32),
        ),
    )


class TorchMmTests(unittest.TestCase):
    def assert_matches_matmul(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_supported_rank_two_calls_reuse_existing_matmul(self):
        for case, left, right in mm_layout_cases(torch):
            expected = left.matmul(right)
            calls = (
                ("positional", lambda: torch.mm(left, right)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=right)),
                ("input alias x", lambda: torch.mm(x=left, mat2=right)),
                ("input alias a", lambda: torch.mm(a=left, mat2=right)),
                ("input alias x1", lambda: torch.mm(x1=left, mat2=right)),
                ("out none", lambda: torch.mm(left, right, out=None)),
            )
            for style, call in calls:
                self.assert_matches_matmul(call(), expected, case=(case, style))

    def test_rank_out_and_api_boundaries_remain_narrow(self):
        left = torch.ones((2, 2))
        right = torch.ones((2, 2))
        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(left, right, out=torch.zeros((2, 2)))

        rank_cases = (
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((2, 2)), torch.ones((2,))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
        )
        for left_arg, right_arg in rank_cases:
            with self.subTest(left=left_arg.shape, right=right_arg.shape):
                with self.assertRaisesRegex(RuntimeError, "requires two rank-2 tensors"):
                    torch.mm(left_arg, right_arg)

        grad_left = torch.ones((1, 1), requires_grad=True)
        grad_right = torch.ones((1, 1), requires_grad=True)
        result = torch.mm(grad_left, grad_right)
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)

        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "bmm"))
        self.assertFalse(hasattr(torch, "addmm"))
        self.assertFalse(hasattr(torch.Tensor, "mm"))

    def test_callable_metadata_copy_pickle_reload_and_exports(self):
        function = torch.mm
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertTrue(
            function.__doc__.startswith("\nmm(input, mat2, *, out=None) -> Tensor\n\n")
        )
        self.assertIn("For broadcasting matrix products, see :func:`torch.matmul`", function.__doc__)
        self.assertRegex(
            repr(function), r"^<built-in method mm of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mm, function)

        for mutation in (
            lambda: setattr(owner, "mm", None),
            lambda: delattr(owner, "mm"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.mm, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        direct_namespace = {}
        exec("from torch_rs import mm", direct_namespace)
        self.assertIs(direct_namespace["mm"], function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mm"], function)
        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.mm, function)


if __name__ == "__main__":
    unittest.main()
