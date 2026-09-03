import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


MM_DOC = """
mm(input, mat2, *, out=None) -> Tensor

Performs a matrix multiplication of the matrices :attr:`input` and :attr:`mat2`.

If :attr:`input` is a :math:`(n \\times m)` tensor, :attr:`mat2` is a
:math:`(m \\times p)` tensor, :attr:`out` will be a :math:`(n \\times p)` tensor.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.
          For broadcasting matrix products, see :func:`torch.matmul`.

The current native implementation supports exact CPU ``float32`` rank-2 tensor
operands by delegating to the existing rank-2 matrix multiplication kernel.
Concrete ``out`` tensors, :func:`torch.bmm`, :func:`torch.addmm`, rank-1 or
batched ``matmul`` behavior, dtype/device expansion, tensor subclasses, and
unsupported autograd cases remain unsupported.
"""

MM_OVERLOADS = (
    "but expected one of:\n"
    " * (Tensor input, Tensor mat2, *, Tensor out = None)\n"
    " * (Tensor input, Tensor mat2, torch.dtype out_dtype, *, Tensor out = None)\n"
)


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist()
    )[1]

    noncontiguous_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    noncontiguous_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)

    return (
        (
            "square",
            module.tensor([[1.0, 2.0], [3.0, 4.0]]),
            module.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),
        (
            "rectangular",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor(
                [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]]
            ),
        ),
        ("empty rows", module.zeros((0, 3)), module.ones((3, 4))),
        ("empty columns", module.ones((2, 3)), module.zeros((3, 0))),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        ("offset", offset_left, offset_right),
        ("noncontiguous", noncontiguous_left, noncontiguous_right),
        ("signed zero", module.tensor([[-0.0]]), module.tensor([[1.0]])),
        (
            "nan and infinities",
            module.tensor(
                [[float("inf"), 1.0], [float("-inf"), -1.0], [float("nan"), 2.0]]
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]]),
        ),
    )


class TopLevelMmTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
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
            expected = torch.matmul(left, right)
            calls = (
                ("positional", lambda: torch.mm(left, right)),
                ("canonical keywords", lambda: torch.mm(input=left, mat2=right)),
                ("left x alias", lambda: torch.mm(x=left, mat2=right)),
                ("left a alias", lambda: torch.mm(a=left, mat2=right)),
                ("left x1 alias", lambda: torch.mm(x1=left, mat2=right)),
                ("mixed positional keyword", lambda: torch.mm(left, mat2=right)),
                ("out none", lambda: torch.mm(left, right, out=None)),
            )
            for style, call in calls:
                self.assert_tensor_matches(call(), expected, case=(case, style))

    def test_preserves_current_autograd_boundary(self):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        result = torch.mm(left, right)
        expected = torch.matmul(left, right)
        self.assert_tensor_matches(result, expected, case="requires-grad operands")
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^element 0 of tensors does not require grad and does not have a grad_fn$",
        ):
            result.sum().backward()
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

    def test_modes_and_overrides_receive_distinct_callable_and_can_forward(self):
        left = torch.tensor([[1.0]])
        right = torch.tensor([[2.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            (lambda: torch.mm(left, right), (left, right), None),
            (lambda: torch.mm(input=left, mat2=right), (), ("input", "mat2")),
            (lambda: torch.mm(x=left, mat2=right, out=None), (), ("x", "mat2", "out")),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.mm)
            self.assertIsNot(function, torch.matmul)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            if expected_keywords is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), expected_keywords)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.mm(input=left, mat2=right, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(forwarded, torch.matmul(left, right), case="forwarded modes")

        events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", func, types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", func, types, args, kwargs))
                return marker

        self.assertIs(torch.mm(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.mm)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

    def test_unsupported_surface_is_narrow_and_documented(self):
        left = torch.ones((1, 1))
        right = torch.ones((1, 1))
        destination = torch.tensor([[17.0]])

        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(left, right, out=destination)
        self.assertEqual(destination.tolist(), [[17.0]])

        rank_cases = (
            (torch.tensor(1.0), torch.ones((1, 1))),
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
            (torch.ones((2, 2)), torch.ones((1, 2, 2))),
        )
        for rank_left, rank_right in rank_cases:
            with self.subTest(left=rank_left.shape, right=rank_right.shape):
                with self.assertRaises(RuntimeError) as raised:
                    torch.mm(rank_left, rank_right)
                self.assertIn("requires two rank-2 tensors", str(raised.exception))

        for name in ("bmm", "addmm"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))

        for phrase in (
            "Concrete ``out`` tensors",
            ":func:`torch.bmm`",
            ":func:`torch.addmm`",
            "rank-1 or\nbatched ``matmul`` behavior",
            "dtype/device expansion",
            "tensor subclasses",
            "unsupported autograd cases",
        ):
            self.assertIn(phrase, torch.mm.__doc__)

    def test_binding_metadata_import_reload_copy_and_pickle(self):
        tensor = torch.tensor([[1.0]])
        cases = (
            (
                lambda: torch.mm(),
                "mm() received an invalid combination of arguments - got (), "
                f"{MM_OVERLOADS}",
            ),
            (
                lambda: torch.mm(tensor),
                "mm() received an invalid combination of arguments - got (Tensor), "
                f"{MM_OVERLOADS}",
            ),
            (
                lambda: torch.mm(tensor, tensor, tensor),
                "mm() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, Tensor), {MM_OVERLOADS}",
            ),
            (
                lambda: torch.mm([], tensor),
                "mm(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.mm(tensor, []),
                "mm(): argument 'mat2' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.mm(input=None, mat2=tensor),
                "mm(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.mm(input=tensor, other=tensor),
                'mm() missing 1 required positional arguments: "mat2"',
            ),
            (
                lambda: torch.mm(tensor, tensor, input=tensor),
                "mm() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, input=Tensor), {MM_OVERLOADS}",
            ),
            (
                lambda: torch.mm(tensor, tensor, mat2=tensor),
                "mm() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, mat2=Tensor), {MM_OVERLOADS}",
            ),
            (
                lambda: torch.mm(tensor, tensor, extra=True),
                "mm() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, extra=bool), {MM_OVERLOADS}",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        function = torch.mm
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertIsNot(function, torch.matmul)
        self.assertEqual(function.__name__, "mm")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mm")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, MM_DOC)
        self.assertRegex(
            repr(function),
            r"^<built-in method mm of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mm, function)
        self.assertIsNot(owner.mm, owner.matmul)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        imported = {}
        wildcard_namespace = {}
        exec("from torch_rs import mm", imported)
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(importlib.import_module("torch_rs").mm, function)
        self.assertIs(imported["mm"], function)
        self.assertIs(wildcard_namespace["mm"], function)
        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertEqual(torch._C.__all__.count("mm"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))

        package = torch
        native = torch._C
        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch.mm, function)
        self.assertEqual(torch.__all__.count("mm"), 1)
        self.assertIs(importlib.reload(native), native)
        self.assertIs(torch.mm, function)
        self.assertEqual(torch._C.__all__.count("mm"), 1)


if __name__ == "__main__":
    unittest.main()
