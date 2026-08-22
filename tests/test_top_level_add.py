import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nadd(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.\n\n"
    ".. math::\n"
    "    \\text{{out}}_i = \\text{{input}}_i + \\text{{alpha}} \\times \\text{{other}}_i\n\n\n"
    "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    other (Tensor or Number): the tensor or number to add to :attr:`input`.\n\n"
    "Keyword arguments:\n"
    "    alpha (Number): the multiplier for :attr:`other`.\n"
    "    out (Tensor, optional): the output tensor.\n\n"
    "Examples::\n\n"
    "    >>> a = torch.randn(4)\n"
    "    >>> a\n"
    "    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])\n"
    "    >>> torch.add(a, 20)\n"
    "    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])\n\n"
    "    >>> b = torch.randn(4)\n"
    "    >>> b\n"
    "    tensor([-0.9732, -0.3497,  0.6245,  0.4022])\n"
    "    >>> c = torch.randn(4, 1)\n"
    "    >>> c\n"
    "    tensor([[ 0.3743],\n"
    "            [-1.7724],\n"
    "            [-0.5811],\n"
    "            [-0.8017]])\n"
    "    >>> torch.add(b, c, alpha=10)\n"
    "    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],\n"
    "            [-18.6971, -18.0736, -17.0994, -17.3216],\n"
    "            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],\n"
    "            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])\n"
)


class TopLevelAddTests(unittest.TestCase):
    def assert_tensor(self, tensor, expected_bits, *, shape, stride, requires_grad):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.requires_grad, requires_grad)
        self.assertEqual(tensor.is_leaf, not requires_grad)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        np.testing.assert_array_equal(
            np.asarray(tensor).reshape(-1).view(np.uint32), expected_bits
        )

    def test_scalar_orders_argument_forms_layout_and_ieee_bits(self):
        source = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)[1]
        expected = np.asarray([[6.0], [7.0], [8.0]], dtype=np.float32).view(
            np.uint32
        ).reshape(-1)
        unit_tensor = torch.tensor(1.0)
        calls = (
            torch.add(source, 2),
            torch.add(2, source),
            torch.add(input=source, other=2, alpha=1, out=None),
            torch.add(x=source, x2=np.int64(2), alpha=np.float32(1.0)),
            torch.add(x1=np.float32(2.0), x2=source, alpha=unit_tensor),
        )
        for result in calls:
            self.assert_tensor(
                result,
                expected,
                shape=(3, 1),
                stride=(1, 3),
                requires_grad=False,
            )
            self.assertNotEqual(result.data_ptr(), source.data_ptr())

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        result = torch.add(-2.0, empty)
        self.assertEqual(result.shape, (3, 0, 2))
        self.assertEqual(result.stride(), (1, 3, 0))

        tensor_bits = np.asarray(
            (0x7FC1_2345, 0xFFC5_4321, 0x0000_0000, 0x8000_0000),
            dtype=np.uint32,
        )
        scalar_bits = np.asarray((0x7FC6_789A,), dtype=np.uint32)
        tensor = torch.tensor(memoryview(tensor_bits.view(np.float32)))
        scalar = scalar_bits.view(np.float32)[0]
        np.testing.assert_array_equal(
            np.asarray(torch.add(tensor, scalar)).view(np.uint32),
            [0x7FC6_789A, 0x7FC6_789A, 0x7FC6_789A, 0x7FC6_789A],
        )
        np.testing.assert_array_equal(
            np.asarray(torch.add(scalar, tensor)).view(np.uint32),
            [0x7FC1_2345, 0xFFC5_4321, 0x7FC6_789A, 0x7FC6_789A],
        )

    def test_autograd_and_no_grad_reuse_scalar_addition(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        output = torch.add(5.0, leaf.transpose(0, 1))
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        output.sum().backward()
        np.testing.assert_array_equal(np.asarray(leaf.grad), np.ones((2, 2)))

        repeated = torch.tensor([2.0, -3.0], requires_grad=True)
        result = torch.add(repeated, 1.0)
        result.sum().backward()
        result.sum().backward()
        np.testing.assert_array_equal(np.asarray(repeated.grad), [2.0, 2.0])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(empty, 7).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.numel(), 0)

        tracked = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.add(tracked, 2)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(torch.add(tracked, 2).requires_grad)

    def test_modes_observe_supported_and_native_unsupported_calls(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            (lambda: torch.add(tensor, 2), (tensor, 2), None),
            (lambda: torch.add(2, tensor), (2, tensor), None),
            (
                lambda: torch.add(input=tensor, other=2, alpha=1, out=None),
                (),
                ("input", "other", "alpha", "out"),
            ),
            (lambda: torch.add(tensor, other), (tensor, other), None),
            (lambda: torch.add(2, 3), (2, 3), None),
            (lambda: torch.add(tensor, 2, alpha=2), (tensor, 2), ("alpha",)),
            (lambda: torch.add(tensor, 2, out=destination), (tensor, 2), ("out",)),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(None if kwargs is None else tuple(kwargs), expected_keywords)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.add(input=2, other=tensor, alpha=1, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(result.tolist(), [3.0])

        invalid = RecordingMode()
        with invalid:
            with self.assertRaises(TypeError):
                torch.add(tensor, [])
        self.assertEqual(invalid.calls, [])

    def test_metadata_errors_and_native_scope(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        cases = (
            (
                lambda: torch.add(),
                TypeError,
                "add() received an invalid combination of arguments - got (), but expected (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor),
                TypeError,
                "add() received an invalid combination of arguments - got (Tensor), but expected (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor, 2, 1),
                TypeError,
                "add() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.add([], tensor),
                TypeError,
                "add(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.add(tensor, []),
                TypeError,
                "add(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.add(tensor, 2, alpha=None),
                TypeError,
                "add(): argument 'alpha' must be Number, not NoneType",
            ),
            (
                lambda: torch.add(tensor, 2, alpha=True),
                RuntimeError,
                "Boolean alpha only supported for Boolean results.",
            ),
            (
                lambda: torch.add(tensor, np.uint64(2**63)),
                TypeError,
                "an integer is required",
            ),
            (
                lambda: torch.add(tensor, 2**64),
                OverflowError,
                "int too big to convert",
            ),
            (
                lambda: torch.add(-(2**63) - 1, tensor),
                OverflowError,
                "can't convert negative int to unsigned",
            ),
        )
        for call, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(TypeError, "tensor-tensor addition"):
            torch.add(tensor, other)
        with self.assertRaisesRegex(TypeError, "scalar-scalar addition"):
            torch.add(1, 2)
        with self.assertRaisesRegex(
            RuntimeError, r"^torch\.add only supports alpha=1$"
        ):
            torch.add(tensor, 2, alpha=0)

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, 2, out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(TypeError, "tensor-tensor addition"):
            torch.add(extreme, extreme)
        with self.assertRaisesRegex(RuntimeError, "only supports alpha=1"):
            torch.add(extreme, 1, alpha=2)
        with self.assertRaisesRegex(RuntimeError, "out.*not supported"):
            torch.add(extreme, 1, out=destination)

        function = torch.add
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "add")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.add")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        owner = function.__reduce__()[1][0]
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.add, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)
        self.assertEqual(torch.__all__.count("add"), 1)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["add"], function)


if __name__ == "__main__":
    unittest.main()
