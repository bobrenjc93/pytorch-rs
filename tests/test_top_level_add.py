import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


ADD_DOC = (
    "\nadd(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.\n\n"
    ".. math::\n"
    "    \\text{{out}}_i = \\text{{input}}_i + \\text{{alpha}} \\times "
    "\\text{{other}}_i\n\n\n"
    "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and "
    "complex inputs.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    other (Tensor or Number): the tensor or number to add to :attr:`input`.\n\n"
    "Keyword args:\n"
    "    alpha (Number): the multiplier for :attr:`other`.\n"
    "    out (Tensor, optional): the output tensor.\n"
)


class TopLevelAddTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
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

    def test_tensor_tensor_values_broadcast_empties_and_special_bits(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left + right
        calls = (
            ("positional tensors", lambda: torch.add(left, right)),
            ("canonical keywords", lambda: torch.add(input=left, other=right)),
            ("x aliases", lambda: torch.add(x=left, x2=right)),
            ("x1 aliases", lambda: torch.add(x1=left, x2=right)),
            ("default alpha", lambda: torch.add(left, right, alpha=1)),
            ("default float alpha", lambda: torch.add(left, right, alpha=1.0)),
            ("default numpy alpha", lambda: torch.add(left, right, alpha=np.int64(1))),
            (
                "default numpy float alpha",
                lambda: torch.add(left, right, alpha=np.float32(1.0)),
            ),
            ("explicit out none", lambda: torch.add(left, right, out=None)),
        )
        for case, call in calls:
            self.assert_tensor_matches(call(), expected, case=case)

        offset = left[1]
        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(offset.is_contiguous())
        self.assert_tensor_matches(
            torch.add(offset, torch.ones((3, 1))),
            offset + torch.ones((3, 1)),
            case="offset noncontiguous input",
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.add(empty, broadcast),
            empty + broadcast,
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        zeros = torch.zeros((5,))
        result = torch.add(special, zeros)
        self.assert_tensor_matches(result, special + zeros, case="signed zero nan infinity")
        self.assertFalse(result.is_set_to(special))
        self.assertFalse(result.is_set_to(zeros))
        if result.numel():
            self.assertNotEqual(result.data_ptr(), special.data_ptr())
            self.assertNotEqual(result.data_ptr(), zeros.data_ptr())

    def test_autograd_no_grad_and_shared_operands_reuse_addition_path(self):
        function_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        function_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        function_output = torch.add(
            function_left.transpose(0, 1), function_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) + operator_right.transpose(
            0, 1
        )
        self.assert_tensor_matches(function_output, operator_output, case="tracked views")
        function_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            function_left.grad, operator_left.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            function_right.grad, operator_right.grad, case="right gradient"
        )

        shared_function = torch.tensor([2.0, -3.0], requires_grad=True)
        shared_operator = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(shared_function, shared_function).sum().backward()
        (shared_operator + shared_operator).sum().backward()
        self.assert_tensor_matches(
            shared_function.grad, shared_operator.grad, case="shared operand gradient"
        )

        empty_function = torch.zeros((2, 0, 3), requires_grad=True)
        empty_operator = torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(empty_function, torch.ones((1, 1, 3))).sum().backward()
        (empty_operator + torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            empty_function.grad, empty_operator.grad, case="empty gradient"
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            output = torch.add(left.transpose(0, 1), right.transpose(0, 1))
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.add(left, right.transpose(0, 1)).requires_grad)

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        left = torch.tensor([2.0])
        right = torch.tensor([3.0])
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
            (lambda: torch.add(left, right), (left, right), None),
            (lambda: torch.add(left, 4.0), (left, 4.0), None),
            (lambda: torch.add(4.0, left), (4.0, left), None),
            (
                lambda: torch.add(input=left, other=right, alpha=2),
                (),
                ("input", "other", "alpha"),
            ),
            (
                lambda: torch.add(left, right, out=destination),
                (left, right),
                ("out",),
            ),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(keywords=expected_keywords):
                self.assertIs(function, torch.add)
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
                actual = torch.add(input=left, other=right, alpha=np.float32(1.0))
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left + right, case="forwarded modes")

        for call in (
            lambda: torch.add([], right),
            lambda: torch.add(left, []),
            lambda: torch.add(left, right, alpha=[]),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

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

        self.assertIs(torch.add(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.add(input=left, other=RightOverride(), alpha=2), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other", "alpha"))

        events.clear()
        self.assertIs(torch.add(left, right, alpha=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("alpha",))

        events.clear()
        self.assertIs(torch.add(left, right, out=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("out",))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.add(DecliningOverride(), left)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.add'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_import_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        cases = (
            (
                lambda: torch.add(),
                "add() received an invalid combination of arguments - got "
                "(), but expected (Tensor input, Tensor other, *, Number alpha "
                "= 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor),
                "add() received an invalid combination of arguments - got "
                "(Tensor), but expected (Tensor input, Tensor other, *, Number "
                "alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor, tensor, tensor),
                "add() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.add([], tensor),
                "add() received an invalid combination of arguments - got "
                "(list, Tensor), but expected (Tensor input, Tensor other, *, "
                "Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor, []),
                "add() received an invalid combination of arguments - got "
                "(Tensor, list), but expected (Tensor input, Tensor other, *, "
                "Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(input=None, other=tensor),
                "add() received an invalid combination of arguments - got "
                "(other=Tensor, input=NoneType, ), but expected (Tensor input, "
                "Tensor other, *, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor, tensor, input=tensor),
                "add() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.add(tensor, tensor, x2=tensor),
                "add() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.add(tensor, tensor, extra=True),
                "add() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.add(tensor, tensor, alpha=[]),
                "add(): argument 'alpha' must be Number, not list",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        for call in (
            lambda: torch.add(tensor, 1),
            lambda: torch.add(1, tensor),
            lambda: torch.add(1, 2),
            lambda: torch.add(tensor, np.float32(1.0)),
        ):
            with self.subTest(boundary="scalar operand"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^add\(\): only exact native CPU float32 Tensor/Tensor "
                    r"operands are supported$",
                ):
                    call()

        for alpha in (2, np.float32(2.0)):
            with self.subTest(boundary="nondefault alpha", alpha=alpha):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^add\(\): alpha values other than 1 are not supported$",
                ):
                    torch.add(tensor, tensor, alpha=alpha)
        for alpha in (True, False, np.bool_(True), np.bool_(False)):
            with self.subTest(boundary="boolean alpha", alpha=alpha):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^add\(\): boolean alpha is not supported$",
                ):
                    torch.add(tensor, tensor, alpha=alpha)

        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaises(TypeError):
            torch.add(tensor, tensor, dtype=torch.float32)
        with self.assertRaises(TypeError):
            torch.add(tensor, tensor, device=torch.device("cpu"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.add(torch.tensor([1.0], device="cuda"), tensor)
        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "float64"))
        self.assertFalse(hasattr(torch, "add_"))
        self.assertFalse(hasattr(torch.Tensor, "add_"))

        function = torch.add
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "add")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.add")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, ADD_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method add of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.add, function)
        for mutation in (
            lambda: setattr(owner, "add", None),
            lambda: delattr(owner, "add"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.add, function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        self.assertEqual(torch.__all__.count("add"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["add"], function)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.add, function)


if __name__ == "__main__":
    unittest.main()
