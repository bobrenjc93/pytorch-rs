import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

The native implementation currently supports only exact native CPU float32
Tensor/Tensor operands with omitted or default-equivalent ``alpha`` and omitted
or ``None`` ``out``. Scalar operands, scalar-only calls, nondefault or boolean
``alpha``, concrete ``out`` tensors, dtype/device extension keywords, tensor
subclasses without ``__torch_function__`` handling, and in-place variants remain
unsupported.
"""


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
            self.assertFalse(actual.is_set_to(expected))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def supported_calls(self, left, right):
        return (
            ("positional", lambda: torch.add(left, right)),
            ("canonical keywords", lambda: torch.add(input=left, other=right)),
            ("x aliases", lambda: torch.add(x=left, x2=right)),
            ("x1 aliases", lambda: torch.add(x1=left, x2=right)),
            ("mixed keyword", lambda: torch.add(left, other=right)),
            ("out none", lambda: torch.add(left, right, out=None)),
            ("alpha int one", lambda: torch.add(left, right, alpha=1)),
            ("alpha float one", lambda: torch.add(left, right, alpha=1.0)),
            ("alpha numpy int one", lambda: torch.add(left, right, alpha=np.int64(1))),
            (
                "alpha numpy float one",
                lambda: torch.add(left, right, alpha=np.float32(1.0)),
            ),
        )

    def test_supported_tensor_tensor_calls_reuse_tensor_add_values_and_layouts(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left.add(right)
        for case, call in self.supported_calls(left, right):
            self.assert_tensor_matches(call(), expected, case=("broadcast", case))

        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        offset = strided[1]
        offset_expected = offset.add(torch.ones((1, 2)))
        self.assert_tensor_matches(
            torch.add(offset, torch.ones((1, 2))),
            offset_expected,
            case="offset noncontiguous",
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        empty_broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.add(empty, empty_broadcast, out=None),
            empty.add(empty_broadcast),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        zeros = torch.tensor(memoryview(special_bits[:2].repeat(3)[:5].view(np.float32)))
        self.assert_tensor_matches(
            torch.add(special, zeros),
            special.add(zeros),
            case="signed zero nan infinity",
        )

    def test_autograd_empty_broadcast_and_no_grad_reuse_tensor_add_path(self):
        function_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        function_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        method_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        function_output = torch.add(
            function_left.transpose(0, 1), function_right.transpose(0, 1)
        )
        method_output = method_left.transpose(0, 1).add(method_right.transpose(0, 1))
        self.assert_tensor_matches(function_output, method_output, case="tracked views")
        function_output.sum().backward()
        method_output.sum().backward()
        self.assert_tensor_matches(
            function_left.grad, method_left.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            function_right.grad, method_right.grad, case="right gradient"
        )

        function_empty = torch.zeros((2, 0, 3), requires_grad=True)
        method_empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(function_empty, torch.ones((1, 1, 3))).sum().backward()
        method_empty.add(torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            function_empty.grad, method_empty.grad, case="empty gradient"
        )

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            output = torch.add(no_grad_left.transpose(0, 1), no_grad_right.transpose(0, 1))
        self.assertFalse(output.requires_grad)
        self.assertIsNone(no_grad_left.grad)
        self.assertIsNone(no_grad_right.grad)
        self.assertTrue(torch.add(no_grad_left, no_grad_right.transpose(0, 1)).requires_grad)

    def test_torch_function_modes_and_overrides_observe_calls(self):
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
            ("tensor/tensor", lambda: torch.add(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.add(left, 4.0), (left, 4.0), None),
            (
                "concrete out",
                lambda: torch.add(input=left, other=right, out=destination),
                (),
                ("input", "other", "out"),
            ),
            (
                "nondefault alpha",
                lambda: torch.add(input=left, other=right, alpha=2),
                (),
                ("input", "other", "alpha"),
            ),
        )
        for case, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
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
                forwarded = torch.add(input=left, other=right, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(forwarded, left.add(right), case="forwarded modes")

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
        self.assertIs(torch.add(input=left, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("alpha", func, types, args, kwargs))
                return marker

        events.clear()
        self.assertIs(torch.add(left, right, alpha=AlphaOverride()), marker)
        self.assertEqual(events[0][2], (AlphaOverride,))
        self.assertEqual(tuple(events[0][4]), ("alpha",))

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

    def test_unsupported_surface_is_explicit_and_nonmutating(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        destination = torch.tensor([17.0])
        unsupported_message = (
            "add(): only exact native CPU float32 Tensor input and Tensor other "
            "operands are supported"
        )

        for case, call in (
            ("tensor scalar", lambda: torch.add(tensor, 2.0)),
            ("scalar tensor", lambda: torch.add(2.0, tensor)),
            ("scalar scalar", lambda: torch.add(2.0, 3.0)),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    NotImplementedError, f"^{re.escape(unsupported_message)}$"
                ):
                    call()

        for alpha in (2, 0.5, np.int64(2), np.float32(0.5)):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^add\(\): alpha values other than 1 are not supported$",
                ):
                    torch.add(tensor, other, alpha=alpha)

        for alpha in (True, np.bool_(True), False, np.bool_(False)):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    RuntimeError, r"^Boolean alpha only supported for Boolean results\.$"
                ):
                    torch.add(tensor, other, alpha=alpha)

        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, other, out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        cases = (
            (
                lambda: torch.add(),
                "add() received an invalid combination of arguments - got (), but expected "
                "(Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor),
                "add() received an invalid combination of arguments - got (Tensor), "
                "but expected (Tensor input, Tensor other, *, Number alpha = 1, "
                "Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor, other, other),
                "add() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.add([], other),
                "add(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.add(tensor, []),
                "add(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.add(tensor, other, dtype=torch.float32),
                "add() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.add(tensor, other, device=torch.device("cpu")),
                "add() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.add(tensor, other, input=tensor),
                "add() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.add(input=tensor, x1=tensor, other=other),
                "add() got an unexpected keyword argument 'x1'",
            ),
            (
                lambda: torch.add(x=tensor, x1=tensor, other=other),
                "add() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.add(tensor, other, x2=other),
                "add() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.add(tensor, other, alpha=object()),
                "add(): argument 'alpha' must be Number, not object",
            ),
            (
                lambda: torch.add(tensor, other, out=[]),
                "add(): argument 'out' must be Tensor, not list",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")
        self.assertFalse(hasattr(torch, "add_"))
        self.assertNotIn("add_", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "add_"))
        self.assertFalse(hasattr(tensor, "add_"))

    def test_callable_metadata_pickling_and_exports(self):
        function = torch.add
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "add")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.add")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
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
