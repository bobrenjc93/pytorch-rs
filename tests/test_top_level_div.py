import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TopLevelDivisionTests(unittest.TestCase):
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

    def test_tensor_tensor_values_layouts_empties_and_ieee_edges(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left / right

        for name in ("div", "divide"):
            function = getattr(torch, name)
            calls = (
                ("positional tensors", lambda function=function: function(left, right)),
                (
                    "canonical keywords",
                    lambda function=function: function(input=left, other=right),
                ),
                ("x aliases", lambda function=function: function(x=left, x2=right)),
                ("x1 aliases", lambda function=function: function(x1=left, x2=right)),
                (
                    "rounding none",
                    lambda function=function: function(
                        left, right, rounding_mode=None
                    ),
                ),
                (
                    "out none",
                    lambda function=function: function(left, right, out=None),
                ),
            )
            for case, call in calls:
                self.assert_tensor_matches(call(), expected, case=(name, case))

        offset_noncontiguous = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ).transpose(0, 1)[1]
        self.assertGreater(offset_noncontiguous.storage_offset(), 0)
        self.assertFalse(offset_noncontiguous.is_contiguous())
        offset_other = torch.tensor([2.0, -0.0])
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(torch, name)(offset_noncontiguous, offset_other),
                offset_noncontiguous / offset_other,
                case=(name, "offset noncontiguous"),
            )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(torch, name)(empty, broadcast),
                empty / broadcast,
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        divisors = torch.tensor([1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0])
        for name in ("div", "divide"):
            result = getattr(torch, name)(special, divisors)
            self.assert_tensor_matches(
                result,
                special / divisors,
                case=(name, "signed zero nan infinity"),
            )
            self.assertFalse(result.is_set_to(special))
            self.assertFalse(result.is_set_to(divisors))
            if result.numel():
                self.assertNotEqual(result.data_ptr(), special.data_ptr())
                self.assertNotEqual(result.data_ptr(), divisors.data_ptr())

    def test_scalar_tensor_values_layouts_empties_and_ieee_edges(self):
        base = torch.tensor([[1.0, -2.0, 0.0], [4.5, -6.0, 3.5]])
        scalars = (
            ("python bool", True),
            ("python int", -2),
            ("python float", 2.5),
            ("numpy bool", np.bool_(True)),
            ("numpy int", np.int64(3)),
            ("numpy float signed zero", np.float32(-0.0)),
            ("python inf", float("inf")),
            ("python nan", float("nan")),
        )
        for name in ("div", "divide"):
            function = getattr(torch, name)
            for case, scalar in scalars:
                self.assert_tensor_matches(
                    function(base, scalar),
                    base / scalar,
                    case=(name, "tensor/scalar", case),
                )
                self.assert_tensor_matches(
                    function(input=base, other=scalar),
                    base / scalar,
                    case=(name, "keyword tensor/scalar", case),
                )

        denominator = torch.tensor([4.0, -0.0])
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(torch, name)(2.0, denominator),
                2.0 / denominator,
                case=(name, "scalar/tensor"),
            )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(torch, name)(empty, -0.0),
                empty / -0.0,
                case=(name, "strided empty scalar"),
            )

    def test_active_autograd_is_rejected_but_no_grad_uses_native_division(self):
        for name in ("div", "divide"):
            function = getattr(torch, name)
            left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            right = torch.tensor([[5.0], [7.0]], requires_grad=True)

            with self.subTest(name=name, case="tensor operands"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(left.transpose(0, 1), right.transpose(0, 1))
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)

            with self.subTest(name=name, case="tensor scalar"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(left, 2.0)

            with self.subTest(name=name, case="scalar tensor"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(2.0, left)

            with self.subTest(name=name, case="no_grad"):
                with torch.no_grad():
                    tensor_output = function(
                        left.transpose(0, 1), right.transpose(0, 1)
                    )
                    scalar_output = function(left, 2.0)
                    reflected_output = function(2.0, left)
                    expected_tensor_output = left.transpose(0, 1) / right.transpose(
                        0, 1
                    )
                    expected_reflected_output = 2.0 / left
                self.assertFalse(tensor_output.requires_grad)
                self.assertTrue(tensor_output.is_leaf)
                self.assert_tensor_matches(
                    tensor_output,
                    expected_tensor_output,
                    case=(name, "no_grad tensor"),
                )
                self.assertFalse(scalar_output.requires_grad)
                self.assertTrue(scalar_output.is_leaf)
                self.assertFalse(reflected_output.requires_grad)
                self.assert_tensor_matches(
                    reflected_output,
                    expected_reflected_output,
                    case=(name, "no_grad reflected scalar"),
                )

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([2.0])
        right = torch.tensor([4.0])
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("div", "divide"):
            function = getattr(torch, name)
            calls = (
                ("tensor/tensor", lambda function=function: function(left, right), (left, right), None),
                ("tensor/scalar", lambda function=function: function(left, 4.0), (left, 4.0), None),
                ("scalar/tensor", lambda function=function: function(4.0, left), (4.0, left), None),
                ("scalar/scalar", lambda function=function: function(4.0, 2.0), (4.0, 2.0), None),
                (
                    "canonical keywords",
                    lambda function=function: function(input=left, other=right),
                    (),
                    ("input", "other"),
                ),
                (
                    "aliases",
                    lambda function=function: function(x1=left, x2=4.0),
                    (),
                    ("x1", "x2"),
                ),
                (
                    "rounding",
                    lambda function=function: function(
                        left, right, rounding_mode="floor"
                    ),
                    (left, right),
                    ("rounding_mode",),
                ),
                (
                    "out",
                    lambda function=function: function(left, right, out=destination),
                    (left, right),
                    ("out",),
                ),
            )
            for case, call, expected_args, expected_keywords in calls:
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                dispatched, dispatch_types, args, kwargs = mode.calls[0]
                with self.subTest(name=name, case=case):
                    self.assertIs(dispatched, function)
                    self.assertEqual(dispatch_types, ())
                    self.assertEqual(args, expected_args)
                    if expected_keywords is None:
                        self.assertIsNone(kwargs)
                    else:
                        self.assertEqual(tuple(kwargs), expected_keywords)

            invalid_mode = RecordingMode()
            with invalid_mode:
                with self.assertRaises(TypeError):
                    function(left, [])
            self.assertEqual(invalid_mode.calls, [])

            order = []

            class ForwardingMode(torch.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    order.append(self.label)
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = function(input=left, other=right, out=None)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(forwarded, left / right, case=(name, "forwarded"))

    def test_operand_rounding_and_out_overrides_order_types_and_declining_errors(self):
        native = torch.tensor([2.0])
        marker = object()
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

        for name in ("div", "divide"):
            function = getattr(torch, name)
            events.clear()
            self.assertIs(function(LeftOverride(), RightOverride()), marker)
            self.assertEqual([event[0] for event in events], ["left", "right"])
            for _, dispatched, dispatch_types, args, kwargs in events:
                self.assertIs(dispatched, function)
                self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
                self.assertEqual(len(args), 2)
                self.assertIsNone(kwargs)

            events.clear()
            self.assertIs(function(input=native, other=RightOverride()), marker)
            _, dispatched, dispatch_types, args, kwargs = events[0]
            self.assertIs(dispatched, function)
            self.assertEqual(dispatch_types, (RightOverride,))
            self.assertEqual(args, ())
            self.assertEqual(tuple(kwargs), ("input", "other"))

            class RoundingOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(("rounding", func, types, args, kwargs))
                    return marker

            events.clear()
            self.assertIs(function(native, native, rounding_mode=RoundingOverride()), marker)
            _, dispatched, dispatch_types, args, kwargs = events[0]
            self.assertIs(dispatched, function)
            self.assertEqual(dispatch_types, (RoundingOverride,))
            self.assertEqual(tuple(kwargs), ("rounding_mode",))
            self.assertIs(kwargs["rounding_mode"].__class__, RoundingOverride)

            class OutOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(("out", func, types, args, kwargs))
                    return marker

            events.clear()
            self.assertIs(function(native, native, out=OutOverride()), marker)
            _, dispatched, dispatch_types, args, kwargs = events[0]
            self.assertIs(dispatched, function)
            self.assertEqual(dispatch_types, (OutOverride,))
            self.assertEqual(tuple(kwargs), ("out",))
            self.assertIs(kwargs["out"].__class__, OutOverride)

            subclass_order = []

            class BaseOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    subclass_order.append("base")
                    return marker

            class DerivedOverride(BaseOverride):
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    subclass_order.append(("derived", types))
                    return marker

            self.assertIs(function(BaseOverride(), DerivedOverride()), marker)
            self.assertEqual(
                subclass_order, [("derived", (DerivedOverride, BaseOverride))]
            )

            class DecliningOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    return NotImplemented

            with self.assertRaises(TypeError) as raised:
                function(DecliningOverride(), native)
            self.assertEqual(
                str(raised.exception),
                f"Multiple dispatch failed for 'torch.{name}'; all __torch_function__ "
                "handlers returned NotImplemented:\n\n"
                f"  - tensor subclass <class '{DecliningOverride.__module__}."
                f"{DecliningOverride.__qualname__}'>\n\n"
                "For more information, try re-running with TORCH_LOGS=not_implemented",
            )

    def test_unsupported_surface_errors_do_not_mutate_inputs(self):
        tensor = torch.tensor([4.0])
        other = torch.tensor([2.0])
        destination = torch.tensor([17.0])

        for name in ("div", "divide"):
            function = getattr(torch, name)
            with self.subTest(name=name, boundary="rounding floor"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): non-None rounding_mode is not supported$",
                ):
                    function(tensor, other, rounding_mode="floor")
            with self.subTest(name=name, boundary="rounding trunc"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): non-None rounding_mode is not supported$",
                ):
                    function(tensor, 2.0, rounding_mode="trunc")
            with self.subTest(name=name, boundary="out tensor"):
                with self.assertRaisesRegex(
                    RuntimeError, rf"^{name}\(\): the 'out' argument is not supported$"
                ):
                    function(tensor, other, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(name=name, boundary="out scalar path"):
                with self.assertRaisesRegex(
                    RuntimeError, rf"^{name}\(\): the 'out' argument is not supported$"
                ):
                    function(tensor, 2.0, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(name=name, boundary="scalar only"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): scalar-scalar division is not supported; "
                    r"at least one operand must be Tensor$",
                ):
                    function(2, 3)
            with self.subTest(name=name, boundary="dtype keyword"):
                with self.assertRaisesRegex(TypeError, f"^{name}\\(\\) received"):
                    function(tensor, other, dtype=torch.float32)
            with self.subTest(name=name, boundary="device keyword"):
                with self.assertRaisesRegex(TypeError, f"^{name}\\(\\) received"):
                    function(tensor, other, device=torch.device("cpu"))

            class TensorLike:
                pass

            with self.subTest(name=name, boundary="unsupported operand"):
                with self.assertRaisesRegex(TypeError, f"^{name}\\(\\)"):
                    function(tensor, TensorLike())

            with self.subTest(name=name, boundary="unsupported subclass"):
                with self.assertRaisesRegex(
                    TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
                ):
                    type("TensorSubclass", (torch.Tensor,), {})

        for name in ("div_", "divide_"):
            self.assertFalse(hasattr(torch, name))
            self.assertFalse(hasattr(torch.Tensor, name))

    def test_callable_metadata_copy_pickle_reload_and_exports(self):
        functions = {"div": torch.div, "divide": torch.divide}
        self.assertIsNot(functions["div"], functions["divide"])

        for name, function in functions.items():
            with self.subTest(name=name, contract=True):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
                self.assertEqual(function.__module__, "torch")
                self.assertIsNone(function.__text_signature__)
                self.assertTrue(
                    function.__doc__.startswith(
                        f"\n{name}(input, other, *, rounding_mode=None, out=None) -> Tensor\n\n"
                    )
                )
                if name == "div":
                    for unsupported in (
                        "Non-``None`` rounding modes",
                        "concrete\n``out`` tensors",
                        "scalar-only calls",
                        "unsupported dtype/device/subclass operands",
                        "active autograd recording remain unsupported",
                    ):
                        self.assertIn(unsupported, function.__doc__)
                self.assertRegex(
                    repr(function),
                    rf"^<built-in method {name} of type object at 0x[0-9a-f]+>$",
                )
                with self.assertRaises(ValueError):
                    inspect.signature(function)

            owner = function.__reduce__()[1][0]
            self.assertEqual(owner.__name__, "_VariableFunctionsClass")
            self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
            self.assertEqual(owner.__module__, "torch_rs._C")
            self.assertIs(owner, torch._C._VariableFunctionsClass)
            self.assertIs(getattr(owner, name), function)
            for mutation in (
                lambda name=name: setattr(owner, name, None),
                lambda name=name: delattr(owner, name),
            ):
                with self.assertRaises(TypeError):
                    mutation()
                self.assertIs(getattr(owner, name), function)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(torch.__all__.count(name), 1)

        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], torch.div)
        self.assertIs(wildcard_namespace["divide"], torch.divide)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.div, functions["div"])
        self.assertIs(torch.divide, functions["divide"])


if __name__ == "__main__":
    unittest.main()
