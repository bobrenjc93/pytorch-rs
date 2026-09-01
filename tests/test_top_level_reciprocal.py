import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
reciprocal(input, *, out=None) -> Tensor

Returns a new tensor with the reciprocal of the elements of :attr:`input`

.. math::
    \\text{out}_{i} = \\frac{1}{\\text{input}_{i}}

.. note::
    Unlike NumPy's reciprocal, torch.reciprocal supports integral inputs. Integral
    inputs to reciprocal are automatically :ref:`promoted <type-promotion-doc>` to
    the default scalar type.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.4595, -2.1219, -1.4314,  0.7298])
    >>> torch.reciprocal(a)
    tensor([-2.1763, -0.4713, -0.6986,  1.3702])
"""


class TopLevelReciprocalTests(unittest.TestCase):
    def assert_matches_division(
        self, actual, expected, *, case, expected_stride=None
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(
                actual.stride(),
                expected.stride() if expected_stride is None else expected_stride,
            )
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.reciprocal(source)),
            ("input", lambda: torch.reciprocal(input=source)),
            ("x", lambda: torch.reciprocal(x=source)),
            ("a", lambda: torch.reciprocal(a=source)),
            ("x1", lambda: torch.reciprocal(x1=source)),
            ("out none", lambda: torch.reciprocal(source, out=None)),
            ("alias and out none", lambda: torch.reciprocal(x=source, out=None)),
        )

    @staticmethod
    def reciprocal_gradient(values, upstream):
        values = np.asarray(values, dtype=np.float32)
        upstream = np.asarray(upstream, dtype=np.float32)
        with np.errstate(all="ignore"):
            reciprocal = np.float32(1.0) / values
            return -(upstream * (reciprocal * reciprocal))

    def autograd_case(self, case):
        if case == "scalar":
            leaf = torch.tensor(4.0, requires_grad=True)
            return leaf, leaf, None, np.asarray(-0.0625, dtype=np.float32)
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return (
                leaf,
                leaf.transpose(0, 2)[1],
                None,
                np.zeros((2, 0, 3), dtype=np.float32),
            )
        if case == "offset":
            leaf = torch.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                requires_grad=True,
            )
            expected = np.zeros((2, 3, 4), dtype=np.float32)
            expected[1] = self.reciprocal_gradient(
                np.arange(13, 25, dtype=np.float32).reshape(3, 4),
                np.float32(1.0),
            )
            return leaf, leaf[1], None, expected
        if case == "noncontiguous":
            leaf = torch.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                requires_grad=True,
            )
            source = leaf.transpose(0, 2)[1]
            weights = torch.tensor(
                np.arange(1, 7, dtype=np.float32).reshape(3, 2).tolist()
            )
            expected = np.zeros((2, 3, 4), dtype=np.float32)
            expected[:, :, 1] = self.reciprocal_gradient(
                np.asarray(source, dtype=np.float32),
                np.asarray(weights, dtype=np.float32),
            ).transpose(1, 0)
            return leaf, source, weights, expected
        raise AssertionError(f"unknown reciprocal autograd case: {case}")

    def test_supported_calls_reuse_reflected_division_bits_and_unary_layouts(self):
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        cases = (
            ("scalar", torch.tensor(-0.0), None),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1], None),
            ("empty singleton trailing", torch.zeros((0, 1)), (1, 1)),
            ("empty singleton middle", torch.zeros((0, 1, 2)), (2, 2, 1)),
            ("empty singleton surrounding", torch.zeros((1, 0, 1)), (1, 1, 1)),
            ("offset", strided[1], None),
            ("noncontiguous", strided, None),
            (
                "numerical edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
                None,
            ),
        )

        for case, source, expected_stride in cases:
            expected = 1.0 / source
            for form, call in self.supported_calls(source):
                output = call()
                self.assert_matches_division(
                    output,
                    expected,
                    case=(case, form),
                    expected_stride=expected_stride,
                )
                if source.numel():
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())

    def test_autograd_records_reciprocal_backward_and_no_grad_is_allowed(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form, _ in self.supported_calls(torch.tensor(1.0)):
                leaf, source, weights, expected = self.autograd_case(case)
                output = dict(self.supported_calls(source))[form]()
                with self.subTest(case=case, form=form, mode="forward"):
                    self.assertTrue(output.requires_grad)
                    self.assertFalse(output.is_leaf)
                    self.assertFalse(output.is_set_to(source))
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            output
                        ),
                        ", grad_fn=<ReciprocalBackward0>",
                    )
                loss = output if output.numel() == 1 else output.sum()
                if weights is not None:
                    loss = (output * weights).sum()
                loss.backward()
                with self.subTest(case=case, form=form, mode="gradient"):
                    np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            torch.reciprocal(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.reciprocal(extreme)

        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        for form, call in self.supported_calls(source):
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    actual = call()
                    expected = 1.0 / source
                self.assert_matches_division(
                    actual, expected, case=(form, "no_grad")
                )

        detached = source.detach()
        self.assert_matches_division(
            torch.reciprocal(detached), 1.0 / detached, case="detached input"
        )

    def test_reciprocal_backward_special_values_accumulation_and_higher_order_boundary(
        self,
    ):
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3E80_0000,
                0x3F80_0000,
                0x4000_0000,
                0x4080_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        expected_gradient_bits = np.asarray(
            (
                0xFF80_0000,
                0x7F80_0000,
                0xFFC0_0000,
                0xFFC0_0000,
                0xFF80_0000,
                0x7F80_0000,
                0xC100_0000,
                0x3F00_0000,
                0xBE80_0000,
                0x3D80_0000,
                0x8000_0000,
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        leaf = torch.tensor(memoryview(input_bits.view(np.float32)), requires_grad=True)
        weights = torch.tensor(memoryview(weight_bits.view(np.float32)))
        loss = (torch.reciprocal(leaf) * weights).sum()
        loss.backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad, dtype=np.float32).view(np.uint32),
            expected_gradient_bits,
        )
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        tail_input_bits = np.asarray(
            (0x3F80_0000, 0x3F80_0000, 0x3F80_0000, 0xFFC5_4321),
            dtype=np.uint32,
        )
        tail_weight_bits = np.asarray(
            (0x3F80_0000, 0x3F80_0000, 0x3F80_0000, 0xFFC0_BBBB),
            dtype=np.uint32,
        )
        tail_expected_gradient_bits = np.asarray(
            (0xBF80_0000, 0xBF80_0000, 0xBF80_0000, 0x7FC0_BBBB),
            dtype=np.uint32,
        )
        tail_leaf = torch.tensor(
            memoryview(tail_input_bits.view(np.float32)), requires_grad=True
        )
        tail_weights = torch.tensor(memoryview(tail_weight_bits.view(np.float32)))
        (torch.reciprocal(tail_leaf) * tail_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(tail_leaf.grad, dtype=np.float32).view(np.uint32),
            tail_expected_gradient_bits,
        )

        accumulated = torch.tensor([1.0, 2.0, -4.0], requires_grad=True)
        torch.reciprocal(accumulated).sum().backward()
        torch.reciprocal(input=accumulated).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(accumulated.grad, dtype=np.float32),
            np.asarray([-2.0, -0.5, -0.125], dtype=np.float32),
        )

        higher_order = torch.tensor(2.0, requires_grad=True)
        higher_order_loss = torch.reciprocal(higher_order)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertEqual(higher_order.grad.item(), -0.25)

    def test_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([0.0, -0.0, 2.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0, 23.0])
        for form, call in (
            ("positional", lambda: torch.reciprocal(source, out=destination)),
            (
                "keyword",
                lambda: torch.reciprocal(input=source, out=destination),
            ),
            ("alias", lambda: torch.reciprocal(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^reciprocal\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

        with torch.no_grad():
            self.assert_matches_division(
                torch.reciprocal(source, out=None),
                1.0 / source,
                case="explicit out none",
            )

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([4.0], requires_grad=True)
        destination = torch.tensor([0.0])
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
            self.assertIs(
                torch.reciprocal(input=tensor, out=destination), marker
            )
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.reciprocal)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.reciprocal(Override()), marker)
        self.assertIs(torch.reciprocal(torch.tensor([4.0]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.reciprocal)
            self.assertEqual(dispatch_types, (Override,))

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(
            torch.reciprocal(BaseOverride(), out=DerivedOverride()), marker
        )
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([4.0])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.reciprocal(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.25])

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.reciprocal(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_copying_pickling_and_exports(self):
        function = torch.reciprocal
        self.assertIs(function, torch._C.reciprocal)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "reciprocal")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.reciprocal"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method reciprocal of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.reciprocal, function)
        for action in (
            lambda: setattr(owner, "reciprocal", None),
            lambda: delattr(owner, "reciprocal"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.reciprocal, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("reciprocal"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["reciprocal"], function)

    def test_tensor_method_is_exposed_while_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([2.0])
        self.assertTrue(hasattr(torch.Tensor, "reciprocal"))
        self.assertFalse(hasattr(torch.Tensor, "reciprocal_"))
        self.assertEqual(tensor.reciprocal().tolist(), [0.5])
        self.assertFalse(hasattr(tensor, "reciprocal_"))
        self.assertFalse(hasattr(torch, "reciprocal_"))
        self.assertNotIn("reciprocal_", torch.__all__)

    def test_other_dtypes_and_devices_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([2.0], device="cuda")

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.reciprocal(),
                'reciprocal() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.reciprocal(tensor, tensor),
                "reciprocal() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.reciprocal(tensor, input=tensor),
                "reciprocal() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reciprocal(out=tensor),
                'reciprocal() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.reciprocal(extra=tensor),
                'reciprocal() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.reciprocal(1, extra=True),
                "reciprocal(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.reciprocal(input=[]),
                "reciprocal(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.reciprocal(tensor, out=[]),
                "reciprocal(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.reciprocal(tensor, extra=True, out=[]),
                "reciprocal(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.reciprocal(tensor, extra=True),
                "reciprocal() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reciprocal(input=tensor, a=tensor),
                "reciprocal() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.reciprocal(a=tensor, x=tensor, out=None),
                "reciprocal() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.reciprocal(x=tensor, a=tensor, out=None),
                "reciprocal() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.reciprocal(
                    np.zeros((2, 3), dtype=np.float32)
                ),
                "reciprocal(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
