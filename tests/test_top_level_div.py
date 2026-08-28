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
div(input, other, *, rounding_mode=None, out=None) -> Tensor

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.

.. math::
    \\text{out}_i = \\frac{\\text{input}_i}{\\text{other}_i}

.. note::
    By default, this performs a "true" division like Python 3.
    See the :attr:`rounding_mode` argument for floor division.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
Always promotes integer types to the default scalar type.

Args:
    input (Tensor): the dividend
    other (Tensor or Number): the divisor

Keyword args:
    rounding_mode (str, optional): Type of rounding applied to the result:

        * None - default behavior. Performs no rounding and, if both :attr:`input` and
          :attr:`other` are integer types, promotes the inputs to the default scalar type.
          Equivalent to true division in Python (the ``/`` operator) and NumPy's ``np.true_divide``.
        * ``"trunc"`` - rounds the results of the division towards zero.
          Equivalent to C-style integer division.
        * ``"floor"`` - rounds the results of the division down.
          Equivalent to floor division in Python (the ``//`` operator) and NumPy's ``np.floor_divide``.

    out (Tensor, optional): the output tensor.

Examples::

    >>> x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> torch.div(x, 0.5)
    tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

    >>> a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
    ...                   [ 0.1815, -1.0111,  0.9805, -1.5923],
    ...                   [ 0.1062,  1.4581,  0.7759, -1.2344],
    ...                   [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> torch.div(a, b)
    tensor([[-0.4620, -6.6051,  0.5676,  1.2639],
            [ 0.2260, -3.4509, -1.2086,  6.8990],
            [ 0.1322,  4.9764, -0.9564,  5.3484],
            [-0.2278, -0.1068, -1.4678,  6.3938]])

    >>> torch.div(a, b, rounding_mode='trunc')
    tensor([[-0., -6.,  0.,  1.],
            [ 0., -3., -1.,  6.],
            [ 0.,  4., -0.,  5.],
            [-0., -0., -1.,  6.]])

    >>> torch.div(a, b, rounding_mode='floor')
    tensor([[-1., -7.,  0.,  1.],
            [ 0., -4., -2.,  6.],
            [ 0.,  4., -1.,  5.],
            [-1., -1., -2.,  6.]])

"""


class TopLevelDivTests(unittest.TestCase):
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

    def test_supported_tensor_and_scalar_calls_reuse_operator_division(self):
        left = torch.tensor([[[1.0, -4.0], [2.0, -5.0], [3.0, -6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-3.0], [4.0]])
        tensor_expected = left / right
        for case, call in (
            ("positional tensors", lambda: torch.div(left, right)),
            (
                "canonical tensor keywords",
                lambda: torch.div(input=left, other=right),
            ),
            (
                "explicit default keywords",
                lambda: torch.div(
                    input=left, other=right, rounding_mode=None, out=None
                ),
            ),
        ):
            self.assert_tensor_matches(call(), tensor_expected, case=case)

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            for case, call, expected in (
                ("tensor/scalar", lambda scalar=scalar: torch.div(offset, scalar), offset / scalar),
                (
                    "keyword tensor/scalar",
                    lambda scalar=scalar: torch.div(input=offset, other=scalar),
                    offset / scalar,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(case, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.div(empty, broadcast), empty / broadcast, case="strided empty"
        )

        numerator_bits = np.asarray(
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
        denominator_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
            ),
            dtype=np.uint32,
        )
        numerator = torch.tensor(memoryview(numerator_bits.view(np.float32)))
        denominator = torch.tensor(memoryview(denominator_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.div(numerator, denominator),
            numerator / denominator,
            case="IEEE edge values",
        )

    def test_autograd_is_rejected_before_output_planning_and_no_grad_is_allowed(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        divisor = torch.tensor([2.0, -0.0])

        for case, call in (
            ("tensor/tensor", lambda: torch.div(source, divisor)),
            ("tensor/scalar", lambda: torch.div(source, 2.0)),
            ("right requires grad", lambda: torch.div(divisor, source)),
        ):
            with self.subTest(case=case, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^div\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(case=case, mode="no_grad"):
                with torch.no_grad():
                    actual = call()
                    expected = (
                        source / divisor
                        if case == "tensor/tensor"
                        else source / 2.0
                        if case == "tensor/scalar"
                        else divisor / source
                    )
                self.assertFalse(actual.requires_grad)
                self.assert_tensor_matches(actual, expected, case=(case, "no_grad"))

        extreme = torch.zeros(
            (0, sys.maxsize // 2 + 1, 1), requires_grad=True
        )
        extreme_divisor = torch.ones((1, 1, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^div\(\): autograd recording is not supported$",
        ):
            torch.div(extreme, extreme_divisor)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.div(extreme, extreme_divisor)

        detached = source.detach()
        self.assert_tensor_matches(
            torch.div(detached, 2.0), detached / 2.0, case="detached input"
        )

    def test_concrete_rounding_mode_and_out_are_rejected_without_mutation(self):
        source = torch.tensor([0.0, -0.0, 2.0], requires_grad=True)
        divisor = torch.tensor([2.0, 2.0, -0.0])
        destination = torch.tensor([17.0, 19.0, 23.0])

        for form, call in (
            ("positional", lambda: torch.div(source, divisor, out=destination)),
            (
                "keyword",
                lambda: torch.div(input=source, other=divisor, out=destination),
            ),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^div\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

        for rounding_mode in ("trunc", "floor", "other"):
            with self.subTest(rounding_mode=rounding_mode):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^div\(\): rounding_mode is not supported; only None is implemented$",
                ):
                    torch.div(source, divisor, rounding_mode=rounding_mode)

        with torch.no_grad():
            self.assert_tensor_matches(
                torch.div(source, divisor, rounding_mode=None, out=None),
                source / divisor,
                case="explicit default options",
            )

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
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
            ("tensor/tensor", lambda: torch.div(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.div(left, 4.0), (left, 4.0), None),
            (
                "canonical keywords",
                lambda: torch.div(input=left, other=right),
                (),
                ("input", "other"),
            ),
            (
                "default options",
                lambda: torch.div(input=left, other=right, rounding_mode=None, out=None),
                (),
                ("input", "other", "rounding_mode", "out"),
            ),
            (
                "concrete out",
                lambda: torch.div(left, right, out=destination),
                (left, right),
                ("out",),
            ),
            (
                "concrete rounding mode",
                lambda: torch.div(left, right, rounding_mode="trunc"),
                (left, right),
                ("rounding_mode",),
            ),
        )
        for case, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.div)
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
                actual = torch.div(input=left, other=4.0, rounding_mode=None, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left / 4.0, case="forwarded modes")

        for call in (
            lambda: torch.div([], right),
            lambda: torch.div(left, []),
            lambda: torch.div(4.0, left),
            lambda: torch.div(left, right, rounding_mode=True),
            lambda: torch.div(x1=left, x2=right),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_operand_overrides_order_types_and_declining_errors(self):
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

        self.assertIs(torch.div(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.div)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.div(input=native, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.div)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("out", func, types, args, kwargs))
                return marker

        events.clear()
        self.assertIs(torch.div(native, native, out=OutOverride()), marker)
        self.assertEqual(events[0][1], torch.div)
        self.assertEqual(events[0][2], (OutOverride,))

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

        self.assertIs(torch.div(BaseOverride(), native, out=DerivedOverride()), marker)
        self.assertEqual(
            subclass_order, [("derived", (DerivedOverride, BaseOverride))]
        )

        class ScalarOverride(int):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("scalar", func, types, args, kwargs))
                return marker

        events.clear()
        self.assertIs(torch.div(ScalarOverride(4), native), marker)
        self.assertEqual(events[0][1], torch.div)
        self.assertEqual(events[0][2], (ScalarOverride,))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.div(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.div'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_pickling_exports_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        overloads = (
            "but expected one of:\n"
            " * (Tensor input, Tensor other, *, Tensor out = None)\n"
            " * (Tensor input, Tensor other, *, str rounding_mode, Tensor out = None)\n"
            " * (Tensor input, Number other, *, str rounding_mode)\n"
        )
        cases = (
            (
                lambda: torch.div(),
                "div() received an invalid combination of arguments - got (), "
                f"{overloads}",
            ),
            (
                lambda: torch.div(tensor),
                "div() received an invalid combination of arguments - got "
                f"(Tensor), {overloads}",
            ),
            (
                lambda: torch.div(tensor, tensor, tensor),
                "div() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, Tensor), {overloads}",
            ),
            (
                lambda: torch.div([], tensor),
                "div(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.div(tensor, []),
                "div(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.div(input=None, other=tensor),
                "div(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.div(tensor, tensor, input=tensor),
                "div() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, input=Tensor), {overloads}",
            ),
            (
                lambda: torch.div(tensor, tensor, extra=True),
                "div() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, extra=bool), {overloads}",
            ),
            (
                lambda: torch.div(tensor, tensor, rounding_mode=True),
                "div() received an invalid combination of arguments - got "
                f"(Tensor, Tensor, rounding_mode=bool), {overloads}",
            ),
            (
                lambda: torch.div(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.div(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.div(2, tensor),
                "div(): argument 'input' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        for alias_kwargs in (
            {"x": tensor, "other": tensor},
            {"a": tensor, "other": tensor},
            {"x1": tensor, "x2": tensor},
            {"input": tensor, "x2": tensor},
        ):
            with self.subTest(alias_kwargs=tuple(alias_kwargs)):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^div\(\) received an invalid combination of arguments",
                ):
                    torch.div(**alias_kwargs)

        with self.assertRaisesRegex(
            TypeError,
            r"^div\(\) received an invalid combination of arguments - got "
            r"\(Tensor, Tensor, out=list\), but expected one of:",
        ):
            torch.div(tensor, tensor, out=[])

        function = torch.div
        self.assertIs(function, torch._C.div)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "div")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.div")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method div of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.div, function)
        for action in (
            lambda: setattr(owner, "div", None),
            lambda: delattr(owner, "div"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.div, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("div"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertFalse(hasattr(torch, "divide"))
        self.assertFalse(hasattr(torch, "true_divide"))
        self.assertFalse(hasattr(torch.Tensor, "div"))
        self.assertFalse(hasattr(tensor, "div"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], function)


if __name__ == "__main__":
    unittest.main()
