import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
square(input: Tensor, *, out: Optional[Tensor]) -> Tensor

Returns a new tensor with the square of the elements of :attr:`input`.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-2.0755,  1.0226,  0.0831,  0.4806])
    >>> torch.square(a)
    tensor([ 4.3077,  1.0457,  0.0069,  0.2310])
"""


class TopLevelSquareTests(unittest.TestCase):
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
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.square(source)),
            ("input", lambda: torch.square(input=source)),
            ("x", lambda: torch.square(x=source)),
            ("a", lambda: torch.square(a=source)),
            ("x1", lambda: torch.square(x1=source)),
            ("out none", lambda: torch.square(source, out=None)),
            ("alias and out none", lambda: torch.square(x=source, out=None)),
        )

    @staticmethod
    def make_autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.0, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None

        leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        if case == "offset":
            source = leaf[1]
            weights = torch.tensor(
                np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist()
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)
            weights = torch.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2).tolist()
            )
            return leaf, source, weights
        raise AssertionError(f"unknown square autograd case: {case}")

    def test_supported_calls_reuse_tensor_square_values_layouts_and_fresh_storage(self):
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
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

        for case, source in cases:
            expected = source.square()
            for form, call in self.supported_calls(source):
                output = call()
                self.assert_tensor_matches(output, expected, case=(case, form))
                self.assertFalse(output.is_set_to(source))

    def test_autograd_scalar_empty_offset_and_noncontiguous_reuses_tensor_vjp(self):
        forms = tuple(form for form, _ in self.supported_calls(torch.tensor(0.0)))
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                function_leaf, function_source, function_weights = (
                    self.make_autograd_case(case)
                )
                method_leaf, method_source, method_weights = self.make_autograd_case(
                    case
                )
                output = dict(self.supported_calls(function_source))[form]()
                method_output = method_source.square()

                self.assert_tensor_matches(
                    output, method_output, case=(case, form, "output")
                )
                if function_weights is None:
                    function_loss = output if case == "scalar" else output.sum()
                    method_loss = (
                        method_output if case == "scalar" else method_output.sum()
                    )
                else:
                    function_loss = (output * function_weights).sum()
                    method_loss = (method_output * method_weights).sum()
                function_loss.backward()
                method_loss.backward()
                self.assert_tensor_matches(
                    function_leaf.grad,
                    method_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_repeated_backward_graph_freeing_no_grad_and_detach_reuse_method_path(self):
        function_leaf = torch.tensor([2.0, -3.0], requires_grad=True)
        method_leaf = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.square(function_leaf).sum().backward()
        method_leaf.square().sum().backward()
        torch.square(input=function_leaf).sum().backward()
        method_leaf.square().sum().backward()
        self.assert_tensor_matches(
            function_leaf.grad, method_leaf.grad, case="accumulated gradient"
        )

        function_freed = torch.tensor([2.0, -3.0], requires_grad=True)
        method_freed = torch.tensor([2.0, -3.0], requires_grad=True)
        function_loss = torch.square(function_freed).sum()
        method_loss = method_freed.square().sum()
        function_loss.backward()
        method_loss.backward()
        with self.assertRaises(RuntimeError) as function_raised:
            function_loss.backward()
        with self.assertRaises(RuntimeError) as method_raised:
            method_loss.backward()
        self.assertEqual(str(function_raised.exception), str(method_raised.exception))

        no_grad_function_leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        no_grad_method_leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        with torch.no_grad():
            function_output = torch.square(
                no_grad_function_leaf.transpose(0, 2)[1], out=None
            )
            method_output = no_grad_method_leaf.transpose(0, 2)[1].square()
        self.assert_tensor_matches(
            function_output, method_output, case="no_grad output"
        )
        self.assertIsNone(no_grad_function_leaf.grad)
        self.assertTrue(torch.square(no_grad_function_leaf).requires_grad)

        detached = no_grad_function_leaf.detach().transpose(0, 2)[1]
        self.assert_tensor_matches(
            torch.square(detached), detached.square(), case="detached input"
        )

    def test_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([2.0, -3.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.square(source, out=destination)),
            ("keyword", lambda: torch.square(input=source, out=destination)),
            ("alias", lambda: torch.square(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^square\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])
                self.assertEqual(source.tolist(), [2.0, -3.0])

        self.assert_tensor_matches(
            torch.square(source, out=None), source.square(), case="explicit out none"
        )

    def test_modes_and_subclass_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([2.0, -3.0], requires_grad=True)
        destination = torch.tensor([0.0, 0.0])
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
            self.assertIs(torch.square(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.square)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.square(Override()), marker)
        self.assertIs(torch.square(tensor, out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.square)
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

        self.assertIs(torch.square(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.square(input=tensor, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [4.0, 9.0])
        forwarded.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [4.0, -6.0])

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
            self.assertIs(torch.square(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.square
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "square")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.square")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method square of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.square, function)
        for action in (
            lambda: setattr(owner, "square", None),
            lambda: delattr(owner, "square"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.square, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("square"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertFalse(hasattr(torch.Tensor, "square_"))
        self.assertFalse(hasattr(torch, "square_"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["square"], function)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([2.0])
        cases = (
            (
                lambda: torch.square(),
                'square() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.square(tensor, tensor),
                "square() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.square(tensor, input=tensor),
                "square() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.square(out=tensor),
                'square() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.square(extra=tensor),
                'square() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.square(1, extra=True),
                "square(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.square(input=[]),
                "square(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.square(tensor, out=[]),
                "square(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.square(tensor, extra=True, out=[]),
                "square(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.square(tensor, extra=True),
                "square() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.square(input=tensor, a=tensor),
                "square() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.square(a=tensor, x=tensor, out=None),
                "square() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.square(x=tensor, a=tensor, out=None),
                "square() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.square(np.zeros((2, 3), dtype=np.float32)),
                "square(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
