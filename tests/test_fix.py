import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .test_trunc import SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_trunc import SPECIAL_OUTPUT_BITS, make_cases


FIX_DOC = """
fix(input, *, out=None) -> Tensor

Alias for :func:`torch.trunc`
"""


class TopLevelFixTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    def assert_matches(self, actual, expected, *, case):
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
                self.tensor_bits(actual), self.tensor_bits(expected)
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.fix(source)),
            ("input", lambda: torch.fix(input=source)),
            ("x", lambda: torch.fix(x=source)),
            ("a", lambda: torch.fix(a=source)),
            ("x1", lambda: torch.fix(x1=source)),
            ("out none", lambda: torch.fix(source, out=None)),
            ("alias and out none", lambda: torch.fix(x=source, out=None)),
        )

    def test_values_ieee_bits_layouts_aliases_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            expected = source.trunc()
            for form, call in self.supported_calls(source):
                output = call()
                self.assert_matches(output, expected, case=(case, form))
                self.assertEqual(output.stride(), expected_stride)
                self.assertFalse(output.is_set_to(source))
                self.assertFalse(output.is_set_to(expected))
                if source.numel():
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                if case == "numerical edges":
                    np.testing.assert_array_equal(
                        self.tensor_bits(output), SPECIAL_OUTPUT_BITS
                    )

    def test_autograd_is_rejected_before_planning_but_no_grad_and_detach_work(self):
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        source = leaf.transpose(0, 2)[1]

        for form, call in self.supported_calls(source):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^fix\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    actual = call()
                    expected = source.trunc()
                self.assert_matches(actual, expected, case=(form, "no_grad"))
                self.assertFalse(actual.is_set_to(source))

        detached = source.detach()
        self.assert_matches(
            torch.fix(detached), detached.trunc(), case="detached input"
        )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^fix\(\): autograd recording is not supported$",
        ):
            torch.fix(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.fix(extreme)

    def test_concrete_out_is_rejected_before_autograd_or_planning(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        for form, call in (
            ("positional", lambda: torch.fix(source, out=destination)),
            ("keyword", lambda: torch.fix(input=source, out=destination)),
            ("alias", lambda: torch.fix(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^fix\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.data_ptr(), destination_pointer)
                self.assertEqual(destination.tolist(), [17.0, 19.0])
                self.assertIsNone(source.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^fix\(\): the 'out' argument is not supported$",
        ):
            torch.fix(extreme, out=destination)

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([1.25], requires_grad=True)
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
            self.assertIs(torch.fix(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.fix)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.fix(Override()), marker)
        self.assertIs(torch.fix(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.fix)
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

        self.assertIs(torch.fix(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([1.25, -1.25])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.fix(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0, -1.0])

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
            self.assertIs(torch.fix(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_distinct_builtin_metadata_copying_pickling_and_exports(self):
        function = torch.fix
        self.assertIs(function, torch._C.fix)
        self.assertIsNot(function, torch.trunc)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "fix")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.fix")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FIX_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method fix of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.fix, function)
        self.assertIsNot(owner.fix, owner.trunc)
        for action in (
            lambda: setattr(owner, "fix", None),
            lambda: delattr(owner, "fix"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.fix, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("fix"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["fix"], function)

    def test_tensor_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertFalse(hasattr(torch.Tensor, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix_"))
        self.assertFalse(hasattr(tensor, "fix"))
        self.assertFalse(hasattr(tensor, "fix_"))
        self.assertFalse(hasattr(torch, "fix_"))
        self.assertNotIn("fix_", torch.__all__)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.fix(),
                'fix() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.fix(tensor, tensor),
                "fix() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.fix(tensor, input=tensor),
                "fix() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.fix(out=tensor),
                'fix() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.fix(1, extra=True),
                "fix(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.fix(input=[]),
                "fix(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.fix(tensor, out=[]),
                "fix(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.fix(tensor, extra=True),
                "fix() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.fix(input=tensor, a=tensor),
                "fix() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.fix(a=tensor, x=tensor, out=None),
                "fix() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.fix(x=tensor, a=tensor, out=None),
                "fix() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
