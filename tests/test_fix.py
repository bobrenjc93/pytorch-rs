import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FIX_DOC = """
fix(input, *, out=None) -> Tensor

Alias for :func:`torch.trunc`
"""

SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
        0xBF7F_FFFF,
        0xBF80_0000,
        0xBFC0_0000,
        0x3FC0_0000,
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


def make_cases(module):
    base = module.tensor(
        np.linspace(-3.75, 3.75, 24, dtype=np.float32)
        .reshape(2, 3, 4)
        .tolist(),
        dtype=module.float32,
    )
    strided = base.transpose(0, 2)
    channels_last = module.tensor(
        np.linspace(-15.0, 15.0, 120, dtype=np.float32)
        .reshape(2, 3, 4, 5)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last)
    channels_last_3d = module.tensor(
        np.linspace(-90.0, 90.0, 720, dtype=np.float32)
        .reshape(2, 3, 4, 5, 6)
        .tolist(),
        dtype=module.float32,
    ).contiguous(memory_format=module.channels_last_3d)
    return (
        ("scalar", module.tensor(-0.0, dtype=module.float32)),
        (
            "empty offset",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
        ),
        ("empty singleton trailing", module.zeros((0, 1))),
        ("empty singleton middle", module.zeros((0, 1, 2))),
        ("empty singleton surrounding", module.zeros((1, 0, 1))),
        ("offset", strided[1]),
        ("noncontiguous", strided),
        ("channels last", channels_last),
        ("channels last 3d", channels_last_3d),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
        ),
    )


class TopLevelFixTests(unittest.TestCase):
    @staticmethod
    def bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

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

    def assert_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(self.bits(actual), self.bits(expected))

    def test_values_ieee_bits_layouts_and_fresh_storage_reuse_trunc_kernel(self):
        for case, source in make_cases(torch):
            expected = source.trunc()
            for form, call in self.supported_calls(source):
                self.assert_matches(call(), expected, source, case=(case, form))

    def test_active_autograd_is_rejected_before_output_planning(self):
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        tracked = (
            torch.tensor(-1.25, requires_grad=True),
            torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
            leaf.transpose(0, 2)[1],
            leaf.transpose(0, 2),
        )
        message = r"^fix\(\): autograd recording is not supported$"
        for case, source in enumerate(tracked):
            for form, call in self.supported_calls(source):
                with self.subTest(case=case, form=form):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            torch.fix(extreme)
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.fix(extreme)

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        leaf = torch.tensor(
            [[-2.75, -0.0, 1.25], [2.5, 4.75, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        detached = source.detach()
        expected = detached.trunc()

        self.assert_matches(torch.fix(detached), expected, detached, case="detached")
        with torch.no_grad():
            actual = torch.fix(source, out=None)
        self.assert_matches(actual, expected, source, case="no_grad")
        self.assertIsNone(leaf.grad)

    def test_concrete_out_is_rejected_before_autograd_or_output_allocation(self):
        source = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        destination_bits = self.bits(destination).copy()

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
                np.testing.assert_array_equal(
                    self.bits(destination), destination_bits
                )
                self.assertIsNone(source.grad)

    def test_modes_and_overrides_observe_original_calls_before_native_limits(self):
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
        self.assertEqual(
            mode.calls,
            [(torch.fix, (), (), {"input": tensor, "out": destination})],
        )

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

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.fix(input=tensor.detach(), out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        forwarding_order.clear()
        with self.assertRaisesRegex(
            RuntimeError, r"^fix\(\): autograd recording is not supported$"
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    torch.fix(tensor)
        self.assertEqual(forwarding_order, ["upper", "lower"])

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

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.fix(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_callable_metadata_exports_and_distinct_alias_identity(self):
        function = torch.fix
        self.assertIs(function, torch._C.fix)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertIsNot(function, torch.trunc)
        self.assertEqual(function.__name__, "fix")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.fix")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FIX_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method fix of type object at 0x[0-9a-f]+>$"
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
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["fix"], function)

    def test_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.25])
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
                lambda: torch.fix(extra=tensor),
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
                lambda: torch.fix(tensor, extra=True, out=[]),
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
            (
                lambda: torch.fix(np.zeros((2, 3), dtype=np.float32)),
                (
                    "fix(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_tensor_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([1.25])
        self.assertTrue(hasattr(torch, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix"))
        self.assertFalse(hasattr(tensor, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix_"))
        self.assertFalse(hasattr(tensor, "fix_"))
        self.assertFalse(hasattr(torch, "fix_"))
        self.assertNotIn("fix_", torch.__all__)


if __name__ == "__main__":
    unittest.main()
