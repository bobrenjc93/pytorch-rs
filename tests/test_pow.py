import inspect
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


POW_UNSUPPORTED = (
    r"^pow\(\): only exact native CPU float32 Tensor bases with exponent 2 or 2\.0 "
    r"are supported$"
)


class TensorPowTests(unittest.TestCase):
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
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def value_cases():
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
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.0, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown pow autograd case: {case}")

    @staticmethod
    def supported_calls(source):
        return (
            ("method int", lambda: source.pow(2)),
            ("method float", lambda: source.pow(2.0)),
            ("method keyword", lambda: source.pow(exponent=2)),
            ("dunder", lambda: source.__pow__(2)),
            ("dunder keyword", lambda: source.__pow__(exponent=2)),
            ("operator", lambda: source**2),
            ("top positional int", lambda: torch.pow(source, 2)),
            ("top positional float", lambda: torch.pow(source, 2.0)),
            ("top keyword exponent", lambda: torch.pow(source, exponent=2)),
            ("top all keywords", lambda: torch.pow(input=source, exponent=2)),
            ("top input alias", lambda: torch.pow(x=source, exponent=2)),
            ("top out none", lambda: torch.pow(source, 2, out=None)),
        )

    def test_values_layouts_and_fresh_storage_reuse_square_path(self):
        for case, source in self.value_cases():
            expected = source.square()
            for form, call in self.supported_calls(source):
                actual = call()
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(source))

    def test_backward_through_sum_reuses_square_vjp(self):
        forms = tuple(form for form, _ in self.supported_calls(torch.tensor(1.0)))
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                pow_leaf, pow_input = self.autograd_case(case)
                square_leaf, square_input = self.autograd_case(case)
                output = dict(self.supported_calls(pow_input))[form]()
                expected = square_input.square()
                self.assert_tensor_matches(output, expected, case=(case, form, "forward"))

                output.sum().backward()
                expected.sum().backward()
                self.assert_tensor_matches(
                    pow_leaf.grad, square_leaf.grad, case=(case, form, "gradient")
                )

    def test_rejects_unsupported_overloads_without_mutating_out(self):
        tensor = torch.tensor([2.0, -3.0], requires_grad=True)
        unsupported = (
            ("method tensor exponent", lambda: tensor.pow(torch.tensor([2.0]))),
            ("top tensor exponent", lambda: torch.pow(tensor, torch.tensor([2.0]))),
            ("scalar base", lambda: torch.pow(2.0, tensor)),
            ("method other integer", lambda: tensor.pow(3)),
            ("top other float", lambda: torch.pow(tensor, 2.5)),
            ("bool exponent", lambda: torch.pow(tensor, True)),
            ("numpy non-two exponent", lambda: torch.pow(tensor, np.float32(3.0))),
        )
        for case, call in unsupported:
            with self.subTest(case=case):
                with self.assertRaisesRegex(NotImplementedError, POW_UNSUPPORTED):
                    call()

        destination = torch.tensor([17.0, 19.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^pow\(\): the 'out' argument is not supported$"
        ):
            torch.pow(tensor, 2, out=destination)
        self.assertEqual(destination.tolist(), [17.0, 19.0])

        with self.assertRaisesRegex(TypeError, r"unsupported operand type"):
            2 ** tensor
        self.assertFalse(hasattr(torch.Tensor, "__rpow__"))
        self.assertFalse(hasattr(torch, "pow_"))
        self.assertFalse(hasattr(torch.Tensor, "pow_"))
        self.assertFalse(hasattr(tensor, "pow_"))
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([2.0], device="cuda")

        with self.assertRaisesRegex(TypeError, "not an acceptable base type"):
            type("TensorSubclass", (torch.Tensor,), {})

    def test_scalar_only_top_level_pow_rejects_before_torch_function_mode(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for call in (
            lambda: torch.pow(2, 2),
            lambda: torch.pow(2, 2.0),
            lambda: torch.pow(2, 2, out=torch.tensor(0.0)),
        ):
            mode = RecordingMode()
            with self.subTest(call=call), mode, self.assertRaises(TypeError):
                call()
            self.assertEqual(mode.calls, [])

    def test_operator_reflected_fallback_for_foreign_rhs(self):
        tensor = torch.tensor([2.0])
        marker = object()

        class ReflectedPower:
            def __rpow__(self, other):
                self.other = other
                return marker

        rhs = ReflectedPower()
        self.assertIs(tensor.__pow__(rhs), NotImplemented)
        self.assertIs(tensor ** rhs, marker)
        self.assertIs(rhs.other, tensor)
        self.assertIs(tensor.__pow__([]), NotImplemented)

    def test_direct_dunder_call_shapes(self):
        tensor = torch.tensor([2.0, -3.0])
        expected = tensor.square()

        self.assert_tensor_matches(
            tensor.__pow__(exponent=2), expected, case="direct keyword exponent"
        )
        self.assert_tensor_matches(
            pow(tensor, 2, None), expected, case="builtin pow modulo none"
        )
        self.assertIs(tensor.__pow__(2, None), NotImplemented)
        self.assertIs(tensor.__pow__(2, modulo=None), NotImplemented)
        self.assertIs(tensor.__pow__(exponent=2, modulo=None), NotImplemented)

    def test_operator_modes_and_overrides_observe_tensorbase_pow(self):
        tensor = torch.tensor([2.0, -3.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "__pow__")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for form, call, expected_tail, expected_kwargs in (
            ("operator", lambda: tensor**2, (2,), {}),
            ("dunder", lambda: tensor.__pow__(2), (2,), {}),
            ("dunder keyword", lambda: tensor.__pow__(exponent=2), (), {"exponent": 2}),
        ):
            with self.subTest(form=form):
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (torch.Tensor,))
                self.assertEqual(len(args), 1 + len(expected_tail))
                self.assertIs(args[0], tensor)
                self.assertEqual(args[1:], expected_tail)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor**2
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(forwarded, tensor.square(), case="forwarded mode")

        class ReflectedPower:
            def __rpow__(self, other):
                self.other = other
                return marker

        rhs = ReflectedPower()
        order.clear()
        with ForwardingMode("reflected"):
            self.assertIs(tensor**rhs, marker)
        self.assertEqual(order, ["reflected"])
        self.assertIs(rhs.other, tensor)

        events = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((func, types, args, kwargs))
                return marker

        for form, call in (
            ("operator override", lambda rhs: tensor**rhs),
            ("dunder override", lambda rhs: tensor.__pow__(rhs)),
        ):
            with self.subTest(form=form):
                events.clear()
                rhs = Override()
                self.assertIs(call(rhs), marker)
                self.assertEqual(len(events), 1)
                function, dispatch_types, args, kwargs = events[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (torch.Tensor, Override))
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], tensor)
                self.assertIs(args[1], rhs)
                self.assertEqual(kwargs, {})

        class ModuloOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((func, types, args, kwargs))
                return marker

        for form, call in (
            ("dunder positional modulo override", lambda modulo: tensor.__pow__(2, modulo)),
            ("builtin positional modulo override", lambda modulo: pow(tensor, 2, modulo)),
        ):
            with self.subTest(form=form):
                events.clear()
                modulo = ModuloOverride()
                self.assertIs(call(modulo), marker)
                self.assertEqual(len(events), 1)
                function, dispatch_types, args, kwargs = events[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, (torch.Tensor, ModuloOverride))
                self.assertEqual(len(args), 3)
                self.assertIs(args[0], tensor)
                self.assertEqual(args[1], 2)
                self.assertIs(args[2], modulo)
                self.assertEqual(kwargs, {})

    def test_malformed_calls_and_public_metadata(self):
        tensor = torch.tensor([2.0])
        method_descriptor = inspect.getattr_static(torch.Tensor, "pow")
        operator_descriptor = inspect.getattr_static(torch.Tensor, "__pow__")
        function = torch.pow

        self.assertIs(type(method_descriptor), types.MethodDescriptorType)
        self.assertIs(type(tensor.pow), types.BuiltinMethodType)
        self.assertIs(type(operator_descriptor), types.MethodDescriptorType)
        self.assertEqual(method_descriptor.__name__, "pow")
        self.assertEqual(method_descriptor.__qualname__, "TensorBase.pow")
        self.assertEqual(tensor.pow.__qualname__, "Tensor.pow")
        self.assertEqual(operator_descriptor.__name__, "__pow__")
        self.assertEqual(tensor.__pow__.__name__, "__pow__")

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "pow")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.pow")
        self.assertEqual(function.__module__, "torch")
        self.assertIn("pow(input, exponent, *, out=None) -> Tensor", function.__doc__)
        self.assertEqual(torch.__all__.count("pow"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["pow"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        malformed = (
            lambda: tensor.pow(),
            lambda: tensor.pow([]),
            lambda: tensor.pow(2, 3),
            lambda: tensor.pow(2, out=None),
            lambda: torch.pow(),
            lambda: torch.pow(tensor),
            lambda: torch.pow([], 2),
            lambda: torch.pow(tensor, []),
            lambda: torch.pow(tensor, 2, extra=True),
        )
        for call in malformed:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_package_reinitialization_keeps_rpow_absent_and_pow_pickleable(self):
        source = r'''
import importlib
import inspect
import pickle
import sys

import torch_rs as torch

def assert_pow_surface(module):
    assert not hasattr(module.Tensor, "__rpow__")
    assert (module.tensor([2.0]) ** 2).tolist() == [4.0]
    descriptor = inspect.getattr_static(module.Tensor, "pow")
    function = module.pow
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(descriptor, protocol=protocol)) is descriptor
        assert pickle.loads(pickle.dumps(function, protocol=protocol)) is function

assert_pow_surface(torch)
for name in list(sys.modules):
    if name == "torch_rs" or name.startswith("torch_rs."):
        del sys.modules[name]
assert_pow_surface(importlib.import_module("torch_rs"))
'''
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
