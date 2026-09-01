import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CosReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("cos differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_values = np.asarray(actual, dtype=np.float32)
            expected_values = expected.detach().cpu().numpy()
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )
            shared_zeros = (actual_values == 0) & (expected_values == 0)
            np.testing.assert_array_equal(
                np.signbit(actual_values[shared_zeros]),
                np.signbit(expected_values[shared_zeros]),
            )

    def make_cases(self, module):
        base = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x3F00_0000,
                0xBF00_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0xD015_02F9,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1]),
            ("contiguous", base),
            ("offset", strided[1]),
            ("strided", strided),
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_cos(module, tensor, form):
        if form == "method":
            return tensor.cos()
        if form == "positional":
            return module.cos(tensor)
        if form == "out none":
            return module.cos(tensor, out=None)
        if form == "alias and out none":
            return module.cos(x=tensor, out=None)
        return module.cos(**{form: tensor})

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(1.5, dtype=module.float32, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf.transpose(0, 2)[1]
        if case == "strided":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown autograd case: {case}")

    def test_scalar_empty_contiguous_offset_strided_and_edge_results_match(self):
        forms = (
            "method",
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for actual_case, expected_case in zip(
            self.make_cases(torch), self.make_cases(reference_torch), strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                actual = self.call_cos(torch, actual_input, form)
                expected = self.call_cos(reference_torch, expected_input, form)
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assert_matches(actual, expected, case=(case, form))

    def test_scalar_empty_offset_and_strided_autograd_match(self):
        forms = (
            "method",
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in ("scalar", "empty", "offset", "strided"):
            for form in forms:
                actual_leaf, actual_input = self.make_autograd_case(torch, case)
                expected_leaf, expected_input = self.make_autograd_case(
                    reference_torch, case
                )
                actual = self.call_cos(torch, actual_input, form)
                expected = self.call_cos(reference_torch, expected_input, form)

                self.assert_matches(actual, expected, case=(case, form, "output"))
                actual.sum().backward()
                expected.sum().backward()
                self.assert_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_vjp_matches_grad_output_times_negative_sine_of_saved_input(self):
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xC000_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0x7F80_0000,
                0xFF80_0000,
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
                0x0000_0000,
                0x7F80_0000,
                0x3F80_0000,
                0xBF80_0000,
            ),
            dtype=np.uint32,
        )
        input_values = input_bits.view(np.float32)
        weight_values = weight_bits.view(np.float32)
        actual_leaf = torch.tensor(memoryview(input_values), requires_grad=True)
        expected_leaf = reference_torch.tensor(input_values, requires_grad=True)
        actual_weights = torch.tensor(memoryview(weight_values))
        expected_weights = reference_torch.tensor(weight_values)

        (actual_leaf.cos() * actual_weights).sum().backward()
        (expected_leaf.cos() * expected_weights).sum().backward()
        expected_formula = expected_weights * -expected_leaf.detach().sin()
        np.testing.assert_array_equal(
            expected_leaf.grad.detach().numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )

    def test_accumulation_no_grad_freed_and_higher_order_boundaries_match(self):
        values = [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]]
        weights = [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_input = actual_leaf.transpose(0, 1)
        expected_input = expected_leaf.transpose(0, 1)

        actual_output = torch.cos(actual_input, out=None)
        expected_output = reference_torch.cos(expected_input, out=None)
        actual_loss = (actual_output * torch.tensor(weights)).sum()
        expected_loss = (expected_output * reference_torch.tensor(weights)).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="first gradient")

        torch.cos(input=actual_input).sum().backward()
        reference_torch.cos(input=expected_input).sum().backward()
        self.assert_matches(
            actual_leaf.grad, expected_leaf.grad, case="accumulated gradient"
        )
        self.assert_error_matches(actual_loss.backward, expected_loss.backward)

        no_grad_actual_leaf = torch.tensor(values, requires_grad=True)
        no_grad_expected_leaf = reference_torch.tensor(values, requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.cos(
                no_grad_actual_leaf.transpose(0, 1), out=None
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.cos(
                no_grad_expected_leaf.transpose(0, 1), out=None
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad output")
        self.assertIsNone(no_grad_actual_leaf.grad)
        self.assertIsNone(no_grad_expected_leaf.grad)
        self.assertTrue(torch.cos(no_grad_actual_leaf).requires_grad)
        self.assertTrue(reference_torch.cos(no_grad_expected_leaf).requires_grad)

        higher_leaf = torch.tensor([0.5], requires_grad=True)
        higher_loss = higher_leaf.cos().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_loss.backward(create_graph=True)
        self.assertIsNone(higher_leaf.grad)

    @staticmethod
    def callable_contract(module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "cos")
        bound = tensor.cos
        function = module.cos
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "function_type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.cos is function,
            "descriptor_doc_signature": descriptor.__doc__.splitlines()[:2],
            "function_doc_signature": function.__doc__.splitlines()[:2],
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "function_text_signature": function.__text_signature__,
            "signature_error": signature_error,
            "all_count": module.__all__.count("cos"),
            "wildcard_identity": wildcard_namespace["cos"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([0.5], dtype=module.float32, requires_grad=True)
destination = module.tensor([0.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "cos")
function = module.cos
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result=marker):
        self.calls = []
        self.result = result

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

method_mode = RecordingMode()
with method_mode:
    method_intercepted = tensor.cos()
method_func, method_types, method_args, method_kwargs = method_mode.calls[0]

function_mode = RecordingMode()
with function_mode:
    function_intercepted = function(input=tensor, out=destination)
function_func, function_types, function_args, function_kwargs = function_mode.calls[0]

forwarding_events = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.events.append(
            [self.label, len(module.overrides._get_current_function_mode_stack())]
        )
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower", forwarding_events):
    with ForwardingMode("upper", forwarding_events):
        forwarded = function(input=tensor, out=None)
forwarded.sum().backward()

extreme = module.zeros((0,), dtype=module.float32).reshape((0, sys.maxsize, 3))
bypass = RecordingMode()
with bypass:
    bypassed = extreme.cos()

class Override:
    calls = []

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        cls.calls.append((func, types, args, kwargs))
        return marker

override_result = function(Override())
override_func, override_types, override_args, override_kwargs = Override.calls[0]

print(json.dumps({
    "method_intercepted": method_intercepted is marker,
    "method_call_count": len(method_mode.calls),
    "method_function_is_descriptor": method_func is descriptor,
    "method_types": method_types == (module.Tensor,),
    "method_arg_count": len(method_args),
    "method_kwargs_is_none": method_kwargs is None,
    "function_intercepted": function_intercepted is marker,
    "function_call_count": len(function_mode.calls),
    "function_is_builtin": function_func is function,
    "function_types_empty": function_types == (),
    "function_arg_count": len(function_args),
    "function_kwargs": tuple(function_kwargs),
    "forwarding_events": forwarding_events,
    "forwarded": forwarded.tolist(),
    "gradient": tuple(float(value) for value in tensor.grad.tolist()),
    "bypassed": bypassed is marker,
    "bypass_calls": len(bypass.calls),
    "override_result": override_result is marker,
    "override_function": override_func is function,
    "override_types": tuple(item.__name__ for item in override_types),
    "override_arg_count": len(override_args),
    "override_kwargs_is_none": override_kwargs is None,
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}, sort_keys=True))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_torch_function_mode_and_override_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_unsupported_operands_out_inplace_and_extension_keywords_match(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        destination = torch.tensor([17.0])
        cases = (
            (lambda: torch.cos(), lambda: reference_torch.cos()),
            (
                lambda: torch.cos(actual, actual),
                lambda: reference_torch.cos(expected, expected),
            ),
            (
                lambda: torch.cos(actual, input=actual),
                lambda: reference_torch.cos(expected, input=expected),
            ),
            (lambda: torch.cos(out=actual), lambda: reference_torch.cos(out=expected)),
            (
                lambda: torch.cos(1, extra=True),
                lambda: reference_torch.cos(1, extra=True),
            ),
            (lambda: torch.cos(input=[]), lambda: reference_torch.cos(input=[])),
            (
                lambda: torch.cos(actual, out=[]),
                lambda: reference_torch.cos(expected, out=[]),
            ),
            (
                lambda: torch.cos(actual, extra=True),
                lambda: reference_torch.cos(expected, extra=True),
            ),
            (
                lambda: torch.cos(input=actual, a=actual),
                lambda: reference_torch.cos(input=expected, a=expected),
            ),
            (
                lambda: torch.cos(a=actual, x=actual, out=None),
                lambda: reference_torch.cos(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.cos(x=actual, a=actual, out=None),
                lambda: reference_torch.cos(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.cos(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.cos(np.zeros((2, 3), dtype=np.float32)),
            ),
            (
                lambda: torch.cos(actual, dtype=torch.float32),
                lambda: reference_torch.cos(expected, dtype=reference_torch.float32),
            ),
            (
                lambda: torch.cos(actual, device=torch.device("cpu")),
                lambda: reference_torch.cos(
                    expected, device=reference_torch.device("cpu")
                ),
            ),
            (
                lambda: actual.cos(out=None),
                lambda: expected.cos(out=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        with self.assertRaisesRegex(
            RuntimeError, r"^cos\(\): the 'out' argument is not supported$"
        ):
            torch.cos(actual, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        self.assertFalse(hasattr(torch.Tensor, "cos_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "cos_"))
        self.assertFalse(hasattr(torch, "cos_"))


if __name__ == "__main__":
    unittest.main()
