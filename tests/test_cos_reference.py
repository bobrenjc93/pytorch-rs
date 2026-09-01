import inspect
import pickle
import re
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
            raise AssertionError("torch.cos differentials require pinned PyTorch 2.13.0")

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

    @staticmethod
    def make_cases(module):
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
                0x3F80_0000,
                0xC000_0000,
                0x4049_0FDB,
                0x5015_02F9,
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
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
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

    def test_scalar_empty_offset_strided_and_edge_results_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
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
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_matches(
                    self.call_cos(torch, actual_input, form),
                    self.call_cos(reference_torch, expected_input, form),
                    case=(case, form),
                )

    def test_scalar_empty_offset_and_strided_autograd_match_pytorch_2_13(self):
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
        expected_formula = expected_weights * (-expected_leaf.detach().sin())
        np.testing.assert_array_equal(
            expected_leaf.grad.detach().numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )

    def test_vjp_nan_grad_outputs_match_pytorch_2_13(self):
        edge_input_bits = np.asarray(
            (
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        nan_weight_bits = np.asarray(
            (
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        input_bits = np.repeat(edge_input_bits, len(nan_weight_bits))
        weight_bits = np.tile(nan_weight_bits, len(edge_input_bits))
        input_values = input_bits.view(np.float32)
        weight_values = weight_bits.view(np.float32)
        actual_leaf = torch.tensor(memoryview(input_values), requires_grad=True)
        expected_leaf = reference_torch.tensor(input_values, requires_grad=True)
        actual_weights = torch.tensor(memoryview(weight_values))
        expected_weights = reference_torch.tensor(weight_values)

        (actual_leaf.cos() * actual_weights).sum().backward()
        (expected_leaf.cos() * expected_weights).sum().backward()
        expected_formula = expected_weights * (-expected_leaf.detach().sin())
        np.testing.assert_array_equal(
            expected_leaf.grad.detach().numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_leaf.grad.detach().numpy().view(np.uint32),
        )

    def test_detach_no_grad_freed_graph_and_higher_order_match_pytorch_2_13(self):
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)

        actual_detached_input = actual_leaf.detach().transpose(0, 1).cos()
        expected_detached_input = expected_leaf.detach().transpose(0, 1).cos()
        self.assert_matches(
            actual_detached_input,
            expected_detached_input,
            case="detached input",
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.transpose(0, 1).cos()
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.transpose(0, 1).cos()
        self.assert_matches(
            actual_untracked,
            expected_untracked,
            case="no_grad output",
        )

        actual_loss = actual_leaf.transpose(0, 1).cos().sum()
        expected_loss = expected_leaf.transpose(0, 1).cos().sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="gradient")
        self.assert_error_matches(actual_loss.backward, expected_loss.backward)

        with self.assertRaises(NotImplementedError):
            torch.tensor([0.5], requires_grad=True).cos().sum().backward(
                create_graph=True
            )

    def callable_contract(self, module):
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
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "method_doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "function_type": type(function).__name__,
            "function_is_builtin": type(function) is types.BuiltinFunctionType,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__,
            "function_doc": function.__doc__,
            "function_text_signature": function.__text_signature__,
            "function_repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "function_signature_error": signature_error,
            "variable_owner_name": owner.__name__,
            "variable_owner_qualname": owner.__qualname__,
            "variable_owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "variable_owner_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.cos is function,
            "all_count": module.__all__.count("cos"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["cos"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def dispatch_observation(self, module):
        tensor = module.tensor([0.5], requires_grad=True)
        destination = module.tensor([0.0])
        descriptor = inspect.getattr_static(module.Tensor, "cos")
        function = module.cos
        marker = object()
        observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for call in (lambda: tensor.cos(), lambda: function(input=tensor, out=None)):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    func is descriptor or func is function,
                    dispatch_types == (module.Tensor,)
                    if func is descriptor
                    else dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
            )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        override_result = function(Override())
        out_override_result = function(module.tensor([1.0]), out=Override())

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label, events):
                self.label = label
                self.events = events

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.events.append(
                    [
                        self.label,
                        len(module.overrides._get_current_function_mode_stack()),
                    ]
                )
                return func(*args, **(kwargs or {}))

        class RaisingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0
                self.handler_stack_depth = None

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                self.handler_stack_depth = len(
                    module.overrides._get_current_function_mode_stack()
                )
                raise ValueError("cos mode failed")

        recovery_events = []
        raising = RaisingMode()
        with ForwardingMode("lower", recovery_events):
            try:
                with raising:
                    tensor.cos()
            except Exception as error:
                raising_error = (type(error).__name__, str(error))
                raising_stack_inside_lower = len(
                    module.overrides._get_current_function_mode_stack()
                )
            else:
                raising_error = None
                raising_stack_inside_lower = None
            recovered = tensor.cos()

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                try:
                    tensor.cos()
                except Exception as error:
                    declining_error = (type(error).__name__, str(error))
                    declining_stack_inside = len(
                        module.overrides._get_current_function_mode_stack()
                    )
                else:
                    declining_error = None
                    declining_stack_inside = None
        finally:
            sys.setrecursionlimit(old_recursion_limit)

        return (
            observations,
            override_result is marker,
            out_override_result is marker,
            tuple(
                (
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                )
                for func, dispatch_types, _, kwargs in override_calls
            ),
            (
                declining_error,
                len(declining.calls),
                declining_stack_inside,
            ),
            (
                raising_error,
                raising.calls,
                raising.handler_stack_depth,
                raising_stack_inside_lower,
                recovery_events,
                recovered.tolist(),
                len(module.overrides._get_current_function_mode_stack()),
            ),
        )

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_declining_override_and_binding_errors_match_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.cos(Override()),
            lambda: reference_torch.cos(Override()),
        )
        self.assert_error_matches(
            lambda: torch.cos(torch.tensor([1.0]), out=Override()),
            lambda: reference_torch.cos(
                reference_torch.tensor([1.0]), out=Override()
            ),
        )

        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
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
            (
                lambda: torch.cos(out=actual),
                lambda: reference_torch.cos(out=expected),
            ),
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
                lambda: torch.cos(actual, dtype=torch.float32),
                lambda: reference_torch.cos(expected, dtype=reference_torch.float32),
            ),
            (
                lambda: torch.cos(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.cos(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
