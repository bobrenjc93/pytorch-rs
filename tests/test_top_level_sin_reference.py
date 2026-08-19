import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelSinReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.sin differentials require pinned PyTorch 2.13.0")

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
    def call_sin(module, tensor, form):
        if form == "positional":
            return module.sin(tensor)
        if form == "out none":
            return module.sin(tensor, out=None)
        if form == "alias and out none":
            return module.sin(x=tensor, out=None)
        return module.sin(**{form: tensor})

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
                    self.call_sin(torch, actual_input, form),
                    self.call_sin(reference_torch, expected_input, form),
                    case=(case, form),
                )

    def test_scalar_empty_offset_and_strided_autograd_match_pytorch_2_13(self):
        forms = (
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
                actual = self.call_sin(torch, actual_input, form)
                expected = self.call_sin(reference_torch, expected_input, form)

                self.assert_matches(actual, expected, case=(case, form, "output"))
                actual.sum().backward()
                expected.sum().backward()
                self.assert_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_accumulation_no_grad_and_freed_graph_match_pytorch_2_13(self):
        values = [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]]
        weights = [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_input = actual_leaf.transpose(0, 1)
        expected_input = expected_leaf.transpose(0, 1)

        actual_output = torch.sin(actual_input, out=None)
        expected_output = reference_torch.sin(expected_input, out=None)
        self.assert_matches(actual_output, expected_output, case="tracked output")
        actual_loss = (actual_output * torch.tensor(weights)).sum()
        expected_loss = (
            expected_output * reference_torch.tensor(weights)
        ).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_matches(
            actual_leaf.grad, expected_leaf.grad, case="first gradient"
        )

        torch.sin(input=actual_input).sum().backward()
        reference_torch.sin(input=expected_input).sum().backward()
        self.assert_matches(
            actual_leaf.grad, expected_leaf.grad, case="accumulated gradient"
        )
        self.assert_error_matches(actual_loss.backward, expected_loss.backward)

        no_grad_actual_leaf = torch.tensor(values, requires_grad=True)
        no_grad_expected_leaf = reference_torch.tensor(values, requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.sin(
                no_grad_actual_leaf.transpose(0, 1), out=None
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sin(
                no_grad_expected_leaf.transpose(0, 1), out=None
            )
        self.assert_matches(
            actual_untracked, expected_untracked, case="no_grad output"
        )
        self.assertIsNone(no_grad_actual_leaf.grad)
        self.assertIsNone(no_grad_expected_leaf.grad)
        self.assertTrue(torch.sin(no_grad_actual_leaf).requires_grad)
        self.assertTrue(reference_torch.sin(no_grad_expected_leaf).requires_grad)

        actual_detached = no_grad_actual_leaf.detach().transpose(0, 1)
        expected_detached = no_grad_expected_leaf.detach().transpose(0, 1)
        self.assert_matches(
            torch.sin(actual_detached),
            reference_torch.sin(expected_detached),
            case="detached input",
        )

        boundary_actual_leaf = torch.tensor(values, requires_grad=True)
        boundary_expected_leaf = reference_torch.tensor(values, requires_grad=True)
        with torch.no_grad():
            boundary_actual_input = boundary_actual_leaf.transpose(0, 1)
        with reference_torch.no_grad():
            boundary_expected_input = boundary_expected_leaf.transpose(0, 1)
        boundary_actual_loss = torch.sin(boundary_actual_input).sum()
        boundary_expected_loss = reference_torch.sin(
            boundary_expected_input
        ).sum()
        boundary_actual_loss.backward()
        boundary_expected_loss.backward()
        self.assertIsNone(boundary_actual_leaf.grad)
        self.assertIsNone(boundary_expected_leaf.grad)
        self.assert_error_matches(
            boundary_actual_loss.backward, boundary_expected_loss.backward
        )

    def callable_contract(self, module):
        function = module.sin
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
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.sin is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("sin"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sin"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_ownership_documentation_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def dispatch_observation(self, module):
        tensor = module.tensor([0.0], requires_grad=True)
        destination = module.tensor([0.0])
        function = module.sin
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
        )
        for call, keyword_names in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(input=value), "input"),
            (lambda value: function(tensor, out=value), "out"),
            (lambda value: function(x=value, out=None), "x"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), out=DerivedOverride())

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)
        forwarded.sum().backward()
        forwarded_observation = (
            forwarded.requires_grad,
            forwarded.is_leaf,
            tuple(forwarded.shape),
            forwarded.stride(),
            forwarded.storage_offset(),
            tuple(np.asarray(forwarded.detach()).reshape(-1)),
            tuple(np.asarray(tensor.grad).reshape(-1)),
        )

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function(),
            lambda: function([], out=destination),
            lambda: function(tensor, out=[]),
            lambda: function(tensor, extra=True),
            lambda: function(tensor, tensor),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            forwarded_observation,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_declining_override_diagnostic_matches_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.sin(Override()),
            lambda: reference_torch.sin(Override()),
        )
        self.assert_error_matches(
            lambda: torch.sin(torch.tensor([1.0]), out=Override()),
            lambda: reference_torch.sin(
                reference_torch.tensor([1.0]), out=Override()
            ),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.sin(), lambda: reference_torch.sin()),
            (
                lambda: torch.sin(actual, actual),
                lambda: reference_torch.sin(expected, expected),
            ),
            (
                lambda: torch.sin(actual, input=actual),
                lambda: reference_torch.sin(expected, input=expected),
            ),
            (
                lambda: torch.sin(out=actual),
                lambda: reference_torch.sin(out=expected),
            ),
            (
                lambda: torch.sin(1, extra=True),
                lambda: reference_torch.sin(1, extra=True),
            ),
            (lambda: torch.sin(input=[]), lambda: reference_torch.sin(input=[])),
            (
                lambda: torch.sin(actual, out=[]),
                lambda: reference_torch.sin(expected, out=[]),
            ),
            (
                lambda: torch.sin(actual, extra=True, out=[]),
                lambda: reference_torch.sin(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.sin(actual, extra=True),
                lambda: reference_torch.sin(expected, extra=True),
            ),
            (
                lambda: torch.sin(input=actual, a=actual),
                lambda: reference_torch.sin(input=expected, a=expected),
            ),
            (
                lambda: torch.sin(a=actual, x=actual, out=None),
                lambda: reference_torch.sin(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.sin(x=actual, a=actual, out=None),
                lambda: reference_torch.sin(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.sin(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.sin(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
