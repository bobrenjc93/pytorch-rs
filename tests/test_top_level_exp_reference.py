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
class TopLevelExpReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.exp differentials require pinned PyTorch 2.13.0")

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
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float32),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )

    def make_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0xC2C8_0000,
                0xC2D0_0000,
                0x42B0_0000,
                0x42B2_0000,
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
    def call_exp(module, tensor, form):
        if form == "positional":
            return module.exp(tensor)
        if form == "out none":
            return module.exp(tensor, out=None)
        if form == "alias and out none":
            return module.exp(x=tensor, out=None)
        return module.exp(**{form: tensor})

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(1.5, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1], None

        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "contiguous":
            weights = module.tensor(values.tolist(), dtype=module.float32)
            return leaf, leaf, weights
        if case == "offset":
            source = leaf[1]
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 12, dtype=np.float32)
                .reshape(3, 4)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        if case == "strided":
            source = leaf.transpose(0, 2)
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 24, dtype=np.float32)
                .reshape(4, 3, 2)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        raise AssertionError(f"unknown torch.exp autograd case: {case}")

    def test_scalar_empty_offset_strided_and_edge_results_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = ("positional", "input", "x", "a", "x1", "out none", "alias and out none")
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_matches(
                    self.call_exp(torch, actual_input, form),
                    self.call_exp(reference_torch, expected_input, form),
                    case=(case, form),
                )

    def test_scalar_empty_contiguous_offset_and_strided_autograd_match_pytorch_2_13(
        self,
    ):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in ("scalar", "empty", "contiguous", "offset", "strided"):
            for form in forms:
                actual_leaf, actual_input, actual_weights = self.make_autograd_case(
                    torch, case
                )
                expected_leaf, expected_input, expected_weights = (
                    self.make_autograd_case(reference_torch, case)
                )
                actual_output = self.call_exp(torch, actual_input, form)
                expected_output = self.call_exp(
                    reference_torch, expected_input, form
                )
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=(case, form, "forward"),
                )

                if actual_weights is None:
                    actual_loss = (
                        actual_output if case == "scalar" else actual_output.sum()
                    )
                    expected_loss = (
                        expected_output if case == "scalar" else expected_output.sum()
                    )
                else:
                    actual_loss = (actual_output * actual_weights).sum()
                    expected_loss = (expected_output * expected_weights).sum()
                actual_loss.backward()
                expected_loss.backward()
                self.assert_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_composition_accumulation_repeated_backward_and_no_grad_match(self):
        snapshots = []
        for module in (torch, reference_torch):
            composed = module.tensor(
                [-1.0, 0.5, 2.0], dtype=module.float32, requires_grad=True
            )
            module.exp(composed.sin(), out=None).sum().backward()
            composed_gradient = np.asarray(composed.grad, dtype=np.float32).copy()

            accumulated = module.tensor(
                [-1.0, 0.0, 1.0, 4.0],
                dtype=module.float32,
                requires_grad=True,
            )
            module.exp(accumulated, out=None).sum().backward()
            first = np.asarray(accumulated.grad, dtype=np.float32).copy()
            module.exp(input=accumulated).sum().backward()
            second = np.asarray(accumulated.grad, dtype=np.float32).copy()

            freed = module.tensor(
                [-1.0, 0.0, 1.0], dtype=module.float32, requires_grad=True
            )
            loss = module.exp(freed, out=None).sum()
            loss.backward()
            try:
                loss.backward()
            except Exception as error:
                repeated_backward = type(error).__name__, str(error)
            else:
                self.fail("torch.exp graph unexpectedly supported repeated backward")
            snapshots.append((composed_gradient, first, second, repeated_backward))

        for index in range(3):
            np.testing.assert_allclose(
                snapshots[0][index],
                snapshots[1][index],
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
            )
        self.assertEqual(snapshots[0][3], snapshots[1][3])

        actual_leaf = torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]
        with torch.no_grad():
            actual = torch.exp(actual_input, out=None)
        with reference_torch.no_grad():
            expected = reference_torch.exp(expected_input, out=None)
        self.assert_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_leaf.grad)
        self.assertTrue(torch.exp(actual_leaf, out=None).requires_grad)

    def callable_contract(self, module):
        function = module.exp
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
            "owner_callable_identity": owner.exp is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("exp"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["exp"] is function,
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
        tensor = module.tensor([1.0])
        destination = module.tensor([0.0])
        function = module.exp
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
            tuple(np.asarray(forwarded).reshape(-1)),
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
            lambda: torch.exp(Override()),
            lambda: reference_torch.exp(Override()),
        )
        self.assert_error_matches(
            lambda: torch.exp(torch.tensor([1.0]), out=Override()),
            lambda: reference_torch.exp(
                reference_torch.tensor([1.0]), out=Override()
            ),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.exp(), lambda: reference_torch.exp()),
            (
                lambda: torch.exp(actual, actual),
                lambda: reference_torch.exp(expected, expected),
            ),
            (
                lambda: torch.exp(actual, input=actual),
                lambda: reference_torch.exp(expected, input=expected),
            ),
            (
                lambda: torch.exp(out=actual),
                lambda: reference_torch.exp(out=expected),
            ),
            (
                lambda: torch.exp(1, extra=True),
                lambda: reference_torch.exp(1, extra=True),
            ),
            (lambda: torch.exp(input=[]), lambda: reference_torch.exp(input=[])),
            (
                lambda: torch.exp(actual, out=[]),
                lambda: reference_torch.exp(expected, out=[]),
            ),
            (
                lambda: torch.exp(actual, extra=True, out=[]),
                lambda: reference_torch.exp(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.exp(actual, extra=True),
                lambda: reference_torch.exp(expected, extra=True),
            ),
            (
                lambda: torch.exp(input=actual, a=actual),
                lambda: reference_torch.exp(input=expected, a=expected),
            ),
            (
                lambda: torch.exp(a=actual, x=actual, out=None),
                lambda: reference_torch.exp(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.exp(x=actual, a=actual, out=None),
                lambda: reference_torch.exp(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.exp(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.exp(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
