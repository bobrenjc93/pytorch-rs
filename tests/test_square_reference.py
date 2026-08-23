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
class TensorSquareReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.square differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    @staticmethod
    def tensor_cases(module):
        base = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
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
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            module.tensor(memoryview(special_bits.view(np.float32))),
        )

    @staticmethod
    def call_top_level_square(module, tensor, form):
        if form == "positional":
            return module.square(tensor)
        if form == "out none":
            return module.square(tensor, out=None)
        if form == "alias and out none":
            return module.square(x=tensor, out=None)
        return module.square(**{form: tensor})

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(-3.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1], None

        leaf = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            source = leaf[1]
            weights = module.tensor(
                np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)
            weights = module.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2).tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        raise AssertionError(f"unknown square autograd case: {case}")

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            actual_output = actual.square()
            expected_output = expected.square()
            self.assert_tensor_matches(actual_output, expected_output, case=case)
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))

    def test_top_level_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case, (actual_input, expected_input) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for form in forms:
                actual = self.call_top_level_square(torch, actual_input, form)
                expected = self.call_top_level_square(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))

    def test_seeded_random_values_match_pytorch_2_13_bitwise(self):
        rng = np.random.default_rng(0x5A_A2E)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(16):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 8, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-1.0e20, 1.0e20, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
            values = values.reshape(shape)
            actual_input = (
                torch.zeros(shape, dtype=torch.float32)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            self.assert_tensor_matches(
                actual_input.square(), expected_input.square(), case=(case, shape)
            )

    def test_autograd_scalar_empty_offset_and_noncontiguous_match_pytorch_2_13(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_leaf, actual_input, actual_weights = self.autograd_case(torch, case)
            expected_leaf, expected_input, expected_weights = self.autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.square()
            expected_output = expected_input.square()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "forward")
            )

            if actual_weights is None:
                actual_loss = actual_output if case == "scalar" else actual_output.sum()
                expected_loss = (
                    expected_output if case == "scalar" else expected_output.sum()
                )
            else:
                actual_loss = (actual_output * actual_weights).sum()
                expected_loss = (expected_output * expected_weights).sum()
            actual_loss.backward()
            expected_loss.backward()
            self.assert_tensor_matches(
                actual_leaf.grad, expected_leaf.grad, case=(case, "gradient")
            )

    def test_top_level_autograd_and_no_grad_match_pytorch_2_13(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input, actual_weights = self.autograd_case(
                    torch, case
                )
                expected_leaf, expected_input, expected_weights = self.autograd_case(
                    reference_torch, case
                )
                actual = self.call_top_level_square(torch, actual_input, form)
                expected = self.call_top_level_square(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(
                    actual, expected, case=(case, form, "output")
                )

                if actual_weights is None:
                    actual_loss = actual if case == "scalar" else actual.sum()
                    expected_loss = expected if case == "scalar" else expected.sum()
                else:
                    actual_loss = (actual * actual_weights).sum()
                    expected_loss = (expected * expected_weights).sum()
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )

        for case in ("scalar", "empty", "offset", "noncontiguous"):
            _, actual_input, _ = self.autograd_case(torch, case)
            _, expected_input, _ = self.autograd_case(reference_torch, case)
            with torch.no_grad():
                actual = torch.square(actual_input, out=None)
            with reference_torch.no_grad():
                expected = reference_torch.square(expected_input, out=None)
            self.assert_tensor_matches(
                actual, expected, case=(case, "top-level no_grad")
            )
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))

    def test_pow_backward_overflow_subnormal_and_nonfinite_bits_match_pytorch_2_13(self):
        input_bits = np.asarray(
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
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x3F00_0000,
                0x3F00_0000,
                0x0000_0001,
                0x0000_0001,
                0x0000_0000,
                0x8000_0000,
                0x3E80_0000,
                0x3E80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        results = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            output = leaf.square()
            (output * weights).sum().backward()
            results.append((output, leaf.grad))

        self.assert_tensor_matches(
            results[0][0], results[1][0], case="special forward"
        )
        self.assert_tensor_matches(
            results[0][1], results[1][1], case="special gradient"
        )

    def test_seeded_finite_pow_backward_bits_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x50_A2E)
        input_bits = rng.integers(0, 1 << 32, size=100_000, dtype=np.uint32)
        weight_bits = rng.integers(0, 1 << 32, size=100_000, dtype=np.uint32)
        for bits in (input_bits, weight_bits):
            nonfinite = bits & np.uint32(0x7F80_0000) == np.uint32(0x7F80_0000)
            bits[nonfinite] ^= np.uint32(0x0080_0000)

        input_bits[:6] = np.asarray(
            (
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
            ),
            dtype=np.uint32,
        )
        weight_bits[:6] = np.asarray(
            (
                0x3E80_0000,
                0x3E80_0000,
                0x3F00_0000,
                0x3F00_0000,
                0x0000_0001,
                0x0000_0001,
            ),
            dtype=np.uint32,
        )

        gradients = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            (leaf.square() * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).view(np.uint32).copy())

        np.testing.assert_array_equal(gradients[0], gradients[1])

    def test_pow_backward_node_identity_matches_pytorch_2_13(self):
        errors = []
        for module in (torch, reference_torch):
            probability = module.tensor(
                [2.0], dtype=module.float32, requires_grad=True
            ).square()
            try:
                module.nn.functional.dropout(
                    module.tensor([1.0], dtype=module.float32),
                    p=probability,
                    training=False,
                )
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail("dropout unexpectedly accepted a squared tensor probability")

        self.assertEqual(errors[0], errors[1])
        self.assertIn("grad_fn=<PowBackward0>", errors[0][1])

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.square unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([2.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "square")
        bound = tensor.square
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.square(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.square(1, 2),
                    lambda: tensor.square(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def top_level_callable_contract(self, module):
        function = module.square
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
            "owner_callable_identity": owner.square is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("square"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["square"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    def test_repeated_backward_graph_freeing_and_no_grad_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            accumulated = module.tensor(
                [2.0, -3.0], dtype=module.float32, requires_grad=True
            )
            accumulated.square().sum().backward()
            first = np.asarray(accumulated.grad).copy()
            accumulated.square().sum().backward()
            second = np.asarray(accumulated.grad).copy()

            freed = module.tensor(
                [2.0, -3.0], dtype=module.float32, requires_grad=True
            )
            loss = freed.square().sum()
            loss.backward()
            second_backward_error = self.error(loss.backward)
            snapshots.append((first, second, second_backward_error))

        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
        self.assertEqual(snapshots[0][2], snapshots[1][2])

        for case in ("scalar", "empty", "offset", "noncontiguous"):
            _, actual_input, _ = self.autograd_case(torch, case)
            _, expected_input, _ = self.autograd_case(reference_torch, case)
            with torch.no_grad():
                actual_output = actual_input.square()
            with reference_torch.no_grad():
                expected_output = expected_input.square()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "no_grad")
            )
            self.assertFalse(actual_output.is_set_to(actual_input))
            self.assertFalse(expected_output.is_set_to(expected_input))

    def top_level_dispatch_observation(self, module):
        tensor = module.tensor([2.0, -3.0], requires_grad=True)
        destination = module.tensor([0.0, 0.0])
        function = module.square
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        for call, keyword_names in (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
        ):
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

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        override_observations = []
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

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
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

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forwarding_order,
            forwarded_observation,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
        )

    def test_top_level_modes_and_subclass_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_dispatch_observation(torch),
            self.top_level_dispatch_observation(reference_torch),
        )

    def test_top_level_declining_override_diagnostic_matches_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assert_error_matches(
            lambda: torch.square(Override()),
            lambda: reference_torch.square(Override()),
        )
        self.assert_error_matches(
            lambda: torch.square(torch.tensor([2.0]), out=Override()),
            lambda: reference_torch.square(
                reference_torch.tensor([2.0]), out=Override()
            ),
        )

    def test_top_level_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([2.0])
        expected = reference_torch.tensor([2.0])
        cases = (
            (lambda: torch.square(), lambda: reference_torch.square()),
            (
                lambda: torch.square(actual, actual),
                lambda: reference_torch.square(expected, expected),
            ),
            (
                lambda: torch.square(actual, input=actual),
                lambda: reference_torch.square(expected, input=expected),
            ),
            (
                lambda: torch.square(out=actual),
                lambda: reference_torch.square(out=expected),
            ),
            (
                lambda: torch.square(extra=actual),
                lambda: reference_torch.square(extra=expected),
            ),
            (
                lambda: torch.square(1, extra=True),
                lambda: reference_torch.square(1, extra=True),
            ),
            (
                lambda: torch.square(input=[]),
                lambda: reference_torch.square(input=[]),
            ),
            (
                lambda: torch.square(actual, out=[]),
                lambda: reference_torch.square(expected, out=[]),
            ),
            (
                lambda: torch.square(actual, extra=True, out=[]),
                lambda: reference_torch.square(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.square(actual, extra=True),
                lambda: reference_torch.square(expected, extra=True),
            ),
            (
                lambda: torch.square(input=actual, a=actual),
                lambda: reference_torch.square(input=expected, a=expected),
            ),
            (
                lambda: torch.square(a=actual, x=actual, out=None),
                lambda: reference_torch.square(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.square(x=actual, a=actual, out=None),
                lambda: reference_torch.square(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.square(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.square(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([2.0, -3.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "square")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

recording = RecordingMode(marker)
with recording:
    intercepted = tensor.square()
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = tensor.square()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.square()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.square(1)
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_is_descriptor": function is descriptor,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "invalid_error": invalid_error,
    "invalid_calls": len(invalid.calls),
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

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_unsupported_surface_remains_explicit(self):
        self.assertTrue(hasattr(torch, "square"))
        self.assertTrue(hasattr(reference_torch, "square"))
        self.assertFalse(hasattr(torch.Tensor, "square_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "square_"))
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaises(RuntimeError):
            torch.tensor([2.0], device="cuda")
        self.assertEqual(reference_torch.device("cuda").type, "cuda")


if __name__ == "__main__":
    unittest.main()
