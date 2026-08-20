import importlib
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
class Atleast1dReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "atleast_1d differentials require pinned PyTorch 2.13.0"
            )

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def normalize_error(self, error):
        return (
            str(error)
            .replace("torch_rs.functional", "torch.functional")
            .replace("torch_rs.torch_rs", "torch")
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize_error(actual_raised.exception),
            str(expected_raised.exception),
        )

    def make_layout_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("scalar-offset", base.transpose(0, 2)[3, 2, 1]),
            ("vector-offset", base[1, 2]),
            ("matrix-offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
        )

    def observe_result(self, module, source, result):
        direct = source if len(source.shape) else source.reshape((1,))
        return (
            result is source,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            str(result.layout),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def observe_layout(self, module, source):
        return self.observe_result(module, source, module.atleast_1d(source))

    def test_values_strides_offsets_aliasing_and_metadata_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for (name, actual_source), (expected_name, expected_source) in zip(
            actual_cases, expected_cases, strict=True
        ):
            with self.subTest(case=name):
                self.assertEqual(name, expected_name)
                actual = self.observe_layout(torch, actual_source)
                expected = self.observe_layout(reference_torch, expected_source)
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

    def test_sequence_values_layouts_aliasing_and_empties_match_pytorch_2_13(self):
        for sequence_type in (tuple, list):
            actual_cases = self.make_layout_cases(torch)
            expected_cases = self.make_layout_cases(reference_torch)
            actual_results = torch.atleast_1d(
                sequence_type(source for _, source in actual_cases)
            )
            expected_results = reference_torch.atleast_1d(
                sequence_type(source for _, source in expected_cases)
            )
            with self.subTest(sequence_type=sequence_type.__name__):
                self.assertIs(type(actual_results), tuple)
                self.assertIs(type(expected_results), tuple)
                self.assertEqual(len(actual_results), len(expected_results))

            for (
                (name, actual_source),
                (expected_name, expected_source),
                actual_result,
                expected_result,
            ) in zip(
                actual_cases,
                expected_cases,
                actual_results,
                expected_results,
                strict=True,
            ):
                with self.subTest(
                    sequence_type=sequence_type.__name__, case=name
                ):
                    self.assertEqual(name, expected_name)
                    actual = self.observe_result(
                        torch, actual_source, actual_result
                    )
                    expected = self.observe_result(
                        reference_torch, expected_source, expected_result
                    )
                    self.assertEqual(actual[:-1], expected[:-1])
                    np.testing.assert_array_equal(actual[-1], expected[-1])

        for empty in ((), []):
            actual = torch.atleast_1d(empty)
            expected = reference_torch.atleast_1d(empty)
            with self.subTest(empty_type=type(empty).__name__):
                self.assertIs(type(actual), tuple)
                self.assertIs(type(expected), tuple)
                self.assertEqual(actual, expected)

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [1.0, 2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        source = leaf[1]
        result = module.atleast_1d(source)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source.reshape((1,))),
        )
        loss = result.sum()
        loss.backward()
        loss.backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def no_grad_outcome(self, module):
        scalar = module.tensor(3.0, dtype=module.float32, requires_grad=True)
        vector_leaf = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        vector = vector_leaf * 2.0
        with module.no_grad():
            scalar_result = module.atleast_1d(scalar)
            vector_result = module.atleast_1d(vector)
        (scalar_result * scalar_result).sum().backward()
        return (
            (
                tuple(scalar_result.shape),
                scalar_result.stride(),
                scalar_result.storage_offset(),
                scalar_result.requires_grad,
                scalar_result.is_leaf,
                scalar_result.data_ptr() == scalar.data_ptr(),
                scalar.grad,
                scalar_result.grad,
            ),
            (
                vector_result is vector,
                vector_result.requires_grad,
                vector_result.is_leaf,
            ),
        )

    def sequence_autograd_outcome(self, module, sequence_type):
        leaf = module.tensor(
            [1.0, 2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        scalar = leaf[1]
        results = module.atleast_1d(sequence_type((scalar, leaf)))
        scalar_result, vector_result = results
        metadata = (
            type(results) is tuple,
            tuple(scalar_result.shape),
            scalar_result.stride(),
            scalar_result.storage_offset(),
            scalar_result.requires_grad,
            scalar_result.is_leaf,
            scalar_result.data_ptr() == scalar.data_ptr(),
            scalar_result.is_set_to(scalar.reshape((1,))),
            vector_result is leaf,
        )
        loss = scalar_result.sum()
        loss.backward()
        loss.backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def sequence_no_grad_outcome(self, module, sequence_type):
        scalar = module.tensor(3.0, dtype=module.float32, requires_grad=True)
        vector_leaf = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        vector = vector_leaf * 2.0
        with module.no_grad():
            results = module.atleast_1d(sequence_type((scalar, vector)))
        scalar_result, vector_result = results
        (scalar_result * scalar_result).sum().backward()
        return (
            type(results) is tuple,
            (
                tuple(scalar_result.shape),
                scalar_result.stride(),
                scalar_result.storage_offset(),
                scalar_result.requires_grad,
                scalar_result.is_leaf,
                scalar_result.data_ptr() == scalar.data_ptr(),
                scalar.grad,
                scalar_result.grad,
            ),
            (
                vector_result is vector,
                vector_result.requires_grad,
                vector_result.is_leaf,
            ),
        )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        actual_metadata, actual_grad = self.autograd_outcome(torch)
        expected_metadata, expected_grad = self.autograd_outcome(reference_torch)
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_grad, expected_grad)
        self.assertEqual(
            self.no_grad_outcome(torch),
            self.no_grad_outcome(reference_torch),
        )

    def test_sequence_autograd_and_no_grad_match_pytorch_2_13(self):
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                actual_metadata, actual_grad = self.sequence_autograd_outcome(
                    torch, sequence_type
                )
                expected_metadata, expected_grad = self.sequence_autograd_outcome(
                    reference_torch, sequence_type
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(actual_grad, expected_grad)
                self.assertEqual(
                    self.sequence_no_grad_outcome(torch, sequence_type),
                    self.sequence_no_grad_outcome(
                        reference_torch, sequence_type
                    ),
                )

    def mode_contract(self, module):
        function = module.atleast_1d
        source = module.tensor(2.0, dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            accepting_result = function(source)

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(source)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                function(source)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(
                    r"0x[0-9a-f]+",
                    "0x<address>",
                    self.normalize_error(error),
                ),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        def normalize_call(call):
            func, dispatch_types, args, kwargs = call
            return (
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                args == (source,),
                kwargs,
            )

        return {
            "accepting": (
                accepting_result is marker,
                tuple(map(normalize_call, accepting.calls)),
            ),
            "forwarding": tuple(
                (label, normalize_call((func, types, args, kwargs)))
                for label, func, types, args, kwargs in order
            ),
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.data_ptr() == source.data_ptr(),
            ),
            "declining": declining_error,
            "declining_calls": tuple(map(normalize_call, declining.calls)),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def override_contract(self, module):
        function = module.atleast_1d
        marker = object()
        events = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("override", func, types, args, kwargs))
                return marker

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(("mode", func, types, args, kwargs))
                return self.result

        value = Override()
        direct = function(value)
        direct_events = tuple(events)

        events.clear()
        with RecordingMode(marker):
            accepting = function(value)
        accepting_events = tuple(events)

        events.clear()
        with RecordingMode(NotImplemented):
            declining = function(value)
        declining_events = tuple(events)

        def normalize(events):
            return tuple(
                (
                    label,
                    func is function,
                    tuple(item.__name__ for item in types),
                    args == (value,),
                    kwargs,
                )
                for label, func, types, args, kwargs in events
            )

        return (
            direct is marker,
            normalize(direct_events),
            accepting is marker,
            normalize(accepting_events),
            declining is marker,
            normalize(declining_events),
        )

    def test_operand_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.override_contract(torch),
            self.override_contract(reference_torch),
        )

    def test_metadata_signature_documentation_exports_and_pickle_match(self):
        actual_functional = importlib.import_module("torch_rs.functional")
        expected_functional = importlib.import_module("torch.functional")
        actual = torch.atleast_1d
        expected = reference_torch.atleast_1d

        self.assertIs(actual, actual_functional.atleast_1d)
        self.assertIs(expected, expected_functional.atleast_1d)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            actual_functional.__all__.count("atleast_1d"),
            expected_functional.__all__.count("atleast_1d"),
        )
        self.assertEqual(
            torch.__all__.count("atleast_1d"),
            reference_torch.__all__.count("atleast_1d"),
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )

    def test_single_input_errors_match_pytorch_2_13(self):
        actual_tensor = torch.tensor(1.0)
        expected_tensor = reference_torch.tensor(1.0)
        cases = (
            (lambda: torch.atleast_1d(None), lambda: reference_torch.atleast_1d(None)),
            (lambda: torch.atleast_1d(1), lambda: reference_torch.atleast_1d(1)),
            (
                lambda: torch.atleast_1d(np.zeros((2,), dtype=np.float32)),
                lambda: reference_torch.atleast_1d(
                    np.zeros((2,), dtype=np.float32)
                ),
            ),
            (
                lambda: torch.atleast_1d(input=actual_tensor),
                lambda: reference_torch.atleast_1d(input=expected_tensor),
            ),
            (
                lambda: torch.atleast_1d(tensors=actual_tensor),
                lambda: reference_torch.atleast_1d(tensors=expected_tensor),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_variadic_and_mixed_forms_remain_unsupported(self):
        source = torch.tensor(1.0)
        unsupported = (
            lambda: torch.atleast_1d(source, source),
        )
        for call in unsupported:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError,
                "^atleast_1d\\(\\) only supports a single Tensor input$",
            ):
                call()

        expected = reference_torch.tensor(1.0)
        self.assertEqual(reference_torch.atleast_1d(), ())
        self.assertEqual(len(reference_torch.atleast_1d(expected, expected)), 2)

        sequence_error = (
            "atleast_1d() sequence inputs only support an exact tuple or list "
            "of exact Tensors"
        )
        mixed_sequences = (
            (source, None),
            [source, 1],
            ((source,),),
        )
        for sequence in mixed_sequences:
            with self.subTest(sequence=sequence), self.assertRaisesRegex(
                TypeError, f"^{re.escape(sequence_error)}$"
            ):
                torch.atleast_1d(sequence)

        with self.assertRaises(TypeError):
            reference_torch.atleast_1d((expected, None))

    def test_inner_overrides_remain_unsupported_and_outer_dispatch_matches(self):
        sequence_error = (
            "atleast_1d() sequence inputs only support an exact tuple or list "
            "of exact Tensors"
        )
        actual_source = torch.tensor(1.0)

        class ActualOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        with self.assertRaisesRegex(
            TypeError, f"^{re.escape(sequence_error)}$"
        ):
            torch.atleast_1d((actual_source, ActualOverride()))
        self.assertEqual(ActualOverride.calls, [])

        marker = object()

        class ExpectedOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        expected_source = reference_torch.tensor(1.0)
        self.assertIs(
            reference_torch.atleast_1d(
                (expected_source, ExpectedOverride())
            ),
            marker,
        )
        self.assertEqual(len(ExpectedOverride.calls), 1)

        def outer_override_outcome(module, sequence_type):
            function = module.atleast_1d
            source = module.tensor(1.0, dtype=module.float32)
            outer_marker = object()

            class OuterOverride(sequence_type):
                calls = []

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    cls.calls.append((func, types, args, kwargs))
                    return outer_marker

            sequence = OuterOverride((source,))
            result = function(sequence)
            func, dispatch_types, args, kwargs = OuterOverride.calls[0]
            return (
                result is outer_marker,
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                args == (sequence,),
                kwargs,
            )

        for sequence_type in (tuple, list):
            with self.subTest(outer_override=sequence_type.__name__):
                self.assertEqual(
                    outer_override_outcome(torch, sequence_type),
                    outer_override_outcome(reference_torch, sequence_type),
                )

        def spoofed_sequence_outcome(module):
            function = module.atleast_1d
            spoofed_marker = object()

            class SpoofedSequence:
                calls = []

                @property
                def __class__(self):
                    return tuple

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    cls.calls.append((func, types, args, kwargs))
                    return spoofed_marker

            value = SpoofedSequence()
            result = function(value)
            func, dispatch_types, args, kwargs = SpoofedSequence.calls[0]
            return (
                isinstance(value, tuple),
                result is spoofed_marker,
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                args == (value,),
                kwargs,
            )

        self.assertEqual(
            spoofed_sequence_outcome(torch),
            spoofed_sequence_outcome(reference_torch),
        )

        class ActualMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        actual_mode = ActualMode()
        with actual_mode:
            actual_result = torch.atleast_1d((actual_source,))
        self.assertIs(actual_result, marker)
        self.assertEqual(len(actual_mode.calls), 1)

        class ExpectedMode(reference_torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        expected_mode = ExpectedMode()
        with expected_mode:
            expected_result = reference_torch.atleast_1d((expected_source,))
        self.assertIs(expected_result, marker)
        self.assertEqual(len(expected_mode.calls), 1)

        def normalize_mode_call(call, function, source):
            func, dispatch_types, args, kwargs = call
            return (
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                args == ((source,),),
                kwargs,
            )

        self.assertEqual(
            normalize_mode_call(
                actual_mode.calls[0], torch.atleast_1d, actual_source
            ),
            normalize_mode_call(
                expected_mode.calls[0],
                reference_torch.atleast_1d,
                expected_source,
            ),
        )


if __name__ == "__main__":
    unittest.main()
