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
class Atleast3dReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "atleast_3d differentials require pinned PyTorch 2.13.0"
            )

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def grad_list(self, tensor, module):
        return (
            None
            if tensor.grad is None
            else self.tensor_array(tensor.grad, module).tolist()
        )

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
        empty_strided = (
            module.zeros((2, 0, 3), dtype=module.float32)
            .transpose(0, 2)[1]
            .transpose(0, 1)[1]
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("scalar-offset", base.transpose(0, 2)[3, 2, 1]),
            ("vector-offset", base[1, 2]),
            (
                "vector-strided",
                base.transpose(0, 2)[3].transpose(0, 1)[1],
            ),
            ("empty-vector", module.zeros((0,), dtype=module.float32)),
            ("empty-vector-strided", empty_strided),
            ("matrix-offset", base[1]),
            ("matrix-transposed", base[1].transpose(0, 1)),
            ("matrix-noncontiguous", base.transpose(0, 2)[2]),
            (
                "empty-matrix-leading",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            (
                "empty-matrix-trailing",
                module.zeros((2, 0, 3), dtype=module.float32)[1].transpose(
                    0, 1
                ),
            ),
            ("rank-three", base.transpose(0, 2)),
            (
                "rank-four-empty",
                module.zeros((1, 2, 0, 3), dtype=module.float32),
            ),
        )

    def observe_result(self, module, source, result):
        repeated = module.atleast_3d(source)
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
            result.is_set_to(repeated),
            self.tensor_array(result, module).copy(),
        )

    def observe_layout(self, module, source):
        return self.observe_result(module, source, module.atleast_3d(source))

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
            actual_results = torch.atleast_3d(
                sequence_type(source for _, source in actual_cases)
            )
            expected_results = reference_torch.atleast_3d(
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
            actual = torch.atleast_3d(empty)
            expected = reference_torch.atleast_3d(empty)
            with self.subTest(empty_type=type(empty).__name__):
                self.assertIs(type(actual), tuple)
                self.assertIs(type(expected), tuple)
                self.assertEqual(actual, expected)

    def autograd_outcome(self, module):
        scalar_leaf = module.tensor(
            [1.0, 2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        scalar_source = scalar_leaf[1]
        scalar_result = module.atleast_3d(scalar_source)
        scalar_metadata = (
            tuple(scalar_result.shape),
            scalar_result.stride(),
            scalar_result.storage_offset(),
            scalar_result.requires_grad,
            scalar_result.is_leaf,
            scalar_result.output_nr,
            scalar_result.data_ptr() == scalar_source.data_ptr(),
            scalar_result.is_set_to(scalar_source.reshape((1, 1, 1))),
        )
        scalar_loss = scalar_result.sum()
        scalar_loss.backward()
        scalar_loss.backward()

        vector_leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        vector_source = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
        vector_result = module.atleast_3d(vector_source)
        vector_metadata = (
            tuple(vector_result.shape),
            vector_result.stride(),
            vector_result.storage_offset(),
            vector_result.requires_grad,
            vector_result.is_leaf,
            vector_result.output_nr,
            vector_result.data_ptr() == vector_source.data_ptr(),
            vector_result.is_set_to(module.atleast_3d(vector_source)),
        )
        vector_loss = vector_result.sum()
        vector_loss.backward()
        vector_loss.backward()

        matrix_leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        matrix_source = matrix_leaf[1].transpose(0, 1)
        matrix_result = module.atleast_3d(matrix_source)
        matrix_metadata = (
            tuple(matrix_result.shape),
            matrix_result.stride(),
            matrix_result.storage_offset(),
            matrix_result.requires_grad,
            matrix_result.is_leaf,
            matrix_result.output_nr,
            matrix_result.data_ptr() == matrix_source.data_ptr(),
            matrix_result.is_set_to(module.atleast_3d(matrix_source)),
        )
        matrix_loss = matrix_result.sum()
        matrix_loss.backward()
        matrix_loss.backward()

        empty_vector_leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        empty_vector_source = (
            empty_vector_leaf.transpose(0, 2)[1].transpose(0, 1)[1]
        )
        empty_vector_result = module.atleast_3d(empty_vector_source)
        empty_vector_metadata = (
            tuple(empty_vector_result.shape),
            empty_vector_result.stride(),
            empty_vector_result.storage_offset(),
            empty_vector_result.requires_grad,
            empty_vector_result.is_leaf,
            empty_vector_result.data_ptr() == empty_vector_source.data_ptr(),
        )
        empty_vector_result.sum().backward()

        empty_matrix_leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        empty_matrix_source = empty_matrix_leaf.transpose(0, 2)[1]
        empty_matrix_result = module.atleast_3d(empty_matrix_source)
        empty_matrix_metadata = (
            tuple(empty_matrix_result.shape),
            empty_matrix_result.stride(),
            empty_matrix_result.storage_offset(),
            empty_matrix_result.requires_grad,
            empty_matrix_result.is_leaf,
            empty_matrix_result.data_ptr() == empty_matrix_source.data_ptr(),
        )
        empty_matrix_result.sum().backward()

        return (
            (scalar_metadata, self.grad_list(scalar_leaf, module)),
            (vector_metadata, self.grad_list(vector_leaf, module)),
            (matrix_metadata, self.grad_list(matrix_leaf, module)),
            (
                empty_vector_metadata,
                tuple(empty_vector_leaf.grad.shape),
                self.grad_list(empty_vector_leaf, module),
            ),
            (
                empty_matrix_metadata,
                tuple(empty_matrix_leaf.grad.shape),
                self.grad_list(empty_matrix_leaf, module),
            ),
        )

    def no_grad_outcome(self, module):
        scalar = module.tensor(3.0, dtype=module.float32, requires_grad=True)
        vector = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        matrix = module.tensor(
            [[1.0, 2.0]], dtype=module.float32, requires_grad=True
        )
        rank_three_leaf = module.zeros(
            (1, 2, 3), dtype=module.float32, requires_grad=True
        )
        rank_three = rank_three_leaf * 2.0
        with module.no_grad():
            scalar_result = module.atleast_3d(scalar)
            vector_result = module.atleast_3d(vector)
            matrix_result = module.atleast_3d(matrix)
            rank_three_result = module.atleast_3d(rank_three)

        view_outcomes = []
        for result, source in (
            (scalar_result, scalar),
            (vector_result, vector),
            (matrix_result, matrix),
        ):
            (result * result).sum().backward()
            view_outcomes.append(
                (
                    tuple(result.shape),
                    result.stride(),
                    result.storage_offset(),
                    result.requires_grad,
                    result.is_leaf,
                    result.output_nr,
                    result.data_ptr() == source.data_ptr(),
                    self.grad_list(source, module),
                    self.grad_list(result, module),
                )
            )

        rank_three_result.sum().backward()
        return (
            tuple(view_outcomes),
            (
                rank_three_result is rank_three,
                rank_three_result.requires_grad,
                rank_three_result.is_leaf,
                rank_three_result.output_nr,
                self.grad_list(rank_three_leaf, module),
            ),
        )

    def sequence_autograd_outcome(self, module, sequence_type):
        scalar_leaf = module.tensor(
            [1.0, 2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        scalar = scalar_leaf[1]
        vector_leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        vector = vector_leaf.transpose(0, 2)[3].transpose(0, 1)[1]
        matrix_leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        matrix = matrix_leaf[1].transpose(0, 1)
        empty_vector_leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        empty_vector = (
            empty_vector_leaf.transpose(0, 2)[1].transpose(0, 1)[1]
        )
        empty_matrix_leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        empty_matrix = empty_matrix_leaf.transpose(0, 2)[1]
        rank_three_leaf = module.zeros(
            (1, 2, 3), dtype=module.float32, requires_grad=True
        )
        rank_three = rank_three_leaf * 2.0

        sources = (
            scalar,
            vector,
            matrix,
            empty_vector,
            empty_matrix,
            rank_three,
        )
        results = module.atleast_3d(sequence_type(sources))
        metadata = (
            type(results) is tuple,
            tuple(
                (
                    result is source,
                    tuple(result.shape),
                    result.stride(),
                    result.storage_offset(),
                    result.requires_grad,
                    result.is_leaf,
                    result.output_nr,
                    result.data_ptr() == source.data_ptr(),
                    result.is_set_to(module.atleast_3d(source)),
                )
                for source, result in zip(sources, results, strict=True)
            ),
        )

        for index, result in enumerate(results):
            loss = result.sum()
            loss.backward()
            if index < 3:
                loss.backward()

        return (
            metadata,
            self.grad_list(scalar_leaf, module),
            self.grad_list(vector_leaf, module),
            self.grad_list(matrix_leaf, module),
            self.grad_list(empty_vector_leaf, module),
            self.grad_list(empty_matrix_leaf, module),
            self.grad_list(rank_three_leaf, module),
        )

    def sequence_no_grad_outcome(self, module, sequence_type):
        scalar = module.tensor(
            3.0, dtype=module.float32, requires_grad=True
        )
        vector = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        matrix = module.tensor(
            [[1.0, 2.0]], dtype=module.float32, requires_grad=True
        )
        rank_three_leaf = module.zeros(
            (1, 2, 3), dtype=module.float32, requires_grad=True
        )
        rank_three = rank_three_leaf * 2.0
        sources = (scalar, vector, matrix, rank_three)
        with module.no_grad():
            results = module.atleast_3d(sequence_type(sources))

        for result in results[:3]:
            (result * result).sum().backward()
        results[3].sum().backward()
        return (
            type(results) is tuple,
            tuple(
                (
                    result is source,
                    tuple(result.shape),
                    result.stride(),
                    result.storage_offset(),
                    result.requires_grad,
                    result.is_leaf,
                    result.output_nr,
                    result.data_ptr() == source.data_ptr(),
                    None if index == 3 else self.grad_list(source, module),
                    None if index == 3 else self.grad_list(result, module),
                )
                for index, (source, result) in enumerate(
                    zip(sources, results, strict=True)
                )
            ),
            self.grad_list(rank_three_leaf, module),
        )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_outcome(torch),
            self.autograd_outcome(reference_torch),
        )
        self.assertEqual(
            self.no_grad_outcome(torch),
            self.no_grad_outcome(reference_torch),
        )

    def test_sequence_autograd_and_no_grad_match_pytorch_2_13(self):
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                self.assertEqual(
                    self.sequence_autograd_outcome(torch, sequence_type),
                    self.sequence_autograd_outcome(
                        reference_torch, sequence_type
                    ),
                )
                self.assertEqual(
                    self.sequence_no_grad_outcome(torch, sequence_type),
                    self.sequence_no_grad_outcome(
                        reference_torch, sequence_type
                    ),
                )

    def mode_contract(self, module):
        function = module.atleast_3d
        source = module.tensor([[1.0, 2.0]], dtype=module.float32)
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
        function = module.atleast_3d
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
        actual = torch.atleast_3d
        expected = reference_torch.atleast_3d

        self.assertIs(actual, actual_functional.atleast_3d)
        self.assertIs(expected, expected_functional.atleast_3d)
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
            actual_functional.__all__.count("atleast_3d"),
            expected_functional.__all__.count("atleast_3d"),
        )
        self.assertEqual(
            torch.__all__.count("atleast_3d"),
            reference_torch.__all__.count("atleast_3d"),
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
            (lambda: torch.atleast_3d(None), lambda: reference_torch.atleast_3d(None)),
            (lambda: torch.atleast_3d(1), lambda: reference_torch.atleast_3d(1)),
            (
                lambda: torch.atleast_3d(np.zeros((2,), dtype=np.float32)),
                lambda: reference_torch.atleast_3d(
                    np.zeros((2,), dtype=np.float32)
                ),
            ),
            (
                lambda: torch.atleast_3d(input=actual_tensor),
                lambda: reference_torch.atleast_3d(input=expected_tensor),
            ),
            (
                lambda: torch.atleast_3d(tensors=actual_tensor),
                lambda: reference_torch.atleast_3d(tensors=expected_tensor),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_zero_input_results_and_modes_match_pytorch_2_13(self):
        actual = torch.atleast_3d()
        expected = reference_torch.atleast_3d()
        self.assertIs(type(actual), tuple)
        self.assertIs(type(expected), tuple)
        self.assertEqual(actual, expected)

        def mode_outcome(module, forwarding):
            function = module.atleast_3d
            native_function = module._C._VariableFunctionsClass.atleast_3d
            marker = object()
            calls = []

            class Mode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    calls.append((func, types, args, kwargs))
                    if forwarding:
                        return func(*args, **(kwargs or {}))
                    return marker if func is native_function else NotImplemented

            with Mode():
                result = function()
            func, dispatch_types, args, kwargs = calls[0]
            return (
                len(calls),
                result == () if forwarding else result is marker,
                func is native_function,
                dispatch_types,
                args,
                kwargs,
            )

        for forwarding in (False, True):
            with self.subTest(forwarding=forwarding):
                self.assertEqual(
                    mode_outcome(torch, forwarding),
                    mode_outcome(reference_torch, forwarding),
                )

    def test_multiple_positional_and_mixed_forms_remain_unsupported(self):
        source = torch.tensor(1.0)
        unsupported = (
            lambda: torch.atleast_3d(source, source),
        )
        for call in unsupported:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError,
                "^atleast_3d\\(\\) only supports a single Tensor input$",
            ):
                call()

        expected = reference_torch.tensor(1.0)
        self.assertEqual(len(reference_torch.atleast_3d(expected, expected)), 2)
        self.assertEqual(len(reference_torch.atleast_3d((expected,))), 1)

        sequence_error = (
            "atleast_3d() sequence inputs only support an exact tuple or list "
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
                torch.atleast_3d(sequence)

        with self.assertRaises(TypeError):
            reference_torch.atleast_3d((expected, None))

    def test_inner_overrides_remain_unsupported_and_outer_dispatch_matches(self):
        sequence_error = (
            "atleast_3d() sequence inputs only support an exact tuple or list "
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
            torch.atleast_3d((actual_source, ActualOverride()))
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
            reference_torch.atleast_3d(
                (expected_source, ExpectedOverride())
            ),
            marker,
        )
        self.assertEqual(len(ExpectedOverride.calls), 1)

        def outer_override_outcome(module, sequence_type):
            function = module.atleast_3d
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
            function = module.atleast_3d
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
            actual_result = torch.atleast_3d((actual_source,))
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
            expected_result = reference_torch.atleast_3d((expected_source,))
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
                actual_mode.calls[0], torch.atleast_3d, actual_source
            ),
            normalize_mode_call(
                expected_mode.calls[0],
                reference_torch.atleast_3d,
                expected_source,
            ),
        )


if __name__ == "__main__":
    unittest.main()
