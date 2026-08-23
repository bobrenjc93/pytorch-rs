import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


UNSUPPORTED = "atleast_1d() only supports a single Tensor input"
UNSUPPORTED_SEQUENCE = (
    "atleast_1d() sequence inputs only support an exact tuple or list of "
    "exact Tensors"
)


class Atleast1dTests(unittest.TestCase):
    def test_ranked_tensors_are_returned_exactly(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            base[1, 2],
            base[1],
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        for source in cases:
            with self.subTest(shape=source.shape, stride=source.stride()):
                result = torch.atleast_1d(source)
                self.assertIs(result, source)

    def test_scalars_become_shared_storage_reshape_views(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        for source in (torch.tensor(-0.0), base.transpose(0, 2)[3, 2, 1]):
            with self.subTest(offset=source.storage_offset()):
                result = torch.atleast_1d(source)
                direct = source.reshape((1,))
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, (1,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(direct))
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))

    def test_tuple_and_list_sequences_use_native_views(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        sources = (
            torch.tensor(-0.0),
            base.transpose(0, 2)[3, 2, 1],
            base[1, 2],
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                result = torch.atleast_1d(sequence_type(sources))
                self.assertIs(type(result), tuple)
                self.assertEqual(len(result), len(sources))

                for source, item in zip(sources[:2], result[:2], strict=True):
                    direct = source.reshape((1,))
                    self.assertIsNot(item, source)
                    self.assertEqual(item.shape, (1,))
                    self.assertEqual(item.stride(), (1,))
                    self.assertEqual(item.storage_offset(), source.storage_offset())
                    self.assertEqual(item.data_ptr(), source.data_ptr())
                    self.assertTrue(item.is_set_to(direct))
                    np.testing.assert_array_equal(
                        np.asarray(item), np.asarray(direct)
                    )

                for source, item in zip(sources[2:], result[2:], strict=True):
                    self.assertIs(item, source)

        for empty in ((), []):
            with self.subTest(empty_type=type(empty).__name__):
                result = torch.atleast_1d(empty)
                self.assertIs(type(result), tuple)
                self.assertEqual(result, ())

    def test_variadic_tensors_use_native_views_in_order(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        sources = (
            torch.tensor(-0.0),
            base.transpose(0, 2)[3, 2, 1],
            base[1, 2],
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        result = torch.atleast_1d(*sources)
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), len(sources))

        for source, item in zip(sources[:2], result[:2], strict=True):
            direct = source.reshape((1,))
            self.assertIsNot(item, source)
            self.assertEqual(item.shape, (1,))
            self.assertEqual(item.stride(), (1,))
            self.assertEqual(item.storage_offset(), source.storage_offset())
            self.assertEqual(item.data_ptr(), source.data_ptr())
            self.assertTrue(item.is_set_to(direct))
            self.assertIs(item.dtype, source.dtype)
            self.assertEqual(item.device, source.device)
            self.assertEqual(item.layout, source.layout)
            np.testing.assert_array_equal(np.asarray(item), np.asarray(direct))

        for source, item in zip(sources[2:], result[2:], strict=True):
            self.assertIs(item, source)

    def test_autograd_repeated_backward_and_no_grad(self):
        leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        source = leaf[1]
        result = torch.atleast_1d(source)
        loss = result.sum()
        loss.backward()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [0.0, 2.0, 0.0])

        no_grad_source = torch.tensor(3.0, requires_grad=True)
        with torch.no_grad():
            no_grad_result = torch.atleast_1d(no_grad_source)
        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertEqual(no_grad_result.data_ptr(), no_grad_source.data_ptr())
        (no_grad_result * no_grad_result).sum().backward()
        self.assertIsNone(no_grad_source.grad)
        self.assertIsNone(no_grad_result.grad)

    def test_sequence_autograd_repeated_backward_and_no_grad(self):
        for sequence_type in (tuple, list):
            with self.subTest(sequence_type=sequence_type.__name__):
                leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
                scalar = leaf[1]
                scalar_result, vector_result = torch.atleast_1d(
                    sequence_type((scalar, leaf))
                )
                self.assertTrue(scalar_result.requires_grad)
                self.assertFalse(scalar_result.is_leaf)
                self.assertEqual(scalar_result.data_ptr(), scalar.data_ptr())
                self.assertIs(vector_result, leaf)

                loss = scalar_result.sum()
                loss.backward()
                loss.backward()
                self.assertEqual(leaf.grad.tolist(), [0.0, 2.0, 0.0])

                no_grad_scalar = torch.tensor(3.0, requires_grad=True)
                vector_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
                vector = vector_leaf * 2.0
                with torch.no_grad():
                    scalar_result, vector_result = torch.atleast_1d(
                        sequence_type((no_grad_scalar, vector))
                    )
                self.assertTrue(scalar_result.requires_grad)
                self.assertTrue(scalar_result.is_leaf)
                self.assertEqual(
                    scalar_result.data_ptr(), no_grad_scalar.data_ptr()
                )
                self.assertIs(vector_result, vector)
                self.assertTrue(vector_result.requires_grad)
                self.assertFalse(vector_result.is_leaf)
                (scalar_result * scalar_result).sum().backward()
                self.assertIsNone(no_grad_scalar.grad)
                self.assertIsNone(scalar_result.grad)

    def test_variadic_autograd_repeated_backward_and_no_grad(self):
        leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        scalar = leaf[1]
        scalar_result, vector_result = torch.atleast_1d(scalar, leaf)
        self.assertTrue(scalar_result.requires_grad)
        self.assertFalse(scalar_result.is_leaf)
        self.assertEqual(scalar_result.data_ptr(), scalar.data_ptr())
        self.assertIs(vector_result, leaf)

        loss = scalar_result.sum()
        loss.backward()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [0.0, 2.0, 0.0])

        no_grad_scalar = torch.tensor(3.0, requires_grad=True)
        vector_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        vector = vector_leaf * 2.0
        with torch.no_grad():
            scalar_result, vector_result = torch.atleast_1d(
                no_grad_scalar, vector
            )
        self.assertTrue(scalar_result.requires_grad)
        self.assertTrue(scalar_result.is_leaf)
        self.assertEqual(scalar_result.data_ptr(), no_grad_scalar.data_ptr())
        self.assertIs(vector_result, vector)
        self.assertTrue(vector_result.requires_grad)
        self.assertFalse(vector_result.is_leaf)
        (scalar_result * scalar_result).sum().backward()
        self.assertIsNone(no_grad_scalar.grad)
        self.assertIsNone(scalar_result.grad)

    def test_modes_and_overrides_receive_the_public_function(self):
        source = torch.tensor(2.0)
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.atleast_1d(value), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.atleast_1d)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {})

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.atleast_1d(source)
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.atleast_1d for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == (source,) for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result.data_ptr(), source.data_ptr())

    def test_inner_sequence_override_dispatch_is_explicitly_unsupported(self):
        source = torch.tensor(2.0)

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        value = Override()
        for sequence in ((source, value), [source, value]):
            with self.subTest(sequence_type=type(sequence).__name__):
                with self.assertRaisesRegex(
                    TypeError, f"^{re.escape(UNSUPPORTED_SEQUENCE)}$"
                ):
                    torch.atleast_1d(sequence)
        self.assertEqual(Override.calls, [])

    def test_variadic_operand_overrides_remain_explicitly_unsupported(self):
        source = torch.tensor(2.0)

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        value = Override()
        for args in ((source, value), (value, source)):
            with self.subTest(args=args), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                torch.atleast_1d(*args)
        self.assertEqual(Override.calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        for args in ((source, value), (value, source), (source, None)):
            with self.subTest(mode_args=args), mode, self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                torch.atleast_1d(*args)
        self.assertEqual(mode.calls, [])

    def test_variadic_exact_tensors_dispatch_through_nested_modes(self):
        scalar = torch.tensor(1.0)
        vector = torch.tensor([2.0, 3.0])
        sources = (scalar, vector)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            result = torch.atleast_1d(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [accepting],
            )
        self.assertIs(result, marker)
        self.assertEqual(
            accepting.calls,
            [(torch.atleast_1d, (torch.Tensor,), sources, {})],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        self.label,
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                results = torch.atleast_1d(*sources)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )

        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.atleast_1d for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == sources for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(calls[0][5], (lower,))
        self.assertEqual(calls[1][5], ())
        self.assertEqual(
            torch.overrides._get_current_function_mode_stack(), []
        )
        self.assertEqual(tuple(result.shape for result in results), ((1,), (2,)))
        self.assertTrue(
            all(
                result.data_ptr() == source.data_ptr()
                for result, source in zip(results, sources, strict=True)
            )
        )

    def test_variadic_declining_and_raising_modes_restore_the_stack(self):
        sources = (torch.tensor(1.0), torch.tensor([2.0, 3.0]))

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return NotImplemented

        declining = DecliningMode()
        with declining:
            with self.assertRaisesRegex(
                TypeError,
                "^no implementation found for "
                "'torch_rs\\.functional\\.atleast_1d' on types that implement "
                "__torch_function__: \\[\\] nor in mode ",
            ):
                torch.atleast_1d(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [declining],
            )
        self.assertEqual(
            declining.calls,
            [
                (torch.atleast_1d, (torch.Tensor,), sources, {}),
                (torch.atleast_1d, (), sources, {}),
            ],
        )

        expected_error = ValueError("mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        raising = RaisingMode()
        with raising:
            with self.assertRaises(ValueError) as raised:
                torch.atleast_1d(*sources)
            self.assertIs(raised.exception, expected_error)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [raising],
            )
        self.assertEqual(
            torch.overrides._get_current_function_mode_stack(), []
        )

    def test_variadic_mode_fallback_disables_nested_override_dispatch(self):
        sources = (torch.tensor(1.0), torch.tensor([2.0, 3.0]))
        marker = object()
        nested_marker = object()
        nested_outcomes = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return nested_marker

        value = Override()

        class FallbackMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                if types:
                    return NotImplemented
                probe = torch.overrides.has_torch_function_unary(value)
                try:
                    torch.neg(value)
                except Exception as error:
                    nested_outcomes.append(
                        (probe, type(error).__name__, str(error))
                    )
                else:
                    nested_outcomes.append((probe, "result", None))
                return marker

        mode = FallbackMode()
        with mode:
            result = torch.atleast_1d(*sources)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )

        self.assertIs(result, marker)
        self.assertEqual(
            mode.calls,
            [
                (torch.atleast_1d, (torch.Tensor,), sources, {}),
                (torch.atleast_1d, (), sources, {}),
            ],
        )
        self.assertEqual(
            nested_outcomes,
            [
                (
                    False,
                    "TypeError",
                    "neg(): argument 'input' (position 1) must be Tensor, not Override",
                )
            ],
        )
        self.assertEqual(Override.calls, [])

        self.assertIs(torch.neg(value), nested_marker)
        self.assertEqual(len(Override.calls), 1)

    def test_variadic_disabled_retry_reaches_lower_mode_without_tensor_type(self):
        sources = (torch.tensor(1.0), torch.tensor([2.0, 3.0]))
        marker = object()
        calls = []

        class LowerMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        "lower",
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return marker

        class UpperMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        "upper",
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(
                            torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                if types:
                    return NotImplemented
                return func(*args, **(kwargs or {}))

        lower = LowerMode()
        upper = UpperMode()
        with lower:
            with upper:
                result = torch.atleast_1d(*sources)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )

        self.assertIs(result, marker)
        self.assertEqual(
            calls,
            [
                (
                    "upper",
                    torch.atleast_1d,
                    (torch.Tensor,),
                    sources,
                    {},
                    (lower,),
                ),
                ("upper", torch.atleast_1d, (), sources, {}, (lower,)),
                ("lower", torch.atleast_1d, (), sources, {}, ()),
            ],
        )
        self.assertEqual(
            torch.overrides._get_current_function_mode_stack(), []
        )

    def test_outer_sequence_overrides_and_modes_precede_the_fast_path(self):
        source = torch.tensor(2.0)
        marker = object()

        class TupleOverride(tuple):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        class ListOverride(list):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for sequence in (TupleOverride((source,)), ListOverride([source])):
            override_type = type(sequence)
            with self.subTest(override_type=override_type.__name__):
                self.assertIs(torch.atleast_1d(sequence), marker)
                function, dispatch_types, args, kwargs = override_type.calls[0]
                self.assertIs(function, torch.atleast_1d)
                self.assertEqual(dispatch_types, (override_type,))
                self.assertEqual(args, (sequence,))
                self.assertEqual(kwargs, {})

        class SpoofedSequence:
            calls = []

            @property
            def __class__(self):
                return tuple

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        spoofed = SpoofedSequence()
        self.assertTrue(isinstance(spoofed, tuple))
        self.assertIs(torch.atleast_1d(spoofed), marker)
        function, dispatch_types, args, kwargs = SpoofedSequence.calls[0]
        self.assertIs(function, torch.atleast_1d)
        self.assertEqual(dispatch_types, (SpoofedSequence,))
        self.assertEqual(args, (spoofed,))
        self.assertEqual(kwargs, {})

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        sequence = (source,)
        mode = RecordingMode(marker)
        with mode:
            result = torch.atleast_1d(sequence)
        self.assertIs(result, marker)
        self.assertEqual(
            mode.calls,
            [(torch.atleast_1d, (), (sequence,), {})],
        )

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            result = torch.atleast_1d(sequence)
        self.assertEqual(calls, [(torch.atleast_1d, (), (sequence,), {})])
        self.assertIs(type(result), tuple)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].data_ptr(), source.data_ptr())

    def test_function_metadata_exports_and_pickle(self):
        function = torch.atleast_1d
        self.assertIs(function, torch.functional.atleast_1d)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "atleast_1d")
        self.assertEqual(function.__qualname__, "atleast_1d")
        self.assertEqual(function.__module__, "torch_rs.functional")
        self.assertEqual(str(inspect.signature(function)), "(*tensors)")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(torch.__all__.count("atleast_1d"), 1)
        self.assertEqual(torch.functional.__all__.count("atleast_1d"), 1)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["atleast_1d"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_single_input_errors_and_unsupported_forms(self):
        invalid_message = (
            "atleast_1d() received an invalid combination of arguments - got "
            "(NoneType), but expected one of:\n * (Tensor input)\n      didn't "
            "match because some of the arguments have invalid types: "
            "(!NoneType!)\n * (tuple of Tensors tensors)\n      didn't match "
            "because some of the arguments have invalid types: (!NoneType!)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_message)}$"):
            torch.atleast_1d(None)
        with self.assertRaisesRegex(
            TypeError,
            "^atleast_1d\\(\\) got an unexpected keyword argument 'input'$",
        ):
            torch.atleast_1d(input=torch.tensor(1.0))

        source = torch.tensor(1.0)
        unsupported_calls = (
            lambda: torch.atleast_1d(source, None),
            lambda: torch.atleast_1d(None, source),
            lambda: torch.atleast_1d(source, source, 1),
        )
        for call in unsupported_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                call()

        mixed_sequences = (
            (source, None),
            [source, 1],
            ((source,),),
        )
        for sequence in mixed_sequences:
            with self.subTest(sequence=sequence), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED_SEQUENCE)}$"
            ):
                torch.atleast_1d(sequence)


if __name__ == "__main__":
    unittest.main()
