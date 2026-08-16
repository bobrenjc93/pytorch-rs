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
class TensorMovedimReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.movedim(int, int) differentials require pinned PyTorch 2.13.0"
            )

    def assert_matches(
        self,
        actual,
        expected,
        *,
        case,
        actual_source=None,
        expected_source=None,
        materialize=True,
    ):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            if materialize:
                np.testing.assert_allclose(
                    np.asarray(actual),
                    expected.detach().cpu().numpy(),
                    rtol=2.0e-6,
                    atol=1.0e-6,
                    equal_nan=True,
                )
            if actual_source is not None:
                self.assertEqual(
                    actual.data_ptr() == actual_source.data_ptr(),
                    expected.data_ptr() == expected_source.data_ptr(),
                )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_tensor(self, module, values, shape):
        if values.size == 0:
            return module.zeros(shape, dtype=module.float32)
        return module.tensor(
            values.item() if shape == () else values.tolist(),
            dtype=module.float32,
        )

    def test_seeded_shapes_strides_aliases_and_consumers_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xA10C_ED1)
        shapes = [(), (0,), (2, 0, 3), (1, 3, 2), (2, 3, 4)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            actual = self.make_tensor(torch, values, shape)
            expected = self.make_tensor(reference_torch, values, shape)

            for chain in range(4):
                rank = len(actual.shape)
                dimensions = [0, -1] if rank == 0 else list(range(-rank, rank))
                source = dimensions[int(rng.integers(0, len(dimensions)))]
                destination = dimensions[int(rng.integers(0, len(dimensions)))]
                actual_source = actual
                expected_source = expected
                style = (case + chain) % 3
                if style == 0:
                    actual = actual.movedim(source, destination)
                    expected = expected.movedim(source, destination)
                elif style == 1:
                    actual = actual.movedim(
                        source=source, destination=destination
                    )
                    expected = expected.movedim(
                        source=source, destination=destination
                    )
                else:
                    actual = actual.movedim(
                        destination=destination, source=source
                    )
                    expected = expected.movedim(
                        destination=destination, source=source
                    )
                self.assert_matches(
                    actual,
                    expected,
                    actual_source=actual_source,
                    expected_source=expected_source,
                    case=f"view-{case}-{chain}",
                )

            for operation, actual_output, expected_output in (
                ("clone", actual.clone(), expected.clone()),
                ("sin", actual.sin(), expected.sin()),
                ("scalar", actual + 1.25, expected + 1.25),
                ("reshape", actual.reshape(-1), expected.reshape(-1)),
                ("sum", actual.sum(), expected.sum()),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=f"{operation}-{case}",
                )

    def test_offset_empty_extreme_and_autograd_views_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_offset_source = torch.tensor(values.tolist()).transpose(0, 2)[1]
        expected_offset_source = reference_torch.tensor(values).transpose(0, 2)[1]
        actual_offset = actual_offset_source.movedim(-1, 0)
        expected_offset = expected_offset_source.movedim(-1, 0)
        self.assert_matches(
            actual_offset,
            expected_offset,
            actual_source=actual_offset_source,
            expected_source=expected_offset_source,
            case="offset-strided-view",
        )

        actual_extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        expected_extreme = reference_torch.zeros((sys.maxsize, 0, 2, 2))
        self.assert_matches(
            actual_extreme.movedim(0, 2),
            expected_extreme.movedim(0, 2),
            actual_source=actual_extreme,
            expected_source=expected_extreme,
            case="extreme-empty-view",
            materialize=False,
        )
        self.assert_error_matches(
            lambda: actual_extreme.movedim(1, 3),
            lambda: expected_extreme.movedim(1, 3),
        )

        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 2, 3)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = actual_leaf.movedim(-1, 0)
        expected = expected_leaf.movedim(-1, 0)
        self.assert_matches(
            actual,
            expected,
            actual_source=actual_leaf,
            expected_source=expected_leaf,
            case="tracked-view",
        )
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="gradient")

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.movedim(0, -1).sum().backward()
        expected_empty.movedim(0, -1).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty-gradient")

        with torch.no_grad():
            actual_untracked = actual_leaf.movedim(source=0, destination=1)
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.movedim(source=0, destination=1)
        self.assert_matches(
            actual_untracked,
            expected_untracked,
            actual_source=actual_leaf,
            expected_source=expected_leaf,
            case="no-grad-view",
        )

    def test_integer_binding_conversion_and_errors_match_pytorch_2_13(self):
        class IntegerSubclass(int):
            pass

        class CustomIndex:
            def __index__(self):
                return 0

        class UserOverflow(np.int64):
            def __index__(self):
                raise OverflowError("user overflow")

        class SpoofedInteger:
            @property
            def __class__(self):
                return np.int64

            def __index__(self):
                raise AssertionError("spoofed integers must not be converted")

        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        accepted = (
            (IntegerSubclass(0), IntegerSubclass(2)),
            (np.int8(-1), np.int32(0)),
            (np.uint64(2), np.int64(1)),
        )
        for case, (source, destination) in enumerate(accepted):
            self.assert_matches(
                actual.movedim(source, destination),
                expected.movedim(source, destination),
                actual_source=actual,
                expected_source=expected,
                case=f"integer-{case}",
            )

        cases = (
            (lambda: actual.movedim(), lambda: expected.movedim()),
            (lambda: actual.movedim(0), lambda: expected.movedim(0)),
            (lambda: actual.movedim(0, 1, 2), lambda: expected.movedim(0, 1, 2)),
            (
                lambda: actual.movedim(source=0),
                lambda: expected.movedim(source=0),
            ),
            (
                lambda: actual.movedim(0, source=1),
                lambda: expected.movedim(0, source=1),
            ),
            (
                lambda: actual.movedim(0, 1, extra=True),
                lambda: expected.movedim(0, 1, extra=True),
            ),
            (lambda: actual.movedim(True, 0), lambda: expected.movedim(True, 0)),
            (
                lambda: actual.movedim(np.bool_(False), 0),
                lambda: expected.movedim(np.bool_(False), 0),
            ),
            (
                lambda: actual.movedim(CustomIndex(), 0),
                lambda: expected.movedim(CustomIndex(), 0),
            ),
            (lambda: actual.movedim(1.5, 0), lambda: expected.movedim(1.5, 0)),
            (lambda: actual.movedim(0, "1"), lambda: expected.movedim(0, "1")),
            (lambda: actual.movedim(2**100, 0), lambda: expected.movedim(2**100, 0)),
            (lambda: actual.movedim(0, 2**100), lambda: expected.movedim(0, 2**100)),
            (
                lambda: actual.movedim(np.uint64(2**63), 0),
                lambda: expected.movedim(np.uint64(2**63), 0),
            ),
            (
                lambda: actual.movedim(2**100, UserOverflow(0)),
                lambda: expected.movedim(2**100, UserOverflow(0)),
            ),
            (
                lambda: actual.movedim(SpoofedInteger(), 2),
                lambda: expected.movedim(SpoofedInteger(), 2),
            ),
            (
                lambda: actual.movedim(0, SpoofedInteger()),
                lambda: expected.movedim(0, SpoofedInteger()),
            ),
            (lambda: actual.movedim(3, 0), lambda: expected.movedim(3, 0)),
            (lambda: actual.movedim(-4, 0), lambda: expected.movedim(-4, 0)),
            (lambda: actual.movedim(0, 3), lambda: expected.movedim(0, 3)),
            (lambda: actual.movedim(0, -4), lambda: expected.movedim(0, -4)),
            (
                lambda: actual.movedim(0, **{"bad\0tail": 1}),
                lambda: expected.movedim(0, **{"bad\0tail": 1}),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_scalar = torch.tensor(1.0)
        expected_scalar = reference_torch.tensor(1.0)
        self.assert_error_matches(
            lambda: actual_scalar.movedim(-2, 0),
            lambda: expected_scalar.movedim(-2, 0),
        )

    def test_stateful_integer_conversion_order_matches_pytorch_2_13(self):
        def stateful_dimensions():
            state = {"destination_converted": False, "calls": []}

            class StatefulInteger(np.int64):
                def __new__(cls, role):
                    value = np.int64.__new__(cls, 0)
                    value.role = role
                    return value

                def __index__(self):
                    state["calls"].append(self.role)
                    if self.role == "destination":
                        state["destination_converted"] = True
                        return 1
                    return 0 if state["destination_converted"] else 2

            return state, StatefulInteger("source"), StatefulInteger("destination")

        actual_state, actual_source, actual_destination = stateful_dimensions()
        expected_state, expected_source, expected_destination = stateful_dimensions()
        actual = torch.zeros((2, 3, 4)).movedim(
            actual_source, actual_destination
        )
        expected = reference_torch.zeros((2, 3, 4)).movedim(
            expected_source, expected_destination
        )
        self.assert_matches(actual, expected, case="stateful-conversion")
        self.assertEqual(actual_state["calls"], expected_state["calls"])
        self.assertEqual(actual_state["calls"], ["destination", "source"])

    def keyword_subclass_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)

        class PlainKeyword(str):
            pass

        class TrueKeyword(str):
            def __eq__(self, other):
                return True

            __hash__ = str.__hash__

        class FalseKeyword(str):
            def __eq__(self, other):
                return False

            __hash__ = str.__hash__

        class RaisingKeyword(str):
            def __eq__(self, other):
                raise RuntimeError("keyword equality failure")

            __hash__ = str.__hash__

        class MismatchedHashKeyword(str):
            def __eq__(self, other):
                return True

            def __hash__(self):
                return 0

        def outcome(call):
            try:
                result = call()
            except Exception as error:
                return type(error).__name__, str(error)
            return "ok", tuple(result.shape), result.stride()

        return tuple(
            outcome(call)
            for call in (
                lambda: tensor.movedim(
                    **{PlainKeyword("source"): 0, "destination": 1}
                ),
                lambda: tensor.movedim(
                    **{TrueKeyword("source"): 0, "destination": 1}
                ),
                lambda: tensor.movedim(
                    **{FalseKeyword("source"): 0, "destination": 1}
                ),
                lambda: tensor.movedim(
                    **{RaisingKeyword("source"): 0, "destination": 1}
                ),
                lambda: tensor.movedim(
                    **{MismatchedHashKeyword("source"): 0, "destination": 1}
                ),
                lambda: tensor.movedim(
                    **{FalseKeyword("unknown"): 0, "other": 1}
                ),
            )
        )

    def test_keyword_subclass_binding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.keyword_subclass_contract(torch),
            self.keyword_subclass_contract(reference_torch),
        )

    def callable_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "movedim")
        bound = tensor.movedim

        def error(call):
            try:
                call()
            except Exception as raised:
                return type(raised).__name__, str(raised)
            self.fail("Tensor.movedim unexpectedly accepted an invalid descriptor call")

        return {
            "descriptor_type": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound) is types.BuiltinMethodType,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "text_signature": descriptor.__text_signature__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_doc": bound.__doc__,
            "bound_text_signature": bound.__text_signature__,
            "objclass_name": descriptor.__objclass__.__name__,
            "objclass_module": descriptor.__objclass__.__module__,
            "descriptor_module": getattr(descriptor, "__module__", "missing"),
            "bound_module": bound.__module__,
            "descriptor_repr": repr(descriptor),
            "descriptor_identity": module.Tensor.movedim is descriptor,
            "descriptor_get": descriptor.__get__(None, module.Tensor) is descriptor,
            "signature_errors": tuple(
                error(
                    lambda callable_object=callable_object: inspect.signature(
                        callable_object
                    )
                )[0]
                for callable_object in (descriptor, bound)
            ),
            "call_errors": (
                error(lambda: descriptor()),
                error(lambda: descriptor(1, 0, 1)),
                error(
                    lambda: descriptor(
                        self=tensor, source=0, destination=1
                    )
                ),
            ),
            "call_shape": tuple(descriptor(tensor, 0, -1).shape),
        }

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "movedim")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            positional_result = tensor.movedim(0, -1)
        positional_call = positional.calls[0]

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.movedim(destination=-1, source=0)
        keyword_call = keyword.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.movedim(source=0, destination=-1)

        invalid = RecordingMode(marker)
        try:
            with invalid:
                tensor.movedim(True, 0)
        except Exception as raised:
            invalid_error = type(raised).__name__, str(raised)
        else:
            invalid_error = None

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.movedim(2**100, -4)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.movedim(0, 1)
        except Exception as raised:
            declining_error = (
                type(raised).__name__,
                re.sub(r"0x[0-9a-f]+", "0xADDR", str(raised)),
            )
        else:
            declining_error = None

        positional_function, positional_types, positional_args, positional_kwargs = (
            positional_call
        )
        keyword_function, keyword_types, keyword_args, keyword_kwargs = keyword_call
        return {
            "positional_intercepted": positional_result is marker,
            "positional_function": positional_function is descriptor,
            "positional_types": positional_types,
            "positional_receiver": positional_args[0] is tensor,
            "positional_metadata": positional_args[1:],
            "positional_kwargs": positional_kwargs,
            "keyword_intercepted": keyword_result is marker,
            "keyword_function": keyword_function is descriptor,
            "keyword_types": keyword_types,
            "keyword_receiver": len(keyword_args) == 1 and keyword_args[0] is tensor,
            "keyword_kwargs": keyword_kwargs,
            "forwarding_order": order,
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_alias": forwarded.data_ptr() == tensor.data_ptr(),
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "deferred_intercepted": deferred_result is marker,
            "deferred_calls": len(deferred.calls),
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_top_level_integer_views_and_autograd_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist())
        expected_base = reference_torch.tensor(values)
        sources = (
            (
                "scalar",
                torch.tensor(2.5),
                reference_torch.tensor(2.5),
                0,
                -1,
            ),
            (
                "empty",
                torch.zeros((2, 0, 3)),
                reference_torch.zeros((2, 0, 3)),
                -1,
                0,
            ),
            (
                "offset",
                actual_base.transpose(0, 2)[1],
                expected_base.transpose(0, 2)[1],
                -1,
                0,
            ),
            (
                "noncontiguous",
                actual_base.transpose(0, 2),
                expected_base.transpose(0, 2),
                0,
                2,
            ),
        )
        for case, actual_source, expected_source, source, destination in sources:
            for style in range(4):
                if style == 0:
                    actual = torch.movedim(actual_source, source, destination)
                    expected = reference_torch.movedim(
                        expected_source, source, destination
                    )
                elif style == 1:
                    actual = torch.movedim(
                        actual_source,
                        source=source,
                        destination=destination,
                    )
                    expected = reference_torch.movedim(
                        expected_source,
                        source=source,
                        destination=destination,
                    )
                elif style == 2:
                    actual = torch.movedim(
                        input=actual_source,
                        source=source,
                        destination=destination,
                    )
                    expected = reference_torch.movedim(
                        input=expected_source,
                        source=source,
                        destination=destination,
                    )
                else:
                    actual = torch.movedim(
                        destination=destination,
                        input=actual_source,
                        source=source,
                    )
                    expected = reference_torch.movedim(
                        destination=destination,
                        input=expected_source,
                        source=source,
                    )
                self.assert_matches(
                    actual,
                    expected,
                    actual_source=actual_source,
                    expected_source=expected_source,
                    case=f"top-level-{case}-{style}",
                )

        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(
            4, 2, 3
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = torch.movedim(actual_leaf, -1, 0)
        expected = reference_torch.movedim(expected_leaf, -1, 0)
        self.assert_matches(
            actual,
            expected,
            actual_source=actual_leaf,
            expected_source=expected_leaf,
            case="top-level-autograd",
        )
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.movedim(actual_empty, 0, -1).sum().backward()
        reference_torch.movedim(expected_empty, 0, -1).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad),
            expected_empty.grad.detach().cpu().numpy(),
        )

        with torch.no_grad():
            actual_untracked = torch.movedim(
                input=actual_leaf, source=0, destination=1
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.movedim(
                input=expected_leaf, source=0, destination=1
            )
        self.assertEqual(
            (actual_untracked.requires_grad, actual_untracked.is_leaf),
            (expected_untracked.requires_grad, expected_untracked.is_leaf),
        )

    def test_top_level_integer_binding_and_errors_match_pytorch_2_13(self):
        class IntegerSubclass(int):
            pass

        class CustomIndex:
            def __index__(self):
                return 0

        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        for alias in ("input", "x", "a", "x1"):
            actual_result = torch.movedim(
                **{alias: actual, "source": np.int64(0), "destination": -1}
            )
            expected_result = reference_torch.movedim(
                **{
                    alias: expected,
                    "source": np.int64(0),
                    "destination": -1,
                }
            )
            self.assert_matches(
                actual_result,
                expected_result,
                actual_source=actual,
                expected_source=expected,
                case=f"top-level-alias-{alias}",
            )

        actual_integer = torch.movedim(
            actual, IntegerSubclass(0), np.uint64(2)
        )
        expected_integer = reference_torch.movedim(
            expected, IntegerSubclass(0), np.uint64(2)
        )
        self.assert_matches(
            actual_integer,
            expected_integer,
            actual_source=actual,
            expected_source=expected,
            case="top-level-integer-subclasses",
        )

        cases = (
            (lambda: torch.movedim(), lambda: reference_torch.movedim()),
            (lambda: torch.movedim(actual), lambda: reference_torch.movedim(expected)),
            (
                lambda: torch.movedim(actual, 0),
                lambda: reference_torch.movedim(expected, 0),
            ),
            (
                lambda: torch.movedim(actual, 0, 1, 2),
                lambda: reference_torch.movedim(expected, 0, 1, 2),
            ),
            (
                lambda: torch.movedim(source=0, destination=1),
                lambda: reference_torch.movedim(source=0, destination=1),
            ),
            (
                lambda: torch.movedim(actual, destination=1),
                lambda: reference_torch.movedim(expected, destination=1),
            ),
            (
                lambda: torch.movedim(actual, 0, source=1),
                lambda: reference_torch.movedim(expected, 0, source=1),
            ),
            (
                lambda: torch.movedim(actual, 0, 1, extra=True),
                lambda: reference_torch.movedim(expected, 0, 1, extra=True),
            ),
            (lambda: torch.movedim(1, 0, 1), lambda: reference_torch.movedim(1, 0, 1)),
            (
                lambda: torch.movedim(actual, True, 0),
                lambda: reference_torch.movedim(expected, True, 0),
            ),
            (
                lambda: torch.movedim(actual, CustomIndex(), 0),
                lambda: reference_torch.movedim(expected, CustomIndex(), 0),
            ),
            (
                lambda: torch.movedim(actual, 1.5, 0),
                lambda: reference_torch.movedim(expected, 1.5, 0),
            ),
            (
                lambda: torch.movedim(actual, 0, "1"),
                lambda: reference_torch.movedim(expected, 0, "1"),
            ),
            (
                lambda: torch.movedim(actual, 2**100, 0),
                lambda: reference_torch.movedim(expected, 2**100, 0),
            ),
            (
                lambda: torch.movedim(actual, 0, np.uint64(2**63)),
                lambda: reference_torch.movedim(
                    expected, 0, np.uint64(2**63)
                ),
            ),
            (
                lambda: torch.movedim(actual, 3, 0),
                lambda: reference_torch.movedim(expected, 3, 0),
            ),
            (
                lambda: torch.movedim(actual, 0, -4),
                lambda: reference_torch.movedim(expected, 0, -4),
            ),
            (
                lambda: torch.movedim(actual, 0, **{"bad\0tail": 1}),
                lambda: reference_torch.movedim(
                    expected, 0, **{"bad\0tail": 1}
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def top_level_callable_contract(self, module):
        function = module.movedim
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
            "owner_callable_identity": owner.movedim is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("movedim"),
            "has_moveaxis": hasattr(module, "moveaxis"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["movedim"] is function,
            "wildcard_has_moveaxis": "moveaxis" in wildcard_namespace,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        actual = self.top_level_callable_contract(torch)
        expected = self.top_level_callable_contract(reference_torch)
        self.assertFalse(actual.pop("has_moveaxis"))
        self.assertTrue(expected.pop("has_moveaxis"))
        self.assertFalse(actual.pop("wildcard_has_moveaxis"))
        self.assertTrue(expected.pop("wildcard_has_moveaxis"))
        self.assertEqual(actual, expected)

    def top_level_mode_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            positional_result = module.movedim(tensor, 0, -1)
        positional_call = positional.calls[0]

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = module.movedim(
                destination=-1, input=tensor, source=0
            )
        keyword_call = keyword.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.movedim(
                    tensor, source=0, destination=-1
                )

        invalid = RecordingMode(marker)
        try:
            with invalid:
                module.movedim(tensor, True, 0)
        except Exception as raised:
            invalid_error = type(raised).__name__, str(raised)
        else:
            invalid_error = None

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = module.movedim(tensor, 2**100, -4)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    module.movedim(tensor, 0, 1)
        except Exception as raised:
            declining_error = (
                type(raised).__name__,
                re.sub(r"0x[0-9a-f]+", "0xADDR", str(raised)),
            )
        else:
            declining_error = None

        positional_function, positional_types, positional_args, positional_kwargs = (
            positional_call
        )
        keyword_function, keyword_types, keyword_args, keyword_kwargs = keyword_call
        return {
            "positional_intercepted": positional_result is marker,
            "positional_function": positional_function is module.movedim,
            "positional_types": positional_types,
            "positional_receiver": positional_args[0] is tensor,
            "positional_metadata": positional_args[1:],
            "positional_kwargs": positional_kwargs,
            "keyword_intercepted": keyword_result is marker,
            "keyword_function": keyword_function is module.movedim,
            "keyword_types": keyword_types,
            "keyword_args": keyword_args,
            "keyword_keys": tuple(keyword_kwargs),
            "keyword_receiver": keyword_kwargs["input"] is tensor,
            "keyword_source": keyword_kwargs["source"],
            "keyword_destination": keyword_kwargs["destination"],
            "forwarding_order": order,
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_alias": forwarded.data_ptr() == tensor.data_ptr(),
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "deferred_intercepted": deferred_result is marker,
            "deferred_calls": len(deferred.calls),
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
        }

    def test_top_level_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_mode_contract(torch),
            self.top_level_mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
