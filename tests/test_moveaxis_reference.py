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
class MoveaxisReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.moveaxis(int, int) differentials require pinned PyTorch 2.13.0"
            )

    def assert_matches(
        self,
        actual,
        expected,
        *,
        actual_source=None,
        expected_source=None,
        case,
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            if actual_source is not None:
                self.assertEqual(actual.data_ptr(), actual_source.data_ptr())
                self.assertEqual(expected.data_ptr(), expected_source.data_ptr())
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual),
                expected.detach().cpu().numpy(),
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_integer_views_aliasing_and_negative_axes_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist())
        expected_base = reference_torch.tensor(values)
        cases = (
            (
                "scalar",
                torch.tensor([2.5, 3.5])[1],
                reference_torch.tensor([2.5, 3.5])[1],
                0,
                -1,
            ),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2),
                reference_torch.zeros((2, 0, 3)).transpose(0, 2),
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
                -1,
            ),
        )
        for name, actual_source, expected_source, source, destination in cases:
            calls = (
                (
                    torch.moveaxis(actual_source, source, destination),
                    reference_torch.moveaxis(expected_source, source, destination),
                ),
                (
                    torch.moveaxis(
                        actual_source,
                        source=source,
                        destination=destination,
                    ),
                    reference_torch.moveaxis(
                        expected_source,
                        source=source,
                        destination=destination,
                    ),
                ),
                (
                    torch.moveaxis(
                        input=actual_source,
                        source=source,
                        destination=destination,
                    ),
                    reference_torch.moveaxis(
                        input=expected_source,
                        source=source,
                        destination=destination,
                    ),
                ),
                (
                    torch.moveaxis(
                        destination=destination,
                        input=actual_source,
                        source=source,
                    ),
                    reference_torch.moveaxis(
                        destination=destination,
                        input=expected_source,
                        source=source,
                    ),
                ),
            )
            for style, (actual, expected) in enumerate(calls):
                self.assert_matches(
                    actual,
                    expected,
                    actual_source=actual_source,
                    expected_source=expected_source,
                    case=(name, style),
                )

    def test_autograd_empty_backward_and_no_grad_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(
            4, 2, 3
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = torch.moveaxis(actual_leaf, -1, 0)
        expected = reference_torch.moveaxis(expected_leaf, -1, 0)
        self.assert_matches(
            actual,
            expected,
            actual_source=actual_leaf,
            expected_source=expected_leaf,
            case="autograd-view",
        )
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.detach().cpu().numpy(),
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.moveaxis(actual_empty, 0, -1).sum().backward()
        reference_torch.moveaxis(expected_empty, 0, -1).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad),
            expected_empty.grad.detach().cpu().numpy(),
        )

        with torch.no_grad():
            actual_untracked = torch.moveaxis(
                input=actual_leaf,
                source=0,
                destination=1,
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.moveaxis(
                input=expected_leaf,
                source=0,
                destination=1,
            )
        self.assertEqual(
            (actual_untracked.requires_grad, actual_untracked.is_leaf),
            (expected_untracked.requires_grad, expected_untracked.is_leaf),
        )

    def test_integer_binding_and_errors_match_pytorch_2_13(self):
        class IntegerSubclass(int):
            pass

        class CustomIndex:
            def __index__(self):
                return 0

        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        for alias in ("input", "x", "a", "x1"):
            actual_result = torch.moveaxis(
                **{alias: actual, "source": np.int64(0), "destination": -1}
            )
            expected_result = reference_torch.moveaxis(
                **{alias: expected, "source": np.int64(0), "destination": -1}
            )
            self.assert_matches(
                actual_result,
                expected_result,
                actual_source=actual,
                expected_source=expected,
                case=("input-alias", alias),
            )

        self.assert_matches(
            torch.moveaxis(actual, IntegerSubclass(0), np.uint64(2)),
            reference_torch.moveaxis(
                expected,
                IntegerSubclass(0),
                np.uint64(2),
            ),
            actual_source=actual,
            expected_source=expected,
            case="integer-subclasses",
        )

        cases = (
            (lambda: torch.moveaxis(), lambda: reference_torch.moveaxis()),
            (lambda: torch.moveaxis(actual), lambda: reference_torch.moveaxis(expected)),
            (
                lambda: torch.moveaxis(actual, 0),
                lambda: reference_torch.moveaxis(expected, 0),
            ),
            (
                lambda: torch.moveaxis(actual, 0, 1, 2),
                lambda: reference_torch.moveaxis(expected, 0, 1, 2),
            ),
            (
                lambda: torch.moveaxis(source=0, destination=1),
                lambda: reference_torch.moveaxis(source=0, destination=1),
            ),
            (
                lambda: torch.moveaxis(actual, destination=1),
                lambda: reference_torch.moveaxis(expected, destination=1),
            ),
            (
                lambda: torch.moveaxis(actual, 0, source=1),
                lambda: reference_torch.moveaxis(expected, 0, source=1),
            ),
            (
                lambda: torch.moveaxis(actual, 0, 1, extra=True),
                lambda: reference_torch.moveaxis(expected, 0, 1, extra=True),
            ),
            (
                lambda: torch.moveaxis(1, 0, 1),
                lambda: reference_torch.moveaxis(1, 0, 1),
            ),
            (
                lambda: torch.moveaxis(actual, True, 0),
                lambda: reference_torch.moveaxis(expected, True, 0),
            ),
            (
                lambda: torch.moveaxis(actual, CustomIndex(), 0),
                lambda: reference_torch.moveaxis(expected, CustomIndex(), 0),
            ),
            (
                lambda: torch.moveaxis(actual, 1.5, 0),
                lambda: reference_torch.moveaxis(expected, 1.5, 0),
            ),
            (
                lambda: torch.moveaxis(actual, 0, "1"),
                lambda: reference_torch.moveaxis(expected, 0, "1"),
            ),
            (
                lambda: torch.moveaxis(actual, 2**100, 0),
                lambda: reference_torch.moveaxis(expected, 2**100, 0),
            ),
            (
                lambda: torch.moveaxis(actual, 0, np.uint64(2**63)),
                lambda: reference_torch.moveaxis(
                    expected,
                    0,
                    np.uint64(2**63),
                ),
            ),
            (
                lambda: torch.moveaxis(actual, 3, 0),
                lambda: reference_torch.moveaxis(expected, 3, 0),
            ),
            (
                lambda: torch.moveaxis(actual, 0, -4),
                lambda: reference_torch.moveaxis(expected, 0, -4),
            ),
            (
                lambda: torch.moveaxis(actual, 0, **{"bad\0tail": 1}),
                lambda: reference_torch.moveaxis(
                    expected,
                    0,
                    **{"bad\0tail": 1},
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_sequence_axes_and_tensor_method_remain_deliberately_unsupported(self):
        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        for source, destination in (
            ((0, 2), (2, 0)),
            ([0, 2], [2, 0]),
            ((), ()),
        ):
            with self.subTest(source=source, destination=destination):
                with self.assertRaises(TypeError):
                    torch.moveaxis(actual, source, destination)
                reference_torch.moveaxis(expected, source, destination)
        self.assertFalse(hasattr(torch.Tensor, "moveaxis"))
        self.assertTrue(hasattr(reference_torch.Tensor, "moveaxis"))

    def mode_contract(self, module):
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
            positional_result = module.moveaxis(tensor, 0, -1)
        positional_call = positional.calls[0]

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = module.moveaxis(
                destination=-1,
                input=tensor,
                source=0,
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
                forwarded = module.moveaxis(tensor, source=0, destination=-1)

        invalid = RecordingMode(marker)
        try:
            with invalid:
                module.moveaxis(tensor, True, 0)
        except Exception as raised:
            invalid_error = type(raised).__name__, str(raised)
        else:
            invalid_error = None

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = module.moveaxis(tensor, 2**100, -4)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    module.moveaxis(tensor, 0, 1)
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
            "positional_function": positional_function is module.moveaxis,
            "positional_types": positional_types,
            "positional_receiver": positional_args[0] is tensor,
            "positional_metadata": positional_args[1:],
            "positional_kwargs": positional_kwargs,
            "keyword_intercepted": keyword_result is marker,
            "keyword_function": keyword_function is module.moveaxis,
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

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def override_contract(self, module):
        calls = []
        marker = object()

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = module.moveaxis(value, 0, 1)
        function, dispatch_types, args, kwargs = calls[0]
        return {
            "intercepted": result is marker,
            "function": function is module.moveaxis,
            "types": dispatch_types == (Override,),
            "receiver": args[0] is value,
            "metadata": args[1:],
            "kwargs": kwargs,
        }

    def test_tensor_like_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.override_contract(torch),
            self.override_contract(reference_torch),
        )

    def callable_contract(self, module):
        function = module.moveaxis
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.moveaxis is function,
            "distinct_from_movedim": function is not module.movedim,
            "all_count": module.__all__.count("moveaxis"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["moveaxis"] is function,
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


if __name__ == "__main__":
    unittest.main()
