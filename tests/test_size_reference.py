import enum
import inspect
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


class Dimension(enum.IntEnum):
    FIRST = 0
    SECOND = 1


class IntegerSubclass(int):
    pass


class CustomIndex:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 1


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.size(dim) differentials require pinned PyTorch 2.13.0"
            )

    def make_cases(self, module):
        base = module.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            dtype=module.float32,
        )
        return (
            module.tensor(3.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            base[1],
            base.transpose(0, 2),
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
        )

    def size_contract(self, tensor):
        shape = tuple(tensor.shape)
        metadata = (
            shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        results = []
        for axis, expected in enumerate(shape):
            for dimension in (axis, axis - len(shape)):
                positional = tensor.size(dimension)
                keyword = tensor.size(dim=dimension)
                results.append(
                    (
                        dimension,
                        type(positional).__name__,
                        positional,
                        type(keyword).__name__,
                        keyword,
                        expected,
                    )
                )
        return {
            "shape": shape,
            "results": tuple(results),
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_scalar_empty_offset_strided_and_extreme_metadata_match(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        self.assertGreater(actual_cases[2].storage_offset(), 0)
        self.assertGreater(expected_cases[2].storage_offset(), 0)
        self.assertFalse(actual_cases[3].is_contiguous())
        self.assertFalse(expected_cases[3].is_contiguous())
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                self.assertEqual(
                    self.size_contract(actual),
                    self.size_contract(expected),
                )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.size(dim) unexpectedly accepted the operation")

    def integer_and_error_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        accepted = (
            0,
            IntegerSubclass(1),
            Dimension.FIRST,
            Dimension.SECOND,
            np.int8(-1),
            np.int64(-2),
            np.uint64(2),
        )
        accepted_results = tuple(
            (
                type(value).__name__,
                tensor.size(value),
                tensor.size(dim=value),
            )
            for value in accepted
        )

        custom = CustomIndex()
        invalid = (
            True,
            np.bool_(False),
            custom,
            1.0,
            "1",
            2**63,
            -(2**63) - 1,
            np.uint64(2**63),
            3,
            -4,
            2**63 - 1,
            -(2**63),
        )
        errors = tuple(
            (
                type(value).__name__,
                self.error(lambda value=value: tensor.size(value)),
                self.error(lambda value=value: tensor.size(dim=value)),
            )
            for value in invalid
        )

        scalar = module.tensor(3.0, dtype=module.float32)
        scalar_errors = tuple(
            (
                self.error(lambda dimension=dimension: scalar.size(dimension)),
                self.error(
                    lambda dimension=dimension: scalar.size(dim=dimension)
                ),
            )
            for dimension in (-1, 0)
        )
        return {
            "accepted": accepted_results,
            "errors": errors,
            "scalar_errors": scalar_errors,
            "custom_index_calls": custom.calls,
        }

    def test_integer_acceptance_rejection_overflow_and_range_match(self):
        self.assertEqual(
            self.integer_and_error_contract(torch),
            self.integer_and_error_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "size")
        bound = tensor.size
        invalid_calls = (
            lambda: tensor.size(0, 1),
            lambda: tensor.size(0, dim=1),
            lambda: tensor.size(foo=0),
            lambda: tensor.size(dim=0, foo=1),
            lambda: tensor.size(True, dim=0),
            lambda: tensor.size(2**63, foo=1),
            lambda: descriptor(),
            lambda: descriptor(1, 0),
            lambda: descriptor(self=tensor, dim=0),
            lambda: descriptor.__get__(1, int),
        )
        signatures = []
        for callable_object in (descriptor, bound):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                signatures.append(type(error).__name__)
            else:
                signatures.append(None)
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_method_descriptor": type(descriptor)
            is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "is_builtin_method": type(bound) is types.BuiltinMethodType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "repr": repr(descriptor),
            "class_identity": module.Tensor.size is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "positional_value": descriptor(tensor, 0),
            "keyword_value": descriptor(tensor, dim=-1),
            "signatures": tuple(signatures),
            "invalid_errors": tuple(
                self.error(action) for action in invalid_calls
            ),
            "has_top_level_size": hasattr(module, "size"),
        }

    def test_descriptor_metadata_and_shared_binding_errors_match(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "size")
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
            positional_result = tensor.size(1)
        positional_function, positional_types, positional_args, positional_kwargs = (
            positional.calls[0]
        )

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.size(dim=-1)
        keyword_function, keyword_types, keyword_args, keyword_kwargs = (
            keyword.calls[0]
        )

        deferred = []
        for dimension in (2**63, 10):
            mode = RecordingMode(marker)
            with mode:
                result = tensor.size(dimension)
            deferred.append((result is marker, len(mode.calls)))

        custom = CustomIndex()
        rejected = []
        for dimension in (True, custom):
            mode = RecordingMode(marker)
            rejected.append(
                (
                    self.error(
                        lambda dimension=dimension, mode=mode: self.size_in_mode(
                            tensor, dimension, mode
                        )
                    ),
                    len(mode.calls),
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.size(dim=-1)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.size(0)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "positional_result": positional_result is marker,
            "positional_call_count": len(positional.calls),
            "positional_function": positional_function is descriptor,
            "positional_types": positional_types,
            "positional_args": len(positional_args) == 2
            and positional_args[0] is tensor
            and positional_args[1] == 1,
            "positional_kwargs": positional_kwargs,
            "keyword_result": keyword_result is marker,
            "keyword_call_count": len(keyword.calls),
            "keyword_function": keyword_function is descriptor,
            "keyword_types": keyword_types,
            "keyword_args": len(keyword_args) == 1
            and keyword_args[0] is tensor,
            "keyword_kwargs": keyword_kwargs,
            "deferred": tuple(deferred),
            "rejected": tuple(rejected),
            "custom_index_calls": custom.calls,
            "forwarding_order": order,
            "forwarded_type": type(forwarded).__name__,
            "forwarded": forwarded,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    @staticmethod
    def size_in_mode(tensor, dimension, mode):
        with mode:
            return tensor.size(dimension)

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
