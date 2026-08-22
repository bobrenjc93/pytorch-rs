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


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NewZerosReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("new_zeros differentials require pinned PyTorch 2.13.0")

    def source(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        return leaf, (leaf * 2.0).transpose(0, 1)

    def tensor_observation(self, module, source, result):
        return {
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "values": np.asarray(result.detach()).reshape(-1).tolist(),
            "dtype": str(result.dtype),
            "canonical_dtype": result.dtype is module.float32,
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "is_contiguous": result.is_contiguous(),
            "storage_offset": result.storage_offset(),
            "fresh_storage": result.data_ptr() != source.data_ptr(),
            "not_same_view": not result.is_set_to(source),
        }

    def test_results_metadata_and_fresh_storage_match_pytorch_2_13(self):
        size_factories = (
            lambda module: 2,
            lambda module: IntSubclass(2),
            lambda module: np.int64(2),
            lambda module: np.uint32(2),
            lambda module: IndexDimension(2),
            lambda module: (2, 3),
            lambda module: [2, 0, 3],
            lambda module: module.Size([2, 1, 3]),
            lambda module: (),
        )
        option_factories = (
            lambda module: {},
            lambda module: {
                "dtype": None,
                "device": None,
                "requires_grad": None,
                "layout": None,
                "pin_memory": None,
            },
            lambda module: {
                "dtype": module.float32,
                "device": "cpu",
                "requires_grad": False,
                "layout": module.strided,
                "pin_memory": False,
            },
            lambda module: {
                "device": module.device("cpu:0"),
                "requires_grad": True,
            },
        )

        for size_factory in size_factories:
            for option_factory in option_factories:
                actual_source = self.source(torch)[1]
                expected_source = self.source(reference_torch)[1]
                actual_size = size_factory(torch)
                expected_size = size_factory(reference_torch)
                actual_options = option_factory(torch)
                expected_options = option_factory(reference_torch)
                with self.subTest(size=actual_size, options=actual_options):
                    actual = actual_source.new_zeros(actual_size, **actual_options)
                    expected = expected_source.new_zeros(
                        expected_size, **expected_options
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual_source, actual),
                        self.tensor_observation(
                            reference_torch, expected_source, expected
                        ),
                    )

    def test_keyword_size_and_autograd_leaf_behavior_match(self):
        observations = []
        for module in (torch, reference_torch):
            leaf, source = self.source(module)
            with module.no_grad():
                result = source.new_zeros(size=[2, 3], requires_grad=True)
            result.sum().backward()
            observations.append(
                {
                    "result": self.tensor_observation(module, source, result),
                    "gradient": result.grad.tolist(),
                    "source_leaf_gradient": leaf.grad,
                }
            )
        self.assertEqual(observations[0], observations[1])

    def error(self, action):
        try:
            action()
        except Exception as error:
            message = str(error).splitlines()[0]
            if "Overflow when unpacking long long" in message:
                message = message.rstrip('"')
            return type(error).__name__, message
        self.fail("new_zeros unexpectedly accepted the operation")

    def error_contract(self, module):
        source = module.tensor([1.0], dtype=module.float32)
        return (
            self.error(lambda: source.new_zeros()),
            self.error(lambda: source.new_zeros(None)),
            self.error(lambda: source.new_zeros(size=2)),
            self.error(lambda: source.new_zeros(range(2))),
            self.error(lambda: source.new_zeros(True)),
            self.error(lambda: source.new_zeros(np.bool_(True))),
            self.error(lambda: source.new_zeros([True])),
            self.error(lambda: source.new_zeros(size=[2.0])),
            self.error(lambda: source.new_zeros((2, -1, 3))),
            self.error(lambda: source.new_zeros(2**63)),
            self.error(lambda: source.new_zeros((2, 2**63))),
            self.error(lambda: source.new_zeros(sys.maxsize)),
            self.error(lambda: source.new_zeros((0, sys.maxsize, 2))),
            self.error(lambda: source.new_zeros((2,), dtype=object())),
            self.error(lambda: source.new_zeros((2,), layout=object())),
            self.error(lambda: source.new_zeros((2,), device=object())),
            self.error(lambda: source.new_zeros((2,), pin_memory=1)),
            self.error(lambda: source.new_zeros((2,), requires_grad=1)),
            self.error(lambda: source.new_zeros((2,), unexpected=True)),
            self.error(lambda: source.new_zeros((2,), size=(3,))),
            self.error(lambda: source.new_zeros((-1,), device="not-a-device")),
        )

    def test_supported_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch), self.error_contract(reference_torch)
        )

    def index_conversion_observation(self, module):
        source = module.tensor([1.0], dtype=module.float32)
        scalar_calls = []
        sequence_calls = []

        class ScalarIndex:
            def __index__(self):
                value = (2, 3, 4)[len(scalar_calls)]
                scalar_calls.append(value)
                return value

        class SequenceIndex:
            def __index__(self):
                value = (2, 3)[len(sequence_calls)]
                sequence_calls.append(value)
                return value

        scalar = source.new_zeros(ScalarIndex())
        sequence = source.new_zeros([SequenceIndex()])
        return {
            "scalar_calls": scalar_calls,
            "scalar_shape": tuple(scalar.shape),
            "sequence_calls": sequence_calls,
            "sequence_shape": tuple(sequence.shape),
        }

    def test_index_conversion_order_matches_pytorch_2_13(self):
        self.assertEqual(
            self.index_conversion_observation(torch),
            self.index_conversion_observation(reference_torch),
        )

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "new_zeros")
        source = module.tensor([1.0], dtype=module.float32)
        try:
            inspect.signature(descriptor)
        except Exception as error:
            signature_error = type(error).__name__, str(error)
        else:
            signature_error = None
        return {
            "type": type(descriptor).__name__,
            "method_descriptor": type(descriptor) is types.MethodDescriptorType,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "owner": (
                descriptor.__objclass__.__name__,
                descriptor.__objclass__.__module__,
            ),
            "has_module": hasattr(descriptor, "__module__"),
            "repr": repr(descriptor),
            "doc": descriptor.__doc__,
            "signature_error": signature_error,
            "call": descriptor(source, (2,)).tolist(),
            "no_receiver": self.error(lambda: descriptor()),
            "wrong_receiver": self.error(lambda: descriptor(1, (2,))),
            "keyword_receiver": self.error(
                lambda: descriptor(self=source, size=(2,))
            ),
        }

    def test_tensorbase_descriptor_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        source = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "new_zeros")
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
            positional_result = source.new_zeros((2, 3))
        positional_call = positional.calls[0]

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = source.new_zeros(
                size=[2, 3], requires_grad=True, layout=module.strided
            )
        keyword_call = keyword.calls[0]

        index_calls = []

        class DeferredIndex:
            def __index__(self):
                value = (2, 3, 4)[len(index_calls)]
                index_calls.append(value)
                return value

        deferred_index = DeferredIndex()
        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = source.new_zeros(
                deferred_index, device="not-a-device"
            )

        invalid = RecordingMode(marker)
        invalid_error = self.error(
            lambda: self.call_inside_mode(
                invalid, lambda: source.new_zeros(object())
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
                forwarded = source.new_zeros((2, 0, 3), requires_grad=True)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        declining_error = self.error(
            lambda: self.call_inside_two_modes(
                lower, declining, lambda: source.new_zeros((2,))
            )
        )

        positional_function, positional_types, positional_args, positional_kwargs = (
            positional_call
        )
        keyword_function, keyword_types, keyword_args, keyword_kwargs = keyword_call
        return {
            "positional_result": positional_result is marker,
            "positional_function": positional_function is descriptor,
            "positional_types": positional_types,
            "positional_receiver": positional_args[0] is source,
            "positional_size": positional_args[1:],
            "positional_kwargs": positional_kwargs,
            "keyword_result": keyword_result is marker,
            "keyword_function": keyword_function is descriptor,
            "keyword_types": keyword_types,
            "keyword_receiver": keyword_args == (source,),
            "keyword_size": keyword_kwargs["size"],
            "keyword_requires_grad": keyword_kwargs["requires_grad"],
            "keyword_layout": repr(keyword_kwargs["layout"]),
            "deferred_result": deferred_result is marker,
            "deferred_calls": len(deferred.calls),
            "deferred_index_calls": index_calls,
            "deferred_argument_identity": deferred.calls[0][2][1]
            is deferred_index,
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding_order": order,
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.requires_grad,
                forwarded.is_leaf,
            ),
            "declining_error": (
                declining_error[0],
                re.sub(r"0x[0-9a-f]+", "0xADDR", declining_error[1]),
            ),
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    @staticmethod
    def call_inside_mode(mode, action):
        with mode:
            return action()

    @staticmethod
    def call_inside_two_modes(lower, upper, action):
        with lower:
            with upper:
                return action()

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )


if __name__ == "__main__":
    unittest.main()
