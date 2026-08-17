import copy
import inspect
import multiprocessing.reduction as multiprocessing_reduction
import pickle
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorToReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.to() differentials require PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]
        return (
            leaf,
            tracked,
            (
                module.tensor(-3.5, dtype=module.float32),
                module.zeros((2, 0, 3), dtype=module.float32),
                source,
                strided,
                offset,
                leaf,
                tracked,
                tracked.detach(),
            ),
        )

    def identity_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        result = tensor.to()
        return {
            "result_is_receiver": result is tensor,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
                str(result.dtype),
                str(result.device),
                str(result.layout),
                result.requires_grad,
                result.is_leaf,
            ),
            "empty_expansions_are_receiver": (
                tensor.to(*()) is tensor,
                tensor.to(**{}) is tensor,
            ),
        }

    def test_empty_conversion_identity_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.identity_contract(actual),
                    self.identity_contract(expected),
                )

        actual_tracked.to().sum().backward()
        expected_tracked.to().sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def no_grad_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)
        with module.no_grad():
            result = tracked.to()
        result.sum().backward()
        return {
            "result_is_receiver": result is tracked,
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "shape": tuple(result.shape),
            "stride": tuple(result.stride()),
            "offset": result.storage_offset(),
            "pointer_unchanged": result.data_ptr() == tracked.data_ptr(),
            "gradient": leaf.grad.tolist(),
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def test_no_grad_identity_matches_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_contract(torch),
            self.no_grad_contract(reference_torch),
        )

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to")
        bound = tensor.to

        def signature_outcome(callable_object):
            try:
                return "signature", str(inspect.signature(callable_object))
            except Exception as error:
                return "error", type(error).__name__

        reducer, (owner, name) = descriptor.__reduce__()
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
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
                signature_outcome(descriptor),
                signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "bound_result_is_receiver": bound() is tensor,
            "reducer": (
                reducer.__name__,
                owner is descriptor.__objclass__,
                owner.__name__,
                owner.__module__,
                name,
            ),
            "copy_identities": (
                copy.copy(descriptor) is descriptor,
                copy.deepcopy(descriptor) is descriptor,
            ),
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol=protocol))
                is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_descriptor_docs_ownership_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def multiprocessing_pickle_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "to")
        return tuple(
            pickle.loads(
                multiprocessing_reduction.ForkingPickler.dumps(
                    descriptor,
                    protocol=protocol,
                )
            )
            is descriptor
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        )

    def test_multiprocessing_pickling_matches_pytorch_2_13(self):
        self.assertEqual(
            self.multiprocessing_pickle_contract(torch),
            self.multiprocessing_pickle_contract(reference_torch),
        )

    def error_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to")
        calls = (
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
        outcomes = []
        for call in calls:
            try:
                call()
            except Exception as error:
                outcomes.append((type(error).__name__, str(error)))
            else:
                outcomes.append(None)
        return tuple(outcomes)

    def test_unbound_no_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch),
            self.error_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to")
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
            intercepted = tensor.to()
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
                forwarded = tensor.to()

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.to()
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_is_descriptor": function is descriptor,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "types_empty": dispatch_types == (),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded_is_receiver": forwarded is tensor,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_conversion_overloads_remain_deliberately_unsupported(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        actual_other = torch.tensor([2.0], dtype=torch.float32)
        unsupported_calls = (
            lambda: actual.to(torch.float32),
            lambda: actual.to(dtype=torch.float32),
            lambda: actual.to(None),
            lambda: actual.to(dtype=None),
            lambda: actual.to(torch.device("cpu")),
            lambda: actual.to("cpu"),
            lambda: actual.to(device=torch.device("cpu")),
            lambda: actual.to(device=None),
            lambda: actual.to(actual_other),
            lambda: actual.to(copy=False),
            lambda: actual.to(copy=True),
            lambda: actual.to(non_blocking=False),
            lambda: actual.to(non_blocking=True),
            lambda: actual.to(memory_format=None),
            lambda: actual.to(memory_format=torch.preserve_format),
            lambda: actual.to(memory_format=torch.contiguous_format),
        )
        for call in unsupported_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        expected = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        expected_other = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32
        )
        supported_results = (
            expected.to(reference_torch.float32),
            expected.to(dtype=reference_torch.float32),
            expected.to(reference_torch.device("cpu")),
            expected.to(device=reference_torch.device("cpu")),
            expected.to(expected_other),
            expected.to(copy=False),
            expected.to(non_blocking=False),
            expected.to(memory_format=reference_torch.preserve_format),
        )
        for result in supported_results:
            self.assertIs(result, expected)


if __name__ == "__main__":
    unittest.main()
