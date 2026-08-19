import inspect
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
class TensorUnsqueezeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("unsqueeze differentials require pinned PyTorch 2.13.0")

    def layout_cases(self, module):
        values = [float(value) for value in range(48)]
        offset_noncontiguous = module.tensor(values).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)
        return (
            module.tensor(3.0),
            module.tensor([1.0, 2.0, 3.0]),
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            offset_noncontiguous,
            module.zeros((2, 0, 3)),
            module.zeros((2, 0, 3)).transpose(0, 2)[1],
        )

    def view_contract(self, source):
        outputs = (
            source.unsqueeze(0),
            source.unsqueeze(dim=0),
            source.unsqueeze(-(source.dim() + 1)),
            source.unsqueeze(dim=-(source.dim() + 1)),
        )
        repeated = source.unsqueeze(0)
        return {
            "source": (
                source.tolist(),
                tuple(source.shape),
                source.stride(),
                source.storage_offset(),
            ),
            "outputs": tuple(
                (
                    output.tolist(),
                    tuple(output.shape),
                    output.stride(),
                    output.storage_offset(),
                    output.data_ptr() == source.data_ptr(),
                    output.is_set_to(repeated),
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                    str(output.dtype),
                    str(output.device),
                )
                for output in outputs
            ),
        }

    def test_layout_offsets_aliasing_and_empties_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    self.view_contract(actual),
                    self.view_contract(expected),
                )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        result = source.unsqueeze(-4)
        metadata = (
            result.requires_grad,
            result.is_leaf,
            result.output_nr,
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == source.data_ptr(),
        )
        (result * 3.0).sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            untracked = no_grad_source.unsqueeze(dim=0)

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_result = empty.unsqueeze(0)
        empty_result.sum().backward()

        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
                untracked.data_ptr() == no_grad_source.data_ptr(),
            ),
            "empty": (
                tuple(empty_result.shape),
                empty_result.stride(),
                empty_result.storage_offset(),
                tuple(empty.grad.shape),
                empty.grad.tolist(),
            ),
        }

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("unsqueeze unexpectedly accepted an invalid call")

    def error_contract(self, module):
        tensor = module.zeros((2, 3))

        class IndexOnly:
            def __index__(self):
                return 0

        return (
            self.error(lambda: tensor.unsqueeze()),
            self.error(lambda: tensor.unsqueeze(0, 1)),
            self.error(lambda: tensor.unsqueeze(0, dim=0)),
            self.error(lambda: tensor.unsqueeze(extra=0)),
            self.error(lambda: tensor.unsqueeze(0, extra=True)),
            self.error(lambda: tensor.unsqueeze(None)),
            self.error(lambda: tensor.unsqueeze(0.0)),
            self.error(lambda: tensor.unsqueeze(True)),
            self.error(lambda: tensor.unsqueeze(dim="0")),
            self.error(lambda: tensor.unsqueeze(IndexOnly())),
            self.error(lambda: tensor.unsqueeze(None, extra=True)),
            self.error(lambda: tensor.unsqueeze(2**100)),
            self.error(lambda: tensor.unsqueeze(3)),
            self.error(lambda: tensor.unsqueeze(-4)),
            self.error(lambda: module.tensor(1.0).unsqueeze(1)),
            self.error(lambda: module.tensor(1.0).unsqueeze(-2)),
            tuple(tensor.unsqueeze(np.int64(-3)).shape),
            tuple(tensor.unsqueeze(np.uint32(0)).shape),
        )

    def test_binding_conversion_and_range_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch),
            self.error_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3))
        descriptor = inspect.getattr_static(module.Tensor, "unsqueeze")
        bound = tensor.unsqueeze

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail("unsqueeze unexpectedly exposed an inspectable signature")

        return {
            "descriptor_type": type(descriptor).__name__,
            "is_method_descriptor": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
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
            "class_identity": module.Tensor.unsqueeze is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "descriptor_signature": signature_error(descriptor),
            "bound_signature": signature_error(bound),
            "call": (
                tuple(descriptor(tensor, 0).shape),
                descriptor(tensor, 0).stride(),
            ),
            "no_receiver": self.error(lambda: descriptor()),
            "wrong_receiver": self.error(lambda: descriptor(1, 0)),
            "keyword_receiver": self.error(
                lambda: descriptor(self=tensor, dim=0)
            ),
        }

    def test_tensorbase_descriptor_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3))
        descriptor = inspect.getattr_static(module.Tensor, "unsqueeze")
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
            positional_result = tensor.unsqueeze(0)
        positional_call = positional.calls[0]

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.unsqueeze(dim=-3)
        keyword_call = keyword.calls[0]

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.unsqueeze(2**100)

        nonleading = RecordingMode(marker)
        with nonleading:
            nonleading_result = tensor.unsqueeze(1)

        invalid = RecordingMode(marker)
        invalid_error = self.error(
            lambda: self.call_inside_mode(invalid, lambda: tensor.unsqueeze(None))
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
                forwarded = tensor.unsqueeze(dim=-3)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.unsqueeze(0)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0xADDR", str(error)),
            )
        else:
            declining_error = None

        positional_function, positional_types, positional_args, positional_kwargs = (
            positional_call
        )
        keyword_function, keyword_types, keyword_args, keyword_kwargs = keyword_call
        return {
            "positional_result": positional_result is marker,
            "positional_function": positional_function is descriptor,
            "positional_types": positional_types,
            "positional_receiver": positional_args[0] is tensor,
            "positional_metadata": positional_args[1:],
            "positional_kwargs": positional_kwargs,
            "keyword_result": keyword_result is marker,
            "keyword_function": keyword_function is descriptor,
            "keyword_types": keyword_types,
            "keyword_receiver": keyword_args == (tensor,),
            "keyword_kwargs": keyword_kwargs,
            "deferred_result": deferred_result is marker,
            "deferred_calls": len(deferred.calls),
            "nonleading_result": nonleading_result is marker,
            "nonleading_calls": len(nonleading.calls),
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding_order": order,
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
            ),
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
        }

    @staticmethod
    def call_inside_mode(mode, action):
        with mode:
            return action()

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
