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

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def make_layout_cases(self, module):
        base = module.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("scalar-offset", base[1, 1, 2, 3]),
            ("vector-offset", base[1, 1, 2]),
            ("matrix-offset", base[1, 1]),
            ("noncontiguous-offset", base[1].transpose(0, 1)),
            ("empty", module.zeros((0, 3), dtype=module.float32)),
            (
                "empty-strided-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
        )

    def call(self, tensor, form):
        if form == "positional":
            return tensor.unsqueeze(0)
        if form == "keyword":
            return tensor.unsqueeze(dim=0)
        if form == "negative":
            return tensor.unsqueeze(-(tensor.dim() + 1))
        if form == "negative_keyword":
            return tensor.unsqueeze(dim=-(tensor.dim() + 1))
        raise AssertionError(f"unknown call form: {form}")

    def layout_contract(self, module, tensor, form):
        result = self.call(tensor, form)
        repeated = tensor.unsqueeze(0)
        return (
            self.tensor_array(result, module).copy(),
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == tensor.data_ptr(),
            result.is_set_to(repeated),
            result.requires_grad,
            result.is_leaf,
            result.output_nr,
            str(result.dtype),
            str(result.device),
            str(result.layout),
        )

    def test_shapes_strides_offsets_aliasing_and_empties_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for (name, actual), (expected_name, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(name, expected_name)
            for form in ("positional", "keyword", "negative", "negative_keyword"):
                with self.subTest(case=name, form=form):
                    actual_contract = self.layout_contract(torch, actual, form)
                    expected_contract = self.layout_contract(
                        reference_torch, expected, form
                    )
                    np.testing.assert_array_equal(
                        actual_contract[0], expected_contract[0]
                    )
                    self.assertEqual(actual_contract[1:], expected_contract[1:])

    def autograd_contract(self, module):
        repeated_leaf = module.tensor(
            [1.0, 2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        repeated_result = repeated_leaf.unsqueeze(0)
        repeated_loss = repeated_result.sum()
        repeated_loss.backward()
        repeated_loss.backward()

        leaf = module.tensor(
            [float(value) for value in range(48)],
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        result = source.unsqueeze(-4)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(source.unsqueeze(0)),
            result.requires_grad,
            result.is_leaf,
            result.output_nr,
        )
        (result * 3.0).sum().backward()

        empty_leaf = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        empty_source = empty_leaf.transpose(0, 2)[1]
        empty_result = empty_source.unsqueeze(0)
        empty_metadata = (
            tuple(empty_result.shape),
            empty_result.stride(),
            empty_result.storage_offset(),
            empty_result.data_ptr() == empty_source.data_ptr(),
            empty_result.is_set_to(empty_source.unsqueeze(0)),
            empty_result.requires_grad,
            empty_result.is_leaf,
            empty_result.output_nr,
        )
        empty_result.sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            no_grad_result = no_grad_source.unsqueeze(dim=0)
        no_grad_metadata = (
            tuple(no_grad_result.shape),
            no_grad_result.stride(),
            no_grad_result.storage_offset(),
            no_grad_result.data_ptr() == no_grad_source.data_ptr(),
            no_grad_result.is_set_to(no_grad_source.unsqueeze(0)),
            no_grad_result.requires_grad,
            no_grad_result.is_leaf,
            no_grad_result.output_nr,
        )
        (no_grad_result * no_grad_result).sum().backward()

        return {
            "repeated": self.tensor_array(repeated_leaf.grad, module).copy(),
            "metadata": metadata,
            "gradient": self.tensor_array(leaf.grad, module).copy(),
            "empty_metadata": empty_metadata,
            "empty_gradient": self.tensor_array(empty_leaf.grad, module).copy(),
            "no_grad_metadata": no_grad_metadata,
            "no_grad_source_gradient": no_grad_source.grad,
            "no_grad_result_gradient": no_grad_result.grad,
        }

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        actual = self.autograd_contract(torch)
        expected = self.autograd_contract(reference_torch)
        np.testing.assert_array_equal(actual.pop("repeated"), expected.pop("repeated"))
        np.testing.assert_array_equal(actual.pop("gradient"), expected.pop("gradient"))
        np.testing.assert_array_equal(
            actual.pop("empty_gradient"), expected.pop("empty_gradient")
        )
        self.assertEqual(actual, expected)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("unsqueeze unexpectedly accepted an invalid call")

    def binding_error_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)

        class IndexOnly:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return 0

        index_only = IndexOnly()
        errors = (
            self.error(lambda: tensor.unsqueeze()),
            self.error(lambda: tensor.unsqueeze(0, 1)),
            self.error(lambda: tensor.unsqueeze(0, dim=0)),
            self.error(lambda: tensor.unsqueeze(extra=0)),
            self.error(lambda: tensor.unsqueeze(dim=0, extra=True)),
            self.error(lambda: tensor.unsqueeze("0", extra=True)),
            self.error(lambda: tensor.unsqueeze(dim="0", extra=True)),
            self.error(lambda: tensor.unsqueeze(True)),
            self.error(lambda: tensor.unsqueeze(np.bool_(False))),
            self.error(lambda: tensor.unsqueeze(0.0)),
            self.error(lambda: tensor.unsqueeze(None)),
            self.error(lambda: tensor.unsqueeze(index_only)),
            self.error(lambda: tensor.unsqueeze(2**100)),
            self.error(lambda: tensor.unsqueeze(3)),
            self.error(lambda: tensor.unsqueeze(-4)),
            self.error(lambda: module.tensor(1.0).unsqueeze(1)),
        )
        accepted = (
            tuple(tensor.unsqueeze(0).shape),
            tuple(tensor.unsqueeze(dim=-3).shape),
            tuple(tensor.unsqueeze(np.int64(0)).shape),
        )
        return errors, index_only.calls, accepted

    def test_binding_integer_conversion_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.binding_error_contract(torch),
            self.binding_error_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "unsqueeze")
        bound = tensor.unsqueeze

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail(f"{module.__name__} unexpectedly exposed an unsqueeze signature")

        unbound_errors = (
            self.error(lambda: descriptor()),
            self.error(lambda: descriptor(self=tensor, dim=0)),
            self.error(lambda: descriptor(1, 0)),
        )
        return {
            "descriptor_type": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound) is types.BuiltinMethodType,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "owner": (
                descriptor.__objclass__.__name__,
                descriptor.__objclass__.__module__,
            ),
            "descriptor_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "repr": repr(descriptor),
            "identity": module.Tensor.unsqueeze is descriptor,
            "get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "subclass_dictionary": "unsqueeze" in module.Tensor.__dict__,
            "signature_error": signature_error(descriptor),
            "bound_signature_error": signature_error(bound),
            "positional_shape": tuple(descriptor(tensor, 0).shape),
            "keyword_shape": tuple(descriptor(tensor, dim=-3).shape),
            "unbound_errors": unbound_errors,
        }

    def test_tensorbase_descriptor_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "unsqueeze")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        records = []
        calls = (
            lambda: tensor.unsqueeze(0),
            lambda: tensor.unsqueeze(dim=0),
            lambda: tensor.unsqueeze(-3),
            lambda: tensor.unsqueeze(1),
            lambda: tensor.unsqueeze(9),
            lambda: tensor.unsqueeze(2**100),
        )
        for call in calls:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            function, dispatch_types, args, kwargs = mode.calls[0]
            records.append(
                (
                    result is marker,
                    len(mode.calls),
                    function is descriptor,
                    function.__qualname__,
                    dispatch_types,
                    tuple("input" if argument is tensor else argument for argument in args),
                    kwargs,
                )
            )

        invalid = RecordingMode(marker)
        invalid_error = self.error(lambda: self.call_inside_mode(invalid, tensor))

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(
                    (
                        self.label,
                        func is descriptor,
                        dispatch_types,
                        tuple(
                            "input" if argument is tensor else argument
                            for argument in args
                        ),
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.unsqueeze(dim=-3)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor.unsqueeze(0)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "invalid": invalid_error + (len(invalid.calls),),
            "forwarding": tuple(order),
            "forwarded": self.layout_contract(module, tensor, "negative")[1:],
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def call_inside_mode(self, mode, tensor):
        with mode:
            return tensor.unsqueeze("0")

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
