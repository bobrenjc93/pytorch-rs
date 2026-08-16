import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorUnbindReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("unbind differentials require pinned PyTorch 2.13.0")

    def layout_cases(self, module):
        offset_noncontiguous = module.tensor(
            [float(value) for value in range(48)]
        ).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        return (
            module.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            offset_noncontiguous,
            module.zeros((2, 0, 3)),
            module.zeros((0, 2)),
        )

    def call_unbind(self, tensor, form):
        if form == "default":
            return tensor.unbind()
        if form == "positional":
            return tensor.unbind(0)
        if form == "keyword":
            return tensor.unbind(dim=0)
        if form == "negative":
            return tensor.unbind(-tensor.dim())
        raise AssertionError(f"unknown call form: {form}")

    def view_contract(self, tensor, form):
        outputs = self.call_unbind(tensor, form)
        return {
            "result_type": type(outputs).__name__,
            "source": (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
            ),
            "outputs": tuple(
                (
                    output.tolist(),
                    tuple(output.shape),
                    output.stride(),
                    output.storage_offset(),
                    output.data_ptr() == tensor[index].data_ptr(),
                    output.is_set_to(tensor[index]),
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                    str(output.dtype),
                    str(output.device),
                )
                for index, output in enumerate(outputs)
            ),
        }

    def test_values_layout_offsets_and_aliasing_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for form in ("default", "positional", "keyword", "negative"):
                with self.subTest(case=case, form=form):
                    self.assertEqual(
                        self.view_contract(actual, form),
                        self.view_contract(expected, form),
                    )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        outputs = source.unbind()
        output_metadata = tuple(
            (
                output.output_nr,
                output.requires_grad,
                output.is_leaf,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.data_ptr() == source[index].data_ptr(),
                output.is_set_to(source[index]),
            )
            for index, output in enumerate(outputs)
        )
        (outputs[0] * outputs[2]).sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        with module.no_grad():
            no_grad_outputs = no_grad_source.unbind()

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_outputs = empty.unbind()
        empty_metadata = tuple(
            (
                output.output_nr,
                output.requires_grad,
                output.is_leaf,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.is_set_to(empty[index]),
            )
            for index, output in enumerate(empty_outputs)
        )
        empty_outputs[1].sum().backward()

        return {
            "output_metadata": output_metadata,
            "gradient": leaf.grad.tolist(),
            "no_grad": tuple(
                (
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                    output.is_set_to(no_grad_source[index]),
                )
                for index, output in enumerate(no_grad_outputs)
            ),
            "empty_metadata": empty_metadata,
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
            "empty_outer": module.zeros((0, 2), requires_grad=True).unbind(),
        }

    def test_output_numbers_autograd_no_grad_and_empty_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("unbind unexpectedly accepted an invalid call")

    def supported_error_contract(self, module):
        tensor = module.zeros((2, 3))
        scalar = module.tensor(1.0)
        return (
            self.error(lambda: tensor.unbind(0, 0)),
            self.error(lambda: tensor.unbind(0, dim=0)),
            self.error(lambda: tensor.unbind(extra=0)),
            self.error(lambda: tensor.unbind(None)),
            self.error(lambda: tensor.unbind(0.0)),
            self.error(lambda: tensor.unbind(True)),
            self.error(lambda: tensor.unbind(dim="0")),
            self.error(lambda: tensor.unbind("0", extra=True)),
            self.error(lambda: tensor.unbind(2**100)),
            self.error(lambda: scalar.unbind()),
            self.error(lambda: scalar.unbind(0)),
            self.error(lambda: scalar.unbind(-1)),
            self.error(lambda: scalar.unbind(1)),
            self.error(lambda: scalar.unbind(-2)),
            self.error(lambda: tensor.unbind(2)),
        )

    def test_supported_call_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_error_contract(torch),
            self.supported_error_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3))
        descriptor = inspect.getattr_static(module.Tensor, "unbind")
        bound = tensor.unbind
        try:
            inspect.signature(descriptor)
        except Exception as error:
            descriptor_signature_error = type(error).__name__
        else:
            self.fail(f"{module.__name__} exposed a descriptor signature")
        try:
            inspect.signature(bound)
        except Exception as error:
            bound_signature_error = type(error).__name__
        else:
            self.fail(f"{module.__name__} exposed a bound signature")

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
            "class_identity": module.Tensor.unbind is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "call_lengths": (
                len(descriptor(tensor)),
                len(descriptor(tensor, 0)),
                len(descriptor(tensor, dim=0)),
            ),
            "descriptor_signature_error": descriptor_signature_error,
            "bound_signature_error": bound_signature_error,
            "no_receiver": self.error(lambda: descriptor()),
            "keyword_receiver": self.error(lambda: descriptor(self=tensor)),
            "wrong_receiver": self.error(lambda: descriptor(1)),
        }

    def test_descriptor_metadata_and_binding_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3))
        descriptor = inspect.getattr_static(module.Tensor, "unbind")
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
            lambda: tensor.unbind(),
            lambda: tensor.unbind(0),
            lambda: tensor.unbind(dim=0),
            lambda: tensor.unbind(1),
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
                    type(function).__name__,
                    function.__qualname__,
                    dispatch_types,
                    tuple("self" if argument is tensor else argument for argument in args),
                    kwargs,
                )
            )

        invalid = RecordingMode(marker)
        try:
            with invalid:
                tensor.unbind("0")
        except Exception as error:
            invalid_error = type(error).__name__, str(error)
        else:
            self.fail(f"{module.__name__} accepted an invalid dimension")

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
                            "self" if argument is tensor else argument for argument in args
                        ),
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.unbind(dim=0)

        return {
            "records": tuple(records),
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding_order": tuple(order),
            "forwarded": self.view_contract(tensor, "default"),
            "forwarded_matches": all(
                output.is_set_to(tensor[index])
                for index, output in enumerate(forwarded)
            ),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
