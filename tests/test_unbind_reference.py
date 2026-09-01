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

    def call_unbind(self, tensor, form, dimension=0):
        if form == "default":
            return tensor.unbind()
        if form == "positional":
            return tensor.unbind(dimension)
        if form == "keyword":
            return tensor.unbind(dim=dimension)
        if form == "negative":
            axis = dimension if dimension >= 0 else dimension + tensor.dim()
            return tensor.unbind(axis - tensor.dim())
        raise AssertionError(f"unknown call form: {form}")

    def view_contract(self, tensor, form, dimension=0):
        outputs = self.call_unbind(tensor, form, dimension)
        return self.output_contract(tensor, outputs, dimension)

    def output_contract(self, tensor, outputs, dimension=0):
        axis = dimension if dimension >= 0 else dimension + tensor.dim()
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
                    output.data_ptr() == tensor.select(axis, index).data_ptr(),
                    output.is_set_to(tensor.select(axis, index)),
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                    str(output.dtype),
                    str(output.device),
                )
                for index, output in enumerate(outputs)
            ),
        }

    def call_top_level_unbind(self, module, tensor, form, dimension=0):
        if form == "default":
            return module.unbind(tensor)
        if form == "positional":
            return module.unbind(tensor, dimension)
        if form == "keyword":
            return module.unbind(tensor, dim=dimension)
        if form == "all_keywords":
            return module.unbind(input=tensor, dim=dimension)
        if form == "alias":
            return module.unbind(x=tensor, dim=dimension)
        if form == "negative":
            axis = dimension if dimension >= 0 else dimension + tensor.dim()
            return module.unbind(tensor, axis - tensor.dim())
        raise AssertionError(f"unknown top-level call form: {form}")

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

    def arbitrary_dimension_cases(self, module):
        contiguous = module.tensor([float(value) for value in range(24)]).reshape(
            2, 3, 4
        )
        offset = module.tensor([float(value) for value in range(120)]).reshape(
            2, 3, 4, 5
        )[1]
        noncontiguous = module.tensor(
            [float(value) for value in range(48)]
        ).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        empty_middle = module.zeros((2, 3, 0, 4))
        empty_unbound = module.zeros((2, 0, 3))
        return (
            ("contiguous middle", contiguous, 1),
            ("contiguous trailing", contiguous, 2),
            ("offset middle", offset, 1),
            ("noncontiguous middle", noncontiguous, 1),
            ("negative trailing", noncontiguous, -1),
            ("empty retained dimension", empty_middle, 1),
            ("empty unbound dimension", empty_unbound, 1),
        )

    def test_arbitrary_dimension_views_match_pytorch_2_13(self):
        actual_cases = self.arbitrary_dimension_cases(torch)
        expected_cases = self.arbitrary_dimension_cases(reference_torch)
        for (case, actual, actual_dimension), (_, expected, expected_dimension) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for form in ("positional", "keyword"):
                with self.subTest(case=case, form=form):
                    self.assertEqual(
                        self.view_contract(actual, form, actual_dimension),
                        self.view_contract(expected, form, expected_dimension),
                    )

    def test_top_level_values_layout_offsets_and_aliasing_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for form in (
                "default",
                "positional",
                "keyword",
                "all_keywords",
                "alias",
                "negative",
            ):
                with self.subTest(case=case, form=form):
                    actual_outputs = self.call_top_level_unbind(torch, actual, form)
                    expected_outputs = self.call_top_level_unbind(
                        reference_torch, expected, form
                    )
                    self.assertEqual(
                        self.output_contract(actual, actual_outputs),
                        self.output_contract(expected, expected_outputs),
                    )

    def test_top_level_arbitrary_dimension_views_match_pytorch_2_13(self):
        actual_cases = self.arbitrary_dimension_cases(torch)
        expected_cases = self.arbitrary_dimension_cases(reference_torch)
        for (case, actual, actual_dimension), (_, expected, expected_dimension) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for form in ("positional", "keyword", "all_keywords", "alias"):
                with self.subTest(case=case, form=form):
                    actual_outputs = self.call_top_level_unbind(
                        torch, actual, form, actual_dimension
                    )
                    expected_outputs = self.call_top_level_unbind(
                        reference_torch, expected, form, expected_dimension
                    )
                    self.assertEqual(
                        self.output_contract(actual, actual_outputs, actual_dimension),
                        self.output_contract(
                            expected, expected_outputs, expected_dimension
                        ),
                    )

    def autograd_contract(self, module, *, top_level=False):
        call = module.unbind if top_level else lambda input: input.unbind()
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        outputs = call(source)
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
            no_grad_outputs = call(no_grad_source)

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_outputs = call(empty)
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
            "empty_outer": call(module.zeros((0, 2), requires_grad=True)),
        }

    def test_output_numbers_autograd_no_grad_and_empty_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def test_top_level_output_numbers_autograd_no_grad_and_empty_match_pytorch_2_13(
        self,
    ):
        self.assertEqual(
            self.autograd_contract(torch, top_level=True),
            self.autograd_contract(reference_torch, top_level=True),
        )

    def arbitrary_dimension_autograd_contract(self, module, *, top_level=False):
        def call(tensor, dimension):
            if top_level:
                return module.unbind(tensor, dimension)
            return tensor.unbind(dimension)

        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        outputs = call(source, 1)
        output_metadata = tuple(
            (
                output.output_nr,
                output.requires_grad,
                output.is_leaf,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.data_ptr() == source.select(1, index).data_ptr(),
                output.is_set_to(source.select(1, index)),
            )
            for index, output in enumerate(outputs)
        )
        loss = outputs[0].sum()
        for output in outputs[1:]:
            loss = loss + output.sum()
        loss.backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        with module.no_grad():
            no_grad_outputs = call(no_grad_source, 1)

        empty = module.zeros((2, 3, 0, 4), requires_grad=True)
        empty_outputs = call(empty, 1)
        empty_metadata = tuple(
            (
                output.output_nr,
                output.requires_grad,
                output.is_leaf,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.is_set_to(empty.select(1, index)),
            )
            for index, output in enumerate(empty_outputs)
        )
        empty_outputs[2].sum().backward()

        return {
            "output_metadata": output_metadata,
            "gradient": leaf.grad.tolist(),
            "no_grad": tuple(
                (
                    output.output_nr,
                    output.requires_grad,
                    output.is_leaf,
                    output.is_set_to(no_grad_source.select(1, index)),
                )
                for index, output in enumerate(no_grad_outputs)
            ),
            "empty_metadata": empty_metadata,
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
            "empty_unbound": call(module.zeros((2, 0, 3), requires_grad=True), 1),
        }

    def test_arbitrary_dimension_autograd_no_grad_and_empty_match_pytorch_2_13(
        self,
    ):
        self.assertEqual(
            self.arbitrary_dimension_autograd_contract(torch),
            self.arbitrary_dimension_autograd_contract(reference_torch),
        )

    def test_top_level_arbitrary_dimension_autograd_no_grad_and_empty_match_pytorch_2_13(
        self,
    ):
        self.assertEqual(
            self.arbitrary_dimension_autograd_contract(torch, top_level=True),
            self.arbitrary_dimension_autograd_contract(reference_torch, top_level=True),
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

    def top_level_supported_error_contract(self, module):
        tensor = module.zeros((2, 3))
        scalar = module.tensor(1.0)
        return (
            self.error(lambda: module.unbind()),
            self.error(lambda: module.unbind(dim=0)),
            self.error(lambda: module.unbind(tensor, 0, 0)),
            self.error(lambda: module.unbind(tensor, input=tensor)),
            self.error(lambda: module.unbind(tensor, 0, dim=0)),
            self.error(lambda: module.unbind(tensor, extra=0)),
            self.error(lambda: module.unbind(x=tensor, extra=0)),
            self.error(lambda: module.unbind([], 0)),
            self.error(lambda: module.unbind(input=[], dim=0)),
            self.error(lambda: module.unbind(tensor, None)),
            self.error(lambda: module.unbind(tensor, 0.0)),
            self.error(lambda: module.unbind(tensor, True)),
            self.error(lambda: module.unbind(tensor, dim="0")),
            self.error(lambda: module.unbind(tensor, "0", extra=True)),
            self.error(lambda: module.unbind(tensor, 2**100)),
            self.error(lambda: module.unbind(scalar)),
            self.error(lambda: module.unbind(scalar, -1)),
            self.error(lambda: module.unbind(tensor, 2)),
            len(module.unbind(tensor, np.int64(0))),
        )

    def test_top_level_supported_call_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_supported_error_contract(torch),
            self.top_level_supported_error_contract(reference_torch),
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

    def top_level_mode_contract(self, module):
        tensor = module.zeros((2, 3))
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
            lambda: module.unbind(tensor),
            lambda: module.unbind(tensor, 0),
            lambda: module.unbind(tensor, dim=0),
            lambda: module.unbind(input=tensor, dim=0),
            lambda: module.unbind(tensor, 1),
            lambda: module.unbind(tensor, 2**100),
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
                    function is module.unbind,
                    function.__qualname__,
                    dispatch_types,
                    tuple("input" if argument is tensor else argument for argument in args),
                    {
                        key: "input" if value is tensor else value
                        for key, value in kwargs.items()
                    }
                    if kwargs
                    else kwargs,
                )
            )

        invalid = RecordingMode(marker)
        try:
            with invalid:
                module.unbind(tensor, "0")
        except Exception as error:
            invalid_error = type(error).__name__, str(error), len(invalid.calls)
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
                        func is module.unbind,
                        dispatch_types,
                        tuple(
                            "input" if argument is tensor else argument
                            for argument in args
                        ),
                        {
                            key: "input" if value is tensor else value
                            for key, value in kwargs.items()
                        }
                        if kwargs
                        else kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.unbind(input=tensor, dim=0)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                module.unbind(tensor)
        except Exception as error:
            declining_error = type(error).__name__, str(error).splitlines()[0]
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "invalid": invalid_error,
            "forwarding_order": tuple(order),
            "forwarded": self.output_contract(tensor, forwarded),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_top_level_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_mode_contract(torch),
            self.top_level_mode_contract(reference_torch),
        )

    def top_level_override_contract(self, module):
        marker = object()
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                calls.append(
                    (
                        func is module.unbind,
                        func.__qualname__,
                        dispatch_types == (Override,),
                        tuple("input" if argument is value else argument for argument in args),
                        {
                            key: "input" if item is value else item
                            for key, item in kwargs.items()
                        }
                        if kwargs
                        else kwargs,
                    )
                )
                return marker

        value = Override()
        replacements = tuple(
            call() is marker
            for call in (
                lambda: module.unbind(value),
                lambda: module.unbind(value, 0),
                lambda: module.unbind(input=value, dim=0),
                lambda: module.unbind(value, 1),
                lambda: module.unbind(value, 2**100),
            )
        )
        call_count = len(calls)
        try:
            module.unbind(value, "0")
        except Exception as error:
            invalid = type(error).__name__, str(error), len(calls) - call_count
        else:
            self.fail(f"{module.__name__} dispatched an invalid dimension")

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.unbind(DecliningOverride())
        except Exception as error:
            declining = type(error).__name__, str(error).splitlines()[0]
        else:
            self.fail(f"{module.__name__} accepted a declining override")

        return replacements, tuple(calls), invalid, declining

    def test_top_level_tensor_like_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_override_contract(torch),
            self.top_level_override_contract(reference_torch),
        )

    def top_level_callable_contract(self, module):
        function = module.unbind
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
            "owner_callable_identity": owner.unbind is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("unbind"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["unbind"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
