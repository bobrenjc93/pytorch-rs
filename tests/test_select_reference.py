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


@unittest.skipIf(reference_torch is None, "install the reference PyTorch package")
class TensorSelectReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("select differentials require pinned PyTorch 2.13.0")

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("select unexpectedly accepted an invalid call")

    def source(self, module, *, requires_grad=False):
        values = [float(value) for value in range(48)]
        return module.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)

    def view_contract(self, module):
        source = self.source(module)
        leading = source[1]
        final = source.permute(2, 0, 1)[2]
        calls = (
            (source.select(0, 1), leading),
            (source.select(0, index=1), leading),
            (source.select(dim=0, index=1), leading),
            (source.select(index=1, dim=0), leading),
            (source.select(-3, -2), leading),
            (source.select(2, 2), final),
            (source.select(-1, 2), final),
            (source.select(index=2, dim=-1), final),
            (source.select(2, -2), final),
        )
        rows = []
        for selected, direct in calls:
            rows.append(
                {
                    "values": selected.tolist(),
                    "shape": tuple(selected.shape),
                    "stride": selected.stride(),
                    "offset": selected.storage_offset(),
                    "data_ptr_matches_direct": selected.data_ptr()
                    == direct.data_ptr(),
                    "is_set_to_direct": selected.is_set_to(direct),
                    "same_dtype": selected.dtype is source.dtype,
                    "same_device": selected.device == source.device,
                }
            )
        empty_source = module.zeros((2, 0, 3))
        for empty in (
            empty_source.select(0, 1),
            empty_source.select(-1, 1),
        ):
            rows.append(
                {
                    "empty_shape": tuple(empty.shape),
                    "empty_stride": empty.stride(),
                    "empty_offset": empty.storage_offset(),
                    "empty_data_ptr": empty.data_ptr(),
                    "empty_values": empty.tolist(),
                }
            )
        return rows

    def test_values_layout_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(
            self.view_contract(torch),
            self.view_contract(reference_torch),
        )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = source.select(-1, 2)
        metadata = (
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
        )
        (selected.transpose(0, 1) * 3.0).sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            untracked = no_grad_source.select(dim=-1, index=1)

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_selected = empty.select(-1, 1)
        empty_selected.sum().backward()
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
                untracked.is_set_to(no_grad_source.permute(1, 0)[1]),
            ),
            "empty_metadata": (
                empty_selected.requires_grad,
                empty_selected.is_leaf,
                empty_selected.output_nr,
                tuple(empty_selected.shape),
                empty_selected.stride(),
                empty_selected.storage_offset(),
                empty_selected.data_ptr(),
                empty_selected.is_set_to(empty.permute(2, 0, 1)[1]),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_autograd_no_grad_and_downstream_operations_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def error_contract(self, module):
        tensor = module.zeros((2, 3, 4))
        scalar = module.tensor(1.0)
        empty = module.zeros((0, 2))
        return (
            self.error(lambda: tensor.select()),
            self.error(lambda: tensor.select(0)),
            self.error(lambda: tensor.select(index=1)),
            self.error(lambda: tensor.select(0, 1, 2)),
            self.error(lambda: tensor.select(0, 1, dim=0)),
            self.error(lambda: tensor.select(0, 1, index=0)),
            self.error(lambda: tensor.select(0, 1, extra=0)),
            self.error(lambda: tensor.select(None, 0)),
            self.error(lambda: tensor.select(dim="0", index=0)),
            self.error(lambda: tensor.select(0, True)),
            self.error(lambda: tensor.select(dim=0, index=1.0)),
            self.error(lambda: tensor.select(2**100, "bad")),
            self.error(lambda: tensor.select(2**100, 0)),
            self.error(lambda: tensor.select(0, 2**100)),
            self.error(lambda: tensor.select(0, 2)),
            self.error(lambda: tensor.select(-3, -3)),
            self.error(lambda: tensor.select(2, 4)),
            self.error(lambda: tensor.select(-1, -5)),
            self.error(lambda: module.zeros((2, 3, 0)).select(-1, 0)),
            self.error(lambda: tensor.select(3, 0)),
            self.error(lambda: tensor.select(-4, 0)),
            self.error(lambda: scalar.select(0, 0)),
            self.error(lambda: scalar.select(-2, 99)),
            self.error(lambda: empty.select(0, 0)),
            tuple(tensor.select(np.int64(0), np.int32(1)).shape),
            tuple(tensor.select(np.int64(-1), np.int32(1)).shape),
        )

    def test_supported_binding_bounds_and_scalar_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch),
            self.error_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3))
        descriptor = inspect.getattr_static(module.Tensor, "select")
        bound = tensor.select

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail("select unexpectedly exposed an inspectable signature")

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
            "class_identity": module.Tensor.select is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "descriptor_signature": signature_error(descriptor),
            "bound_signature": signature_error(bound),
            "call_shape": tuple(descriptor(tensor, 0, 1).shape),
            "no_receiver": self.error(lambda: descriptor()),
            "wrong_receiver": self.error(lambda: descriptor(1, 0, 0)),
            "keyword_receiver": self.error(
                lambda: descriptor(self=tensor, dim=0, index=1)
            ),
        }

    def test_tensorbase_descriptor_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(module.Tensor, "select")
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
            positional_result = tensor.select(2, 1)
        positional_call = positional.calls[0]

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.select(index=1, dim=-1)
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
                forwarded = tensor.select(dim=-1, index=1)

        index_calls = []

        class CustomIndex:
            def __index__(self):
                index_calls.append("index")
                return 1

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.select(2**100, CustomIndex())

        invalid = RecordingMode(marker)
        invalid_error = self.error(
            lambda: self.call_inside_mode(invalid, lambda: tensor.select(True, 0))
        )

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.select(0, 1)
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
            "forwarding_order": order,
            "forwarded_shape": tuple(forwarded.shape),
            "forwarded_stride": forwarded.stride(),
            "forwarded_offset": forwarded.storage_offset(),
            "forwarded_is_set_to": forwarded.is_set_to(
                tensor.permute(2, 0, 1)[1]
            ),
            "deferred_result": deferred_result is marker,
            "deferred_calls": len(deferred.calls),
            "index_calls": index_calls,
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
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

    def test_custom_index_conversion_order_matches_pytorch_2_13(self):
        def observation(module):
            calls = []

            class StatefulIndex:
                def __index__(self):
                    calls.append("index")
                    return (0, 1, 0)[len(calls) - 1]

            selected = module.zeros((2, 3)).select(-1, StatefulIndex())
            return calls, selected.storage_offset(), tuple(selected.shape)

        self.assertEqual(observation(torch), observation(reference_torch))

    def top_level_view_contract(self, module):
        source = self.source(module)
        direct = source[1]
        calls = (
            module.select(source, 0, 1),
            module.select(source, 0, index=1),
            module.select(source, dim=0, index=1),
            module.select(input=source, dim=0, index=1),
            module.select(index=1, input=source, dim=0),
            module.select(x=source, dim=0, index=1),
            module.select(a=source, dim=0, index=1),
            module.select(x1=source, dim=0, index=1),
            module.select(source, -3, -2),
        )
        observations = tuple(
            {
                "values": selected.tolist(),
                "shape": tuple(selected.shape),
                "stride": selected.stride(),
                "offset": selected.storage_offset(),
                "data_ptr_matches_direct": selected.data_ptr() == direct.data_ptr(),
                "is_set_to_direct": selected.is_set_to(direct),
                "same_dtype": selected.dtype is source.dtype,
                "same_device": selected.device == source.device,
            }
            for selected in calls
        )
        empty = module.select(module.zeros((2, 0, 3)), 0, 1)
        scalar = module.select(module.tensor([1.0, 2.0, 3.0]), -1, -1)
        return observations, {
            "empty_shape": tuple(empty.shape),
            "empty_stride": empty.stride(),
            "empty_offset": empty.storage_offset(),
            "empty_data_ptr": empty.data_ptr(),
            "empty_values": empty.tolist(),
            "scalar_shape": tuple(scalar.shape),
            "scalar_stride": scalar.stride(),
            "scalar_offset": scalar.storage_offset(),
            "scalar_value": scalar.item(),
        }

    def test_top_level_values_layout_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_view_contract(torch),
            self.top_level_view_contract(reference_torch),
        )

    def top_level_autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = module.select(source, -3, 1)
        metadata = (
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
            selected.data_ptr() == source[1].data_ptr(),
            selected.is_set_to(source[1]),
        )
        (selected.transpose(0, 1) * 3.0).sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            untracked = module.select(input=no_grad_source, dim=0, index=1)

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_selected = module.select(empty, 0, 1)
        empty_selected.sum().backward()
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
                untracked.is_set_to(no_grad_source[1]),
            ),
            "empty": (
                empty_selected.requires_grad,
                empty_selected.is_leaf,
                empty_selected.output_nr,
                tuple(empty_selected.shape),
                empty_selected.stride(),
                empty_selected.storage_offset(),
                empty_selected.data_ptr(),
                empty_selected.is_set_to(empty[1]),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_top_level_autograd_no_grad_and_output_number_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_autograd_contract(torch),
            self.top_level_autograd_contract(reference_torch),
        )

    def top_level_error_contract(self, module):
        tensor = module.zeros((2, 3, 4))
        scalar = module.tensor(1.0)
        empty = module.zeros((0, 2))
        return (
            self.error(lambda: module.select()),
            self.error(lambda: module.select(tensor)),
            self.error(lambda: module.select(tensor, 0)),
            self.error(lambda: module.select(dim=0, index=1)),
            self.error(lambda: module.select(tensor, 0, 1, 2)),
            self.error(lambda: module.select(tensor, 0, 1, input=tensor)),
            self.error(lambda: module.select(tensor, 0, 1, dim=0)),
            self.error(lambda: module.select(tensor, 0, 1, index=0)),
            self.error(lambda: module.select(tensor, 0, 1, extra=0)),
            self.error(lambda: module.select(x=tensor, dim=0, index=1, extra=0)),
            self.error(lambda: module.select([], 0, 1)),
            self.error(lambda: module.select(input=[], dim=0, index=1)),
            self.error(lambda: module.select(tensor, None, 0)),
            self.error(lambda: module.select(tensor, dim="0", index=0)),
            self.error(lambda: module.select(tensor, 0, True)),
            self.error(lambda: module.select(tensor, dim=0, index=1.0)),
            self.error(lambda: module.select(tensor, 2**100, "bad")),
            self.error(lambda: module.select(tensor, 2**100, 0)),
            self.error(lambda: module.select(tensor, 0, 2**100)),
            self.error(lambda: module.select(tensor, 0, 2)),
            self.error(lambda: module.select(tensor, -3, -3)),
            self.error(lambda: module.select(tensor, 3, 0)),
            self.error(lambda: module.select(tensor, -4, 0)),
            self.error(lambda: module.select(scalar, 0, 0)),
            self.error(lambda: module.select(scalar, -2, 99)),
            self.error(lambda: module.select(empty, 0, 0)),
            tuple(module.select(tensor, np.int64(0), np.int32(1)).shape),
        )

    def test_top_level_supported_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_error_contract(torch),
            self.top_level_error_contract(reference_torch),
        )

    def top_level_mode_contract(self, module):
        tensor = module.zeros((2, 3, 4))
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        records = []
        calls = (
            lambda: module.select(tensor, 0, 1),
            lambda: module.select(tensor, 0, index=1),
            lambda: module.select(input=tensor, dim=0, index=1),
            lambda: module.select(x=tensor, dim=0, index=1),
            lambda: module.select(tensor, 1, 0),
            lambda: module.select(tensor, 2**100, 0),
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
                    function is module.select,
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

        index_calls = []

        class CustomIndex:
            def __index__(self):
                index_calls.append("index")
                return 1

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = module.select(tensor, 2**100, CustomIndex())

        invalid = RecordingMode(marker)
        invalid_error = self.error(
            lambda: self.call_inside_mode(
                invalid, lambda: module.select(tensor, "0", 1)
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
                forwarded = module.select(input=tensor, dim=0, index=1)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                module.select(tensor, 0, 1)
        except Exception as error:
            declining_error = type(error).__name__, str(error).splitlines()[0]
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "deferred": deferred_result is marker,
            "deferred_calls": len(deferred.calls),
            "index_calls": index_calls,
            "invalid": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding_order": order,
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.is_set_to(tensor[1]),
            ),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_top_level_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_mode_contract(torch),
            self.top_level_mode_contract(reference_torch),
        )

    def top_level_override_contract(self, module):
        marker = object()
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        func is module.select,
                        func.__qualname__,
                        types == (Override,),
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
                lambda: module.select(value, 0, 1),
                lambda: module.select(input=value, dim=0, index=1),
                lambda: module.select(x=value, dim=0, index=1),
                lambda: module.select(value, 1, 0),
                lambda: module.select(value, 2**100, 2**100),
            )
        )
        call_count = len(calls)
        invalid = self.error(lambda: module.select(value, 0, "1"))
        invalid = (*invalid, len(calls) - call_count)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.select(DecliningOverride(), 0, 1)
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
        function = module.select
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
            "owner_callable_identity": owner.select is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("select"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["select"] is function,
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
