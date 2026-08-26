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
class TensorReshapeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("reshape differentials require pinned PyTorch 2.13.0")

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("reshape unexpectedly accepted an invalid call")

    def layout_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist())
        offset_source = base[1]
        non_contiguous = base.transpose(0, 1)
        cases = (
            ("tuple", base, lambda: module.reshape(base, (4, 6))),
            ("list", base, lambda: module.reshape(base, [4, 6])),
            ("size", base, lambda: module.reshape(base, module.Size([4, 6]))),
            (
                "keyword",
                base,
                lambda: module.reshape(input=base, shape=(4, 6)),
            ),
            (
                "offset view",
                offset_source,
                lambda: module.reshape(offset_source, (2, 6)),
            ),
            (
                "noncontiguous copy",
                non_contiguous,
                lambda: module.reshape(non_contiguous, (6, 4)),
            ),
        )
        observations = []
        for name, source, call in cases:
            output = call()
            method_output = source.reshape(tuple(output.shape))
            observations.append(
                {
                    "name": name,
                    "values": output.tolist(),
                    "shape": tuple(output.shape),
                    "stride": output.stride(),
                    "offset": output.storage_offset(),
                    "contiguous": output.is_contiguous(),
                    "aliases_source_pointer": output.data_ptr() == source.data_ptr(),
                    "matches_method_storage": output.is_set_to(method_output),
                    "fresh_wrapper": output is not source,
                    "same_dtype": output.dtype is source.dtype,
                    "same_device": output.device == source.device,
                }
            )

        scalar = module.reshape(module.tensor([7.0]), ())
        empty = module.reshape(module.zeros((0,)), module.Size([2, 0, 3]))
        return observations, {
            "scalar": (
                scalar.item(),
                tuple(scalar.shape),
                scalar.stride(),
                scalar.storage_offset(),
            ),
            "empty": (
                empty.tolist(),
                tuple(empty.shape),
                empty.stride(),
                empty.storage_offset(),
                empty.data_ptr(),
            ),
        }

    def test_values_strides_offsets_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(
            self.layout_contract(torch),
            self.layout_contract(reference_torch),
        )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = module.reshape(leaf.transpose(0, 1), (6,))
        metadata = (
            output.requires_grad,
            output.is_leaf,
            output.output_nr,
            tuple(output.shape),
            output.stride(),
            output.storage_offset(),
        )
        weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()

        view_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        with module.no_grad():
            view = module.reshape(view_leaf, (2, 1))
            copy = module.reshape(
                module.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
                ).transpose(0, 1),
                (6,),
            )

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_output = module.reshape(empty, (0, 6))
        empty_output.sum().backward()
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "view_no_grad": (
                view.requires_grad,
                view.is_leaf,
                view.output_nr,
                view.data_ptr() == view_leaf.data_ptr(),
            ),
            "copy_no_grad": (copy.requires_grad, copy.is_leaf, copy.output_nr),
            "empty_output": (
                empty_output.requires_grad,
                empty_output.is_leaf,
                empty_output.output_nr,
                tuple(empty_output.shape),
                empty_output.stride(),
            ),
            "empty_gradient": (tuple(empty.grad.shape), empty.grad.tolist()),
        }

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def error_contract(self, module):
        tensor = module.zeros((6,))
        return (
            self.error(lambda: module.reshape()),
            self.error(lambda: module.reshape(tensor)),
            self.error(lambda: module.reshape(shape=(2, 3))),
            self.error(lambda: module.reshape(tensor, 2, 3)),
            self.error(lambda: module.reshape(tensor, (2, 3), input=tensor)),
            self.error(lambda: module.reshape(tensor, (2, 3), shape=(2, 3))),
            self.error(lambda: module.reshape(tensor, (2, 3), extra=True)),
            self.error(lambda: module.reshape([], (0,))),
            self.error(lambda: module.reshape(input=None, shape=())),
            self.error(lambda: module.reshape(tensor, 6)),
            self.error(lambda: module.reshape(tensor, shape=6)),
            self.error(lambda: module.reshape(tensor, (2.0, 3))),
            self.error(lambda: module.reshape(tensor, shape=(2.0, 3))),
            self.error(lambda: module.reshape(tensor, (2, 3.0))),
            self.error(lambda: module.reshape(tensor, (True, 6))),
            self.error(lambda: module.reshape(tensor, ((2, 3),))),
            self.error(lambda: module.reshape(tensor, (4, 2))),
            self.error(lambda: module.reshape(tensor, (-1, -1))),
            tuple(module.reshape(module.zeros((2,)), (2, True)).shape),
        )

    def test_binding_and_shape_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch),
            self.error_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((6,))
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
            lambda: module.reshape(tensor, (2, 3)),
            lambda: module.reshape(tensor, shape=[2, 3]),
            lambda: module.reshape(input=tensor, shape=module.Size([2, 3])),
            lambda: module.reshape(x=tensor, shape=(2, 3)),
            lambda: module.reshape(tensor, (4, 2)),
            lambda: module.reshape(tensor, (2, 3.0)),
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
                    function is module.reshape,
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
        invalid_error = self.error(
            lambda: self.call_inside_mode(
                invalid, lambda: module.reshape(tensor, (2.0, 3))
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
                forwarded = module.reshape(input=tensor, shape=(2, 3))

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                module.reshape(tensor, (2, 3))
        except Exception as error:
            declining_error = type(error).__name__, str(error).splitlines()[0]
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "invalid": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding_order": order,
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
            ),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    @staticmethod
    def call_inside_mode(mode, call):
        with mode:
            return call()

    def test_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def override_contract(self, module):
        marker = object()
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        func is module.reshape,
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
                lambda: module.reshape(value, (2, 3)),
                lambda: module.reshape(input=value, shape=(2, 3)),
                lambda: module.reshape(x=value, shape=(2, 3)),
                lambda: module.reshape(value, (4, 2)),
                lambda: module.reshape(value, (2, 3.0)),
            )
        )
        call_count = len(calls)
        invalid = self.error(lambda: module.reshape(value, (2.0, 3)))
        invalid = (*invalid, len(calls) - call_count)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.reshape(DecliningOverride(), (2, 3))
        except Exception as error:
            declining = type(error).__name__, str(error).splitlines()[0]
        else:
            self.fail(f"{module.__name__} accepted a declining override")
        return replacements, tuple(calls), invalid, declining

    def test_tensor_like_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.override_contract(torch),
            self.override_contract(reference_torch),
        )

    def callable_contract(self, module):
        function = module.reshape
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
            "owner_callable_identity": owner.reshape is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("reshape"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["reshape"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
