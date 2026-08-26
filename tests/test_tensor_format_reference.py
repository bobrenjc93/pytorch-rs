import copy
import inspect
import pickle
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SPECIAL_BITS = (
    0xC020_0000,
    0x3F9E_0652,
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC1_2345,
    0xFFC5_4321,
)
FORMAT_SPECS = ("", ".2f", "+08.2f", "e", "g", "%", "^12")


def scalar_view(module, bits, *, requires_grad=False):
    values = np.asarray((0x3F80_0000, bits), dtype=np.uint32).view(np.float32)
    leaf = module.tensor(
        memoryview(values),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    return leaf[1], leaf


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFormatReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.__format__ differentials require pinned PyTorch 2.13.0"
            )

    def metadata(self, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype).replace("torch_rs", "torch"),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def error(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        self.fail("Tensor.__format__ unexpectedly accepted an invalid call")

    def test_scalar_values_and_format_specs_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            actual, _ = scalar_view(torch, bits)
            expected, _ = scalar_view(reference_torch, bits)
            for format_spec in FORMAT_SPECS:
                with self.subTest(bits=f"{bits:#010x}", format_spec=format_spec):
                    self.assertEqual(
                        format(actual, format_spec),
                        format(expected, format_spec),
                    )

    def test_active_autograd_formatting_and_graph_state_match(self):
        for bits in SPECIAL_BITS:
            outcomes = []
            for module in (torch, reference_torch):
                tensor, leaf = scalar_view(module, bits, requires_grad=True)
                before = self.metadata(tensor), self.metadata(leaf), leaf.grad
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    values = tuple(format(tensor, spec) for spec in FORMAT_SPECS)
                after = self.metadata(tensor), self.metadata(leaf), leaf.grad
                tensor.backward()
                outcomes.append(
                    (
                        values,
                        tuple(
                            (warning.category.__name__, str(warning.message))
                            for warning in caught
                        ),
                        before == after,
                        leaf.grad.tolist(),
                    )
                )
            with self.subTest(bits=f"{bits:#010x}"):
                self.assertEqual(outcomes[0], outcomes[1])

    def non_scalar_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tensors = (
            module.zeros((0,)),
            module.tensor([1.0]),
            leaf,
            (leaf * 2.0).transpose(0, 1),
            module.zeros((2, 0, 3)).transpose(0, 2),
        )
        outcomes = []
        for tensor in tensors:
            before = self.metadata(tensor)
            empty = format(tensor, "")
            after_empty = self.metadata(tensor)
            error = self.error(lambda tensor=tensor: format(tensor, ".2f"))
            after_error = self.metadata(tensor)
            outcomes.append(
                (
                    tuple(tensor.shape),
                    tensor.stride(),
                    empty == str(tensor),
                    type(empty).__name__,
                    error,
                    before == after_empty == after_error,
                )
            )
        before_grad = leaf.grad
        leaf.sum().backward()
        return outcomes, before_grad, leaf.grad.tolist()

    def test_non_scalar_empty_and_nonempty_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.non_scalar_contract(torch),
            self.non_scalar_contract(reference_torch),
        )

    def callable_contract(self, module):
        tensor = module.tensor(1.25)
        function = inspect.getattr_static(module.Tensor, "__format__")
        bound = tensor.__format__
        return {
            "function_type": type(function).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(function) is types.FunctionType,
                type(bound) is types.MethodType,
            ),
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_module": bound.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "bound_doc": bound.__doc__,
            "annotations": function.__annotations__,
            "bound_annotations": bound.__annotations__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "text_signature": hasattr(function, "__text_signature__"),
            "bound_text_signature": hasattr(bound, "__text_signature__"),
            "signatures": (
                str(inspect.signature(function)),
                str(inspect.signature(bound)),
            ),
            "owned_by_tensor": "__format__" in module.Tensor.__dict__,
            "absent_from_bases": all(
                "__format__" not in owner.__dict__
                for owner in module.Tensor.__mro__[1:-1]
            ),
            "module_identity": module._tensor.Tensor is module.Tensor,
            "function_identity": module._tensor.Tensor.__format__ is function,
            "copy_identities": (
                copy.copy(function) is function,
                copy.deepcopy(function) is function,
            ),
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol))
                is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def direct_call_contract(self, module):
        tensor = module.tensor(1.25)
        function = inspect.getattr_static(module.Tensor, "__format__")
        bound = tensor.__format__
        calls = (
            lambda: function(),
            lambda: function(tensor),
            lambda: bound(),
            lambda: function(tensor, "", 1),
            lambda: bound("", 1),
            lambda: function(tensor, "", unexpected=True),
            lambda: function(tensor, "", self=tensor),
            lambda: function(tensor, "", format_spec=""),
            lambda: function(1.0, ""),
            lambda: function(tensor, object()),
        )
        return (
            function(self=tensor, format_spec=".1f"),
            tuple(self.error(call) for call in calls),
        )

    def test_direct_calls_and_binding_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.direct_call_contract(torch),
            self.direct_call_contract(reference_torch),
        )

    def dispatch_contract(self, module):
        function = inspect.getattr_static(module.Tensor, "__format__")
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        override = Override()
        override_result = function(override, ".2f")
        override_call = Override.calls[0]

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        tensor = module.tensor(-0.0)
        recording = RecordingMode(marker)
        with recording:
            intercepted = tensor.__format__(format_spec="+08.2f")
        mode_call = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = format(tensor, "+08.2f")

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                format(tensor, ".2f")
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(
                    r"0x[0-9a-f]+",
                    "0x...",
                    str(error).replace("torch_rs", "torch"),
                ),
            )
        else:
            declining_error = None

        return {
            "override_result": override_result is marker,
            "override_function": override_call[0] is function,
            "override_types": override_call[1] == (Override,),
            "override_args": override_call[2] == (override, ".2f"),
            "override_kwargs": override_call[3],
            "intercepted": intercepted is marker,
            "mode_function": mode_call[0] is function,
            "mode_types": mode_call[1] == (module.Tensor,),
            "mode_args": mode_call[2] == (tensor, "+08.2f"),
            "mode_kwargs": mode_call[3],
            "forwarding_order": order,
            "forwarded": forwarded,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "declining_types": (
                declining.calls[0][1] == (module.Tensor,),
                declining.calls[1][1] == (),
            ),
            "declining_args": all(
                call[2] == (tensor, ".2f") for call in declining.calls
            ),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_override_and_mode_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
