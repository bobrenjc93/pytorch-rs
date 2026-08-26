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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFormatReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.__format__ differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def float32_from_bits(bits):
        return np.asarray(bits, dtype=np.uint32).view(np.float32).item()

    @staticmethod
    def metadata(tensor):
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.__format__ unexpectedly accepted the call")

    def scalar_contract(self, module, value, format_spec, *, offset_view):
        tensor = (
            module.tensor([9.0, value], dtype=module.float32)[1]
            if offset_view
            else module.tensor(value, dtype=module.float32)
        )
        before = self.metadata(tensor)
        formatted = format(tensor, format_spec)
        return {
            "formatted": formatted,
            "formatted_type": type(formatted).__name__,
            "metadata_unchanged": self.metadata(tensor) == before,
        }

    def test_scalar_values_and_specs_match_pytorch_2_13(self):
        values = (
            self.float32_from_bits(0x0000_0000),
            self.float32_from_bits(0x8000_0000),
            self.float32_from_bits(0x3FA0_0000),
            self.float32_from_bits(0xC020_0000),
            self.float32_from_bits(0x7F80_0000),
            self.float32_from_bits(0xFF80_0000),
            self.float32_from_bits(0x7FC0_0000),
            self.float32_from_bits(0xFFC0_0000),
        )
        format_specs = (
            "",
            ".2f",
            "+.3e",
            " 012.4f",
            "#.0f",
            "%",
            "^15",
            "F",
        )

        for value in values:
            for offset_view in (False, True):
                for format_spec in format_specs:
                    with self.subTest(
                        value=value,
                        offset_view=offset_view,
                        format_spec=format_spec,
                    ):
                        self.assertEqual(
                            self.scalar_contract(
                                torch,
                                value,
                                format_spec,
                                offset_view=offset_view,
                            ),
                            self.scalar_contract(
                                reference_torch,
                                value,
                                format_spec,
                                offset_view=offset_view,
                            ),
                        )

    def autograd_contract(self, module):
        leaf = module.tensor(1.25, dtype=module.float32, requires_grad=True)
        output = leaf * -2.0
        leaf_before = self.metadata(leaf)
        output_before = self.metadata(output)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            formatted = (format(leaf, "+.2f"), format(output, ".3e"))
        unchanged = (
            self.metadata(leaf) == leaf_before
            and self.metadata(output) == output_before
            and leaf.grad is None
        )
        output.backward()
        return {
            "formatted": formatted,
            "warnings": tuple(
                (item.category.__name__, str(item.message)) for item in caught
            ),
            "unchanged_before_backward": unchanged,
            "gradient": leaf.grad.item(),
        }

    def test_active_autograd_formatting_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def nonscalar_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        cases = (
            module.tensor([1.25], dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
            ).transpose(0, 1)[1],
            tracked,
        )
        results = []
        for tensor in cases:
            before = self.metadata(tensor)
            empty_formatted = format(tensor, "")
            errors = tuple(
                self.error(lambda spec=spec: format(tensor, spec))
                for spec in (".2f", "x", " >12")
            )
            results.append(
                {
                    "empty_is_str": empty_formatted == str(tensor),
                    "empty_type": type(empty_formatted).__name__,
                    "errors": errors,
                    "metadata_unchanged": self.metadata(tensor) == before,
                }
            )
        tracked.sum().backward()
        return tuple(results), leaf.grad.tolist()

    def test_nonscalar_behavior_and_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.nonscalar_contract(torch),
            self.nonscalar_contract(reference_torch),
        )

    def callable_contract(self, module):
        tensor = module.tensor(1.25, dtype=module.float32)
        function = inspect.getattr_static(module.Tensor, "__format__")
        bound = tensor.__format__
        invalid_calls = (
            lambda: function(),
            lambda: function(tensor),
            lambda: function(tensor, ".2f", "extra"),
            lambda: function(tensor, unexpected=True),
            lambda: bound(self=tensor),
            lambda: function(1, ".2f"),
            lambda: function(tensor, None),
            lambda: function(tensor, 2),
        )
        pickle_identities = tuple(
            pickle.loads(pickle.dumps(function, protocol=protocol)) is function
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        )
        return {
            "function_type": type(function).__name__,
            "bound_type": type(bound).__name__,
            "is_function": type(function) is types.FunctionType,
            "is_method": type(bound) is types.MethodType,
            "repr_shape": bool(
                re.fullmatch(
                    r"<function Tensor\.__format__ at 0x[0-9a-f]+>",
                    repr(function),
                )
            ),
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "annotations": function.__annotations__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "signatures": (
                str(inspect.signature(function)),
                str(inspect.signature(bound)),
            ),
            "owned_by_tensor": "__format__" in module.Tensor.__dict__,
            "absent_from_tensor_bases": all(
                "__format__" not in owner.__dict__
                for owner in module.Tensor.__mro__[1:-1]
            ),
            "module_tensor_identity": module._tensor.Tensor is module.Tensor,
            "module_function_identity": module._tensor.Tensor.__format__
            is function,
            "copy_identities": (
                copy.copy(function) is function,
                copy.deepcopy(function) is function,
            ),
            "pickle_identities": pickle_identities,
            "keyword_results": (
                function(self=tensor, format_spec=".2f"),
                bound(format_spec=".2f"),
            ),
            "errors": tuple(self.error(call) for call in invalid_calls),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        function = inspect.getattr_static(module.Tensor, "__format__")
        marker = "format-marker"

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        override = Override()
        override_result = function(override, ".3f")
        override_function, override_types, override_args, override_kwargs = (
            Override.calls[0]
        )

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        tensor = module.tensor(1.25, dtype=module.float32)
        recording = RecordingMode(marker)
        with recording:
            intercepted = format(tensor, ".2f")
        called_function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = format(tensor, ".2f")

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

        rejected = RecordingMode(marker)
        try:
            with rejected:
                function(tensor)
        except TypeError:
            rejected_before_dispatch = not rejected.calls
        else:
            rejected_before_dispatch = False

        return {
            "override": (
                override_result == marker,
                override_function is function,
                override_types == (Override,),
                override_args == (override, ".3f"),
                override_kwargs,
            ),
            "intercepted": intercepted == marker,
            "recording_call": (
                called_function is function,
                dispatch_types == (module.Tensor,),
                len(args) == 2 and args[0] is tensor and args[1] == ".2f",
                kwargs,
            ),
            "forwarded": forwarded,
            "forwarding_calls": tuple(
                (
                    label,
                    func is function,
                    types == (module.Tensor,),
                    len(call_args) == 2
                    and call_args[0] is tensor
                    and call_args[1] == ".2f",
                    call_kwargs,
                )
                for label, func, types, call_args, call_kwargs in order
            ),
            "declining_error": declining_error,
            "declining_calls": tuple(
                (
                    func is function,
                    types == ((module.Tensor,) if index == 0 else ()),
                    len(call_args) == 2
                    and call_args[0] is tensor
                    and call_args[1] == ".2f",
                    call_kwargs,
                )
                for index, (func, types, call_args, call_kwargs) in enumerate(
                    declining.calls
                )
            ),
            "rejected_before_dispatch": rejected_before_dispatch,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_override_and_mode_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
