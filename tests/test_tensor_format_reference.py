import copy
import inspect
import math
import pickle
import types
import unittest
import warnings

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

    def outcome(self, call):
        try:
            return "value", call()
        except Exception as error:
            return "error", type(error).__name__, str(error)

    def test_scalar_values_and_format_errors_match_pytorch_2_13(self):
        values = (
            0.0,
            -0.0,
            1.23456789,
            1.0e-40,
            1.0e30,
            math.inf,
            -math.inf,
            math.nan,
        )
        format_specs = (
            "",
            ".2f",
            "+08.2f",
            "e",
            ".4e",
            "g",
            "#g",
            "%",
            "020",
            " z",
            "_",
            "^20",
            "s",
            "d",
            "n",
            "=+20.3f",
        )
        for value in values:
            actual = torch.tensor(value, dtype=torch.float32, requires_grad=True)
            expected = reference_torch.tensor(
                value,
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            for format_spec in format_specs:
                with self.subTest(value=value, format_spec=format_spec):
                    self.assertEqual(
                        self.outcome(
                            lambda actual=actual, format_spec=format_spec: format(
                                actual, format_spec
                            )
                        ),
                        self.outcome(
                            lambda expected=expected, format_spec=format_spec: format(
                                expected, format_spec
                            )
                        ),
                    )

    def graph_contract(self, module, *, scalar):
        values = 1.25 if scalar else [1.0, 2.0]
        leaf = module.tensor(values, requires_grad=True)
        tensor = leaf * 2.0
        metadata_before = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
            leaf.grad is None,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            empty = format(tensor, "")
            nonempty = self.outcome(lambda: format(tensor, ".2f"))
        metadata_after = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
            leaf.grad is None,
        )
        (tensor if scalar else tensor.sum()).backward()
        return {
            "empty": empty if scalar else empty == str(tensor),
            "nonempty": nonempty,
            "metadata_unchanged": metadata_before == metadata_after,
            "warnings": tuple(
                (item.category.__name__, str(item.message)) for item in caught
            ),
            "gradient": leaf.grad.tolist(),
        }

    def test_storage_graph_warning_and_non_scalar_behavior_match_pytorch_2_13(self):
        for scalar in (True, False):
            with self.subTest(scalar=scalar):
                self.assertEqual(
                    self.graph_contract(torch, scalar=scalar),
                    self.graph_contract(reference_torch, scalar=scalar),
                )

    def callable_contract(self, module):
        tensor = module.tensor(1.25)
        function = inspect.getattr_static(module.Tensor, "__format__")
        bound = tensor.__format__
        direct_calls = (
            lambda: function(),
            lambda: function(tensor),
            lambda: function(tensor, ".2f", "extra"),
            lambda: bound(),
            lambda: bound(".2f", "extra"),
            lambda: function(1, ".2f"),
            lambda: function(tensor, None),
        )
        return {
            "function_type": type(function).__name__,
            "is_function": type(function) is types.FunctionType,
            "bound_type": type(bound).__name__,
            "is_method": type(bound) is types.MethodType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "annotations": function.__annotations__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "signature": str(inspect.signature(function)),
            "bound_signature": str(inspect.signature(bound)),
            "class_owns": "__format__" in module.Tensor.__dict__,
            "module_owns": module._tensor.Tensor.__format__ is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "direct_outcomes": tuple(self.outcome(call) for call in direct_calls),
            "keyword_result": function(tensor, format_spec=".2f"),
            "keyword_self_result": function(self=tensor, format_spec=".2f"),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def dispatch_contract(self, module):
        function = inspect.getattr_static(module.Tensor, "__format__")
        tensor = module.tensor(1.25)

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __repr__(self):
                return "format-mode"

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        recording = RecordingMode("formatted-by-mode")
        with recording:
            result = format(tensor, ".2f")
        recorded = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(
                    (
                        self.label,
                        func is function,
                        dispatch_types == (module.Tensor,),
                        args[0] is tensor,
                        args[1:],
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = format(tensor, ".2f")

        declining = RecordingMode(NotImplemented)
        declining_outcome = self.outcome(
            lambda: self.format_with_mode(declining, tensor, ".2f")
        )
        if declining_outcome[0] == "error":
            declining_outcome = (
                declining_outcome[0],
                declining_outcome[1],
                declining_outcome[2].replace("torch_rs", "torch"),
            )

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return "formatted-by-override"

        override = Override()
        override_result = function(override, ".2f")
        override_call = Override.calls[0]
        return {
            "result": result,
            "recorded": (
                recorded[0] is function,
                recorded[1] == (module.Tensor,),
                recorded[2][0] is tensor,
                recorded[2][1:],
                recorded[3],
            ),
            "forwarded": forwarded,
            "order": order,
            "declining_outcome": declining_outcome,
            "declining_calls": tuple(
                (
                    call[0] is function,
                    call[1] == (module.Tensor,),
                    call[1] == (),
                    call[2][0] is tensor,
                    call[2][1:],
                    call[3],
                )
                for call in declining.calls
            ),
            "mode_stack_empty": module.overrides._get_current_function_mode_stack()
            == [],
            "override_result": override_result,
            "override_call": (
                override_call[0] is function,
                override_call[1] == (Override,),
                override_call[2][0] is override,
                override_call[2][1:],
                override_call[3],
            ),
        }

    def format_with_mode(self, mode, tensor, format_spec):
        with mode:
            return format(tensor, format_spec)

    def test_override_and_mode_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
