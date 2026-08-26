import importlib
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
class BroadcastTensorsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "broadcast_tensors differentials require pinned PyTorch 2.13.0"
            )

    def make_sources(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        contiguous = module.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist()
        )
        noncontiguous = module.tensor(
            np.arange(12, dtype=np.float32).reshape(4, 3).tolist()
        ).transpose(0, 1)
        return base[1], contiguous, noncontiguous

    def value_contract(self, module):
        sources = self.make_sources(module)
        outputs = module.broadcast_tensors(*sources)
        scalar = module.tensor(-0.0)
        scalar_outputs = module.functional.broadcast_tensors(scalar, scalar)
        empty = module.zeros((2, 0, 3))
        empty_outputs = module.broadcast_tensors(empty)

        def metadata(output, source):
            return (
                output is source,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.data_ptr() == source.data_ptr(),
                str(output.dtype).replace("torch_rs", "torch"),
                str(output.device),
                str(output.layout).replace("torch_rs", "torch"),
                output.requires_grad,
                output.is_leaf,
                output.output_nr,
            )

        return {
            "zero": (type(module.broadcast_tensors()) is tuple, module.broadcast_tensors()),
            "many_type": type(outputs) is tuple,
            "many": tuple(
                metadata(output, source)
                for output, source in zip(outputs, sources, strict=True)
            ),
            "scalar": tuple(
                metadata(output, scalar) for output in scalar_outputs
            ),
            "empty": metadata(empty_outputs[0], empty),
        }

    def test_zero_scalar_empty_offset_and_strided_values_match_pytorch_2_13(self):
        self.assertEqual(
            self.value_contract(torch),
            self.value_contract(reference_torch),
        )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        nonleaf = leaf * 2.0
        view = nonleaf.transpose(0, 1)
        outputs = module.broadcast_tensors(leaf, nonleaf, view)
        before = tuple(
            (
                output is source,
                output.requires_grad,
                output.is_leaf,
                output.output_nr,
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.data_ptr() == source.data_ptr(),
            )
            for output, source in zip(outputs, (leaf, nonleaf, view), strict=True)
        )
        outputs[2].sum().backward()
        gradient = np.asarray(leaf.grad).copy()

        with module.no_grad():
            no_grad_outputs = module.broadcast_tensors(nonleaf, view)
        no_grad = tuple(
            (
                output is source,
                output.requires_grad,
                output.is_leaf,
                output.output_nr,
            )
            for output, source in zip(
                no_grad_outputs, (nonleaf, view), strict=True
            )
        )
        return before, gradient, no_grad

    def test_autograd_and_no_grad_identity_match_pytorch_2_13(self):
        actual_before, actual_gradient, actual_no_grad = self.autograd_contract(torch)
        expected_before, expected_gradient, expected_no_grad = self.autograd_contract(
            reference_torch
        )
        self.assertEqual(actual_before, expected_before)
        np.testing.assert_array_equal(actual_gradient, expected_gradient)
        self.assertEqual(actual_no_grad, expected_no_grad)

    def mode_contract(self, module):
        function = module.broadcast_tensors
        sources = self.make_sources(module)[:2]
        marker = object()

        def mode_stack():
            return module.overrides._get_current_function_mode_stack()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (func, types, args, kwargs, len(mode_stack()))
                )
                return self.result

        def normalize(call):
            func, dispatch_types, args, kwargs, stack_depth = call
            return (
                func is function,
                tuple(item.__name__ for item in dispatch_types),
                tuple(
                    argument is source
                    for argument, source in zip(args, sources, strict=True)
                ),
                kwargs,
                stack_depth,
            )

        accepting = RecordingMode(marker)
        with accepting:
            accepting_result = function(*sources)
            accepting_restored = mode_stack() == [accepting]

        forwarding_calls = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_calls.append(
                    (
                        self.label,
                        func,
                        types,
                        args,
                        kwargs,
                        len(mode_stack()),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = function(*sources)
                forwarding_restored = mode_stack() == [lower, upper]

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                function(*sources)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(
                    r"0x[0-9a-f]+",
                    "0x<address>",
                    str(error).replace("torch_rs", "torch"),
                ),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "accepting": (
                accepting_result is marker,
                accepting_restored,
                tuple(map(normalize, accepting.calls)),
            ),
            "forwarding": tuple(
                (label, normalize((func, types, args, kwargs, stack_depth)))
                for label, func, types, args, kwargs, stack_depth in forwarding_calls
            ),
            "forwarded": tuple(
                output is source
                for output, source in zip(forwarded, sources, strict=True)
            ),
            "forwarding_restored": forwarding_restored,
            "declining": declining_error,
            "declining_calls": tuple(map(normalize, declining.calls)),
            "stack_depth": len(mode_stack()),
        }

    def test_nonempty_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def zero_mode_contract(self, module):
        function = module.broadcast_tensors
        native_function = (
            module._C._VariableFunctions.broadcast_tensors
            if module is reference_torch
            else module._C.broadcast_tensors
        )
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode(marker)
        with accepting:
            accepting_result = function()

        forwarding_calls = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function()

        three_argument_calls = []

        class ThreeArgumentMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=()):
                three_argument_calls.append((func, types, args))
                return marker

        with ThreeArgumentMode():
            three_argument_result = function()

        def normalize(call):
            func, types, args, kwargs = call
            return func is native_function, func is function, types, args, kwargs

        return (
            accepting_result is marker,
            tuple(map(normalize, accepting.calls)),
            tuple(
                (label, *normalize((func, types, args, kwargs)))
                for label, func, types, args, kwargs in forwarding_calls
            ),
            type(forwarded) is tuple,
            forwarded,
            three_argument_result is marker,
            tuple(
                (
                    func is native_function,
                    func is function,
                    types,
                    args,
                )
                for func, types, args in three_argument_calls
            ),
        )

    def test_zero_input_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.zero_mode_contract(torch),
            self.zero_mode_contract(reference_torch),
        )

    def namespace_contract(self, module):
        functional = importlib.import_module(f"{module.__name__}.functional")
        function = module.broadcast_tensors
        functional_namespace = {}
        exec(f"from {module.__name__}.functional import *", functional_namespace)
        top_level_namespace = {}
        exec(f"from {module.__name__} import *", top_level_namespace)
        return (
            module.functional is functional,
            function is functional.broadcast_tensors,
            type(function) is types.FunctionType,
            function.__name__,
            function.__qualname__,
            function.__module__.replace("torch_rs", "torch"),
            str(inspect.signature(function)),
            hasattr(function, "__text_signature__"),
            function.__annotations__,
            function.__doc__,
            functional.__all__.count("broadcast_tensors"),
            module.__all__.count("broadcast_tensors"),
            functional_namespace["broadcast_tensors"] is function,
            top_level_namespace["broadcast_tensors"] is function,
        )

    def test_namespace_metadata_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.namespace_contract(torch),
            self.namespace_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
