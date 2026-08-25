import copy
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .test_trunc import SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_trunc import SPECIAL_OUTPUT_BITS, make_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFixReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.fix differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                self.tensor_values(actual).reshape(-1).view(np.uint32),
                self.tensor_values(expected).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def call_fix(module, tensor, form):
        if form == "method":
            return tensor.fix()
        if form == "positional":
            return module.fix(tensor)
        if form == "out none":
            return module.fix(tensor, out=None)
        if form == "alias and out none":
            return module.fix(x=tensor, out=None)
        return module.fix(**{form: tensor})

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(
                -1.25, dtype=module.float32, requires_grad=True
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1]
        if case == "channels last":
            leaf = module.tensor(
                np.linspace(-15.0, 15.0, 120, dtype=np.float32)
                .reshape(2, 3, 4, 5)
                .tolist(),
                dtype=module.float32,
                requires_grad=True,
            )
            return leaf, leaf.contiguous(memory_format=module.channels_last)
        if case == "channels last 3d":
            leaf = module.tensor(
                np.linspace(-90.0, 90.0, 720, dtype=np.float32)
                .reshape(2, 3, 4, 5, 6)
                .tolist(),
                dtype=module.float32,
                requires_grad=True,
            )
            return leaf, leaf.contiguous(memory_format=module.channels_last_3d)

        leaf = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        if case == "offset":
            return leaf, strided[1]
        if case == "strided":
            return leaf, strided
        raise AssertionError(f"unknown fix autograd case: {case}")

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("torch.fix unexpectedly accepted an invalid call")

    def test_values_ieee_bits_layouts_aliases_and_storage_match_pytorch_2_13(self):
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        forms = (
            "method",
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for (case, actual_input, actual_stride), (
            expected_case,
            expected_input,
            expected_stride,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(actual_stride, expected_stride)
            for form in forms:
                actual = self.call_fix(torch, actual_input, form)
                expected = self.call_fix(reference_torch, expected_input, form)
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )
                if case == "numerical edges":
                    np.testing.assert_array_equal(
                        self.tensor_values(actual).reshape(-1).view(np.uint32),
                        SPECIAL_OUTPUT_BITS,
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(expected).reshape(-1).view(np.uint32),
                        SPECIAL_OUTPUT_BITS,
                    )

    def test_seeded_float32_values_match_pytorch_2_13_exactly(self):
        rng = np.random.default_rng(0xF1_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(20):
            rank = int(rng.integers(0, 6))
            shapes.append(
                tuple(int(value) for value in rng.integers(0, 9, size=rank))
            )

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-1.0e6, 1.0e6, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
            values = values.reshape(shape)

            actual_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            self.assert_tensor_matches(
                torch.fix(actual_input),
                reference_torch.fix(expected_input),
                case=(case, shape),
            )

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def tensor_callable_contract(self, module):
        tensor = module.tensor([1.25], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "fix")
        trunc_descriptor = inspect.getattr_static(module.Tensor, "trunc")
        bound = tensor.fix
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "distinct_from_trunc": descriptor is not trunc_descriptor,
            "distinct_from_function": descriptor is not module.fix,
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.fix(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.fix(1, 2),
                    lambda: tensor.fix(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_tensor_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_callable_contract(torch),
            self.tensor_callable_contract(reference_torch),
        )

    @staticmethod
    def tensor_mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.25], dtype=module.float32, requires_grad=True)
descriptor = inspect.getattr_static(module.Tensor, "fix")
trunc_descriptor = inspect.getattr_static(module.Tensor, "trunc")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

recording = RecordingMode(marker)
with recording:
    intercepted = tensor.fix()
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = tensor.fix()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.fix()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.fix(1)
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_is_descriptor": function is descriptor,
    "function_is_not_trunc": function is not trunc_descriptor,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
    "forwarded_requires_grad": forwarded.requires_grad,
    "forwarded_is_leaf": forwarded.is_leaf,
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "invalid_error": invalid_error,
    "invalid_calls": len(invalid.calls),
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}, sort_keys=True))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_tensor_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_mode_dispatch_observation("torch_rs"),
            self.tensor_mode_dispatch_observation("torch"),
        )

    def callable_contract(self, module):
        function = module.fix
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature": self.signature_outcome(function),
            "distinct_from_trunc": function is not module.trunc,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.fix is function,
            "owner_distinct_from_trunc": owner.fix is not owner.trunc,
            "all_count": module.__all__.count("fix"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["fix"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_distinct_builtin_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    @staticmethod
    def dispatch_observation(module):
        tensor = module.tensor([1.25], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.fix
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
        )
        for call, keyword_names in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            dispatched, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(input=value), "input"),
            (lambda value: function(tensor, out=value), "out"),
            (lambda value: function(x=value, out=None), "x"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            dispatched, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), out=DerivedOverride())

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function(),
            lambda: function([], out=destination),
            lambda: function(tensor, out=[]),
            lambda: function(tensor, extra=True),
            lambda: function(tensor, tensor),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forwarding_order,
            forwarded.tolist(),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_modes_and_subclass_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.25])
        expected = reference_torch.tensor([1.25])
        cases = (
            (lambda: torch.fix(), lambda: reference_torch.fix()),
            (
                lambda: torch.fix(actual, actual),
                lambda: reference_torch.fix(expected, expected),
            ),
            (
                lambda: torch.fix(actual, input=actual),
                lambda: reference_torch.fix(expected, input=expected),
            ),
            (
                lambda: torch.fix(out=actual),
                lambda: reference_torch.fix(out=expected),
            ),
            (
                lambda: torch.fix(1, extra=True),
                lambda: reference_torch.fix(1, extra=True),
            ),
            (lambda: torch.fix(input=[]), lambda: reference_torch.fix(input=[])),
            (
                lambda: torch.fix(actual, out=[]),
                lambda: reference_torch.fix(expected, out=[]),
            ),
            (
                lambda: torch.fix(actual, extra=True),
                lambda: reference_torch.fix(expected, extra=True),
            ),
            (
                lambda: torch.fix(input=actual, a=actual),
                lambda: reference_torch.fix(input=expected, a=expected),
            ),
            (
                lambda: torch.fix(a=actual, x=actual, out=None),
                lambda: reference_torch.fix(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.fix(x=actual, a=actual, out=None),
                lambda: reference_torch.fix(x=expected, a=expected, out=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_autograd_scalar_empty_offset_and_strided_match_pytorch_2_13(self):
        forms = (
            "method",
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in (
            "scalar",
            "empty",
            "offset",
            "strided",
            "channels last",
            "channels last 3d",
        ):
            for form in forms:
                actual_leaf, actual_input = self.make_autograd_case(torch, case)
                expected_leaf, expected_input = self.make_autograd_case(
                    reference_torch, case
                )
                actual_output = self.call_fix(torch, actual_input, form)
                expected_output = self.call_fix(
                    reference_torch, expected_input, form
                )

                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(case, form, "output"),
                )
                self.assertEqual(
                    type(expected_output.grad_fn).__name__, "TruncBackward0"
                )
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        actual_output
                    ),
                    ", grad_fn=<TruncBackward0>",
                )

                actual_loss = (
                    actual_output if case == "scalar" else actual_output.sum()
                )
                expected_loss = (
                    expected_output if case == "scalar" else expected_output.sum()
                )
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "first gradient"),
                )
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "repeated gradient"),
                )

        actual_extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        expected_extreme = reference_torch.zeros(
            (0,), dtype=reference_torch.float32, requires_grad=True
        ).reshape((0, sys.maxsize, 3))
        for actual_call, expected_call in (
            (actual_extreme.fix, expected_extreme.fix),
            (
                lambda: torch.fix(actual_extreme, out=None),
                lambda: reference_torch.fix(expected_extreme, out=None),
            ),
        ):
            self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_zero_vjp_special_upstreams_accumulation_and_composition_match(self):
        for form in ("method", "out none"):
            snapshots = []
            for module in (torch, reference_torch):
                apply = lambda tensor: self.call_fix(module, tensor, form)
                special = module.tensor(
                    [float("nan"), float("inf"), -float("inf"), -0.0],
                    dtype=module.float32,
                )
                leaf = module.tensor(
                    [-1.25, -0.0, 1.75, 4.5],
                    dtype=module.float32,
                    requires_grad=True,
                )
                (apply(leaf) * special).sum().backward()
                special_gradient = (
                    self.tensor_values(leaf.grad)
                    .reshape(-1)
                    .view(np.uint32)
                    .copy()
                )

                accumulated = module.tensor(
                    [-2.0, 0.0, 3.0],
                    dtype=module.float32,
                    requires_grad=True,
                )
                (accumulated * 3.0).sum().backward()
                before_zero = self.tensor_values(accumulated.grad).copy()
                reusable_loss = apply(accumulated).sum()
                reusable_loss.backward()
                after_zero = self.tensor_values(accumulated.grad).copy()
                reusable_loss.backward()
                after_repeated_zero = self.tensor_values(accumulated.grad).copy()

                composed = module.tensor(
                    [-0.5, 0.5], dtype=module.float32, requires_grad=True
                )
                composed_loss = apply(composed.sin()).sum()
                composed_loss.backward()
                composed_gradient = self.tensor_values(composed.grad).copy()
                repeated_composed = self.error(composed_loss.backward)
                snapshots.append(
                    (
                        special_gradient,
                        before_zero,
                        after_zero,
                        after_repeated_zero,
                        composed_gradient,
                        repeated_composed,
                    )
                )

            with self.subTest(form=form):
                for index in range(5):
                    np.testing.assert_array_equal(
                        snapshots[0][index], snapshots[1][index]
                    )
                self.assertEqual(snapshots[0][5], snapshots[1][5])
                np.testing.assert_array_equal(
                    snapshots[0][0], np.zeros((4,), dtype=np.uint32)
                )

    def test_requires_grad_inputs_match_inside_no_grad_and_detached(self):
        values = np.linspace(-3.75, 3.75, 24, dtype=np.float32).reshape(2, 3, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 2)[1]
        expected_input = expected_leaf.transpose(0, 2)[1]

        for form in ("method", "out none"):
            with torch.no_grad():
                actual_no_grad = self.call_fix(torch, actual_input, form)
            with reference_torch.no_grad():
                expected_no_grad = self.call_fix(
                    reference_torch, expected_input, form
                )
            self.assert_tensor_matches(
                actual_no_grad, expected_no_grad, case=(form, "no_grad")
            )

            actual_detached = self.call_fix(torch, actual_input.detach(), form)
            expected_detached = self.call_fix(
                reference_torch, expected_input.detach(), form
            )
            self.assert_tensor_matches(
                actual_detached, expected_detached, case=(form, "detached")
            )

    def test_concrete_out_and_inplace_boundaries_remain_explicit(self):
        actual_input = torch.tensor([1.25, -1.25], requires_grad=True)
        actual_input_pointer = actual_input.data_ptr()
        actual_input_bits = (
            self.tensor_values(actual_input).reshape(-1).view(np.uint32).copy()
        )
        actual_out = torch.tensor([17.0, 19.0])
        actual_out_pointer = actual_out.data_ptr()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^fix\(\): the 'out' argument is not supported$",
        ):
            torch.fix(actual_input, out=actual_out)
        self.assertEqual(actual_out.data_ptr(), actual_out_pointer)
        self.assertEqual(actual_out.tolist(), [17.0, 19.0])
        self.assertEqual(actual_input.data_ptr(), actual_input_pointer)
        np.testing.assert_array_equal(
            self.tensor_values(actual_input).reshape(-1).view(np.uint32),
            actual_input_bits,
        )
        self.assertIsNone(actual_input.grad)

        expected_out = reference_torch.tensor(
            [17.0, 19.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.fix(
                reference_torch.tensor([1.25, -1.25]), out=expected_out
            ),
            expected_out,
        )
        self.assertEqual(expected_out.tolist(), [1.0, -1.0])

        self.assertTrue(hasattr(torch.Tensor, "fix"))
        self.assertTrue(hasattr(reference_torch.Tensor, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "fix_"))
        self.assertFalse(hasattr(torch, "fix_"))
        self.assertTrue(hasattr(reference_torch, "fix_"))
        self.assertNotIn("fix_", torch.__all__)

        for call in (
            lambda: actual_input.fix_(),
            lambda: torch.fix_(actual_input),
            lambda: actual_input.fix(out=None),
        ):
            with self.assertRaises((AttributeError, TypeError)):
                call()
            self.assertEqual(actual_input.data_ptr(), actual_input_pointer)
            np.testing.assert_array_equal(
                self.tensor_values(actual_input).reshape(-1).view(np.uint32),
                actual_input_bits,
            )
            self.assertIsNone(actual_input.grad)


if __name__ == "__main__":
    unittest.main()
