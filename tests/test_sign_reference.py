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
    from .test_sign import SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_sign import SPECIAL_OUTPUT_BITS, make_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSignReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.sign differentials require pinned PyTorch 2.13.0")

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
    def call_top_level_sign(module, tensor, form):
        if form == "positional":
            return module.sign(tensor)
        if form == "out none":
            return module.sign(tensor, out=None)
        if form == "alias and out none":
            return module.sign(x=tensor, out=None)
        return module.sign(**{form: tensor})

    @staticmethod
    def make_autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(-0.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "contiguous":
            return leaf, leaf
        strided = leaf.transpose(0, 2)
        if case == "offset":
            return leaf, strided[1]
        if case == "strided":
            return leaf, strided
        raise AssertionError(f"unknown sign autograd case: {case}")

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.sign unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.25], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "sign")
        bound = tensor.sign
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
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.sign(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.sign(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        for (case, actual_input, actual_stride), (
            expected_case,
            expected_input,
            expected_stride,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(actual_stride, expected_stride)
            actual = actual_input.sign()
            expected = expected_input.sign()
            self.assert_tensor_matches(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )

    def test_top_level_out_none_values_layouts_and_storage_match_pytorch_2_13(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        for (case, actual_input, _), (expected_case, expected_input, _) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            for form in forms:
                actual = self.call_top_level_sign(torch, actual_input, form)
                expected = self.call_top_level_sign(reference_torch, expected_input, form)
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.25], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "sign")
function = module.sign
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result=marker):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

method_mode = RecordingMode()
with method_mode:
    method_intercepted = tensor.sign()
method_func, method_types, method_args, method_kwargs = method_mode.calls[0]

top_level_mode = RecordingMode()
with top_level_mode:
    top_level_intercepted = function(input=tensor, out=None)
top_func, top_types, top_args, top_kwargs = top_level_mode.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = function(input=tensor, out=None)

override_events = []
class Override:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        override_events.append((func is function, tuple(item.__name__ for item in types), len(args), kwargs is None))
        return marker

override_result = function(Override())

subclass_order = []
class BaseOverride:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        subclass_order.append(("base", tuple(item.__name__ for item in types)))
        return marker

class DerivedOverride(BaseOverride):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        subclass_order.append(("derived", tuple(item.__name__ for item in types)))
        return marker

subclass_result = function(BaseOverride(), out=DerivedOverride())

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.sign()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

print(json.dumps({
    "method_intercepted": method_intercepted is marker,
    "method_function_name": method_func.__name__,
    "method_function_qualname": method_func.__qualname__,
    "method_function_is_descriptor": method_func is descriptor,
    "method_types": method_types == (module.Tensor,),
    "method_args": len(method_args) == 1 and method_args[0] is tensor,
    "method_kwargs_is_none": method_kwargs is None,
    "top_level_intercepted": top_level_intercepted is marker,
    "top_level_function": top_func is function,
    "top_level_types": top_types == (),
    "top_level_args": top_args == (),
    "top_level_kwargs": tuple(top_kwargs),
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
    "override_result": override_result is marker,
    "override_events": override_events,
    "subclass_result": subclass_result is marker,
    "subclass_order": subclass_order,
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
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

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_torch_function_mode_observes_default_foreign_tensors_before_rejection(self):
        foreign_input = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        native_input = torch.tensor([1.0], dtype=torch.float32)
        foreign_out = reference_torch.tensor([0.0], dtype=reference_torch.float32)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        input_mode = RecordingMode()
        with input_mode:
            self.assertIs(torch.sign(foreign_input), marker)
        self.assertEqual(len(input_mode.calls), 1)
        function, dispatch_types, args, kwargs = input_mode.calls[0]
        self.assertIs(function, torch.sign)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], foreign_input)
        self.assertIsNone(kwargs)

        out_mode = RecordingMode()
        with out_mode:
            self.assertIs(torch.sign(native_input, out=foreign_out), marker)
        self.assertEqual(len(out_mode.calls), 1)
        function, dispatch_types, args, kwargs = out_mode.calls[0]
        self.assertIs(function, torch.sign)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], native_input)
        self.assertEqual(tuple(kwargs), ("out",))
        self.assertIs(kwargs["out"], foreign_out)
        self.assertEqual(foreign_out.tolist(), [0.0])

        declining_input = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^sign\(\): only exact native CPU float32 Tensor inputs are supported$",
        ):
            with declining_input:
                torch.sign(foreign_input)
        self.assertEqual(len(declining_input.calls), 1)

        declining_out = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^sign\(\): only exact native CPU float32 Tensor inputs are supported$",
        ):
            with declining_out:
                torch.sign(native_input, out=foreign_out)
        self.assertEqual(len(declining_out.calls), 1)
        self.assertEqual(foreign_out.tolist(), [0.0])

    def top_level_callable_contract(self, module):
        function = module.sign
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.sign is function,
            "all_count": module.__all__.count("sign"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sign"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

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
        for case in ("scalar", "empty", "contiguous", "offset", "strided"):
            for form in forms:
                actual_leaf, actual_input = self.make_autograd_case(torch, case)
                expected_leaf, expected_input = self.make_autograd_case(
                    reference_torch, case
                )
                if form == "method":
                    actual_output = actual_input.sign()
                    expected_output = expected_input.sign()
                else:
                    actual_output = self.call_top_level_sign(torch, actual_input, form)
                    expected_output = self.call_top_level_sign(
                        reference_torch, expected_input, form
                    )

                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(case, form, "output"),
                )
                self.assertEqual(
                    type(expected_output.grad_fn).__name__, "SignBackward0"
                )
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        actual_output
                    ),
                    ", grad_fn=<SignBackward0>",
                )

                actual_loss = actual_output if case == "scalar" else actual_output.sum()
                expected_loss = expected_output if case == "scalar" else expected_output.sum()
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

    def test_zero_vjp_special_upstreams_accumulation_and_composition_match(self):
        snapshots = []
        for module in (torch, reference_torch):
            special = module.tensor(
                [float("nan"), float("inf"), -float("inf"), -0.0],
                dtype=module.float32,
            )
            leaf = module.tensor(
                [-1.25, -0.0, 1.75, 4.5],
                dtype=module.float32,
                requires_grad=True,
            )
            (module.sign(leaf, out=None) * special).sum().backward()
            special_gradient = (
                self.tensor_values(leaf.grad).reshape(-1).view(np.uint32).copy()
            )

            accumulated = module.tensor(
                [-2.0, 0.0, 3.0],
                dtype=module.float32,
                requires_grad=True,
            )
            (accumulated * 3.0).sum().backward()
            before_zero = self.tensor_values(accumulated.grad).copy()
            reusable_loss = accumulated.sign().sum()
            reusable_loss.backward()
            after_zero = self.tensor_values(accumulated.grad).copy()
            reusable_loss.backward()
            after_repeated_zero = self.tensor_values(accumulated.grad).copy()

            composed = module.tensor(
                [-0.5, 0.5], dtype=module.float32, requires_grad=True
            )
            composed_loss = composed.sin().sign().sum()
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

        for index in range(5):
            np.testing.assert_array_equal(snapshots[0][index], snapshots[1][index])
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

        with torch.no_grad():
            actual_no_grad = actual_input.sign()
        with reference_torch.no_grad():
            expected_no_grad = expected_input.sign()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        with torch.no_grad():
            actual_top_level_no_grad = torch.sign(actual_input, out=None)
        with reference_torch.no_grad():
            expected_top_level_no_grad = reference_torch.sign(
                expected_input, out=None
            )
        self.assert_tensor_matches(
            actual_top_level_no_grad,
            expected_top_level_no_grad,
            case="top-level no_grad",
        )

        actual_detached = actual_input.detach().sign()
        expected_detached = expected_input.detach().sign()
        self.assert_tensor_matches(actual_detached, expected_detached, case="detached")

    def test_concrete_out_and_unsupported_boundaries_remain_explicit(self):
        actual_input = torch.tensor([1.25, -1.25], requires_grad=True)
        actual_out = torch.tensor([17.0, 19.0])
        actual_out_pointer = actual_out.data_ptr()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sign\(\): the 'out' argument is not supported$",
        ):
            torch.sign(actual_input, out=actual_out)
        self.assertEqual(actual_out.data_ptr(), actual_out_pointer)
        self.assertEqual(actual_out.tolist(), [17.0, 19.0])
        self.assertIsNone(actual_input.grad)

        expected_input = reference_torch.tensor(
            [1.25, -1.25], dtype=reference_torch.float32
        )
        expected_out = reference_torch.tensor(
            [17.0, 19.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.sign(expected_input, out=expected_out), expected_out
        )
        self.assertEqual(expected_out.tolist(), [1.0, -1.0])

        self.assertTrue(hasattr(torch, "sign"))
        self.assertTrue(hasattr(reference_torch, "sign"))
        self.assertIn("sign", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "sign_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "sign_"))
        self.assertFalse(hasattr(torch, "sign_"))
        self.assertNotIn("sign_", torch.__all__)
        self.assertFalse(hasattr(torch, "sgn"))
        self.assertTrue(hasattr(reference_torch, "sgn"))
        self.assertFalse(hasattr(torch.Tensor, "sgn"))
        self.assertTrue(hasattr(reference_torch.Tensor, "sgn"))
        self.assertNotIn("sgn", torch.__all__)

        non_float32 = (
            reference_torch.tensor([1.0], dtype=reference_torch.float64),
            reference_torch.tensor([1], dtype=reference_torch.int64),
        )
        for foreign in non_float32:
            with self.subTest(dtype=str(foreign.dtype)):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^sign\(\): only exact native CPU float32 Tensor inputs are supported$",
                ):
                    torch.sign(foreign)

        if reference_torch.cuda.is_available():
            cuda_tensor = reference_torch.tensor(
                [1.0], dtype=reference_torch.float32, device="cuda"
            )
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^sign\(\): only exact native CPU float32 Tensor inputs are supported$",
            ):
                torch.sign(cuda_tensor)

        class ForeignTensorSubclass(reference_torch.Tensor):
            pass

        foreign_subclass = reference_torch.Tensor._make_subclass(
            ForeignTensorSubclass,
            reference_torch.tensor([1.0], dtype=reference_torch.float32),
            False,
        )
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^sign\(\): only exact native CPU float32 Tensor inputs are supported$",
        ):
            torch.sign(foreign_subclass)

        marker = object()

        class ForeignTensorOverride(reference_torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return marker

        foreign_override = reference_torch.Tensor._make_subclass(
            ForeignTensorOverride,
            reference_torch.tensor([1.0], dtype=reference_torch.float32),
            False,
        )
        self.assertIs(torch.sign(foreign_override), marker)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(TypeError, "returned NotImplemented"):
            torch.sign(DecliningOverride())


if __name__ == "__main__":
    unittest.main()
