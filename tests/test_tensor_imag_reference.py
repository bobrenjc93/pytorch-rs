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

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorImagReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.imag differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        scalar_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = module.tensor(
            memoryview(scalar_bits.view(np.float32)), dtype=module.float32
        )
        base = module.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 3)
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            no_grad_view = leaf.transpose(0, 1)
        return (
            *(scalar_storage[index] for index in range(len(scalar_bits))),
            base,
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            base.contiguous(memory_format=module.channels_last),
            module.zeros((2, 3, 4, 5, 6), dtype=module.float32).contiguous(
                memory_format=module.channels_last_3d
            ),
            leaf,
            (leaf * 3.0).transpose(0, 1)[1],
            no_grad_view,
            module.zeros((1, 0, 1, 1, 1, 1), dtype=module.float32),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return error
        self.fail("Tensor.imag unexpectedly accepted the operation")

    def error_read_contract(self, tensor, action):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        pointer = tensor.data_ptr()
        alias = tensor.detach()
        bits = np.asarray(alias).reshape(-1).view(np.uint32).copy()
        errors = [self.error(lambda: action(tensor)) for _ in range(3)]
        return {
            "errors": tuple((type(error).__name__, str(error)) for error in errors),
            "fresh_errors": len({id(error) for error in errors}) == len(errors),
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
                str(tensor.layout),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            ),
            "pointer_unchanged": tensor.data_ptr() == pointer,
            "storage_alias_unchanged": tensor.is_set_to(alias),
            "bits_unchanged": np.array_equal(
                np.asarray(tensor.detach()).reshape(-1).view(np.uint32), bits
            ),
        }

    def test_float32_errors_and_side_effect_boundaries_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                for access, actual_action, expected_action in (
                    ("property", lambda tensor: tensor.imag, lambda tensor: tensor.imag),
                    ("top-level", torch.imag, reference_torch.imag),
                    (
                        "top-level input",
                        lambda tensor: torch.imag(input=tensor),
                        lambda tensor: reference_torch.imag(input=tensor),
                    ),
                    (
                        "top-level x",
                        lambda tensor: torch.imag(x=tensor),
                        lambda tensor: reference_torch.imag(x=tensor),
                    ),
                    (
                        "top-level a",
                        lambda tensor: torch.imag(a=tensor),
                        lambda tensor: reference_torch.imag(a=tensor),
                    ),
                    (
                        "top-level x1",
                        lambda tensor: torch.imag(x1=tensor),
                        lambda tensor: reference_torch.imag(x1=tensor),
                    ),
                ):
                    with self.subTest(access=access):
                        self.assertEqual(
                            self.error_read_contract(actual, actual_action),
                            self.error_read_contract(expected, expected_action),
                        )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        before = (
            leaf.requires_grad,
            leaf.is_leaf,
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            tuple(non_leaf.shape),
            non_leaf.stride(),
            non_leaf.storage_offset(),
        )
        errors = (
            self.error(lambda: leaf.imag),
            self.error(lambda: module.imag(leaf)),
            self.error(lambda: non_leaf.imag),
            self.error(lambda: module.imag(non_leaf)),
        )
        after = (
            leaf.requires_grad,
            leaf.is_leaf,
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            tuple(non_leaf.shape),
            non_leaf.stride(),
            non_leaf.storage_offset(),
        )
        non_leaf.sum().backward()
        gradient = leaf.grad
        final_errors = (
            self.error(lambda: leaf.imag),
            self.error(lambda: module.imag(leaf)),
        )
        return {
            "before": before,
            "after": after,
            "errors": tuple((type(error).__name__, str(error)) for error in errors),
            "gradient": np.asarray(gradient).copy(),
            "gradient_identity_preserved": leaf.grad is gradient,
            "final_errors": tuple(
                (type(error).__name__, str(error)) for error in final_errors
            ),
        }

    def test_autograd_graph_preservation_matches_pytorch_2_13(self):
        actual = self.autograd_contract(torch)
        expected = self.autograd_contract(reference_torch)
        np.testing.assert_array_equal(
            actual.pop("gradient"), expected.pop("gradient")
        )
        self.assertEqual(actual, expected)

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "imag")
        tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
        replacement = module.tensor([2.0], dtype=module.float32)
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        actions = (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        )
        mutation_errors = tuple(
            (type(error).__name__, str(error))
            for error in (self.error(action) for action in actions)
        )
        receiver_errors = tuple(
            (type(error).__name__, str(error))
            for error in (
                self.error(lambda: descriptor.__get__(1, int)),
                self.error(lambda: descriptor.__set__(1, replacement)),
                self.error(lambda: descriptor.__delete__(1)),
            )
        )
        read_error = self.error(lambda: descriptor.__get__(tensor, module.Tensor))
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.imag is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "read_error": (type(read_error).__name__, str(read_error)),
            "mutation_errors": mutation_errors,
            "receiver_errors": receiver_errors,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_descriptor_and_mutation_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def top_level_callable_contract(self, module):
        function = module.imag
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
            "owner_callable_identity": owner.imag is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "all_count": module.__all__.count("imag"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["imag"] is function,
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    def top_level_binding_error_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        cases = (
            lambda: module.imag(),
            lambda: module.imag(tensor, tensor),
            lambda: module.imag(tensor, input=tensor),
            lambda: module.imag(tensor, out=tensor),
            lambda: module.imag(tensor, dtype=module.float32),
            lambda: module.imag(tensor, device="cpu"),
            lambda: module.imag(input=tensor),
            lambda: module.imag(x=tensor),
            lambda: module.imag(a=tensor),
            lambda: module.imag(x1=tensor),
            lambda: module.imag(extra=tensor),
            lambda: module.imag(1, extra=True),
            lambda: module.imag(input=[]),
            lambda: module.imag(a=1),
            lambda: module.imag(x=[]),
            lambda: module.imag(x1=None),
            lambda: module.imag(a=tensor, x=tensor),
            lambda: module.imag(x=tensor, a=tensor),
            lambda: module.imag(input=tensor, x1=tensor),
        )
        return tuple(
            (type(error).__name__, str(error)) for error in map(self.error, cases)
        )

    def test_top_level_binding_error_precedence_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_binding_error_contract(torch),
            self.top_level_binding_error_contract(reference_torch),
        )

    def top_level_mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import json
import re
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

records = []
for keyword in (None, "input", "x", "a", "x1"):
    mode = RecordingMode(marker)
    with mode:
        if keyword is None:
            intercepted = module.imag(tensor)
        else:
            intercepted = module.imag(**{keyword: tensor})
    function, dispatch_types, args, kwargs = mode.calls[0]
    records.append({
        "keyword": keyword,
        "intercepted": intercepted is marker,
        "call_count": len(mode.calls),
        "function_identity": function is module.imag,
        "types": dispatch_types == (),
        "args": args == ((tensor,) if keyword is None else ()),
        "kwargs_is_none": kwargs is None,
        "kwargs_value": None if kwargs is None else list(kwargs.keys()),
        "kwargs_tensor": kwargs is None or kwargs[keyword] is tensor,
    })

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

try:
    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            module.imag(a=tensor)
except Exception as error:
    forwarding_error = [type(error).__name__, str(error)]
else:
    forwarding_error = None

sys.setrecursionlimit(80)
class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

lower = RecordingMode(marker)
upper = DecliningMode()
try:
    with lower:
        with upper:
            module.imag(tensor)
except Exception as error:
    declining_error = [
        type(error).__name__,
        re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
    ]
else:
    declining_error = None

override_calls = []
class Override:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        override_calls.append((func, types, args, kwargs))
        return marker

value = Override()
override_results = []
for keyword in (None, "input", "x", "a", "x1"):
    if keyword is None:
        result = module.imag(value)
    else:
        result = module.imag(**{keyword: value})
    function, dispatch_types, args, kwargs = override_calls[-1]
    override_results.append({
        "keyword": keyword,
        "result": result is marker,
        "function_identity": function is module.imag,
        "types": dispatch_types == (Override,),
        "args": args == ((value,) if keyword is None else ()),
        "kwargs_value": None if kwargs is None else list(kwargs.keys()),
        "kwargs_override": kwargs is None or kwargs[keyword] is value,
    })

print(json.dumps({
    "records": records,
    "forwarding_order": order,
    "forwarding_error": forwarding_error,
    "declining_error": declining_error,
    "declining_calls_once": upper.calls == 1,
    "lower_skipped": len(lower.calls) == 0,
    "override_results": override_results,
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_top_level_torch_function_mode_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_mode_dispatch_observation("torch_rs"),
            self.top_level_mode_dispatch_observation("torch"),
        )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
replacement = module.tensor([2.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "imag")
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
    intercepted = tensor.imag
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

try:
    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            tensor.imag
except Exception as error:
    forwarding_error = [type(error).__name__, str(error)]
else:
    forwarding_error = None

sys.setrecursionlimit(80)
class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

lower = RecordingMode(marker)
upper = DecliningMode()
try:
    with lower:
        with upper:
            tensor.imag
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

mutation_calls = []
for action in (
    lambda: setattr(tensor, "imag", replacement),
    lambda: delattr(tensor, "imag"),
    lambda: descriptor.__set__(tensor, replacement),
    lambda: descriptor.__delete__(tensor),
):
    mode = RecordingMode(marker)
    try:
        with mode:
            action()
    except Exception as error:
        outcome = [type(error).__name__, str(error)]
    else:
        outcome = None
    mutation_calls.append([len(mode.calls), outcome])

errors = []
for _ in range(3):
    try:
        tensor.imag
    except Exception as error:
        errors.append(error)

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_self": function.__self__ is descriptor,
    "function_equals_descriptor_get": function == descriptor.__get__,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarding_error": forwarding_error,
    "declining_error": declining_error,
    "declining_calls_gt_one": upper.calls > 1,
    "lower_skipped": len(lower.calls) == 0,
    "mutation_calls": mutation_calls,
    "fresh_errors": len({id(error) for error in errors}) == len(errors),
    "ordinary_errors": [[type(error).__name__, str(error)] for error in errors],
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_torch_function_mode_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_scope_keeps_complex_dtypes_and_imaginary_views_unsupported(self):
        self.assertTrue(hasattr(torch.Tensor, "imag"))
        self.assertTrue(hasattr(reference_torch.Tensor, "imag"))
        self.assertTrue(hasattr(torch, "imag"))
        self.assertTrue(hasattr(reference_torch, "imag"))
        self.assertEqual(
            torch.__all__.count("imag"), reference_torch.__all__.count("imag")
        )
        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(dtype=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))


if __name__ == "__main__":
    unittest.main()
