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
class TopLevelImagReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.imag differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def make_cases(module):
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
            leaf,
            (leaf * 3.0).transpose(0, 1)[1],
            (leaf * 3.0).transpose(0, 1)[1].detach(),
            no_grad_view,
        )

    @staticmethod
    def call_imag(module, tensor, form):
        if form == "positional":
            return module.imag(tensor)
        return module.imag(**{form: tensor})

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return error
        raise AssertionError("torch.imag unexpectedly accepted the operation")

    def error_read_contract(self, module, tensor, form):
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
        errors = [
            self.error(lambda: self.call_imag(module, tensor, form))
            for _ in range(3)
        ]
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
        forms = ("positional", "input", "x", "a", "x1")
        for case, (actual, expected) in enumerate(
            zip(self.make_cases(torch), self.make_cases(reference_torch), strict=True)
        ):
            for form in forms:
                with self.subTest(case=case, form=form):
                    self.assertEqual(
                        self.error_read_contract(torch, actual, form),
                        self.error_read_contract(reference_torch, expected, form),
                    )

    @staticmethod
    def autograd_contract(module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        with module.no_grad():
            no_grad_view = leaf.transpose(0, 1)
        before = (
            leaf.requires_grad,
            leaf.is_leaf,
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            tuple(non_leaf.shape),
            non_leaf.stride(),
            non_leaf.storage_offset(),
            no_grad_view.requires_grad,
            no_grad_view.is_leaf,
        )
        errors = (
            TopLevelImagReferenceTests.error(lambda: module.imag(leaf)),
            TopLevelImagReferenceTests.error(lambda: module.imag(non_leaf)),
            TopLevelImagReferenceTests.error(lambda: module.imag(no_grad_view)),
        )
        after = (
            leaf.requires_grad,
            leaf.is_leaf,
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            tuple(non_leaf.shape),
            non_leaf.stride(),
            non_leaf.storage_offset(),
            no_grad_view.requires_grad,
            no_grad_view.is_leaf,
        )
        non_leaf.sum().backward()
        gradient = leaf.grad
        final_error = TopLevelImagReferenceTests.error(lambda: module.imag(input=leaf))
        return {
            "before": before,
            "after": after,
            "errors": tuple((type(error).__name__, str(error)) for error in errors),
            "gradient": np.asarray(gradient).copy(),
            "gradient_identity_preserved": leaf.grad is gradient,
            "final_error": (type(final_error).__name__, str(final_error)),
        }

    def test_autograd_and_no_grad_preservation_matches_pytorch_2_13(self):
        actual = self.autograd_contract(torch)
        expected = self.autograd_contract(reference_torch)
        np.testing.assert_array_equal(
            actual.pop("gradient"), expected.pop("gradient")
        )
        self.assertEqual(actual, expected)

    @staticmethod
    def callable_contract(module):
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
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.imag is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("imag"),
            "wildcard_identity": wildcard_namespace["imag"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_copy_pickling_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    @staticmethod
    def dispatch_observation(module_name):
        source = r'''
import importlib
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
function = module.imag
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result=marker):
        self.calls = []
        self.result = result

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

mode_observations = []
for call in (
    lambda: function(tensor),
    lambda: function(input=tensor),
    lambda: function(x=tensor),
    lambda: function(a=tensor),
    lambda: function(x1=tensor),
):
    mode = RecordingMode()
    with mode:
        result = call()
    func, dispatch_types, args, kwargs = mode.calls[0]
    mode_observations.append([
        result is marker,
        func is function,
        [item.__name__ for item in dispatch_types],
        len(args),
        None if kwargs is None else list(kwargs.keys()),
    ])

override_calls = []
class Override:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        override_calls.append((func, types, args, kwargs))
        return marker

override_results = []
for call in (
    lambda: function(Override()),
    lambda: function(input=Override()),
    lambda: function(x=Override()),
    lambda: function(a=Override()),
    lambda: function(x1=Override()),
):
    override_results.append(call() is marker)
override_observations = [
    [
        func is function,
        [item.__name__ for item in dispatch_types],
        len(args),
        None if kwargs is None else list(kwargs.keys()),
    ]
    for func, dispatch_types, args, kwargs in override_calls
]

forwarding_order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        forwarding_order.append(self.label)
        return func(*args, **(kwargs or {}))

try:
    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            function(input=tensor)
except Exception as error:
    forwarding_error = [type(error).__name__, str(error)]
else:
    forwarding_error = None

class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

declining_mode = DecliningMode()
try:
    with declining_mode:
        function(tensor)
except Exception as error:
    declining_error = [type(error).__name__, str(error).splitlines()[0]]
else:
    declining_error = None

class DecliningOverride:
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return NotImplemented

try:
    function(DecliningOverride())
except Exception as error:
    declining_override_error = [type(error).__name__, str(error).splitlines()[0]]
else:
    declining_override_error = None

invalid_observations = []
for call in (
    lambda: function(),
    lambda: function(tensor, out=None),
    lambda: function(tensor, extra=True),
    lambda: function(Override(), out=None),
    lambda: function(input=tensor, out=Override()),
):
    mode = RecordingMode()
    try:
        with mode:
            call()
    except Exception as error:
        invalid_observations.append([type(error).__name__, str(error), len(mode.calls)])
    else:
        invalid_observations.append(["OK", "", len(mode.calls)])

print(json.dumps({
    "mode_observations": mode_observations,
    "override_results": override_results,
    "override_observations": override_observations,
    "forwarding_order": forwarding_order,
    "forwarding_error": forwarding_error,
    "declining_error": declining_error,
    "declining_calls": declining_mode.calls,
    "declining_override_error": declining_override_error,
    "invalid_observations": invalid_observations,
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

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation("torch_rs"),
            self.dispatch_observation("torch"),
        )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_binding_type_and_out_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.imag(), lambda: reference_torch.imag()),
            (
                lambda: torch.imag(actual, actual),
                lambda: reference_torch.imag(expected, expected),
            ),
            (
                lambda: torch.imag(actual, input=actual),
                lambda: reference_torch.imag(expected, input=expected),
            ),
            (
                lambda: torch.imag(out=actual),
                lambda: reference_torch.imag(out=expected),
            ),
            (
                lambda: torch.imag(1, extra=True),
                lambda: reference_torch.imag(1, extra=True),
            ),
            (lambda: torch.imag(input=[]), lambda: reference_torch.imag(input=[])),
            (
                lambda: torch.imag(actual, out=[]),
                lambda: reference_torch.imag(expected, out=[]),
            ),
            (
                lambda: torch.imag(actual, extra=True, out=[]),
                lambda: reference_torch.imag(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.imag(actual, extra=True),
                lambda: reference_torch.imag(expected, extra=True),
            ),
            (
                lambda: torch.imag(input=actual, a=actual),
                lambda: reference_torch.imag(input=expected, a=expected),
            ),
            (
                lambda: torch.imag(a=actual, x=actual, out=None),
                lambda: reference_torch.imag(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.imag(x=actual, a=actual, out=None),
                lambda: reference_torch.imag(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.imag(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.imag(np.zeros((2, 3), dtype=np.float32)),
            ),
            (
                lambda: torch.imag(actual, out=None),
                lambda: reference_torch.imag(expected, out=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_complex_dtypes_remain_out_of_scope(self):
        self.assertTrue(hasattr(torch, "imag"))
        self.assertTrue(hasattr(reference_torch, "imag"))
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
