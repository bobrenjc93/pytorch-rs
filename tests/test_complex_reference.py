import inspect
import json
import struct
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SPECIAL_BITS = (
    0xC020_0000,
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC1_2345,
    0xFFC5_4321,
)


def python_float_bits(value):
    return struct.unpack("=Q", struct.pack("=d", value))[0]


def python_complex_bits(value):
    return python_float_bits(value.real), python_float_bits(value.imag)


def complex_layouts(module, bits, *, requires_grad=False):
    values = np.asarray((0x3F80_0000, bits), dtype=np.uint32).view(np.float32)
    scalar_leaf = module.tensor(
        memoryview(values[1:]),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    scalar = scalar_leaf.reshape(())

    contiguous = module.tensor(
        memoryview(values[1:]),
        dtype=module.float32,
        requires_grad=requires_grad,
    )

    offset_leaf = module.tensor(
        memoryview(values),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    offset = offset_leaf[1]

    strided_leaf = module.tensor(
        memoryview(values),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    strided = strided_leaf.reshape((1, 2)).transpose(0, 1)[1]
    return (
        ("scalar", scalar, scalar_leaf, [1.0]),
        ("contiguous", contiguous, contiguous, [1.0]),
        ("offset", offset, offset_leaf, [0.0, 1.0]),
        ("strided", strided, strided_leaf, [0.0, 1.0]),
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorComplexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("complex differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_and_layout_metadata_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            actual_layouts = complex_layouts(torch, bits)
            expected_layouts = complex_layouts(reference_torch, bits)
            for actual_case, expected_case in zip(
                actual_layouts, expected_layouts, strict=True
            ):
                actual_layout, actual, _, _ = actual_case
                expected_layout, expected, _, _ = expected_case
                with self.subTest(bits=f"{bits:#010x}", layout=actual_layout):
                    self.assertEqual(actual_layout, expected_layout)
                    self.assertEqual(actual.shape, tuple(expected.shape))
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )
                    actual_values = (complex(actual), actual.__complex__())
                    expected_values = (complex(expected), expected.__complex__())
                    for actual_value, expected_value in zip(
                        actual_values, expected_values, strict=True
                    ):
                        self.assertIs(type(actual_value), type(expected_value))
                        self.assertEqual(
                            python_complex_bits(actual_value),
                            python_complex_bits(expected_value),
                        )

    def test_cardinality_errors_match_pytorch_2_13(self):
        actual_cases = (
            torch.zeros((0,)),
            torch.zeros((2,)),
            torch.zeros((2, 0, 3)).transpose(0, 2),
            torch.zeros((2, 3)).transpose(0, 1),
        )
        expected_cases = (
            reference_torch.zeros((0,)),
            reference_torch.zeros((2,)),
            reference_torch.zeros((2, 0, 3)).transpose(0, 2),
            reference_torch.zeros((2, 3)).transpose(0, 1),
        )
        for actual, expected in zip(actual_cases, expected_cases, strict=True):
            with self.subTest(shape=actual.shape, stride=actual.stride()):
                self.assertEqual(actual.shape, tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assert_error_matches(
                    lambda actual=actual: complex(actual),
                    lambda expected=expected: complex(expected),
                )
                self.assert_error_matches(actual.__complex__, expected.__complex__)

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib, inspect, json, warnings
module = importlib.import_module(MODULE)
descriptor = inspect.getattr_static(module.Tensor, "__complex__")

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

records = []
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for layout, tensor in (
        ("scalar", module.tensor(2.5, requires_grad=True)),
        ("multi", module.tensor([2.5, 2.5], requires_grad=True)),
    ):
        graph_before = [tensor.requires_grad, tensor.is_leaf, tensor.grad is None]
        for conversion in ("builtin", "method"):
            mode = RecordingMode(3.0 + 4.0j)
            with mode:
                if conversion == "builtin":
                    result = complex(tensor)
                else:
                    result = tensor.__complex__()
            function, dispatch_types, args, kwargs = mode.calls[0]
            records.append({
                "layout": layout,
                "conversion": conversion,
                "result": [result.real, result.imag],
                "call_count": len(mode.calls),
                "function_type": type(function).__name__,
                "function_name": function.__name__,
                "function_qualname": function.__qualname__,
                "function_is_descriptor": function is descriptor,
                "types": dispatch_types == (module.Tensor,),
                "args": len(args) == 1 and args[0] is tensor,
                "kwargs_is_none": kwargs is None,
                "graph_unchanged": graph_before == [
                    tensor.requires_grad,
                    tensor.is_leaf,
                    tensor.grad is None,
                ],
            })

tensor = module.tensor(2.5)
marker = object()
arbitrary_mode = RecordingMode(marker)
with arbitrary_mode:
    arbitrary_result = tensor.__complex__()

invalid_mode = RecordingMode(7.0)
try:
    with invalid_mode:
        complex(tensor)
except Exception as error:
    invalid_result = [type(error).__name__, str(error)]
else:
    invalid_result = None

forwarded = {}
for conversion in ("builtin", "method"):
    order = []

    class ForwardingMode(module.overrides.TorchFunctionMode):
        def __init__(self, label):
            self.label = label

        def __torch_function__(self, func, types, args=(), kwargs=None):
            order.append(self.label)
            return func(*args, **(kwargs or {}))

    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            if conversion == "builtin":
                result = complex(tensor)
            else:
                result = tensor.__complex__()
    forwarded[conversion] = {
        "order": order,
        "result": [result.real, result.imag],
    }

print(json.dumps({
    "records": records,
    "warnings": [
        [item.category.__name__, str(item.message)] for item in caught
    ],
    "arbitrary_method_result": arbitrary_result is marker,
    "arbitrary_call_count": len(arbitrary_mode.calls),
    "invalid_builtin_result": invalid_result,
    "invalid_call_count": len(invalid_mode.calls),
    "forwarded": forwarded,
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

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_graph_state_and_backward_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            outcomes = []
            for module in (torch, reference_torch):
                module_outcomes = []
                for layout, tensor, leaf, _ in complex_layouts(
                    module, bits, requires_grad=True
                ):
                    graph_before = (
                        tensor.requires_grad,
                        tensor.is_leaf,
                        leaf.requires_grad,
                        leaf.is_leaf,
                        leaf.grad is None,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        values = (complex(tensor), tensor.__complex__())
                    graph_after = (
                        tensor.requires_grad,
                        tensor.is_leaf,
                        leaf.requires_grad,
                        leaf.is_leaf,
                        leaf.grad is None,
                    )
                    tensor.backward()
                    module_outcomes.append(
                        (
                            layout,
                            tuple(python_complex_bits(value) for value in values),
                            graph_before,
                            graph_after,
                            leaf.grad.tolist(),
                        )
                    )
                outcomes.append(module_outcomes)
            with self.subTest(bits=f"{bits:#010x}"):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_warning_order_shared_once_behavior_and_no_grad_match(self):
        script = r'''
import importlib, json, warnings
torch = importlib.import_module(MODULE)
outputs = {}

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with torch.no_grad():
        value = complex(torch.tensor(-0.0, requires_grad=True))
        outputs["no_grad_value"] = [value.real, value.imag]
        try:
            complex(torch.zeros((2,), requires_grad=True))
        except Exception as error:
            outputs["no_grad_error"] = [type(error).__name__, str(error)]
outputs["no_grad_warnings"] = len(caught)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    outcomes = []
    calls = (
        lambda: complex(torch.zeros((2,), requires_grad=True)),
        lambda: float(torch.zeros((), requires_grad=True)),
        lambda: complex(torch.zeros((), requires_grad=True)),
        lambda: torch.zeros((1,), requires_grad=True).__complex__(),
    )
    for call in calls:
        try:
            value = call()
            if isinstance(value, complex):
                value = [value.real, value.imag]
            outcomes.append(["value", value])
        except Exception as error:
            outcomes.append([type(error).__name__, str(error)])
outputs["grad"] = {
    "outcomes": outcomes,
    "warnings": [
        [item.category.__name__, str(item.message), item.filename, item.lineno]
        for item in caught
    ],
}
print(json.dumps(outputs))
'''

        def run(module):
            module_script = f"MODULE = {module!r}\n" + script
            result = subprocess.run(
                [sys.executable, "-c", module_script],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout)

        self.assertEqual(run("torch_rs"), run("torch"))

    def test_descriptor_and_call_errors_match_pytorch_2_13(self):
        actual = torch.tensor([2.75])
        expected = reference_torch.tensor([2.75])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "__complex__")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "__complex__"
        )

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "__complex__")
            self.assertEqual(descriptor.__qualname__, "TensorBase.__complex__")
            self.assertIsNone(descriptor.__doc__)
            self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
            self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
            assert_no_argument_signature(self, descriptor, "(self, /)")
        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))

        actual_bound = actual.__complex__
        expected_bound = expected.__complex__
        for bound in (actual_bound, expected_bound):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "__complex__")
            self.assertIsNone(bound.__doc__)
            assert_no_argument_signature(self, bound, "()")

        self.assertEqual(
            actual_descriptor(actual), expected_descriptor(expected)
        )
        call_pairs = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (lambda: actual.__complex__(1), lambda: expected.__complex__(1)),
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (
                lambda: actual_descriptor(actual, value=1),
                lambda: expected_descriptor(expected, value=1),
            ),
            (
                lambda: actual.__complex__(value=1),
                lambda: expected.__complex__(value=1),
            ),
            (
                lambda: actual_bound(value=1),
                lambda: expected_bound(value=1),
            ),
            (lambda: actual_descriptor(1.0), lambda: expected_descriptor(1.0)),
            (
                lambda: actual_descriptor.__get__(1.0, torch.Tensor),
                lambda: expected_descriptor.__get__(
                    1.0, reference_torch.Tensor
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
