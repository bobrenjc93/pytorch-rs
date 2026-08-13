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


def float_layouts(module, bits, *, requires_grad=False):
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
class TensorFloatReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("float differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_and_layout_metadata_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            actual_layouts = float_layouts(torch, bits)
            expected_layouts = float_layouts(reference_torch, bits)
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
                    actual_value = float(actual)
                    expected_value = float(expected)
                    self.assertIs(type(actual_value), type(expected_value))
                    self.assertEqual(
                        python_float_bits(actual_value),
                        python_float_bits(expected_value),
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
                    lambda actual=actual: float(actual),
                    lambda expected=expected: float(expected),
                )
                self.assert_error_matches(actual.__float__, expected.__float__)

    def test_graph_state_and_backward_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            outcomes = []
            for module in (torch, reference_torch):
                module_outcomes = []
                for layout, tensor, leaf, _ in float_layouts(
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
                        value = float(tensor)
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
                            python_float_bits(value),
                            graph_before,
                            graph_after,
                            leaf.grad.tolist(),
                        )
                    )
                outcomes.append(module_outcomes)
            with self.subTest(bits=f"{bits:#010x}"):
                self.assertEqual(outcomes[0], outcomes[1])

    def test_warning_message_order_once_behavior_and_no_grad_match(self):
        script = r'''
import importlib, json, warnings
torch = importlib.import_module(MODULE)
outputs = {}

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with torch.no_grad():
        outputs["no_grad_value"] = float(torch.tensor(2.0, requires_grad=True))
outputs["no_grad_warnings"] = len(caught)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    outcomes = []
    for shape in ((2,), (), (1,)):
        tensor = torch.zeros(shape, requires_grad=True)
        try:
            outcomes.append(["value", float(tensor)])
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
        actual = torch.tensor([2.0])
        expected = reference_torch.tensor([2.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "__float__")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "__float__"
        )

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "__float__")
            self.assertEqual(descriptor.__qualname__, "TensorBase.__float__")
            self.assertIsNone(descriptor.__text_signature__)
            self.assertIsNone(descriptor.__doc__)
            self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
            self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
            with self.assertRaises(ValueError):
                inspect.signature(descriptor)
        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))

        actual_bound = actual.__float__
        expected_bound = expected.__float__
        for bound in (actual_bound, expected_bound):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "__float__")
            self.assertIsNone(bound.__text_signature__)
            self.assertIsNone(bound.__doc__)
            with self.assertRaises(ValueError):
                inspect.signature(bound)

        self.assertEqual(actual_descriptor(actual), expected_descriptor(expected))
        call_pairs = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (
                lambda: actual_descriptor(actual, value=1),
                lambda: expected_descriptor(expected, value=1),
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
