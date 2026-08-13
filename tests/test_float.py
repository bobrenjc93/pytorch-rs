import inspect
import json
import re
import struct
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


SPECIAL_BITS = (
    0xC020_0000,
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC1_2345,
    0xFFC5_4321,
)
WARNING = (
    "Converting a tensor with requires_grad=True to a scalar may lead to "
    "unexpected behavior.\nConsider using tensor.detach() first."
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


class TensorFloatTests(unittest.TestCase):
    def test_scalar_contiguous_offset_and_strided_values_are_bit_exact(self):
        for bits in SPECIAL_BITS:
            expected = float(
                np.asarray((bits,), dtype=np.uint32).view(np.float32)[0]
            )
            for layout, tensor, _, _ in float_layouts(torch, bits):
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    if layout == "scalar":
                        self.assertEqual(tensor.shape, ())
                        self.assertEqual(tensor.stride(), ())
                        self.assertEqual(tensor.storage_offset(), 0)
                    elif layout == "contiguous":
                        self.assertEqual(tensor.shape, (1,))
                        self.assertEqual(tensor.stride(), (1,))
                        self.assertEqual(tensor.storage_offset(), 0)
                        self.assertTrue(tensor.is_contiguous())
                    elif layout == "offset":
                        self.assertEqual(tensor.shape, ())
                        self.assertEqual(tensor.stride(), ())
                        self.assertEqual(tensor.storage_offset(), 1)
                    else:
                        self.assertEqual(tensor.shape, (1,))
                        self.assertEqual(tensor.stride(), (2,))
                        self.assertEqual(tensor.storage_offset(), 1)
                        # PyTorch treats singleton dimensions as contiguous even
                        # when their recorded stride is non-canonical.
                        self.assertTrue(tensor.is_contiguous())
                    actual = float(tensor)
                    self.assertIs(type(actual), float)
                    self.assertEqual(
                        python_float_bits(actual),
                        python_float_bits(expected),
                    )

    def test_empty_and_multi_element_errors_match_pytorch_2_13(self):
        message = "only one element tensors can be converted to Python scalars"
        cases = (
            torch.zeros((0,)),
            torch.zeros((2,)),
            torch.zeros((2, 0, 3)).transpose(0, 2),
            torch.zeros((2, 3)).transpose(0, 1),
        )
        for tensor in cases:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                for conversion in (float, tensor.__float__):
                    with self.assertRaisesRegex(
                        ValueError, f"^{re.escape(message)}$"
                    ) as raised:
                        conversion(tensor) if conversion is float else conversion()
                    self.assertIs(type(raised.exception), ValueError)

    def test_requires_grad_conversion_does_not_mutate_graphs(self):
        for bits in SPECIAL_BITS:
            expected = float(
                np.asarray((bits,), dtype=np.uint32).view(np.float32)[0]
            )
            for layout, tensor, leaf, expected_grad in float_layouts(
                torch, bits, requires_grad=True
            ):
                graph_before = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    actual = float(tensor)
                graph_after = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    self.assertEqual(
                        python_float_bits(actual), python_float_bits(expected)
                    )
                    self.assertEqual(graph_after, graph_before)
                    tensor.backward()
                    self.assertEqual(leaf.grad.tolist(), expected_grad)

    def test_requires_grad_warning_is_once_only_and_respects_no_grad(self):
        script = r'''
import json, warnings
import torch_rs as torch

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
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(
            output["grad"]["outcomes"],
            [
                [
                    "ValueError",
                    "only one element tensors can be converted to Python scalars",
                ],
                ["value", 0.0],
                ["value", 0.0],
            ],
        )
        self.assertEqual(len(output["grad"]["warnings"]), 1)
        category, message, filename, lineno = output["grad"]["warnings"][0]
        self.assertEqual(category, "UserWarning")
        self.assertEqual(message.split(" (Triggered internally at ", 1)[0], WARNING)
        self.assertEqual(filename, "<string>")
        self.assertGreater(lineno, 0)
        self.assertEqual(output["no_grad_value"], 2.0)
        self.assertEqual(output["no_grad_warnings"], 0)

    def test_special_method_descriptor_matches_pytorch_2_13(self):
        tensor = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "__float__")
        bound = tensor.__float__

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertEqual(
            repr(descriptor), "<method '__float__' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "__float__")
        self.assertEqual(descriptor.__qualname__, "TensorBase.__float__")
        self.assertIsNone(descriptor.__doc__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)

        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(bound.__name__, "__float__")
        self.assertIsNone(bound.__doc__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor(tensor), 2.0)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(descriptor.__get__(tensor, torch.Tensor)(), 2.0)

        calls = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.__float__() needs an argument",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.__float__() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.__float__() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, value=1),
                "TensorBase.__float__() takes no keyword arguments",
            ),
            (
                lambda: bound(value=1),
                "Tensor.__float__() takes no keyword arguments",
            ),
            (
                lambda: descriptor(1.0),
                "descriptor '__float__' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'float' object",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
