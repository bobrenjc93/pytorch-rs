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

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


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


class TensorComplexTests(unittest.TestCase):
    def test_scalar_contiguous_offset_and_strided_values_are_bit_exact(self):
        for bits in SPECIAL_BITS:
            real = float(np.asarray((bits,), dtype=np.uint32).view(np.float32)[0])
            expected = complex(real, 0.0)
            for layout, tensor, _, _ in complex_layouts(torch, bits):
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
                        self.assertTrue(tensor.is_contiguous())

                    for name, conversion in (
                        ("builtin", lambda: complex(tensor)),
                        ("method", tensor.__complex__),
                    ):
                        with self.subTest(conversion=name):
                            actual = conversion()
                            self.assertIs(type(actual), complex)
                            self.assertEqual(
                                python_complex_bits(actual),
                                python_complex_bits(expected),
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
                for conversion in (lambda: complex(tensor), tensor.__complex__):
                    with self.assertRaisesRegex(
                        ValueError, f"^{re.escape(message)}$"
                    ) as raised:
                        conversion()
                    self.assertIs(type(raised.exception), ValueError)

    def test_mode_dispatches_before_validation_and_forwards(self):
        descriptor = inspect.getattr_static(torch.Tensor, "__complex__")

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for shape in ((), (2,)):
            tensor = torch.full(shape, 2.5, requires_grad=True)
            graph_before = (tensor.requires_grad, tensor.is_leaf, tensor.grad)
            for name, conversion in (
                ("builtin", lambda: complex(tensor)),
                ("method", tensor.__complex__),
            ):
                with self.subTest(shape=shape, conversion=name):
                    mode = RecordingMode(3.0 + 4.0j)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with mode:
                            result = conversion()

                    self.assertEqual(result, 3.0 + 4.0j)
                    self.assertEqual(caught, [])
                    self.assertEqual(
                        (tensor.requires_grad, tensor.is_leaf, tensor.grad),
                        graph_before,
                    )
                    self.assertEqual(len(mode.calls), 1)
                    function, dispatch_types, args, kwargs = mode.calls[0]
                    self.assertIs(function, descriptor)
                    self.assertEqual(dispatch_types, (torch.Tensor,))
                    self.assertEqual(len(args), 1)
                    self.assertIs(args[0], tensor)
                    self.assertIsNone(kwargs)

        tensor = torch.tensor(2.5)
        marker = object()
        mode = RecordingMode(marker)
        with mode:
            self.assertIs(tensor.__complex__(), marker)

        mode = RecordingMode(7.0)
        with mode:
            with self.assertRaisesRegex(
                TypeError, "^__complex__ returned non-complex \\(type float\\)$"
            ):
                complex(tensor)

        for name, conversion in (
            ("builtin", lambda: complex(tensor)),
            ("method", tensor.__complex__),
        ):
            with self.subTest(forwarding=name):
                order = []

                class ForwardingMode(torch.overrides.TorchFunctionMode):
                    def __init__(self, label):
                        self.label = label

                    def __torch_function__(
                        self, func, types, args=(), kwargs=None
                    ):
                        order.append(self.label)
                        return func(*args, **(kwargs or {}))

                with ForwardingMode("lower"):
                    with ForwardingMode("upper"):
                        result = conversion()
                self.assertEqual(result, 2.5 + 0.0j)
                self.assertEqual(order, ["upper", "lower"])

    def test_requires_grad_conversion_does_not_mutate_graphs(self):
        for bits in SPECIAL_BITS:
            real = float(np.asarray((bits,), dtype=np.uint32).view(np.float32)[0])
            expected = complex(real, 0.0)
            for layout, tensor, leaf, expected_grad in complex_layouts(
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
                    actual_values = (complex(tensor), tensor.__complex__())
                graph_after = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    for actual in actual_values:
                        self.assertEqual(
                            python_complex_bits(actual),
                            python_complex_bits(expected),
                        )
                    self.assertEqual(graph_after, graph_before)
                    tensor.backward()
                    self.assertEqual(leaf.grad.tolist(), expected_grad)

    def test_requires_grad_warning_is_shared_once_only_and_respects_no_grad(self):
        script = r'''
import json, warnings
import torch_rs as torch

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
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        output = json.loads(result.stdout)
        self.assertEqual(
            output["no_grad_error"],
            [
                "ValueError",
                "only one element tensors can be converted to Python scalars",
            ],
        )
        self.assertEqual(output["no_grad_value"], [-0.0, 0.0])
        self.assertEqual(output["no_grad_warnings"], 0)
        self.assertEqual(
            output["grad"]["outcomes"],
            [
                [
                    "ValueError",
                    "only one element tensors can be converted to Python scalars",
                ],
                ["value", 0.0],
                ["value", [0.0, 0.0]],
                ["value", [0.0, 0.0]],
            ],
        )
        self.assertEqual(len(output["grad"]["warnings"]), 1)
        category, message, filename, lineno = output["grad"]["warnings"][0]
        self.assertEqual(category, "UserWarning")
        self.assertEqual(message.split(" (Triggered internally at ", 1)[0], WARNING)
        self.assertEqual(filename, "<string>")
        self.assertGreater(lineno, 0)

    def test_special_method_descriptor_matches_pytorch_2_13(self):
        tensor = torch.tensor([2.75])
        descriptor = inspect.getattr_static(torch.Tensor, "__complex__")
        bound = tensor.__complex__

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertEqual(
            repr(descriptor),
            "<method '__complex__' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "__complex__")
        self.assertEqual(descriptor.__qualname__, "TensorBase.__complex__")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        assert_no_argument_signature(self, descriptor, "(self, /)")

        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(bound.__name__, "__complex__")
        self.assertIsNone(bound.__doc__)
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor(tensor), complex(2.75, 0.0))
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(
            descriptor.__get__(tensor, torch.Tensor)(), complex(2.75, 0.0)
        )

        calls = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.__complex__() needs an argument",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.__complex__() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.__complex__(1),
                "TensorBase.__complex__() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.__complex__() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, value=1),
                "TensorBase.__complex__() takes no keyword arguments",
            ),
            (
                lambda: tensor.__complex__(value=1),
                (
                    "Tensor.__complex__() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.__complex__() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(value=1),
                "Tensor.__complex__() takes no keyword arguments",
            ),
            (
                lambda: descriptor(1.0),
                "descriptor '__complex__' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'float' object",
            ),
            (
                lambda: descriptor.__get__(1.0, torch.Tensor),
                "descriptor '__complex__' for 'torch._C.TensorBase' objects "
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
