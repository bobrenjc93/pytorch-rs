import inspect
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


FINITE_CASES = (
    (0xC020_0000, -2),
    (0xBFC0_0000, -1),
    (0x3FC0_0000, 1),
    (0x4020_0000, 2),
    (0x0000_0001, 0),
    (0x8000_0001, 0),
    (0x7F7F_FFFF, 340282346638528859811704183484516925440),
    (0xFF7F_FFFF, -340282346638528859811704183484516925440),
)

NONFINITE_CASES = (
    (0x7F80_0000, OverflowError, "cannot convert float infinity to integer"),
    (0xFF80_0000, OverflowError, "cannot convert float infinity to integer"),
    (0x7FC1_2345, ValueError, "cannot convert float NaN to integer"),
    (0xFFC5_4321, ValueError, "cannot convert float NaN to integer"),
)


def int_layouts(module, bits, *, requires_grad=False):
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


class TensorIntTests(unittest.TestCase):
    def test_finite_values_truncate_toward_zero_for_every_layout(self):
        for bits, expected in FINITE_CASES:
            for layout, tensor, _, _ in int_layouts(torch, bits):
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    if layout == "scalar":
                        self.assertEqual(tensor.shape, ())
                        self.assertEqual(tensor.stride(), ())
                        self.assertEqual(tensor.storage_offset(), 0)
                    elif layout == "contiguous":
                        self.assertEqual(tensor.shape, (1,))
                        self.assertEqual(tensor.stride(), (1,))
                        self.assertEqual(tensor.storage_offset(), 0)
                    elif layout == "offset":
                        self.assertEqual(tensor.shape, ())
                        self.assertEqual(tensor.stride(), ())
                        self.assertEqual(tensor.storage_offset(), 1)
                    else:
                        self.assertEqual(tensor.shape, (1,))
                        self.assertEqual(tensor.stride(), (2,))
                        self.assertEqual(tensor.storage_offset(), 1)

                    for conversion in (int, tensor.__int__):
                        actual = (
                            conversion(tensor) if conversion is int else conversion()
                        )
                        self.assertIs(type(actual), int)
                        self.assertEqual(actual, expected)

    def test_finite_conversion_is_silent_and_preserves_autograd_graphs(self):
        for bits, expected in FINITE_CASES:
            for layout, tensor, leaf, expected_grad in int_layouts(
                torch, bits, requires_grad=True
            ):
                graph_before = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual = int(tensor)
                    method_actual = tensor.__int__()
                graph_after = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )

                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    self.assertEqual(actual, expected)
                    self.assertEqual(method_actual, expected)
                    self.assertEqual(caught, [])
                    self.assertEqual(graph_after, graph_before)
                    tensor.backward()
                    self.assertEqual(leaf.grad.tolist(), expected_grad)

    def test_cardinality_and_nonfinite_errors_are_exact_and_silent(self):
        cardinality_message = (
            "only one element tensors can be converted to Python scalars"
        )
        cardinality_cases = []
        for shape in ((0,), (2,)):
            leaf = torch.zeros(shape, requires_grad=True)
            cardinality_cases.append((leaf, leaf))
        for shape in ((2, 0, 3), (2, 3)):
            leaf = torch.zeros(shape, requires_grad=True)
            cardinality_cases.append((leaf.transpose(0, -1), leaf))

        for tensor, leaf in cardinality_cases:
            graph_before = (
                tensor.requires_grad,
                tensor.is_leaf,
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for conversion in (int, tensor.__int__):
                    with self.assertRaisesRegex(
                        ValueError, f"^{re.escape(cardinality_message)}$"
                    ) as raised:
                        conversion(tensor) if conversion is int else conversion()
                    self.assertIs(type(raised.exception), ValueError)
            graph_after = (
                tensor.requires_grad,
                tensor.is_leaf,
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
            )
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assertEqual(caught, [])
                self.assertEqual(graph_after, graph_before)

        for bits, error_type, message in NONFINITE_CASES:
            for layout, tensor, leaf, expected_grad in int_layouts(
                torch, bits, requires_grad=True
            ):
                graph_before = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    for conversion in (int, tensor.__int__):
                        with self.assertRaises(error_type) as raised:
                            conversion(tensor) if conversion is int else conversion()
                        self.assertEqual(str(raised.exception), message)
                        self.assertIs(type(raised.exception), error_type)
                graph_after = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    self.assertEqual(caught, [])
                    self.assertEqual(graph_after, graph_before)
                    tensor.backward()
                    self.assertEqual(leaf.grad.tolist(), expected_grad)

    def test_special_method_descriptor_matches_pytorch_2_13(self):
        tensor = torch.tensor([2.75])
        descriptor = inspect.getattr_static(torch.Tensor, "__int__")
        bound = tensor.__int__

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertEqual(
            repr(descriptor), "<method '__int__' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "__int__")
        self.assertEqual(descriptor.__qualname__, "TensorBase.__int__")
        self.assertIsNone(descriptor.__doc__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)

        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(bound.__name__, "__int__")
        self.assertIsNone(bound.__doc__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor(tensor), 2)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(descriptor.__get__(tensor, torch.Tensor)(), 2)

        calls = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.__int__() needs an argument",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.__int__() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.__int__(1),
                "TensorBase.__int__() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.__int__() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, value=1),
                "TensorBase.__int__() takes no keyword arguments",
            ),
            (
                lambda: tensor.__int__(value=1),
                "TensorBase.__int__() takes no keyword arguments",
            ),
            (
                lambda: bound(value=1),
                "Tensor.__int__() takes no keyword arguments",
            ),
            (
                lambda: descriptor(1.0),
                "descriptor '__int__' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'float' object",
            ),
            (
                lambda: descriptor.__get__(1.0, torch.Tensor),
                "descriptor '__int__' for 'torch._C.TensorBase' objects "
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
