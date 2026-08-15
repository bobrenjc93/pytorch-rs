import inspect
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


FINITE_BITS = (
    0xC020_0000,
    0xBFC0_0000,
    0x3FC0_0000,
    0x4020_0000,
    0x0000_0001,
    0x8000_0001,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
)

NONFINITE_BITS = (
    0x7F80_0000,
    0xFF80_0000,
    0x7FC1_2345,
    0xFFC5_4321,
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
        ("scalar", scalar, scalar_leaf),
        ("contiguous", contiguous, contiguous),
        ("offset", offset, offset_leaf),
        ("strided", strided, strided_leaf),
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIntReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("int differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_finite_values_layouts_warnings_and_graphs_match_pytorch_2_13(self):
        for bits in FINITE_BITS:
            actual_layouts = int_layouts(torch, bits, requires_grad=True)
            expected_layouts = int_layouts(
                reference_torch, bits, requires_grad=True
            )
            for actual_case, expected_case in zip(
                actual_layouts, expected_layouts, strict=True
            ):
                actual_layout, actual, actual_leaf = actual_case
                expected_layout, expected, expected_leaf = expected_case
                with self.subTest(bits=f"{bits:#010x}", layout=actual_layout):
                    self.assertEqual(actual_layout, expected_layout)
                    self.assertEqual(actual.shape, tuple(expected.shape))
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )

                    actual_graph_before = (
                        actual.requires_grad,
                        actual.is_leaf,
                        actual_leaf.requires_grad,
                        actual_leaf.is_leaf,
                        actual_leaf.grad is None,
                    )
                    expected_graph_before = (
                        expected.requires_grad,
                        expected.is_leaf,
                        expected_leaf.requires_grad,
                        expected_leaf.is_leaf,
                        expected_leaf.grad is None,
                    )
                    with warnings.catch_warnings(record=True) as actual_warnings:
                        warnings.simplefilter("always")
                        actual_values = (int(actual), actual.__int__())
                    with warnings.catch_warnings(record=True) as expected_warnings:
                        warnings.simplefilter("always")
                        expected_values = (int(expected), expected.__int__())
                    actual_graph_after = (
                        actual.requires_grad,
                        actual.is_leaf,
                        actual_leaf.requires_grad,
                        actual_leaf.is_leaf,
                        actual_leaf.grad is None,
                    )
                    expected_graph_after = (
                        expected.requires_grad,
                        expected.is_leaf,
                        expected_leaf.requires_grad,
                        expected_leaf.is_leaf,
                        expected_leaf.grad is None,
                    )

                    self.assertEqual(actual_values, expected_values)
                    self.assertIs(type(actual_values[0]), type(expected_values[0]))
                    self.assertEqual(actual_warnings, expected_warnings)
                    self.assertEqual(actual_warnings, [])
                    self.assertEqual(actual_graph_before, expected_graph_before)
                    self.assertEqual(actual_graph_after, expected_graph_after)
                    self.assertEqual(actual_graph_after, actual_graph_before)

                    actual.backward()
                    expected.backward()
                    self.assertEqual(
                        actual_leaf.grad.tolist(), expected_leaf.grad.tolist()
                    )

    def test_cardinality_and_nonfinite_errors_match_without_side_effects(self):
        actual_cardinality = []
        expected_cardinality = []
        for shape in ((0,), (2,)):
            actual_leaf = torch.zeros(shape, requires_grad=True)
            expected_leaf = reference_torch.zeros(shape, requires_grad=True)
            actual_cardinality.append((actual_leaf, actual_leaf))
            expected_cardinality.append((expected_leaf, expected_leaf))
        for shape in ((2, 0, 3), (2, 3)):
            actual_leaf = torch.zeros(shape, requires_grad=True)
            expected_leaf = reference_torch.zeros(shape, requires_grad=True)
            actual_cardinality.append((actual_leaf.transpose(0, -1), actual_leaf))
            expected_cardinality.append(
                (expected_leaf.transpose(0, -1), expected_leaf)
            )

        for actual_case, expected_case in zip(
            actual_cardinality, expected_cardinality, strict=True
        ):
            actual, actual_leaf = actual_case
            expected, expected_leaf = expected_case
            actual_before = (
                actual.requires_grad,
                actual.is_leaf,
                actual_leaf.requires_grad,
                actual_leaf.is_leaf,
                actual_leaf.grad,
            )
            expected_before = (
                expected.requires_grad,
                expected.is_leaf,
                expected_leaf.requires_grad,
                expected_leaf.is_leaf,
                expected_leaf.grad,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                self.assert_error_matches(
                    lambda actual=actual: int(actual),
                    lambda expected=expected: int(expected),
                )
                self.assert_error_matches(actual.__int__, expected.__int__)
            actual_after = (
                actual.requires_grad,
                actual.is_leaf,
                actual_leaf.requires_grad,
                actual_leaf.is_leaf,
                actual_leaf.grad,
            )
            expected_after = (
                expected.requires_grad,
                expected.is_leaf,
                expected_leaf.requires_grad,
                expected_leaf.is_leaf,
                expected_leaf.grad,
            )
            with self.subTest(shape=actual.shape, stride=actual.stride()):
                self.assertEqual(caught, [])
                self.assertEqual(actual_before, expected_before)
                self.assertEqual(actual_after, expected_after)
                self.assertEqual(actual_after, actual_before)

        for bits in NONFINITE_BITS:
            actual_layouts = int_layouts(torch, bits, requires_grad=True)
            expected_layouts = int_layouts(
                reference_torch, bits, requires_grad=True
            )
            for actual_case, expected_case in zip(
                actual_layouts, expected_layouts, strict=True
            ):
                layout, actual, actual_leaf = actual_case
                _, expected, expected_leaf = expected_case
                actual_before = (
                    actual.requires_grad,
                    actual.is_leaf,
                    actual_leaf.grad,
                )
                expected_before = (
                    expected.requires_grad,
                    expected.is_leaf,
                    expected_leaf.grad,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assert_error_matches(
                        lambda actual=actual: int(actual),
                        lambda expected=expected: int(expected),
                    )
                    self.assert_error_matches(actual.__int__, expected.__int__)
                actual_after = (
                    actual.requires_grad,
                    actual.is_leaf,
                    actual_leaf.grad,
                )
                expected_after = (
                    expected.requires_grad,
                    expected.is_leaf,
                    expected_leaf.grad,
                )
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    self.assertEqual(caught, [])
                    self.assertEqual(actual_before, expected_before)
                    self.assertEqual(actual_after, expected_after)
                    self.assertEqual(actual_after, actual_before)
                    actual.backward()
                    expected.backward()
                    self.assertEqual(
                        actual_leaf.grad.tolist(), expected_leaf.grad.tolist()
                    )

    def test_descriptor_and_invalid_calls_match_pytorch_2_13(self):
        actual = torch.tensor([2.75])
        expected = reference_torch.tensor([2.75])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "__int__")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "__int__"
        )

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "__int__")
            self.assertEqual(descriptor.__qualname__, "TensorBase.__int__")
            self.assertIsNone(descriptor.__doc__)
            self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
            self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
            self.assertFalse(hasattr(descriptor, "__module__"))
            if sys.version_info >= (3, 13):
                self.assertEqual(descriptor.__text_signature__, "($self, /)")
                self.assertEqual(str(inspect.signature(descriptor)), "(self, /)")
            else:
                self.assertIsNone(descriptor.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)
        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))

        actual_bound = actual.__int__
        expected_bound = expected.__int__
        for bound in (actual_bound, expected_bound):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "__int__")
            self.assertIsNone(bound.__doc__)
            if sys.version_info >= (3, 13):
                self.assertEqual(bound.__text_signature__, "($self, /)")
                self.assertEqual(str(inspect.signature(bound)), "()")
            else:
                self.assertIsNone(bound.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(bound)

        self.assertEqual(actual_descriptor(actual), expected_descriptor(expected))
        self.assertIs(
            actual_descriptor.__get__(None, torch.Tensor), actual_descriptor
        )
        call_pairs = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (lambda: actual.__int__(1), lambda: expected.__int__(1)),
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (
                lambda: actual_descriptor(actual, value=1),
                lambda: expected_descriptor(expected, value=1),
            ),
            (
                lambda: actual.__int__(value=1),
                lambda: expected.__int__(value=1),
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
