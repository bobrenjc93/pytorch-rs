import inspect
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
class TensorRavelReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.cpu().detach().numpy()
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_layouts_identity_aliasing_and_lifetimes_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist(), requires_grad=True)
        expected_base = reference_torch.tensor(values, requires_grad=True)
        actual_singleton_base = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        expected_singleton_base = reference_torch.tensor([[0.0, 1.0, 2.0, 3.0]])

        cases = (
            ("scalar", actual_base[0][0][0], expected_base[0][0][0]),
            ("vector", actual_base[0][1], expected_base[0][1]),
            ("ordinary", actual_base, expected_base),
            ("offset", actual_base[1], expected_base[1]),
            ("transpose", actual_base.transpose(0, 2), expected_base.transpose(0, 2)),
            (
                "strided-vector",
                actual_base.transpose(0, 2)[0][0],
                expected_base.transpose(0, 2)[0][0],
            ),
            (
                "singleton-stride",
                actual_singleton_base.transpose(0, 1)[2],
                expected_singleton_base.transpose(0, 1)[2],
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3), requires_grad=True)
                .transpose(0, 2)[1],
            ),
        )
        retained = []
        for case, actual_source, expected_source in cases:
            actual = actual_source.ravel()
            expected = expected_source.ravel()
            self.assertIsNot(actual, actual_source)
            self.assertIsNot(expected, expected_source)
            self.assert_matches(actual, expected, case=case)
            self.assertEqual(
                expected.untyped_storage().data_ptr()
                == expected_source.untyped_storage().data_ptr(),
                expected_source.is_contiguous(),
            )
            retained.append((actual, expected))

        del actual_base, expected_base, actual_singleton_base, expected_singleton_base, cases
        self.assert_matches(retained[4][0], retained[4][1], case="lifetime-copy")
        self.assert_matches(retained[3][0], retained[3][1], case="lifetime-view")

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        gradients = []
        states = []
        scalar_gradients = []
        empty_gradients = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            output = leaf.transpose(0, 1).ravel()
            states.append((output.requires_grad, output.is_leaf))
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            scalar = module.tensor(2.0, requires_grad=True)
            (scalar.ravel() * 7.0).sum().backward()
            scalar_gradients.append(scalar.grad.item())

            empty = module.zeros((2, 0, 3), requires_grad=True)
            empty.ravel().sum().backward()
            empty_gradients.append((empty.grad.shape, np.asarray(empty.grad).copy()))

            source = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            non_contiguous = source.transpose(0, 1)
            with module.no_grad():
                alias = source.ravel()
                copied = non_contiguous.ravel()
            states.append(
                (
                    alias.requires_grad,
                    alias.is_leaf,
                    copied.requires_grad,
                    copied.is_leaf,
                )
            )

        np.testing.assert_array_equal(gradients[0], gradients[1])
        self.assertEqual(states[0], states[2])
        self.assertEqual(states[1], states[3])
        self.assertEqual(scalar_gradients[0], scalar_gradients[1])
        self.assertEqual(empty_gradients[0][0], empty_gradients[1][0])
        np.testing.assert_array_equal(empty_gradients[0][1], empty_gradients[1][1])

    def test_descriptor_and_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        actual_descriptor = inspect.getattr_static(torch.Tensor, "ravel")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "ravel")
        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "ravel")
            if sys.version_info >= (3, 13):
                self.assertEqual(descriptor.__text_signature__, "($self, /)")
                self.assertEqual(str(inspect.signature(descriptor)), "(self, /)")
            else:
                self.assertIsNone(descriptor.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)

        for bound in (actual.ravel, expected.ravel):
            self.assertIs(type(bound), types.BuiltinMethodType)
            if sys.version_info >= (3, 13):
                self.assertEqual(bound.__text_signature__, "($self, /)")
                self.assertEqual(str(inspect.signature(bound)), "()")
            else:
                self.assertIsNone(bound.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(bound)

        self.assert_matches(
            actual_descriptor(actual), expected_descriptor(expected), case="unbound-call"
        )

        actual_bound = actual.ravel
        expected_bound = expected.ravel
        for actual_call, expected_call in (
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (lambda: actual_bound(dim=0), lambda: expected_bound(dim=0)),
            (lambda: actual_bound(input=actual), lambda: expected_bound(input=expected)),
        ):
            self.assert_error_matches(actual_call, expected_call)

        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor([1.0])
            with self.assertRaises(TypeError):
                descriptor(actual if descriptor is actual_descriptor else expected, 1)


if __name__ == "__main__":
    unittest.main()
