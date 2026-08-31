import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorDivReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.div differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_values = np.asarray(actual).reshape(-1)
            expected_values = expected.detach().cpu().numpy().reshape(-1)
            actual_bits = actual_values.view(np.uint32)
            expected_bits = expected_values.view(np.uint32)
            nan_mask = np.isnan(expected_values)
            np.testing.assert_array_equal(np.isnan(actual_values), nan_mask)
            np.testing.assert_array_equal(actual_bits[~nan_mask], expected_bits[~nan_mask])

    def test_values_layouts_ieee_empties_scalars_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [4.0], [0.5]])
        expected_right = reference_torch.tensor([[2.0], [4.0], [0.5]])

        for name in ("div", "divide"):
            actual_method = getattr(actual_left, name)
            expected_method = getattr(expected_left, name)
            calls = (
                (
                    "positional tensors",
                    lambda: actual_method(actual_right),
                    lambda: expected_method(expected_right),
                ),
                (
                    "other keyword",
                    lambda: actual_method(other=actual_right),
                    lambda: expected_method(other=expected_right),
                ),
                (
                    "x2 keyword",
                    lambda: actual_method(x2=actual_right),
                    lambda: expected_method(x2=expected_right),
                ),
                (
                    "rounding none",
                    lambda: actual_method(actual_right, rounding_mode=None),
                    lambda: expected_method(expected_right, rounding_mode=None),
                ),
            )
            for case, actual_call, expected_call in calls:
                self.assert_matches(actual_call(), expected_call(), case=(name, case))

        actual_offset = actual_left[1]
        expected_offset = expected_left[1]
        for scalar in (
            True,
            False,
            -2,
            2.5,
            np.bool_(False),
            np.int64(3),
            np.float32(-0.0),
        ):
            for name in ("div", "divide"):
                self.assert_matches(
                    getattr(actual_offset, name)(scalar),
                    getattr(expected_offset, name)(scalar),
                    case=(name, "offset scalar", type(scalar).__name__, scalar),
                )

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        for name in ("div", "divide"):
            self.assert_matches(
                getattr(actual_empty, name)(torch.ones((1, 1, 2))),
                getattr(expected_empty, name)(reference_torch.ones((1, 1, 2))),
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (
                0x7FC1_2345,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0x8000_0000,
            ),
            dtype=np.uint32,
        )
        denominator_bits = np.asarray(
            (
                0x3F80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x0000_0000,
                0x0000_0000,
                0x4000_0000,
                0x4000_0000,
                0xC000_0000,
                0xC000_0000,
            ),
            dtype=np.uint32,
        )
        actual_numerator = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected_numerator = reference_torch.tensor(memoryview(special_bits.view(np.float32)))
        actual_denominator = torch.tensor(memoryview(denominator_bits.view(np.float32)))
        expected_denominator = reference_torch.tensor(
            memoryview(denominator_bits.view(np.float32))
        )
        for name in ("div", "divide"):
            self.assert_matches(
                getattr(actual_numerator, name)(actual_denominator),
                getattr(expected_numerator, name)(expected_denominator),
                case=(name, "signed zero nan infinity"),
            )

        for name in ("div", "divide"):
            actual_grad = torch.tensor([[2.0, 4.0]], requires_grad=True)
            expected_grad = reference_torch.tensor([[2.0, 4.0]], requires_grad=True)
            with torch.no_grad():
                actual_untracked = getattr(actual_grad.transpose(0, 1), name)(2.0)
            with reference_torch.no_grad():
                expected_untracked = getattr(expected_grad.transpose(0, 1), name)(2.0)
            self.assert_matches(
                actual_untracked, expected_untracked, case=(name, "no_grad")
            )

    @staticmethod
    def callable_contract(module, method_name):
        descriptor = inspect.getattr_static(module.Tensor, method_name)
        tensor = module.tensor([1.0])
        bound = getattr(tensor, method_name)
        try:
            inspect.signature(descriptor)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "descriptor_type": type(descriptor).__name__,
            "descriptor_is_method_descriptor": type(descriptor)
            is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "bound_is_builtin_method": type(bound) is types.BuiltinMethodType,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "descriptor_doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signature_error": signature_error,
            "objclass_name": descriptor.__objclass__.__name__,
            "objclass_module": descriptor.__objclass__.__module__,
            "has_descriptor_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_copy_identity": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy_identity": copy.deepcopy(descriptor) is descriptor,
            "bound_copy_identity": copy.copy(bound) is bound,
            "bound_deepcopy_identity": copy.deepcopy(bound) is bound,
            "descriptor_pickle_identity": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_copy_and_pickle_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

        descriptors = {
            name: inspect.getattr_static(torch.Tensor, name)
            for name in ("div", "divide")
        }
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        for name, descriptor in descriptors.items():
            self.assertIs(inspect.getattr_static(torch.Tensor, name), descriptor)

    def test_common_binding_and_scalar_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_alias = torch.tensor([3.0])
        expected_alias = reference_torch.tensor([3.0])
        for name in ("div", "divide"):
            actual_method = getattr(actual, name)
            expected_method = getattr(expected, name)
            cases = (
                (lambda: actual_method(), lambda: expected_method()),
                (
                    lambda: actual_method(actual, actual),
                    lambda: expected_method(expected, expected),
                ),
                (
                    lambda: actual_method(actual, other=actual),
                    lambda: expected_method(expected, other=expected),
                ),
                (
                    lambda: actual_method(actual, out=actual),
                    lambda: expected_method(expected, out=expected),
                ),
                (
                    lambda: actual_method(actual, dtype=torch.float32),
                    lambda: expected_method(expected, dtype=reference_torch.float32),
                ),
                (
                    lambda: actual_method(x2=actual_alias, other=actual),
                    lambda: expected_method(x2=expected_alias, other=expected),
                ),
                (
                    lambda: actual_method(other=actual, x2=actual_alias),
                    lambda: expected_method(other=expected, x2=expected_alias),
                ),
                (
                    lambda: actual_method(np.uint64(2**63)),
                    lambda: expected_method(np.uint64(2**63)),
                ),
                (lambda: actual_method(2**64), lambda: expected_method(2**64)),
                (
                    lambda: actual_method(-(2**63) - 1),
                    lambda: expected_method(-(2**63) - 1),
                ),
            )
            if name == "div":
                cases = (
                    *cases,
                    (lambda: actual_method([]), lambda: expected_method([])),
                    (lambda: actual_method(None), lambda: expected_method(None)),
                    (
                        lambda: actual_method(dtype=torch.float32),
                        lambda: expected_method(dtype=reference_torch.float32),
                    ),
                )
            for index, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(name=name, case=index):
                    self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
