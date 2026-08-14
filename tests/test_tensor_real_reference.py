import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorRealReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.real differentials require pinned PyTorch 2.13.0"
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
        scalar_storage = module.tensor(memoryview(scalar_bits.view(np.float32)))
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        return (
            *(scalar_storage[index] for index in range(len(scalar_bits))),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            leaf,
            (leaf * 3.0).transpose(0, 1)[1],
        )

    def identity_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        pointer = tensor.data_ptr()
        result = tensor.real
        return {
            "identity": result is tensor,
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
                str(result.dtype),
                str(result.device),
                result.requires_grad,
                result.is_leaf,
            ),
            "pointer_unchanged": result.data_ptr() == pointer,
            "bits": np.asarray(result.detach()).reshape(-1).view(np.uint32).copy(),
        }

    def test_scalar_empty_offset_strided_leaf_and_non_leaf_tensors_match(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_contract = self.identity_contract(actual)
                expected_contract = self.identity_contract(expected)
                np.testing.assert_array_equal(
                    actual_contract.pop("bits"), expected_contract.pop("bits")
                )
                self.assertEqual(actual_contract, expected_contract)

    def test_leaf_and_non_leaf_autograd_identity_matches(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            leaf_result = leaf.real
            non_leaf = (leaf_result * 3.0).transpose(0, 1)[1]
            graph_before = (
                non_leaf.requires_grad,
                non_leaf.is_leaf,
                tuple(non_leaf.shape),
                non_leaf.stride(),
                non_leaf.storage_offset(),
            )
            pointer = non_leaf.data_ptr()
            result = non_leaf.real
            graph_after = (
                result.requires_grad,
                result.is_leaf,
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
            )
            result.sum().backward()
            gradient = leaf.grad
            outcomes.append(
                (
                    leaf_result is leaf,
                    result is non_leaf,
                    result.data_ptr() == pointer,
                    graph_before,
                    graph_after,
                    leaf.real is leaf,
                    leaf.grad is gradient,
                    np.asarray(gradient).copy(),
                )
            )

        self.assertEqual(outcomes[0][:-1], outcomes[1][:-1])
        np.testing.assert_array_equal(outcomes[0][-1], outcomes[1][-1])

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.real unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "real")
        tensor = module.tensor([1.0], dtype=module.float32)
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
            "class_identity": module.Tensor.real is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_identity": descriptor.__get__(tensor, module.Tensor) is tensor,
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_ownership_documentation_and_identity_match(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
