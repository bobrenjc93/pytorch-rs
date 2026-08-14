import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorItemsizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.itemsize differentials require pinned PyTorch 2.13.0"
            )

    def make_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        base = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        return (
            module.tensor(-0.0),
            module.zeros((2, 0, 3)),
            base[1],
            base.transpose(0, 1),
            leaf,
            tracked,
            leaf.grad,
        )

    def itemsize_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        itemsize = tensor.itemsize
        result = {
            "metadata": metadata,
            "itemsize_type": type(itemsize).__name__,
            "itemsize": itemsize,
            "dtype_itemsize": tensor.dtype.itemsize,
            "element_size": tensor.element_size(),
            "nbytes": tensor.nbytes,
            "numel_times_itemsize": tensor.numel() * itemsize,
        }
        result["metadata_unchanged"] = metadata == (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        return result

    def test_scalar_empty_offset_strided_and_autograd_tensors_match(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)

        self.assertGreater(actual_cases[2].storage_offset(), 0)
        self.assertGreater(expected_cases[2].storage_offset(), 0)
        self.assertFalse(actual_cases[3].is_contiguous())
        self.assertFalse(expected_cases[3].is_contiguous())
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                self.assertEqual(
                    self.itemsize_contract(actual),
                    self.itemsize_contract(expected),
                )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.itemsize unexpectedly accepted mutation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "itemsize")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "itemsize", 8),
            lambda: delattr(tensor, "itemsize"),
            lambda: descriptor.__set__(tensor, 8),
            lambda: descriptor.__delete__(tensor),
        )
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
            "class_identity": module.Tensor.itemsize is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": descriptor.__get__(tensor, module.Tensor),
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_ownership_documentation_and_errors_match(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
