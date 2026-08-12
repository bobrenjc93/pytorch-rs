import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsLeafReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_leaf differentials require pinned PyTorch 2.13.0")

    def outcomes(self, module):
        ordinary = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked_operation = leaf * 2.0
        tracked_view = leaf.transpose(0, 1)

        tensors = [
            ordinary,
            ordinary + 1.0,
            ordinary.transpose(0, 1),
            leaf,
            tracked_operation,
            tracked_operation.sum(),
            -leaf,
            leaf.sin(),
            tracked_view,
            leaf.reshape(4),
            leaf[0],
            tracked_operation.detach(),
            tracked_view.detach(),
        ]
        with module.no_grad():
            no_grad_views = [
                leaf.transpose(0, 1),
                tracked_operation.transpose(0, 1),
                leaf.reshape(4),
                leaf[0],
            ]
            tensors.extend((leaf + 1.0, -leaf, leaf.sin(), *no_grad_views))

        tensors.extend(view + 1.0 for view in no_grad_views)
        tracked_operation.sum().backward()
        tensors.append(leaf.grad)
        return [(tensor.requires_grad, tensor.is_leaf) for tensor in tensors]

    def test_leaf_status_matches_pytorch_2_13(self):
        self.assertEqual(self.outcomes(torch), self.outcomes(reference_torch))

    def test_read_only_descriptor_matches_pytorch_2_13(self):
        descriptors = (
            inspect.getattr_static(torch.Tensor, "is_leaf"),
            inspect.getattr_static(reference_torch.Tensor, "is_leaf"),
        )
        for descriptor in descriptors:
            self.assertIs(type(descriptor), types.GetSetDescriptorType)
            self.assertEqual(descriptor.__name__, "is_leaf")
            self.assertFalse(callable(descriptor))

        errors = []
        for module in (torch, reference_torch):
            tensor = module.tensor([1.0])
            module_errors = []
            for action in ("set", "delete"):
                try:
                    if action == "set":
                        tensor.is_leaf = False
                    else:
                        del tensor.is_leaf
                except Exception as error:
                    module_errors.append((type(error).__name__, str(error)))
                else:
                    self.fail(f"{module.__name__} allowed is_leaf to be {action}")
            errors.append(module_errors)

        for actual, expected in zip(errors[0], errors[1], strict=True):
            self.assertEqual(actual[0], expected[0])
            for _, message in (actual, expected):
                self.assertIn("attribute 'is_leaf'", message)
                self.assertTrue(message.endswith("objects is not writable"))


if __name__ == "__main__":
    unittest.main()
