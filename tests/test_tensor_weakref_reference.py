import gc
import unittest
import weakref

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorWeakReferenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("weakref differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_cases(module):
        leaf = module.tensor([1.0, 2.0], requires_grad=True)
        view = leaf.view(2)
        nonleaf = leaf * 2.0
        empty = module.zeros(0)
        nonleaf.sum().backward()
        gradient = leaf.grad
        return (
            ("leaf", leaf),
            ("view", view),
            ("nonleaf", nonleaf),
            ("empty", empty),
            ("gradient", gradient),
        )

    @staticmethod
    def tensor_values(tensor, module):
        if module is reference_torch:
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor)

    def metadata(self, module):
        descriptor = module.Tensor.__dict__["__weakref__"]
        tensor = module.tensor([1.0])
        initial = tensor.__weakref__
        reference = weakref.ref(tensor)
        return {
            "descriptor_type": type(descriptor).__name__,
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "descriptor_doc": descriptor.__doc__,
            "descriptor_owner": descriptor.__objclass__ is module.Tensor,
            "tensor_offset_nonzero": module.Tensor.__weakrefoffset__ != 0,
            "base_offset": module.Tensor.__base__.__weakrefoffset__,
            "base_has_descriptor": "__weakref__" in module.Tensor.__base__.__dict__,
            "initial": initial,
            "after_is_reference": tensor.__weakref__ is reference,
        }

    def behavior(self, module):
        outcomes = []
        for name, tensor in self.tensor_cases(module):
            reference = weakref.ref(tensor)
            proxy = weakref.proxy(tensor)
            with self.assertRaises(TypeError) as raised:
                hash(proxy)
            outcomes.append(
                (
                    name,
                    reference() is tensor,
                    reference is weakref.ref(tensor),
                    proxy is weakref.proxy(tensor),
                    hash(tensor) == id(tensor),
                    hash(reference) == hash(tensor),
                    tensor.__weakref__ is reference,
                    proxy.__class__ is module.Tensor,
                    tuple(proxy.shape),
                    proxy.requires_grad,
                    proxy.is_leaf,
                    proxy.data_ptr() == tensor.data_ptr(),
                    str(proxy) == str(tensor),
                    self.tensor_values(proxy + 1.0, module).tolist(),
                    (type(raised.exception).__name__, str(raised.exception)),
                )
            )
        return outcomes

    @staticmethod
    def lifecycle(module):
        summaries = []
        for name, tensor in TensorWeakReferenceReferenceTests.tensor_cases(module):
            callbacks = []
            reference = weakref.ref(
                tensor,
                lambda dead, kind=name: callbacks.append(
                    (kind, "reference", dead())
                ),
            )
            proxy = weakref.proxy(
                tensor,
                lambda dead, kind=name: callbacks.append(
                    (kind, "proxy", type(dead).__name__)
                ),
            )
            tensor_hash = hash(tensor)
            reference_hash = hash(reference)
            del tensor
            for _ in range(3):
                gc.collect()

            try:
                proxy.shape
            except Exception as error:
                proxy_error = (type(error).__name__, str(error))
            else:
                proxy_error = None
            first_callback_count = len(callbacks)
            for _ in range(3):
                gc.collect()
            summaries.append(
                (
                    name,
                    reference() is None,
                    tensor_hash == reference_hash == hash(reference),
                    sorted(callbacks, key=lambda item: item[1]),
                    first_callback_count,
                    len(callbacks),
                    proxy_error,
                )
            )
        return summaries

    def test_weakref_metadata_matches_pytorch_2_13(self):
        self.assertEqual(self.metadata(torch), self.metadata(reference_torch))

    def test_reference_hash_and_proxy_behavior_match_pytorch_2_13(self):
        self.assertEqual(self.behavior(torch), self.behavior(reference_torch))

    def test_collection_lifecycle_matches_pytorch_2_13(self):
        self.assertEqual(self.lifecycle(torch), self.lifecycle(reference_torch))


if __name__ == "__main__":
    unittest.main()
