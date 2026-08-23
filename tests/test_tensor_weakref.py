import gc
import unittest
import weakref

import numpy as np
import torch_rs as torch


class TensorWeakReferenceTests(unittest.TestCase):
    @staticmethod
    def tensor_factories():
        def leaf():
            return torch.tensor([1.0, 2.0], requires_grad=True)

        def view():
            return torch.tensor([1.0, 2.0], requires_grad=True).view(2)

        def nonleaf():
            return torch.tensor([1.0, 2.0], requires_grad=True) * 2.0

        def empty():
            return torch.zeros(0)

        def gradient():
            source = torch.tensor([1.0, 2.0], requires_grad=True)
            (source * source).sum().backward()
            return source.grad

        return (
            ("leaf", leaf),
            ("view", view),
            ("nonleaf", nonleaf),
            ("empty", empty),
            ("gradient", gradient),
        )

    def test_tensor_exposes_weakref_slot_metadata(self):
        descriptor = torch.Tensor.__dict__["__weakref__"]
        self.assertEqual(type(descriptor).__name__, "getset_descriptor")
        self.assertEqual(descriptor.__name__, "__weakref__")
        self.assertEqual(descriptor.__qualname__, "Tensor.__weakref__")
        self.assertEqual(descriptor.__doc__, "list of weak references to the object")
        self.assertIs(descriptor.__objclass__, torch.Tensor)
        self.assertNotEqual(torch.Tensor.__weakrefoffset__, 0)

        tensor_base = torch.Tensor.__base__
        self.assertEqual(tensor_base.__weakrefoffset__, 0)
        self.assertNotIn("__weakref__", tensor_base.__dict__)

        tensor = torch.tensor([1.0])
        self.assertIsNone(tensor.__weakref__)
        reference = weakref.ref(tensor)
        self.assertIs(tensor.__weakref__, reference)

    def test_ref_identity_hashing_and_proxy_forwarding(self):
        for name, factory in self.tensor_factories():
            with self.subTest(kind=name):
                tensor = factory()
                reference = weakref.ref(tensor)
                proxy = weakref.proxy(tensor)

                self.assertIs(reference(), tensor)
                self.assertIs(reference, weakref.ref(tensor))
                self.assertIs(proxy, weakref.proxy(tensor))
                self.assertEqual(hash(tensor), id(tensor))
                self.assertEqual(hash(reference), hash(tensor))
                self.assertIs(tensor.__weakref__, reference)

                self.assertIs(proxy.__class__, torch.Tensor)
                self.assertEqual(proxy.shape, tensor.shape)
                self.assertEqual(proxy.requires_grad, tensor.requires_grad)
                self.assertEqual(proxy.is_leaf, tensor.is_leaf)
                self.assertEqual(proxy.data_ptr(), tensor.data_ptr())
                self.assertEqual(str(proxy), str(tensor))
                np.testing.assert_array_equal(
                    np.asarray(proxy + 1.0),
                    np.asarray(tensor + 1.0),
                )
                with self.assertRaisesRegex(
                    TypeError, "unhashable type: 'weakref.ProxyType'"
                ):
                    hash(proxy)

    def test_callbacks_run_once_and_weak_handles_do_not_retain_wrappers(self):
        for name, factory in self.tensor_factories():
            with self.subTest(kind=name):
                callbacks = []
                tensor = factory()
                tensor_hash = hash(tensor)
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
                reference_hash = hash(reference)

                del tensor
                for _ in range(3):
                    gc.collect()

                self.assertIsNone(reference())
                self.assertEqual(reference_hash, tensor_hash)
                self.assertEqual(hash(reference), reference_hash)
                self.assertCountEqual(
                    callbacks,
                    [
                        (name, "reference", None),
                        (name, "proxy", "ProxyType"),
                    ],
                )
                with self.assertRaisesRegex(
                    ReferenceError, "weakly-referenced object no longer exists"
                ):
                    proxy.shape

                for _ in range(3):
                    gc.collect()
                self.assertEqual(len(callbacks), 2)

    def test_weak_handles_do_not_change_view_or_autograd_semantics(self):
        callbacks = []
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        view = leaf.view(2)
        nonleaf = view * view
        loss = nonleaf.sum()

        wrappers = (leaf, view, nonleaf, loss)
        references = tuple(
            weakref.ref(
                wrapper,
                lambda dead, position=position: callbacks.append(
                    (position, dead())
                ),
            )
            for position, wrapper in enumerate(wrappers)
        )
        proxy = weakref.proxy(view)

        self.assertEqual(proxy.data_ptr(), leaf.data_ptr())
        self.assertEqual(np.asarray(proxy).tolist(), [1.0, 2.0])
        loss.backward()
        gradient = leaf.grad
        gradient_reference = weakref.ref(
            gradient, lambda dead: callbacks.append((4, dead()))
        )
        np.testing.assert_array_equal(np.asarray(gradient), [2.0, 4.0])
        self.assertIs(leaf.grad, gradient)

        del proxy, gradient, loss, nonleaf, view, leaf, wrappers
        for _ in range(3):
            gc.collect()

        self.assertTrue(all(reference() is None for reference in references))
        self.assertIsNone(gradient_reference())
        self.assertCountEqual(callbacks, [(position, None) for position in range(5)])


if __name__ == "__main__":
    unittest.main()
