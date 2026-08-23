import gc
import inspect
import sys
import types
import unittest
import weakref

import torch_rs as torch


def collect_garbage():
    for _ in range(3):
        gc.collect()


def gradient_tensor():
    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
    (leaf * 4.0).sum().backward()
    return leaf.grad


class TensorWeakReferenceTests(unittest.TestCase):
    def tensor_cases(self):
        view_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        nonleaf_source = torch.tensor([2.0, 3.0], requires_grad=True)
        return (
            ("leaf", torch.tensor([1.0, 2.0], requires_grad=True)),
            ("view", view_source.transpose(0, 1)[1]),
            ("non-leaf", nonleaf_source * 3.0),
            ("empty", torch.zeros((0, 2), requires_grad=True)),
            ("gradient", gradient_tensor()),
        )

    def tensor_factories(self):
        def view():
            return torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            ).transpose(0, 1)[1]

        return (
            ("leaf", lambda: torch.tensor([1.0, 2.0], requires_grad=True)),
            ("view", view),
            (
                "non-leaf",
                lambda: torch.tensor([2.0, 3.0], requires_grad=True) * 3.0,
            ),
            ("empty", lambda: torch.zeros((0, 2), requires_grad=True)),
            ("gradient", gradient_tensor),
        )

    def test_reference_identity_hashing_and_proxy_forwarding(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case):
                self.assertIsNone(tensor.__weakref__)
                strong_references = sys.getrefcount(tensor)

                reference = weakref.ref(tensor)
                proxy = weakref.proxy(tensor)

                self.assertEqual(sys.getrefcount(tensor), strong_references)
                self.assertIs(reference(), tensor)
                self.assertIs(weakref.ref(tensor), reference)
                self.assertIs(weakref.proxy(tensor), proxy)
                self.assertIs(tensor.__weakref__, reference)
                self.assertEqual(weakref.getweakrefcount(tensor), 2)
                self.assertEqual(len(weakref.getweakrefs(tensor)), 2)

                self.assertEqual(hash(tensor), id(tensor))
                self.assertEqual(hash(reference), hash(tensor))
                with self.assertRaisesRegex(
                    TypeError, "^unhashable type: 'weakref.ProxyType'$"
                ):
                    hash(proxy)

                self.assertIs(type(proxy), weakref.ProxyType)
                self.assertIs(proxy.__class__, torch.Tensor)
                self.assertIsInstance(proxy, torch.Tensor)
                self.assertEqual(proxy.shape, tensor.shape)
                self.assertEqual(proxy.stride(), tensor.stride())
                self.assertEqual(proxy.storage_offset(), tensor.storage_offset())
                self.assertEqual(proxy.data_ptr(), tensor.data_ptr())
                self.assertEqual(proxy.tolist(), tensor.tolist())
                self.assertEqual(proxy.requires_grad, tensor.requires_grad)
                self.assertEqual(proxy.is_leaf, tensor.is_leaf)
                self.assertEqual((proxy * 2.0).tolist(), (tensor * 2.0).tolist())

    def test_weakref_descriptor_and_hash_metadata(self):
        tensor_type = torch.Tensor
        tensor_base = tensor_type.__base__
        descriptor = inspect.getattr_static(tensor_type, "__weakref__")
        tensor = torch.tensor([1.0])

        self.assertIn("__weakref__", tensor_type.__dict__)
        self.assertNotIn("__weakref__", tensor_base.__dict__)
        self.assertNotEqual(tensor_type.__weakrefoffset__, 0)
        self.assertEqual(tensor_base.__weakrefoffset__, 0)
        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "__weakref__")
        self.assertEqual(descriptor.__qualname__, "Tensor.__weakref__")
        self.assertEqual(descriptor.__doc__, "list of weak references to the object")
        self.assertIs(descriptor.__objclass__, tensor_type)
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertFalse(hasattr(descriptor, "__text_signature__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute '__weakref__' of 'torch_rs.Tensor' objects>",
        )
        self.assertIs(tensor_type.__weakref__, descriptor)
        self.assertIs(descriptor.__get__(None, tensor_type), descriptor)
        self.assertIsNone(descriptor.__get__(tensor, tensor_type))

        for action in (
            lambda: setattr(tensor, "__weakref__", None),
            lambda: delattr(tensor, "__weakref__"),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        ):
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute '__weakref__' of 'torch_rs.Tensor' objects "
                    "is not writable",
                )

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor '__weakref__' for 'torch_rs.Tensor' objects "
            "doesn't apply to a 'int' object",
        )

        self.assertIs(type(tensor_type.__hash__), types.FunctionType)
        self.assertEqual(tensor_type.__hash__.__name__, "__hash__")
        self.assertEqual(tensor_type.__hash__.__qualname__, "Tensor.__hash__")
        self.assertEqual(tensor_type.__hash__.__module__, "torch_rs._tensor")
        self.assertEqual(str(inspect.signature(tensor_type.__hash__)), "(self)")
        self.assertEqual(tensor_type.__hash__(tensor), id(tensor))
        self.assertEqual(hash(tensor), id(tensor))
        self.assertNotEqual(hash(tensor), object.__hash__(tensor))

    def test_callbacks_fire_once_and_dead_references_match_python(self):
        for case, factory in self.tensor_factories():
            with self.subTest(case=case):
                events = []
                tensor = factory()
                strong_references = sys.getrefcount(tensor)
                reference = weakref.ref(
                    tensor,
                    lambda handle: events.append(("ref", type(handle).__name__)),
                )
                proxy = weakref.proxy(
                    tensor,
                    lambda handle: events.append(("proxy", type(handle).__name__)),
                )
                cached_hash = hash(reference)

                self.assertEqual(sys.getrefcount(tensor), strong_references)
                del tensor
                collect_garbage()

                self.assertIsNone(reference())
                self.assertEqual(
                    events,
                    [("proxy", "ProxyType"), ("ref", "ReferenceType")],
                )
                self.assertEqual(hash(reference), cached_hash)
                with self.assertRaisesRegex(
                    ReferenceError, "^weakly-referenced object no longer exists$"
                ):
                    proxy.tolist()

                collect_garbage()
                self.assertEqual(len(events), 2)

                unhashed_tensor = factory()
                unhashed_reference = weakref.ref(unhashed_tensor)
                del unhashed_tensor
                collect_garbage()
                with self.assertRaisesRegex(TypeError, "^weak object has gone away$"):
                    hash(unhashed_reference)

    def test_weak_references_do_not_change_storage_or_autograd_lifetimes(self):
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        alias = source.data
        source_reference = weakref.ref(source)
        source_proxy = weakref.proxy(source)
        pointer = source_proxy.data_ptr()

        del source, source_proxy
        collect_garbage()
        self.assertIsNone(source_reference())
        self.assertEqual(alias.data_ptr(), pointer)
        self.assertEqual(alias.tolist(), [1.0, 2.0])

        alias_reference = weakref.ref(alias)
        del alias
        collect_garbage()
        self.assertIsNone(alias_reference())

        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        nonleaf = leaf * 4.0
        loss = nonleaf.sum()
        references = tuple(weakref.ref(value) for value in (leaf, nonleaf, loss))
        loss_proxy = weakref.proxy(loss)

        loss_proxy.backward()
        gradient = leaf.grad
        gradient_reference = weakref.ref(gradient)
        self.assertEqual(gradient.tolist(), [4.0, 4.0])

        del gradient, loss_proxy, loss, nonleaf, leaf
        collect_garbage()
        self.assertTrue(all(reference() is None for reference in references))
        self.assertIsNone(gradient_reference())


if __name__ == "__main__":
    unittest.main()
