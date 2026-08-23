import gc
import inspect
import sys
import types
import unittest
import weakref

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def collect_garbage():
    for _ in range(3):
        gc.collect()


def tensor_factories(module):
    def leaf():
        return module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )

    def view():
        return module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        ).transpose(0, 1)[1]

    def nonleaf():
        return (
            module.tensor(
                [2.0, 3.0], dtype=module.float32, requires_grad=True
            )
            * 3.0
        )

    def empty():
        return module.zeros(
            (0, 2), dtype=module.float32, requires_grad=True
        )

    def gradient():
        source = module.tensor(
            [2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        (source * 4.0).sum().backward()
        return source.grad

    return (
        ("leaf", leaf),
        ("view", view),
        ("non-leaf", nonleaf),
        ("empty", empty),
        ("gradient", gradient),
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorWeakReferenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor weak-reference differentials require pinned PyTorch 2.13.0"
            )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("weak-reference operation unexpectedly succeeded")

    def normalize_tensor_name(self, module, value):
        return value.replace(f"{module.Tensor.__module__}.Tensor", "Tensor")

    def metadata_contract(self, module):
        tensor_type = module.Tensor
        tensor_base = tensor_type.__base__
        descriptor = inspect.getattr_static(tensor_type, "__weakref__")
        tensor = module.tensor([1.0], dtype=module.float32)
        read_only_errors = []
        for action in (
            lambda: setattr(tensor, "__weakref__", None),
            lambda: delattr(tensor, "__weakref__"),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        ):
            error_type, message = self.error(action)
            read_only_errors.append(
                (error_type, self.normalize_tensor_name(module, message))
            )
        receiver_error = self.error(lambda: descriptor.__get__(1, int))

        return {
            "tensor_declares_weakref": "__weakref__" in tensor_type.__dict__,
            "implementation_slot_is_hidden": "_torch_rs_weakref_slot"
            not in tensor_type.__dict__,
            "base_declares_weakref": "__weakref__" in tensor_base.__dict__,
            "tensor_has_weakref_offset": tensor_type.__weakrefoffset__ != 0,
            "base_weakref_offset": tensor_base.__weakrefoffset__,
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_is_tensor": descriptor.__objclass__ is tensor_type,
            "has_module": hasattr(descriptor, "__module__"),
            "has_text_signature": hasattr(descriptor, "__text_signature__"),
            "repr": self.normalize_tensor_name(module, repr(descriptor)),
            "class_identity": tensor_type.__weakref__ is descriptor,
            "class_get_identity": descriptor.__get__(None, tensor_type)
            is descriptor,
            "initial_value": descriptor.__get__(tensor, tensor_type),
            "read_only_errors": tuple(read_only_errors),
            "receiver_error": (
                receiver_error[0],
                self.normalize_tensor_name(module, receiver_error[1]),
            ),
            "hash_is_function": type(tensor_type.__hash__) is types.FunctionType,
            "hash_name": tensor_type.__hash__.__name__,
            "hash_qualname": tensor_type.__hash__.__qualname__,
            "hash_module_suffix": tensor_type.__hash__.__module__.split(".")[-1],
            "hash_signature": str(inspect.signature(tensor_type.__hash__)),
            "hash_matches_id": tensor_type.__hash__(tensor) == id(tensor),
        }

    def live_contract(self, module, factory):
        tensor = factory()
        initial_weakref_is_none = tensor.__weakref__ is None
        strong_references = sys.getrefcount(tensor)
        reference = weakref.ref(tensor)
        proxy = weakref.proxy(tensor)
        weakrefs = weakref.getweakrefs(tensor)

        return {
            "initial_weakref_is_none": initial_weakref_is_none,
            "does_not_add_strong_references": sys.getrefcount(tensor)
            == strong_references,
            "referent_identity": reference() is tensor,
            "reference_is_canonical": weakref.ref(tensor) is reference,
            "proxy_is_canonical": weakref.proxy(tensor) is proxy,
            "instance_weakref_identity": tensor.__weakref__ is reference,
            "weakref_count": weakref.getweakrefcount(tensor),
            "weakref_types": tuple(type(value).__name__ for value in weakrefs),
            "tensor_hash_is_id": hash(tensor) == id(tensor),
            "reference_hash_matches": hash(reference) == hash(tensor),
            "object_hash_differs": object.__hash__(tensor) != hash(tensor),
            "proxy_hash_error": self.error(lambda: hash(proxy)),
            "proxy_type": type(proxy).__name__,
            "proxy_class_is_tensor": proxy.__class__ is module.Tensor,
            "proxy_isinstance": isinstance(proxy, module.Tensor),
            "shape": tuple(proxy.shape),
            "stride": proxy.stride(),
            "storage_offset": proxy.storage_offset(),
            "same_pointer": proxy.data_ptr() == tensor.data_ptr(),
            "dtype": str(proxy.dtype),
            "device": str(proxy.device),
            "requires_grad": proxy.requires_grad,
            "is_leaf": proxy.is_leaf,
            "values": proxy.tolist(),
            "scaled_values": (proxy * 2.0).tolist(),
        }

    def slot_lookup_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        original_getweakrefs = weakref.getweakrefs
        weakref.getweakrefs = lambda _: ["spoofed"]
        try:
            initial_value = tensor.__weakref__
            reference = weakref.ref(tensor)
            live_value_is_reference = tensor.__weakref__ is reference
        finally:
            weakref.getweakrefs = original_getweakrefs
        return initial_value, live_value_is_reference

    def collection_contract(self, module, factory):
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
        does_not_add_strong_references = (
            sys.getrefcount(tensor) == strong_references
        )

        del tensor
        collected_before_gc = reference() is None
        collect_garbage()
        events_after_collection = tuple(events)
        collect_garbage()

        unhashed_tensor = factory()
        unhashed_reference = weakref.ref(unhashed_tensor)
        del unhashed_tensor
        collect_garbage()

        return {
            "does_not_add_strong_references": does_not_add_strong_references,
            "collected_before_gc": collected_before_gc,
            "referent_is_none": reference() is None,
            "callback_events": events_after_collection,
            "callbacks_are_exactly_once": tuple(events) == events_after_collection,
            "cached_hash_survives": hash(reference) == cached_hash,
            "dead_proxy_error": self.error(lambda: proxy.tolist()),
            "unhashed_dead_reference_error": self.error(
                lambda: hash(unhashed_reference)
            ),
        }

    def resource_lifetime_contract(self, module):
        source = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        alias = source.data
        source_reference = weakref.ref(source)
        source_proxy = weakref.proxy(source)
        pointer = source_proxy.data_ptr()
        del source
        collect_garbage()
        storage_result = (
            source_reference() is None,
            alias.data_ptr() == pointer,
            alias.tolist(),
            self.error(lambda: source_proxy.tolist()),
        )
        alias_reference = weakref.ref(alias)
        del alias
        collect_garbage()

        untracked_source = module.tensor(
            [3.0], dtype=module.float32, requires_grad=True
        )
        untracked_reference = weakref.ref(untracked_source)
        with module.no_grad():
            untracked_result = untracked_source * 2.0
        del untracked_source
        collect_garbage()
        untracked_result_contract = (
            untracked_reference() is None,
            untracked_result.requires_grad,
            untracked_result.tolist(),
        )

        leaf = module.tensor(
            [2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        leaf_events = []
        leaf_reference = weakref.ref(
            leaf, lambda _: leaf_events.append("leaf collected")
        )
        leaf_proxy = weakref.proxy(leaf)
        nonleaf = leaf * 4.0
        nonleaf_reference = weakref.ref(nonleaf)
        loss = nonleaf.sum()
        del leaf, nonleaf
        collect_garbage()
        graph_retention = (
            leaf_reference() is not None,
            nonleaf_reference() is None,
            tuple(leaf_events),
        )

        loss.backward()
        gradient = leaf_proxy.grad
        gradient_reference = weakref.ref(gradient)
        gradient_result = (gradient is leaf_proxy.grad, gradient.tolist())

        del gradient, loss
        collect_garbage()
        graph_result = (
            leaf_reference() is None,
            nonleaf_reference() is None,
            gradient_reference() is None,
            tuple(leaf_events),
            self.error(lambda: leaf_proxy.grad),
        )
        collect_garbage()
        callbacks_are_exactly_once = tuple(leaf_events) == graph_result[3]

        view_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
        )
        view_events = []
        view_source_reference = weakref.ref(
            view_source, lambda _: view_events.append("source collected")
        )
        view_source_proxy = weakref.proxy(view_source)
        view = view_source.transpose(0, 1)
        view_reference = weakref.ref(view)
        nested_view = view[1]
        del view_source
        collect_garbage()
        view_root_retention = (
            view_source_reference() is not None,
            view_source_proxy.tolist(),
            tuple(view_events),
        )
        del view
        collect_garbage()
        nested_view_retention = (
            view_reference() is None,
            view_source_reference() is not None,
            nested_view.tolist(),
        )
        del nested_view
        collect_garbage()
        view_collection = (
            view_source_reference() is None,
            tuple(view_events),
            self.error(lambda: view_source_proxy.tolist()),
        )
        collect_garbage()
        view_callbacks_are_exactly_once = tuple(view_events) == view_collection[1]

        return (
            storage_result,
            alias_reference() is None,
            untracked_result_contract,
            graph_retention,
            gradient_result,
            graph_result,
            callbacks_are_exactly_once,
            view_root_retention,
            nested_view_retention,
            view_collection,
            view_callbacks_are_exactly_once,
        )

    def test_descriptor_and_hash_metadata_match_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_contract(torch),
            self.metadata_contract(reference_torch),
        )

    def test_live_reference_and_proxy_semantics_match_pytorch_2_13(self):
        actual_factories = tensor_factories(torch)
        expected_factories = tensor_factories(reference_torch)
        for (actual_case, actual_factory), (expected_case, expected_factory) in zip(
            actual_factories, expected_factories, strict=True
        ):
            self.assertEqual(actual_case, expected_case)
            with self.subTest(case=actual_case):
                self.assertEqual(
                    self.live_contract(torch, actual_factory),
                    self.live_contract(reference_torch, expected_factory),
                )

    def test_slot_lookup_ignores_module_monkeypatches_like_pytorch_2_13(self):
        self.assertEqual(
            self.slot_lookup_contract(torch),
            self.slot_lookup_contract(reference_torch),
        )

    def test_collection_semantics_match_pytorch_2_13(self):
        actual_factories = tensor_factories(torch)
        expected_factories = tensor_factories(reference_torch)
        for (actual_case, actual_factory), (expected_case, expected_factory) in zip(
            actual_factories, expected_factories, strict=True
        ):
            self.assertEqual(actual_case, expected_case)
            with self.subTest(case=actual_case):
                self.assertEqual(
                    self.collection_contract(torch, actual_factory),
                    self.collection_contract(reference_torch, expected_factory),
                )

    def test_storage_and_autograd_lifetimes_match_pytorch_2_13(self):
        self.assertEqual(
            self.resource_lifetime_contract(torch),
            self.resource_lifetime_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
