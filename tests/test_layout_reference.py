import copy
import copyreg
import inspect
import pickle
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorLayoutReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("layout differentials require pinned PyTorch 2.13.0")

    def normalize(self, module, value):
        return value.replace(module.layout.__module__, "torch")

    def error(self, module, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, self.normalize(module, str(error))
        self.fail(f"{module.__name__} unexpectedly accepted the operation")

    def layout_contract(self, module):
        layout = module.strided
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        others = (
            None,
            0,
            "torch.strided",
            module.float32,
            module.contiguous_format,
            object(),
        )
        constructor_errors = tuple(
            self.error(module, lambda arguments=arguments: module.layout(*arguments))
            for arguments in ((), ("strided",), (1, 2))
        )
        assignment_errors = tuple(
            self.error(module, action)
            for action in (
                lambda: setattr(layout, "label", "strided"),
                lambda: delattr(layout, "label"),
                lambda: setattr(layout, "__doc__", "layout"),
                lambda: delattr(layout, "__doc__"),
            )
        )
        return {
            "type_identity": type(layout) is module.layout,
            "isinstance": isinstance(layout, module.layout),
            "type_module": self.normalize(module, module.layout.__module__),
            "type_name": module.layout.__name__,
            "type_qualname": module.layout.__qualname__,
            "type_repr": self.normalize(module, repr(module.layout)),
            "type_doc": module.layout.__doc__,
            "metatype_is_type": type(module.layout) is type,
            "object_base": module.layout.__bases__ == (object,),
            "immutable_flag": bool(module.layout.__flags__ & (1 << 8)),
            "repr_descriptor_type": type(
                inspect.getattr_static(module.layout, "__repr__")
            ).__name__,
            "repr_descriptor_owner": (
                inspect.getattr_static(module.layout, "__repr__").__objclass__
                is module.layout
            ),
            "repr": repr(layout),
            "str": str(layout),
            "self_identity": layout is module.strided,
            "self_equality": (layout == layout, layout != layout),
            "direct_self_equality": (
                module.layout.__eq__(layout, layout),
                module.layout.__ne__(layout, layout),
            ),
            "other_equality": tuple((layout == other, layout != other) for other in others),
            "direct_other_equality": tuple(
                (
                    module.layout.__eq__(layout, other) is NotImplemented,
                    module.layout.__ne__(layout, other) is NotImplemented,
                )
                for other in others
            ),
            "hash_type": type(hash(layout)).__name__,
            "hash_is_stable": hash(layout) == hash(module.strided),
            "identity_hash": hash(layout) == object.__hash__(layout),
            "dict_lookup": {layout: "strided"}[module.strided],
            "constructor_errors": constructor_errors,
            "assignment_errors": assignment_errors,
            "layout_in_all": "layout" in module.__all__,
            "strided_in_all": "strided" in module.__all__,
            "layout_in_wildcard": "layout" in wildcard_namespace,
            "strided_in_wildcard": "strided" in wildcard_namespace,
        }

    def test_type_representation_equality_and_hash_match_pytorch_2_13(self):
        self.assertEqual(
            self.layout_contract(torch),
            self.layout_contract(reference_torch),
        )

    def copy_and_pickle_contract(self, module):
        layout = module.strided
        reducer = copyreg.dispatch_table.get(type(layout))
        if reducer is None:
            return {"registered_reducer": False}
        constructor, arguments = reducer(layout)
        return {
            "registered_reducer": True,
            "reducer_arguments": arguments,
            "reducer_identity": constructor(*arguments) is layout,
            "copy_identity": copy.copy(layout) is layout,
            "deepcopy_identity": copy.deepcopy(layout) is layout,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(layout, protocol=protocol)) is layout
                for protocol in range(6)
            ),
        }

    def test_copy_and_pickle_identity_match_pytorch_2_13(self):
        self.assertEqual(
            self.copy_and_pickle_contract(torch),
            self.copy_and_pickle_contract(reference_torch),
        )

    def class_mutation_contract(self, module):
        mutations = (
            (
                "__repr__",
                lambda: setattr(module.layout, "__repr__", lambda _: "broken"),
            ),
            ("__repr__", lambda: delattr(module.layout, "__repr__")),
            (
                "__eq__",
                lambda: setattr(module.layout, "__eq__", lambda *_: True),
            ),
            ("__eq__", lambda: delattr(module.layout, "__eq__")),
            (
                "__hash__",
                lambda: setattr(module.layout, "__hash__", lambda _: 0),
            ),
            ("__hash__", lambda: delattr(module.layout, "__hash__")),
            ("marker", lambda: setattr(module.layout, "marker", object())),
            ("marker", lambda: delattr(module.layout, "marker")),
            (
                "marker",
                lambda: type.__setattr__(module.layout, "marker", object()),
            ),
            ("marker", lambda: type.__delattr__(module.layout, "marker")),
        )
        initial_hash = hash(module.strided)
        errors = tuple(self.error(module, mutation) for _, mutation in mutations)
        return {
            "errors": errors,
            "repr": repr(module.strided),
            "hash_unchanged": hash(module.strided) == initial_hash,
            "self_equality": module.strided == module.strided,
            "other_equality": module.strided == object(),
            "marker_absent": not hasattr(module.layout, "marker"),
        }

    def test_layout_type_immutability_matches_pytorch_2_13(self):
        self.assertEqual(
            self.class_mutation_contract(torch),
            self.class_mutation_contract(reference_torch),
        )

    def tensor_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        offset_view = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        tensors = (
            module.tensor(-0.0),
            module.zeros((2, 0, 3)),
            offset_view,
            module.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
            module.zeros((sys.maxsize, 0))[sys.maxsize - 1],
            leaf,
            tracked,
            tracked.detach(),
        )

        def metadata(tensor):
            return (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
            )

        before = tuple(metadata(tensor) for tensor in tensors)
        identities = tuple(tensor.layout is module.strided for tensor in tensors)
        repeated_identities = tuple(
            tensor.layout is tensors[0].layout for tensor in tensors
        )
        after = tuple(metadata(tensor) for tensor in tensors)
        grad_before = leaf.grad is None
        tracked.sum().backward()
        return {
            "metadata_unchanged": before == after,
            "metadata": before,
            "canonical_identities": identities,
            "shared_identities": repeated_identities,
            "grad_was_absent": grad_before,
            "leaf_gradient": leaf.grad.tolist(),
            "gradient_layout": leaf.grad.layout is module.strided,
        }

    def test_scalar_empty_views_and_autograd_match_without_side_effects(self):
        self.assertEqual(
            self.tensor_contract(torch),
            self.tensor_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "layout")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "layout", module.strided),
            lambda: delattr(tensor, "layout"),
            lambda: descriptor.__set__(tensor, module.strided),
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
            "class_identity": module.Tensor.layout is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "value_identity": descriptor.__get__(tensor, module.Tensor) is module.strided,
            "assignment_errors": tuple(
                self.error(module, action) for action in actions
            ),
            "receiver_error": self.error(
                module, lambda: descriptor.__get__(1, int)
            ),
        }

    def test_descriptor_ownership_and_assignment_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
