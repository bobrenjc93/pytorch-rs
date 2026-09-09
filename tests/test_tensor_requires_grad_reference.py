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
class TensorRequiresGradInplaceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "requires_grad_ differentials require pinned PyTorch 2.13.0"
            )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        self.fail("Tensor.requires_grad_ unexpectedly accepted the operation")

    def grad_payload(self, grad):
        if grad is None:
            return None
        return {
            "values": grad.tolist(),
            "shape": tuple(grad.shape),
            "stride": grad.stride(),
            "storage_offset": grad.storage_offset(),
            "requires_grad": grad.requires_grad,
            "is_leaf": grad.is_leaf,
        }

    def fresh_leaf_contract(self, module):
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32)
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
        )

        default_result = tensor.requires_grad_()
        after_default = tensor.requires_grad, tensor.is_leaf
        repeated_true_result = tensor.requires_grad_(True)
        after_repeated_true = tensor.requires_grad, tensor.is_leaf
        keyword_false_result = tensor.requires_grad_(requires_grad=False)
        after_keyword_false = tensor.requires_grad, tensor.is_leaf
        positional_true_result = tensor.requires_grad_(True)
        after_positional_true = tensor.requires_grad, tensor.is_leaf

        return {
            "same_objects": (
                default_result is tensor,
                repeated_true_result is tensor,
                keyword_false_result is tensor,
                positional_true_result is tensor,
            ),
            "states": (
                after_default,
                after_repeated_true,
                after_keyword_false,
                after_positional_true,
            ),
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
            ),
            "values": tensor.tolist(),
        }

    def test_fresh_leaf_toggles_match_pytorch_2_13(self):
        self.assertEqual(
            self.fresh_leaf_contract(torch),
            self.fresh_leaf_contract(reference_torch),
        )

    def leaf_with_grad_contract(self, module):
        leaf = module.tensor([2.0, 3.0], requires_grad=True)
        (leaf * leaf).sum().backward()
        first_grad = leaf.grad

        disabled_result = leaf.requires_grad_(False)
        disabled_output = leaf * 3.0
        disabled_output_requires_grad = disabled_output.requires_grad
        reenabled_result = leaf.requires_grad_(True)
        (leaf * 3.0).sum().backward()

        disabled_old_graph_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        disabled_old_graph = (disabled_old_graph_leaf * 2.0).sum()
        disabled_old_graph_leaf.requires_grad_(False)
        disabled_old_graph.backward()

        reenabled_old_graph_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        reenabled_old_graph = (reenabled_old_graph_leaf * 2.0).sum()
        reenabled_old_graph_leaf.requires_grad_(False)
        reenabled_old_graph_leaf.requires_grad_(True)
        reenabled_old_graph.backward()

        return {
            "same_objects": (
                disabled_result is leaf,
                reenabled_result is leaf,
            ),
            "first_grad_reused_after_false": leaf.grad is first_grad,
            "first_grad_reused_after_true": leaf.grad is first_grad,
            "final_grad": self.grad_payload(leaf.grad),
            "disabled_output_requires_grad": disabled_output_requires_grad,
            "disabled_old_graph_grad": self.grad_payload(
                disabled_old_graph_leaf.grad
            ),
            "reenabled_old_graph_grad": self.grad_payload(
                reenabled_old_graph_leaf.grad
            ),
        }

    def test_leaves_with_existing_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.leaf_with_grad_contract(torch),
            self.leaf_with_grad_contract(reference_torch),
        )

    def no_grad_contract(self, module):
        tensor = module.tensor([1.0, 2.0])
        with module.no_grad():
            result = tensor.requires_grad_()
            inside_product = tensor * 2.0
            inside = (
                result is tensor,
                tensor.requires_grad,
                tensor.is_leaf,
                inside_product.requires_grad,
            )
        outside_product = tensor * 2.0
        return {
            "inside": inside,
            "outside_product_requires_grad": outside_product.requires_grad,
            "grad_enabled": module.is_grad_enabled(),
        }

    def test_no_grad_context_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_contract(torch), self.no_grad_contract(reference_torch)
        )

    def views_and_non_leaves_contract(self, module):
        leaf = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        non_leaf = leaf * 2.0
        view = non_leaf.transpose(0, 1)

        non_leaf_true = non_leaf.requires_grad_(True)
        view_true = view.requires_grad_(True)
        non_leaf_false = self.error(lambda: non_leaf.requires_grad_(False))
        view_false = self.error(lambda: view.requires_grad_(False))
        view.sum().backward()

        return {
            "same_objects": (non_leaf_true is non_leaf, view_true is view),
            "states": (
                non_leaf.requires_grad,
                non_leaf.is_leaf,
                view.requires_grad,
                view.is_leaf,
            ),
            "errors": (non_leaf_false, view_false),
            "leaf_grad": self.grad_payload(leaf.grad),
            "non_leaf_grad": self.grad_payload(non_leaf.grad),
            "view_grad": self.grad_payload(view.grad),
        }

    def test_views_and_non_leaves_match_pytorch_2_13(self):
        self.assertEqual(
            self.views_and_non_leaves_contract(torch),
            self.views_and_non_leaves_contract(reference_torch),
        )

    def no_grad_leaf_view_contract(self, module):
        base = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with module.no_grad():
            view = base.transpose(0, 1)

        false_result = view.requires_grad_(False)
        first_loss = (view * 3.0).sum()
        first_loss.backward()

        true_result = view.requires_grad_(True)
        second_loss = (view * 3.0).sum()
        second_loss.backward()

        delayed_base = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            delayed_view = delayed_base.transpose(0, 1)
        delayed_view.requires_grad_(True)
        delayed_view.requires_grad_(False)
        delayed_loss = (delayed_view * 5.0).sum()
        delayed_view.requires_grad_(True)
        delayed_loss.backward()

        return {
            "same_objects": (false_result is view, true_result is view),
            "states": (
                view.requires_grad,
                view.is_leaf,
                first_loss.requires_grad,
                second_loss.requires_grad,
            ),
            "view_grad": self.grad_payload(view.grad),
            "base_grad": self.grad_payload(base.grad),
            "delayed_view_grad": self.grad_payload(delayed_view.grad),
            "delayed_base_grad": self.grad_payload(delayed_base.grad),
        }

    def test_no_grad_leaf_views_match_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_leaf_view_contract(torch),
            self.no_grad_leaf_view_contract(reference_torch),
        )

    def no_grad_leaf_view_base_toggle_contract(self, module):
        base = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with module.no_grad():
            view = base.transpose(0, 1)

        initial = (base.requires_grad, view.requires_grad, view.is_leaf)
        base.requires_grad_(False)
        disabled_loss = (view * 2.0).sum()
        after_disable = (
            base.requires_grad,
            view.requires_grad,
            disabled_loss.requires_grad,
        )
        base.requires_grad_(True)
        inherited_loss = (view * 2.0).sum()
        inherited_loss.backward()
        after_reenable = (
            base.requires_grad,
            view.requires_grad,
            inherited_loss.requires_grad,
            self.grad_payload(base.grad),
            self.grad_payload(view.grad),
        )

        view.requires_grad_(True)
        base.requires_grad_(False)
        promoted_loss = (view * 2.0).sum()
        promoted_loss.backward()
        after_promote = (
            base.requires_grad,
            view.requires_grad,
            promoted_loss.requires_grad,
            self.grad_payload(base.grad),
            self.grad_payload(view.grad),
        )

        initially_false = module.tensor([[3.0, 4.0], [5.0, 6.0]])
        with module.no_grad():
            initially_false_view = initially_false.transpose(0, 1)
        false_initial = (
            initially_false.requires_grad,
            initially_false_view.requires_grad,
            initially_false_view.is_leaf,
        )
        initially_false.requires_grad_(True)
        late_loss = (initially_false_view * 5.0).sum()
        late_loss.backward()
        after_late_enable = (
            initially_false.requires_grad,
            initially_false_view.requires_grad,
            late_loss.requires_grad,
            self.grad_payload(initially_false.grad),
            self.grad_payload(initially_false_view.grad),
        )
        initially_false.requires_grad_(False)
        after_late_disable = (
            initially_false.requires_grad,
            initially_false_view.requires_grad,
            (initially_false_view * 5.0).sum().requires_grad,
        )

        return {
            "initial": initial,
            "after_disable": after_disable,
            "after_reenable": after_reenable,
            "after_promote": after_promote,
            "false_initial": false_initial,
            "after_late_enable": after_late_enable,
            "after_late_disable": after_late_disable,
        }

    def test_no_grad_leaf_views_track_base_toggles_until_promoted(self):
        self.assertEqual(
            self.no_grad_leaf_view_base_toggle_contract(torch),
            self.no_grad_leaf_view_base_toggle_contract(reference_torch),
        )

    def promoted_no_grad_view_preexisting_graph_contract(self, module):
        inherited_enabled_base = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with module.no_grad():
            inherited_enabled_view = inherited_enabled_base.transpose(0, 1)
        inherited_enabled_view.requires_grad_(True)
        inherited_enabled_loss = (inherited_enabled_view * 2.0).sum()
        inherited_enabled_view.requires_grad_(False)
        inherited_enabled_loss.backward()
        inherited_enabled = (
            inherited_enabled_base.requires_grad,
            inherited_enabled_view.requires_grad,
            inherited_enabled_loss.requires_grad,
            self.grad_payload(inherited_enabled_base.grad),
            self.grad_payload(inherited_enabled_view.grad),
        )

        inherited_disabled_base = module.tensor(
            [[5.0, 6.0], [7.0, 8.0]], requires_grad=True
        )
        with module.no_grad():
            inherited_disabled_view = inherited_disabled_base.transpose(0, 1)
        inherited_disabled_view.requires_grad_(True)
        inherited_disabled_loss = (inherited_disabled_view * 3.0).sum()
        inherited_disabled_view.requires_grad_(False)
        inherited_disabled_base.requires_grad_(False)
        inherited_disabled_loss.backward()
        inherited_disabled = (
            inherited_disabled_base.requires_grad,
            inherited_disabled_view.requires_grad,
            inherited_disabled_loss.requires_grad,
            self.grad_payload(inherited_disabled_base.grad),
            self.grad_payload(inherited_disabled_view.grad),
        )

        late_enabled_base = module.tensor([[9.0, 10.0], [11.0, 12.0]])
        with module.no_grad():
            late_enabled_view = late_enabled_base.transpose(0, 1)
        late_enabled_view.requires_grad_(True)
        late_enabled_loss = (late_enabled_view * 4.0).sum()
        late_enabled_view.requires_grad_(False)
        late_enabled_base.requires_grad_(True)
        late_enabled_loss.backward()
        late_enabled = (
            late_enabled_base.requires_grad,
            late_enabled_view.requires_grad,
            late_enabled_loss.requires_grad,
            self.grad_payload(late_enabled_base.grad),
            self.grad_payload(late_enabled_view.grad),
        )

        return {
            "inherited_enabled": inherited_enabled,
            "inherited_disabled": inherited_disabled,
            "late_enabled": late_enabled,
        }

    def test_promoted_no_grad_view_preexisting_graph_matches_pytorch_2_13(self):
        self.assertEqual(
            self.promoted_no_grad_view_preexisting_graph_contract(torch),
            self.promoted_no_grad_view_preexisting_graph_contract(reference_torch),
        )

    def nested_no_grad_leaf_view_contract(self, module):
        base = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with module.no_grad():
            view = base.transpose(0, 1)
        view.requires_grad_(True)
        with module.no_grad():
            child = view.transpose(0, 1)

        true_initial = (
            base.requires_grad,
            view.requires_grad,
            child.requires_grad,
            (child * 2.0).sum().requires_grad,
        )
        view.requires_grad_(False)
        true_loss = (child * 2.0).sum()
        true_loss.backward()
        true_after_view_disable = (
            base.requires_grad,
            view.requires_grad,
            child.requires_grad,
            true_loss.requires_grad,
            self.grad_payload(base.grad),
            self.grad_payload(view.grad),
            self.grad_payload(child.grad),
        )
        base.requires_grad_(False)
        true_after_base_disable = (
            base.requires_grad,
            view.requires_grad,
            child.requires_grad,
            (child * 2.0).sum().requires_grad,
        )

        initially_false = module.tensor([[3.0, 4.0], [5.0, 6.0]])
        with module.no_grad():
            false_view = initially_false.transpose(0, 1)
        false_view.requires_grad_(True)
        with module.no_grad():
            false_child = false_view.transpose(0, 1)

        false_initial = (
            initially_false.requires_grad,
            false_view.requires_grad,
            false_child.requires_grad,
            (false_child * 3.0).sum().requires_grad,
        )
        false_view.requires_grad_(False)
        false_after_view_disable = (
            initially_false.requires_grad,
            false_view.requires_grad,
            false_child.requires_grad,
            (false_child * 3.0).sum().requires_grad,
        )
        initially_false.requires_grad_(True)
        false_late_loss = (false_child * 3.0).sum()
        false_late_loss.backward()
        false_after_base_enable = (
            initially_false.requires_grad,
            false_view.requires_grad,
            false_child.requires_grad,
            false_late_loss.requires_grad,
            self.grad_payload(initially_false.grad),
            self.grad_payload(false_view.grad),
            self.grad_payload(false_child.grad),
        )

        return {
            "true_initial": true_initial,
            "true_after_view_disable": true_after_view_disable,
            "true_after_base_disable": true_after_base_disable,
            "false_initial": false_initial,
            "false_after_view_disable": false_after_view_disable,
            "false_after_base_enable": false_after_base_enable,
        }

    def test_nested_no_grad_leaf_views_match_pytorch_2_13(self):
        self.assertEqual(
            self.nested_no_grad_leaf_view_contract(torch),
            self.nested_no_grad_leaf_view_contract(reference_torch),
        )

    def tracked_view_case(self, base, name):
        if name == "slice":
            return base[:1]
        if name == "reshape":
            return base.reshape(4)
        if name == "unsqueeze":
            return base.unsqueeze(0)
        if name == "chunk":
            return base.chunk(2, 0)[0]
        raise AssertionError(f"unknown tracked view case {name}")

    def no_grad_tracked_view_descendant_contract(self, module):
        results = {}
        for name in ("slice", "reshape", "unsqueeze", "chunk"):
            base = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            first = self.tracked_view_case(base, name)
            with module.no_grad():
                child = first.reshape(-1)

            initial = (
                base.requires_grad,
                first.requires_grad,
                first.is_leaf,
                child.requires_grad,
                child.is_leaf,
                (child * 2.0).sum().requires_grad,
            )
            base.requires_grad_(False)
            disabled_loss = (child * 2.0).sum()
            after_disable = (
                base.requires_grad,
                first.requires_grad,
                first.is_leaf,
                child.requires_grad,
                child.is_leaf,
                disabled_loss.requires_grad,
            )
            base.requires_grad_(True)
            after_reenable = (
                base.requires_grad,
                first.requires_grad,
                child.requires_grad,
                (child * 2.0).sum().requires_grad,
            )

            promoted_base = module.tensor(
                [[5.0, 6.0], [7.0, 8.0]], requires_grad=True
            )
            promoted_first = self.tracked_view_case(promoted_base, name)
            with module.no_grad():
                promoted_child = promoted_first.reshape(-1)
            promoted_child.requires_grad_(True)
            promoted_loss = (promoted_child * 3.0).sum()
            promoted_child.requires_grad_(False)
            promoted_base.requires_grad_(False)
            promoted_loss.backward()
            promoted_after_disable = (
                promoted_base.requires_grad,
                promoted_first.requires_grad,
                promoted_child.requires_grad,
                promoted_loss.requires_grad,
                self.grad_payload(promoted_base.grad),
                self.grad_payload(promoted_child.grad),
            )

            results[name] = {
                "initial": initial,
                "after_disable": after_disable,
                "after_reenable": after_reenable,
                "promoted_after_disable": promoted_after_disable,
            }

        base = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        ordinary_non_leaf = base * 2.0
        with module.no_grad():
            child = ordinary_non_leaf.reshape(-1)
        base.requires_grad_(False)
        results["ordinary_non_view_non_leaf"] = (
            base.requires_grad,
            ordinary_non_leaf.requires_grad,
            ordinary_non_leaf.is_leaf,
            child.requires_grad,
            child.is_leaf,
            (child * 2.0).sum().requires_grad,
        )

        return results

    def test_no_grad_descendants_of_tracked_views_match_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_tracked_view_descendant_contract(torch),
            self.no_grad_tracked_view_descendant_contract(reference_torch),
        )

    def invalid_argument_contract(self, module):
        return tuple(
            self.error(call)
            for call in (
                lambda: module.tensor([1.0]).requires_grad_(None),
                lambda: module.tensor([1.0]).requires_grad_(1),
                lambda: module.tensor([1.0]).requires_grad_(0),
                lambda: module.tensor([1.0]).requires_grad_(np.bool_(True)),
                lambda: module.tensor([1.0]).requires_grad_(
                    requires_grad=None
                ),
                lambda: module.tensor([1.0]).requires_grad_(True, False),
                lambda: module.tensor([1.0]).requires_grad_(foo=True),
                lambda: module.tensor([1.0]).requires_grad_(
                    False, requires_grad=True
                ),
            )
        )

    def test_invalid_arguments_match_pytorch_2_13(self):
        self.assertEqual(
            self.invalid_argument_contract(torch),
            self.invalid_argument_contract(reference_torch),
        )

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def descriptor_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "requires_grad_")
        bound = tensor.requires_grad_
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "class_identity": module.Tensor.requires_grad_ is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "descriptor_result_is_self": descriptor(tensor) is tensor,
            "bound_result_is_self": bound(False) is tensor,
            "after_calls_requires_grad": tensor.requires_grad,
        }

    def test_descriptor_documentation_and_call_shape_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "requires_grad_")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.requires_grad_(False)
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.requires_grad_(requires_grad=True)

        invalid = RecordingMode()
        try:
            with invalid:
                tensor.requires_grad_(1)
        except Exception as error:
            invalid_error = type(error).__name__, str(error)
        else:
            self.fail(f"{module.__name__} accepted a requires_grad_ argument")

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_is_descriptor": function is descriptor,
            "types": dispatch_types == (),
            "args": len(args) == 2
            and args[0] is tensor
            and args[1] is False,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded_is_self": forwarded is tensor,
            "after_forward_requires_grad": tensor.requires_grad,
            "invalid_error": invalid_error,
            "invalid_calls": len(invalid.calls),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
