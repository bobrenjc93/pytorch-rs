import inspect
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
class TensorRequiresGradInPlaceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.requires_grad_ differentials require pinned PyTorch 2.13.0"
            )

    def layout_cases(self, module):
        base = module.tensor(
            [float(value) for value in range(24)], dtype=module.float32
        ).reshape(2, 3, 4)
        source = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        detached = (source * 2.0).transpose(0, 1).detach()
        return (
            (module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32), None),
            (module.zeros((2, 0, 3), dtype=module.float32), None),
            (detached, source),
            (base[1], base),
            (base.transpose(0, 2)[1], base),
        )

    def enable_contract(self, module):
        outcomes = []
        for index, (tensor, source) in enumerate(self.layout_cases(module)):
            before = (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.tolist(),
                tensor.requires_grad,
                tensor.is_leaf,
            )
            result = (
                tensor.requires_grad_()
                if index % 2 == 0
                else tensor.requires_grad_(True)
            )
            after = (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.tolist(),
                tensor.requires_grad,
                tensor.is_leaf,
            )
            second_result = tensor.requires_grad_(True)
            (tensor * module.ones(tensor.shape, dtype=module.float32)).sum().backward()
            outcomes.append(
                {
                    "result_is_receiver": result is tensor,
                    "second_result_is_receiver": second_result is tensor,
                    "metadata_preserved": before[:-2] == after[:-2],
                    "before_autograd": before[-2:],
                    "after_autograd": after[-2:],
                    "gradient_shape": tuple(tensor.grad.shape),
                    "gradient_stride": tuple(tensor.grad.stride()),
                    "gradient": tensor.grad.tolist(),
                    "source_grad_is_none": source is None or source.grad is None,
                }
            )
        return outcomes

    def test_leaf_layout_enablement_and_backward_match_pytorch_2_13(self):
        self.assertEqual(
            self.enable_contract(torch),
            self.enable_contract(reference_torch),
        )

    def preexisting_view_contract(self, module):
        base = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
        )
        alias = base[:]
        nested = alias.transpose(0, 1)[1]
        detached = alias.detach()
        before = (
            base.requires_grad,
            alias.requires_grad,
            nested.requires_grad,
            detached.requires_grad,
        )
        result = base.requires_grad_()

        outputs = []
        for view in (alias, nested):
            output = view * 2.0
            output.sum().backward()
            outputs.append(
                (
                    view.requires_grad,
                    view.is_leaf,
                    output.requires_grad,
                    output.is_leaf,
                    view.grad is None,
                )
            )
        base_grad_after_views = base.grad

        promoted = alias.requires_grad_(True)
        weights = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
        )
        (alias * weights).sum().backward()
        return {
            "before": before,
            "base_result_is_receiver": result is base,
            "after": (
                base.requires_grad,
                alias.requires_grad,
                nested.requires_grad,
                detached.requires_grad,
            ),
            "outputs": outputs,
            "base_grad_after_views_is_none": base_grad_after_views is None,
            "promoted_is_receiver": promoted is alias,
            "alias_gradient": alias.grad.tolist(),
            "base_grad_is_none": base.grad is None,
        }

    def test_preexisting_views_follow_base_enablement_like_pytorch_2_13(self):
        self.assertEqual(
            self.preexisting_view_contract(torch),
            self.preexisting_view_contract(reference_torch),
        )

    def gradient_layout_contract(self, module):
        base = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
        )
        leaf = base.transpose(0, 1).detach()
        first = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=module.float32
        )
        second = module.tensor(
            [[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]], dtype=module.float32
        )
        result = leaf.requires_grad_()
        (leaf * first).sum().backward()
        gradient = leaf.grad
        first_state = (
            tuple(gradient.shape),
            tuple(gradient.stride()),
            gradient.storage_offset(),
            gradient.is_contiguous(),
            gradient.tolist(),
        )
        (leaf * second).sum().backward()
        return {
            "result_is_receiver": result is leaf,
            "leaf_stride": tuple(leaf.stride()),
            "first_gradient": first_state,
            "gradient_identity_preserved": leaf.grad is gradient,
            "accumulated_stride": tuple(leaf.grad.stride()),
            "accumulated_values": leaf.grad.tolist(),
        }

    def test_dense_noncontiguous_gradient_layout_matches_pytorch_2_13(self):
        self.assertEqual(
            self.gradient_layout_contract(torch),
            self.gradient_layout_contract(reference_torch),
        )

    def state_contract(self, module):
        leaf = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        (leaf * 2.0).sum().backward()
        cached_grad = leaf.grad
        leaf_pointer = leaf.data_ptr()
        leaf_result = leaf.requires_grad_()
        cached_grad_preserved = leaf.grad is cached_grad
        (leaf * 3.0).sum().backward()

        source = module.tensor(
            [2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        nonleaf = source * 4.0
        nonleaf_pointer = nonleaf.data_ptr()
        nonleaf_result = nonleaf.requires_grad_(True)
        nonleaf.sum().backward()

        view_source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = view_source * 2.0
        with module.no_grad():
            view = tracked.transpose(0, 1)
        view_before = (
            view.requires_grad,
            view.is_leaf,
            tuple(view.shape),
            tuple(view.stride()),
            view.storage_offset(),
        )
        view_pointer = view.data_ptr()
        view_result = view.requires_grad_()
        weights = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
        )
        (view * weights).sum().backward()

        return {
            "leaf_result_is_receiver": leaf_result is leaf,
            "leaf_pointer_preserved": leaf.data_ptr() == leaf_pointer,
            "leaf_cached_grad_preserved": cached_grad_preserved,
            "leaf_gradient": leaf.grad.tolist(),
            "nonleaf_result_is_receiver": nonleaf_result is nonleaf,
            "nonleaf_pointer_preserved": nonleaf.data_ptr() == nonleaf_pointer,
            "nonleaf_requires_grad": nonleaf.requires_grad,
            "nonleaf_is_leaf": nonleaf.is_leaf,
            "source_gradient": source.grad.tolist(),
            "view_before": view_before,
            "view_result_is_receiver": view_result is view,
            "view_metadata_preserved": view_before
            == (
                view.requires_grad,
                view.is_leaf,
                tuple(view.shape),
                tuple(view.stride()),
                view.storage_offset(),
            ),
            "view_pointer_preserved": view.data_ptr() == view_pointer,
            "view_gradient": view.grad.tolist(),
            "view_source_grad_is_none": view_source.grad is None,
        }

    def test_existing_graphs_and_no_grad_leaf_views_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_contract(torch),
            self.state_contract(reference_torch),
        )

    def no_grad_contract(self, module):
        tensor = module.tensor([2.0, 3.0], dtype=module.float32)
        with module.no_grad():
            result = tensor.requires_grad_()
            suppressed = tensor * 5.0
        recorded = tensor * 7.0
        recorded.sum().backward()
        return {
            "result_is_receiver": result is tensor,
            "tensor_requires_grad": tensor.requires_grad,
            "tensor_is_leaf": tensor.is_leaf,
            "suppressed_requires_grad": suppressed.requires_grad,
            "suppressed_is_leaf": suppressed.is_leaf,
            "recorded_requires_grad": recorded.requires_grad,
            "recorded_is_leaf": recorded.is_leaf,
            "gradient": tensor.grad.tolist(),
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def test_no_grad_interaction_matches_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_contract(torch),
            self.no_grad_contract(reference_torch),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("requires_grad_ call unexpectedly succeeded")

    def binding_contract(self, module):
        def outcome(action):
            tensor = module.tensor([1.0], dtype=module.float32)
            error = self.error(lambda: action(tensor))
            return error, tensor.requires_grad, tensor.grad

        descriptor = inspect.getattr_static(module.Tensor, "requires_grad_")
        tensor = module.tensor([1.0], dtype=module.float32)
        return {
            "invalid_calls": (
                outcome(lambda value: value.requires_grad_(1)),
                outcome(lambda value: value.requires_grad_(requires_grad=1)),
                outcome(lambda value: value.requires_grad_(None)),
                outcome(lambda value: value.requires_grad_(np.bool_(True))),
                outcome(lambda value: value.requires_grad_(True, False)),
                outcome(lambda value: value.requires_grad_(foo=True)),
                outcome(
                    lambda value: value.requires_grad_(True, requires_grad=True)
                ),
            ),
            "receiver_errors": (
                self.error(lambda: descriptor()),
                self.error(lambda: descriptor(1)),
                self.error(lambda: descriptor(self=tensor)),
            ),
        }

    def test_strict_boolean_binding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.binding_contract(torch),
            self.binding_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "requires_grad_")
        tensor = module.tensor([1.0], dtype=module.float32)
        bound = tensor.requires_grad_

        def signature_outcome(callable_object):
            try:
                return "signature", str(inspect.signature(callable_object))
            except Exception as error:
                return "error", type(error).__name__

        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "repr": repr(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                signature_outcome(descriptor),
                signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "class_identity": module.Tensor.requires_grad_ is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
        }

    def test_tensorbase_ownership_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "requires_grad_")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        observations = []
        for form, call in (
            ("default", lambda tensor: tensor.requires_grad_()),
            ("positional", lambda tensor: tensor.requires_grad_(True)),
            (
                "keyword",
                lambda tensor: tensor.requires_grad_(requires_grad=True),
            ),
        ):
            tensor = module.tensor([1.0], dtype=module.float32)
            mode = RecordingMode()
            with mode:
                result = call(tensor)
            function, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    form,
                    result is marker,
                    function is descriptor,
                    function.__qualname__,
                    dispatch_types == (),
                    args[0] is tensor,
                    args[1:],
                    kwargs,
                    tensor.requires_grad,
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        forwarded_tensor = module.tensor([1.0], dtype=module.float32)
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = forwarded_tensor.requires_grad_(requires_grad=True)

        invalid_tensor = module.tensor([1.0], dtype=module.float32)
        invalid_mode = RecordingMode()
        try:
            with invalid_mode:
                invalid_tensor.requires_grad_(1)
        except Exception as error:
            invalid_error = type(error).__name__, str(error)
        else:
            invalid_error = None

        declining_tensor = module.tensor([1.0], dtype=module.float32)
        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                declining_tensor.requires_grad_()
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "observations": observations,
            "forwarding_order": order,
            "forwarded_is_receiver": forwarded is forwarded_tensor,
            "forwarded_requires_grad": forwarded_tensor.requires_grad,
            "invalid_error": invalid_error,
            "invalid_mode_calls": len(invalid_mode.calls),
            "invalid_requires_grad": invalid_tensor.requires_grad,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "declining_requires_grad": declining_tensor.requires_grad,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_forwarding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_reference_disabling_bounds_the_intentionally_unsupported_state(self):
        actual = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
        before = (
            actual.requires_grad,
            actual.is_leaf,
            actual.data_ptr(),
            actual.tolist(),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^requires_grad_\(False\) is not supported; only enabling leaf "
            r"tensors is implemented$",
        ):
            actual.requires_grad_(False)
        self.assertEqual(
            (
                actual.requires_grad,
                actual.is_leaf,
                actual.data_ptr(),
                actual.tolist(),
            ),
            before,
        )

        expected = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32, requires_grad=True
        )
        self.assertIs(expected.requires_grad_(False), expected)
        self.assertFalse(expected.requires_grad)


if __name__ == "__main__":
    unittest.main()
