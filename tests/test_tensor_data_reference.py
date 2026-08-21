import gc
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
class TensorDataReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.data differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        return (
            module.tensor(-0.0, dtype=module.float32, requires_grad=True),
            module.tensor(values.tolist(), dtype=module.float32),
            (leaf * 3.0).transpose(0, 2)[1],
            module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            ).transpose(0, 2)[1],
        )

    def alias_contract(self, source):
        first = source.data
        second = source.data
        return {
            "fresh_from_source": first is not source,
            "fresh_per_get": second is not first,
            "shares_source": first.is_set_to(source),
            "aliases_share": second.is_set_to(first),
            "shape": tuple(first.shape),
            "stride": first.stride(),
            "offset": first.storage_offset(),
            "same_pointer": first.data_ptr() == source.data_ptr(),
            "dtype": str(first.dtype),
            "device": str(first.device),
            "requires_grad": first.requires_grad,
            "is_leaf": first.is_leaf,
            "values": np.asarray(first).reshape(-1).view(np.uint32).copy(),
        }

    def test_alias_layout_values_and_autograd_metadata_match_pytorch_2_13(self):
        for actual, expected in zip(
            self.tensor_cases(torch),
            self.tensor_cases(reference_torch),
            strict=True,
        ):
            with self.subTest(shape=tuple(expected.shape), stride=expected.stride()):
                actual_contract = self.alias_contract(actual)
                expected_contract = self.alias_contract(expected)
                np.testing.assert_array_equal(
                    actual_contract.pop("values"), expected_contract.pop("values")
                )
                self.assertEqual(actual_contract, expected_contract)

    def lifetime_values(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def make_alias():
            temporary = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
            )
            return temporary.transpose(0, 2)[1].data

        alias = make_alias()
        gc.collect()
        return (
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
            np.asarray(alias).copy(),
        )

    def test_alias_lifetime_matches_pytorch_2_13(self):
        actual = self.lifetime_values(torch)
        expected = self.lifetime_values(reference_torch)
        self.assertEqual(actual[:-1], expected[:-1])
        np.testing.assert_array_equal(actual[-1], expected[-1])

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        alias = source.data
        detached_loss = (alias * alias).sum()
        source.sum().backward()
        return (
            source.requires_grad,
            source.is_leaf,
            alias.requires_grad,
            alias.is_leaf,
            detached_loss.requires_grad,
            source.is_set_to(alias),
            np.asarray(leaf.grad).copy(),
        )

    def test_autograd_isolation_and_source_graph_preservation_match(self):
        actual = self.autograd_outcome(torch)
        expected = self.autograd_outcome(reference_torch)
        self.assertEqual(actual[:-1], expected[:-1])
        np.testing.assert_array_equal(actual[-1], expected[-1])

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.data unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "data")
        tensor = module.tensor([1.0], dtype=module.float32)
        value = descriptor.__get__(tensor, module.Tensor)
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
            "has_text_signature": hasattr(descriptor, "__text_signature__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.data is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_is_fresh": value is not tensor,
            "value_shares_storage": value.is_set_to(tensor),
            "value_requires_grad": value.requires_grad,
            "value_is_leaf": value.is_leaf,
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_getter_descriptor_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
        descriptor = inspect.getattr_static(module.Tensor, "data")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.data
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.data

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_self": function.__self__ is descriptor,
            "function_equals_descriptor_get": function == descriptor.__get__,
            "types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded_is_fresh": forwarded is not tensor,
            "forwarded_shares_storage": forwarded.is_set_to(tensor),
            "forwarded_requires_grad": forwarded.requires_grad,
            "forwarded_is_leaf": forwarded.is_leaf,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_getter_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
