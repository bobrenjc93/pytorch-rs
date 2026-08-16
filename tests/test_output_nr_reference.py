import inspect
import struct
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorOutputNumberReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "output_nr differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        ordinary = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        empty_leaf = module.zeros((2, 0, 3), requires_grad=True)
        empty_tracked = empty_leaf * 2.0

        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_leaf_view = leaf.transpose(0, 1)
            no_grad_non_leaf_view = tracked.transpose(0, 1)

        recorded_after_no_grad = no_grad_leaf_view + 1.0
        tracked.sum().backward()

        return (
            ordinary,
            ordinary + 1.0,
            ordinary.transpose(0, 1),
            leaf,
            tracked,
            tracked_view,
            tracked.detach(),
            tracked_view.detach(),
            empty_leaf,
            empty_tracked,
            empty_tracked.transpose(0, 2),
            no_grad_output,
            no_grad_leaf_view,
            no_grad_non_leaf_view,
            recorded_after_no_grad,
            leaf.grad,
        )

    def output_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        value = tensor.output_nr
        return {
            "value": value,
            "value_type": type(value).__name__,
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_zero_matches_pytorch_2_13_for_supported_single_output_states(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                actual_contract = self.output_contract(actual)
                expected_contract = self.output_contract(expected)
                self.assertEqual(actual_contract["value"], 0)
                self.assertEqual(expected_contract["value"], 0)
                self.assertEqual(actual_contract, expected_contract)

    def iteration_contract(self, module):
        source = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        iterator = iter(source)
        iterator_type = type(iterator).__name__
        iterator_is_self = iter(iterator) is iterator
        length_hint_before = iterator.__length_hint__()
        rows = tuple(iterator)
        length_hint_after = iterator.__length_hint__()
        indexed = tuple(source[index] for index in range(len(source)))

        row_metadata = tuple(
            (
                row.output_nr,
                row.requires_grad,
                row.is_leaf,
                tuple(row.shape),
                row.stride(),
                row.storage_offset(),
                row.data_ptr() == direct.data_ptr(),
                row.tolist(),
            )
            for row, direct in zip(rows, indexed, strict=True)
        )
        direct_output_numbers = tuple(row.output_nr for row in indexed)
        middle = rows[1]
        identity_results = (+middle, middle.real, middle.contiguous())
        new_results = (middle.detach(), middle.transpose(0, 0), middle[0])
        rows[1].sum().backward()

        signed_source = module.tensor([[1.0], [-0.0]], requires_grad=True)
        signed_rows = tuple(signed_source)
        (signed_rows[0] * signed_rows[1]).sum().backward()
        signed_gradient_bits = tuple(
            struct.unpack(">I", struct.pack(">f", value))[0]
            for row in signed_source.grad.tolist()
            for value in row
        )

        ordinary = module.tensor([[1.0], [2.0], [3.0]])
        ordinary_output_numbers = tuple(row.output_nr for row in ordinary)

        no_grad_source = module.tensor(
            [[1.0], [2.0], [3.0]], requires_grad=True
        )
        with module.no_grad():
            no_grad_rows = tuple(no_grad_source)

        empty_rows = tuple(module.zeros((2, 0), requires_grad=True))
        empty_outer = tuple(module.zeros((0, 2), requires_grad=True))
        try:
            iter(module.tensor(1.0))
        except Exception as error:
            scalar_error = type(error).__name__, str(error)
        else:
            self.fail(f"{module.__name__} allowed scalar tensor iteration")

        return {
            "iterator_type": iterator_type,
            "iterator_is_self": iterator_is_self,
            "length_hints": (length_hint_before, length_hint_after),
            "rows": row_metadata,
            "direct_output_numbers": direct_output_numbers,
            "identity_results": tuple(
                (result is middle, result.output_nr) for result in identity_results
            ),
            "new_result_output_numbers": tuple(
                result.output_nr for result in new_results
            ),
            "gradient": source.grad.tolist(),
            "signed_gradient_bits": signed_gradient_bits,
            "ordinary_output_numbers": ordinary_output_numbers,
            "no_grad_rows": tuple(
                (row.output_nr, row.requires_grad, row.is_leaf)
                for row in no_grad_rows
            ),
            "empty_rows": tuple(
                (row.output_nr, row.numel(), tuple(row.shape))
                for row in empty_rows
            ),
            "empty_outer": empty_outer,
            "scalar_error": scalar_error,
        }

    def test_iteration_output_numbers_match_pytorch_2_13(self):
        self.assertEqual(
            self.iteration_contract(torch),
            self.iteration_contract(reference_torch),
        )

    def iteration_mode_contract(self, module):
        source = module.tensor([[1.0], [2.0]])

        class ReplacingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                if func.__name__ == "unbind":
                    return ("replacement-a", "replacement-b")
                return func(*args, **(kwargs or {}))

        mode = ReplacingMode()
        with mode:
            result = tuple(iter(source))

        dim_function, dim_types, dim_args, dim_kwargs = mode.calls[0]
        unbind_function, unbind_types, unbind_args, unbind_kwargs = mode.calls[1]
        return {
            "result": result,
            "call_count": len(mode.calls),
            "names": tuple(call[0].__name__ for call in mode.calls),
            "function_types": tuple(type(call[0]).__name__ for call in mode.calls),
            "qualnames": tuple(call[0].__qualname__ for call in mode.calls),
            "owner_names": tuple(call[0].__objclass__.__name__ for call in mode.calls),
            "owner_modules": tuple(
                call[0].__objclass__.__module__ for call in mode.calls
            ),
            "dim_types": dim_types == (module.Tensor,),
            "dim_args": dim_args == (source,),
            "dim_kwargs_none": dim_kwargs is None,
            "unbind_types": unbind_types == (),
            "unbind_args": len(unbind_args) == 2
            and unbind_args[0] is source
            and unbind_args[1] == 0,
            "unbind_kwargs_none": unbind_kwargs is None,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_iteration_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.iteration_mode_contract(torch),
            self.iteration_mode_contract(reference_torch),
        )

    def iteration_replacement_protocol_contract(self, module):
        source = module.tensor([[1.0], [2.0]])

        class GetItemOnly:
            def __init__(self):
                self.values = ("replacement-a", "replacement-b")

            def __getitem__(self, index):
                return self.values[index]

        class ReplacingMode(module.overrides.TorchFunctionMode):
            def __init__(self, replacement):
                self.replacement = replacement

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                if func.__name__ == "unbind":
                    return self.replacement
                return func(*args, **(kwargs or {}))

        with ReplacingMode(GetItemOnly()):
            iterator = iter(source)
        sequence_result = type(iterator).__name__, tuple(iterator)

        try:
            with ReplacingMode(object()):
                iter(source)
        except Exception as error:
            non_iterable_error = type(error).__name__, str(error)
        else:
            self.fail(f"{module.__name__} accepted a non-iterable replacement")

        return {
            "sequence_result": sequence_result,
            "non_iterable_error": non_iterable_error,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_iteration_replacement_protocol_matches_pytorch_2_13(self):
        self.assertEqual(
            self.iteration_replacement_protocol_contract(torch),
            self.iteration_replacement_protocol_contract(reference_torch),
        )

    def test_reference_multi_output_nodes_expose_the_unsupported_nonzero_state(self):
        self.assertFalse(hasattr(torch.Tensor, "unbind"))
        self.assertFalse(hasattr(torch.Tensor, "chunk"))
        self.assertTrue(hasattr(reference_torch.Tensor, "unbind"))
        self.assertTrue(hasattr(reference_torch.Tensor, "chunk"))

        source = reference_torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            requires_grad=True,
        )
        unbound = source.unbind(0)
        chunked = source.chunk(3, 0)

        self.assertEqual(tuple(output.output_nr for output in unbound), (0, 1, 2))
        self.assertEqual(tuple(output.output_nr for output in chunked), (0, 1, 2))
        self.assertTrue(all(output.requires_grad for output in unbound + chunked))
        self.assertTrue(any(output.output_nr != 0 for output in unbound))
        self.assertTrue(any(output.output_nr != 0 for output in chunked))

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.output_nr unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "output_nr")
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, "output_nr", 1),
            lambda: delattr(tensor, "output_nr"),
            lambda: descriptor.__set__(tensor, 1),
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
            "class_identity": module.Tensor.output_nr is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": descriptor.__get__(tensor, module.Tensor),
            "value_type": type(
                descriptor.__get__(tensor, module.Tensor)
            ).__name__,
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_tensorbase_ownership_and_read_only_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(module.Tensor, "output_nr")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.output_nr
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
                forwarded = tensor.output_nr

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
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
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
