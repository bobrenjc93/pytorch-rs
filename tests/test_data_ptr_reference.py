import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorDataPtrReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "data_ptr differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def pointer_relationships(self, module):
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        source_ptr = source.data_ptr()
        transposed = source.transpose(0, 1)
        row = source[2]
        strided_row = transposed[1]
        detached = strided_row.detach()
        cloned = strided_row.clone()
        packed = transposed.contiguous()

        empty = module.zeros((3, 0, 4), dtype=module.float32)
        offset_empty = empty[2]
        extreme_empty = module.zeros(
            (sys.maxsize, 0), dtype=module.float32
        )[sys.maxsize - 1]
        return (
            type(source_ptr),
            source_ptr > 0,
            source.data_ptr() == source_ptr,
            transposed.data_ptr() == source_ptr,
            row.storage_offset(),
            row.data_ptr() - source_ptr,
            strided_row.storage_offset(),
            strided_row.data_ptr() - source_ptr,
            detached.data_ptr() == strided_row.data_ptr(),
            cloned.data_ptr() != strided_row.data_ptr(),
            packed.data_ptr() != transposed.data_ptr(),
            empty.data_ptr(),
            offset_empty.storage_offset(),
            offset_empty.data_ptr(),
            offset_empty.detach().data_ptr(),
            offset_empty.clone().data_ptr(),
            extreme_empty.storage_offset(),
            extreme_empty.data_ptr(),
            tuple(
                tensor.const_data_ptr() == tensor.data_ptr()
                for tensor in (
                    source,
                    transposed,
                    row,
                    strided_row,
                    detached,
                    cloned,
                    packed,
                    empty,
                    offset_empty,
                    offset_empty.detach(),
                    offset_empty.clone(),
                    extreme_empty,
                )
            ),
        )

    def test_offsets_empty_views_and_alias_relationships_match_pytorch_2_13(self):
        actual = self.pointer_relationships(torch)
        expected = self.pointer_relationships(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0], int)
        self.assertEqual(actual[5], 8 * 4)
        self.assertEqual(actual[7], 4)
        self.assertEqual(actual[11], 0)
        self.assertEqual(actual[13:16], (0, 0, 0))
        self.assertEqual(actual[16:18], (sys.maxsize - 1, 0))
        self.assertTrue(all(actual[18]))

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 3.0
        view = produced.transpose(0, 1)
        view_ptr = view.data_ptr()
        detached = view.detach()
        cloned = view.clone()
        packed = view.contiguous()
        state_before = (
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.requires_grad,
            view.is_leaf,
        )
        cloned_ptr = cloned.data_ptr()
        cloned.sum().backward()
        state_after = (
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.requires_grad,
            view.is_leaf,
        )
        return (
            produced.data_ptr() == view_ptr,
            detached.data_ptr() == view_ptr,
            cloned_ptr != view_ptr,
            packed.data_ptr() != view_ptr,
            cloned.data_ptr() == cloned_ptr,
            cloned.requires_grad,
            cloned.is_leaf,
            packed.requires_grad,
            packed.is_leaf,
            state_before,
            state_after,
            (
                view.const_data_ptr() == view_ptr,
                produced.const_data_ptr() == produced.data_ptr(),
                detached.const_data_ptr() == detached.data_ptr(),
                cloned.const_data_ptr() == cloned.data_ptr(),
                packed.const_data_ptr() == packed.data_ptr(),
                leaf.grad.const_data_ptr() == leaf.grad.data_ptr(),
            ),
            np.asarray(leaf.grad).copy(),
        )

    def test_live_copies_materialization_and_graph_state_match_pytorch_2_13(self):
        actual = self.autograd_outcome(torch)
        expected = self.autograd_outcome(reference_torch)
        self.assertEqual(actual[:-1], expected[:-1])
        np.testing.assert_array_equal(actual[-1], expected[-1])

    def test_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "data_ptr")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "data_ptr"
        )
        actual_bound = actual.data_ptr
        expected_bound = expected.data_ptr

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual_bound, expected_bound, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                self.signature_outcome(actual_callable),
                self.signature_outcome(expected_callable),
            )

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertIs(type(actual_descriptor(actual)), int)
        self.assertIs(type(expected_descriptor(expected)), int)

        call_pairs = (
            (lambda: actual.data_ptr(1), lambda: expected.data_ptr(1)),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (lambda: actual.data_ptr(dim=0), lambda: expected.data_ptr(dim=0)),
            (
                lambda: actual_bound(unexpected=True),
                lambda: expected_bound(unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, unexpected=True),
                lambda: expected_descriptor(expected, unexpected=True),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(invalid_call=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_const_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "const_data_ptr")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "const_data_ptr"
        )
        actual_bound = actual.const_data_ptr
        expected_bound = expected.const_data_ptr

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual_bound, expected_bound, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__qualname__, expected_callable.__qualname__
            )
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                self.signature_outcome(actual_callable),
                self.signature_outcome(expected_callable),
            )

        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(
            hasattr(actual_descriptor, "__module__"),
            hasattr(expected_descriptor, "__module__"),
        )
        self.assertEqual(actual_bound.__module__, expected_bound.__module__)
        self.assertEqual(actual_descriptor(actual), actual.data_ptr())
        self.assertEqual(expected_descriptor(expected), expected.data_ptr())

        call_pairs = (
            (
                lambda: actual.const_data_ptr(1),
                lambda: expected.const_data_ptr(1),
            ),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (
                lambda: actual.const_data_ptr(dim=0),
                lambda: expected.const_data_ptr(dim=0),
            ),
            (
                lambda: actual_bound(unexpected=True),
                lambda: expected_bound(unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, unexpected=True),
                lambda: expected_descriptor(expected, unexpected=True),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
            (
                lambda: actual_descriptor(self=actual),
                lambda: expected_descriptor(self=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(invalid_call=case):
                self.assert_error_matches(actual_call, expected_call)

    def mode_dispatch_outcome(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "const_data_ptr")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.const_data_ptr()
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
                forwarded = tensor.const_data_ptr()

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_owner": (
                function.__objclass__.__module__,
                function.__objclass__.__name__,
            ),
            "function_is_descriptor": function is descriptor,
            "dispatch_types": dispatch_types == (module.Tensor,),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded_type": type(forwarded).__name__,
            "forwarded_matches_data_ptr": forwarded == tensor.data_ptr(),
        }

    def test_const_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_outcome(torch),
            self.mode_dispatch_outcome(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
