import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


UNSUPPORTED = "atleast_2d() only supports a single Tensor input"


class Atleast2dTests(unittest.TestCase):
    def test_rank_two_and_higher_tensors_are_returned_exactly(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            base[1],
            base.transpose(0, 2),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
        )
        for source in cases:
            with self.subTest(shape=source.shape, stride=source.stride()):
                result = torch.atleast_2d(source)
                self.assertIs(result, source)

    def test_scalars_and_vectors_become_shared_storage_views(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        empty_vector = torch.zeros((0, 2)).transpose(0, 1)[1]
        cases = (
            (torch.tensor(-0.0), (1, 1), (1, 1)),
            (base.transpose(0, 2)[3, 2, 1], (1, 1), (1, 1)),
            (base[1, 2], (1, 4), (4, 1)),
            (base.transpose(0, 2)[1, 1], (1, 2), (24, 12)),
            (empty_vector, (1, 0), (0, 2)),
        )
        for source, shape, stride in cases:
            with self.subTest(
                source_shape=source.shape,
                source_stride=source.stride(),
                offset=source.storage_offset(),
            ):
                result = torch.atleast_2d(source)
                direct = source.reshape(shape)
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.stride(), stride)
                self.assertEqual(result.storage_offset(), source.storage_offset())
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertEqual(result.is_set_to(direct), source.numel() != 0)
                self.assertIs(result.dtype, source.dtype)
                self.assertEqual(result.device, source.device)
                self.assertEqual(result.layout, source.layout)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))

    def test_autograd_repeated_backward_and_no_grad(self):
        scalar_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        scalar = scalar_leaf[1]
        scalar_result = torch.atleast_2d(scalar)
        scalar_loss = scalar_result.sum()
        scalar_loss.backward()
        scalar_loss.backward()
        self.assertEqual(scalar_leaf.grad.tolist(), [0.0, 2.0, 0.0])

        vector_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        vector_result = torch.atleast_2d(vector_leaf)
        vector_loss = vector_result.sum()
        vector_loss.backward()
        vector_loss.backward()
        self.assertEqual(vector_leaf.grad.tolist(), [2.0, 2.0, 2.0])

        no_grad_source = torch.tensor([3.0, 4.0], requires_grad=True)
        with torch.no_grad():
            no_grad_result = torch.atleast_2d(no_grad_source)
        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertEqual(no_grad_result.data_ptr(), no_grad_source.data_ptr())
        (no_grad_result * no_grad_result).sum().backward()
        self.assertIsNone(no_grad_source.grad)
        self.assertIsNone(no_grad_result.grad)

    def test_modes_and_overrides_receive_the_public_function(self):
        source = torch.tensor(2.0)
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.atleast_2d(value), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.atleast_2d)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {})

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.atleast_2d(source)
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.atleast_2d for call in calls))
        self.assertTrue(all(call[2] == (torch.Tensor,) for call in calls))
        self.assertTrue(all(call[3] == (source,) for call in calls))
        self.assertTrue(all(call[4] == {} for call in calls))
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result.data_ptr(), source.data_ptr())

    def test_function_metadata_exports_and_pickle(self):
        function = torch.atleast_2d
        self.assertIs(function, torch.functional.atleast_2d)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "atleast_2d")
        self.assertEqual(function.__qualname__, "atleast_2d")
        self.assertEqual(function.__module__, "torch_rs.functional")
        self.assertEqual(str(inspect.signature(function)), "(*tensors)")
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(torch.__all__.count("atleast_2d"), 1)
        self.assertEqual(torch.functional.__all__.count("atleast_2d"), 1)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["atleast_2d"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_single_input_errors_and_unsupported_forms(self):
        invalid_message = (
            "atleast_2d() received an invalid combination of arguments - got "
            "(NoneType), but expected one of:\n * (Tensor input)\n      didn't "
            "match because some of the arguments have invalid types: "
            "(!NoneType!)\n * (tuple of Tensors tensors)\n      didn't match "
            "because some of the arguments have invalid types: (!NoneType!)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_message)}$"):
            torch.atleast_2d(None)
        with self.assertRaisesRegex(
            TypeError,
            "^atleast_2d\\(\\) got an unexpected keyword argument 'input'$",
        ):
            torch.atleast_2d(input=torch.tensor(1.0))

        source = torch.tensor(1.0)
        unsupported_calls = (
            lambda: torch.atleast_2d(),
            lambda: torch.atleast_2d(source, source),
            lambda: torch.atleast_2d((source,)),
            lambda: torch.atleast_2d([source]),
        )
        for call in unsupported_calls:
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, f"^{re.escape(UNSUPPORTED)}$"
            ):
                call()


if __name__ == "__main__":
    unittest.main()
