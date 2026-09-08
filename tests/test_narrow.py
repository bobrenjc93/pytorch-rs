import inspect
import pickle
import re
import types
import unittest

import numpy as np

import torch_rs as torch


METHOD_DOC = "\nnarrow(dimension, start, length) -> Tensor\n\nSee :func:`torch.narrow`.\n"
FUNCTION_DOC = (
    "\nnarrow(input, dim, start, length) -> Tensor\n\n"
    "Returns a new tensor that is a narrowed version of :attr:`input` tensor. The\n"
    "dimension :attr:`dim` is input from :attr:`start` to ``start + length``. The\n"
    "returned tensor and :attr:`input` tensor share the same underlying storage.\n\n"
    "Args:\n"
    "    input (Tensor): the tensor to narrow\n"
    "    dim (int): the dimension along which to narrow\n"
    "    start (int or Tensor): index of the element to start the narrowed dimension\n"
    "        from. Can be negative, which means indexing from the end of `dim`. If\n"
    "        `Tensor`, it must be an 0-dim integral `Tensor` (bools not allowed)\n"
    "    length (int): length of the narrowed dimension, must be weakly positive\n\n"
    "Example::\n\n"
    "    >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n"
    "    >>> torch.narrow(x, 0, 0, 2)\n"
    "    tensor([[ 1,  2,  3],\n"
    "            [ 4,  5,  6]])\n"
    "    >>> torch.narrow(x, 1, 1, 2)\n"
    "    tensor([[ 2,  3],\n"
    "            [ 5,  6],\n"
    "            [ 8,  9]])\n"
    "    >>> torch.narrow(x, -1, torch.tensor(-1), 1)\n"
    "    tensor([[3],\n"
    "            [6],\n"
    "            [9]])\n"
)


def offset_noncontiguous_source(*, requires_grad=False):
    values = [float(value) for value in range(120)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 3, 4, 5)[
        1
    ].transpose(0, 1)


class TensorNarrowTests(unittest.TestCase):
    def assert_same_view(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertTrue(actual.is_set_to(expected))
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def test_method_and_top_level_narrow_return_shared_slice_views(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        contiguous = torch.tensor(values.tolist())
        transposed = contiguous.transpose(0, 2)
        offset = offset_noncontiguous_source()

        cases = (
            ("contiguous middle", contiguous, 1, 1, 2, contiguous[:, 1:3]),
            ("transposed leading", transposed, 0, 1, 2, transposed[1:3]),
            ("offset noncontiguous", offset, 1, 1, 2, offset[:, 1:3]),
            ("empty length", contiguous, 1, 3, 0, contiguous[:, 3:3]),
            ("full length", contiguous, -2, -3, 3, contiguous[:, 0:3]),
        )
        for case, source, dimension, start, length, expected in cases:
            with self.subTest(case=case, surface="method"):
                self.assert_same_view(source.narrow(dimension, start, length), expected)
            with self.subTest(case=case, surface="top-level"):
                self.assert_same_view(
                    torch.narrow(source, dimension, start, length), expected
                )

    def test_call_forms_input_aliases_and_integer_protocol(self):
        source = offset_noncontiguous_source()
        expected = source[:, 1:3]
        calls = (
            ("method positional", lambda: source.narrow(1, 1, 2)),
            ("method mixed", lambda: source.narrow(1, start=1, length=2)),
            ("method keywords", lambda: source.narrow(dim=1, start=1, length=2)),
            (
                "method reordered keywords",
                lambda: source.narrow(length=2, start=1, dim=1),
            ),
            ("method negative", lambda: source.narrow(-2, -2, 2)),
            ("top positional", lambda: torch.narrow(source, 1, 1, 2)),
            ("top mixed", lambda: torch.narrow(source, 1, start=1, length=2)),
            ("top keywords", lambda: torch.narrow(input=source, dim=1, start=1, length=2)),
            ("top input alias x", lambda: torch.narrow(x=source, dim=1, start=1, length=2)),
            ("top input alias a", lambda: torch.narrow(a=source, dim=1, start=1, length=2)),
            ("top input alias x1", lambda: torch.narrow(x1=source, dim=1, start=1, length=2)),
            ("top negative", lambda: torch.narrow(source, -2, -2, 2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                self.assert_same_view(call(), expected)

        class IntegerSubclass(int):
            pass

        self.assertEqual(
            source.narrow(np.int64(1), IntegerSubclass(1), np.uint32(2)).shape,
            (4, 2, 5),
        )

        conversion_order = []

        class StatefulIndex:
            def __init__(self, name, values):
                self.name = name
                self.values = values
                self.calls = 0

            def __index__(self):
                conversion_order.append(self.name)
                value = self.values[self.calls]
                self.calls += 1
                return value

        start = StatefulIndex("start", [1, 2, 1])
        length = StatefulIndex("length", [2, 1, 2])
        selected = source.narrow(1, start, length)
        self.assert_same_view(selected, expected)
        self.assertEqual(
            conversion_order,
            ["start", "length", "length", "length", "start", "start"],
        )

    def test_backward_through_sum_matches_slice_view(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        source = (leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        selected = source.narrow(0, 1, 2)

        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)
        self.assertEqual(selected.output_nr, 0)
        selected.sum().backward()

        expected_gradient = np.zeros_like(values)
        expected_gradient[:, 1:3, :] = 2.0
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(values.shape),
            expected_gradient,
        )

        empty = torch.zeros((2, 3, 4), requires_grad=True)
        empty.narrow(1, 3, 0).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 3, 4))
        self.assertEqual(empty.grad.tolist(), torch.zeros((2, 3, 4)).tolist())

    def test_bounds_and_unsupported_forms_fail_closed(self):
        tensor = torch.zeros((2, 3, 4))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: tensor.narrow(0, 0, -1),
                RuntimeError,
                "narrow(): length must be non-negative.",
            ),
            (
                lambda: tensor.narrow(0, 3, 0),
                IndexError,
                "start out of range (expected to be in range of [-2, 2], but got 3)",
            ),
            (
                lambda: tensor.narrow(0, -3, 0),
                IndexError,
                "start out of range (expected to be in range of [-2, 2], but got -3)",
            ),
            (
                lambda: tensor.narrow(0, 1, 2),
                RuntimeError,
                "start (1) + length (2) exceeds dimension size (2).",
            ),
            (
                lambda: tensor.narrow(3, 0, 1),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: scalar.narrow(0, 0, 1),
                RuntimeError,
                "narrow() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: tensor.narrow(0, torch.tensor(0.0), 1),
                NotImplementedError,
                "narrow(): tensor-valued start is not supported",
            ),
            (
                lambda: torch.narrow(tensor, 0, torch.tensor(0.0), 1),
                NotImplementedError,
                "narrow(): tensor-valued start is not supported",
            ),
            (
                lambda: tensor.narrow(slice(None), 0, 1),
                TypeError,
                "narrow() received an invalid combination of arguments",
            ),
            (
                lambda: torch.narrow([1.0], 0, 0, 1),
                TypeError,
                "narrow() received an invalid combination of arguments",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}") as raised:
                    call()
                self.assertIn(message, str(raised.exception))

        for call in (
            lambda: tensor.narrow(2**100, 0, 1),
            lambda: tensor.narrow(0, 2**100, 1),
            lambda: tensor.narrow(0, 0, 2**100),
        ):
            with self.assertRaises(ValueError) as raised:
                call()
            self.assertEqual(str(raised.exception), "Overflow when unpacking long long")

        with self.assertRaises(TypeError):
            type("TensorSubclass", (torch.Tensor,), {})

    def test_callable_metadata_exports_pickle_and_torch_function_modes(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "narrow")
        bound = tensor.narrow

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "narrow")
        self.assertEqual(bound.__name__, "narrow")
        self.assertEqual(descriptor.__qualname__, "TensorBase.narrow")
        self.assertEqual(bound.__qualname__, "Tensor.narrow")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)

        function = torch.narrow
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "narrow")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.narrow")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertIn("narrow", torch.__all__)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)

        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(tensor.narrow(0, 0, 1), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, 0, 1))
        self.assertIsNone(kwargs)

        tensor_start = torch.tensor(0.0)
        tensor_start_mode = RecordingMode(marker)
        with tensor_start_mode:
            self.assertIs(tensor.narrow(0, tensor_start, 1), marker)
        function, dispatch_types, args, kwargs = tensor_start_mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, tensor_start, 1))
        self.assertIsNone(kwargs)

        top_level_mode = RecordingMode(marker)
        with top_level_mode:
            self.assertIs(torch.narrow(tensor, 0, 0, 1), marker)
        function, dispatch_types, args, kwargs = top_level_mode.calls[0]
        self.assertIs(function, torch.narrow)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, 0, 1))
        self.assertIsNone(kwargs)

        top_level_tensor_start_mode = RecordingMode(marker)
        with top_level_tensor_start_mode:
            self.assertIs(torch.narrow(tensor, 0, tensor_start, 1), marker)
        function, dispatch_types, args, kwargs = top_level_tensor_start_mode.calls[0]
        self.assertIs(function, torch.narrow)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, tensor_start, 1))
        self.assertIsNone(kwargs)


if __name__ == "__main__":
    unittest.main()
