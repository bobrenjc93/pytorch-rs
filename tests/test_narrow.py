import copy
import importlib
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


def contiguous_source(*, requires_grad=False):
    return torch.tensor(
        [float(value) for value in range(24)], requires_grad=requires_grad
    ).reshape(3, 2, 4)


def offset_source(*, requires_grad=False):
    return torch.tensor(
        [float(value) for value in range(24)], requires_grad=requires_grad
    ).reshape(2, 3, 4)[1]


def noncontiguous_source(*, requires_grad=False):
    return torch.tensor(
        [float(value) for value in range(48)], requires_grad=requires_grad
    ).reshape(2, 2, 3, 4)[1].transpose(0, 1)


class TensorNarrowTests(unittest.TestCase):
    def assert_narrow_view(
        self,
        actual,
        source,
        *,
        start,
        shape,
        stride,
        offset,
        values,
    ):
        self.assertEqual(actual.tolist(), values)
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)
        if actual.numel() == 0:
            self.assertEqual(actual.data_ptr(), 0)
        else:
            self.assertEqual(actual.data_ptr(), source[start].data_ptr())

    def test_method_call_forms_return_contiguous_shared_storage_views(self):
        source = contiguous_source()
        expected_values = [
            [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
            [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
        ]
        calls = (
            ("positional", lambda: source.narrow(0, 1, 2)),
            ("mixed", lambda: source.narrow(0, start=1, length=2)),
            ("keywords", lambda: source.narrow(dim=0, start=1, length=2)),
            ("reordered keywords", lambda: source.narrow(length=2, dim=0, start=1)),
            ("normalized negative dimension", lambda: source.narrow(-3, 1, 2)),
            ("negative start", lambda: source.narrow(dim=-3, start=-2, length=2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                self.assert_narrow_view(
                    call(),
                    source,
                    start=1,
                    shape=(2, 2, 4),
                    stride=(8, 4, 1),
                    offset=8,
                    values=expected_values,
                )

    def test_offset_noncontiguous_empty_length_and_zero_source_views(self):
        offset = offset_source()
        self.assert_narrow_view(
            offset.narrow(0, 1, 2),
            offset,
            start=1,
            shape=(2, 4),
            stride=(4, 1),
            offset=16,
            values=[
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        )

        noncontiguous = noncontiguous_source()
        self.assert_narrow_view(
            noncontiguous.narrow(0, 1, 2),
            noncontiguous,
            start=1,
            shape=(2, 2, 4),
            stride=(4, 12, 1),
            offset=28,
            values=[
                [[28.0, 29.0, 30.0, 31.0], [40.0, 41.0, 42.0, 43.0]],
                [[32.0, 33.0, 34.0, 35.0], [44.0, 45.0, 46.0, 47.0]],
            ],
        )

        contiguous = contiguous_source()
        empty_length = contiguous.narrow(0, 1, 0)
        self.assert_narrow_view(
            empty_length,
            contiguous,
            start=1,
            shape=(0, 2, 4),
            stride=contiguous.stride(),
            offset=8,
            values=[],
        )

        zero_source = torch.zeros((0, 2))
        zero_narrowed = zero_source.narrow(0, 0, 0)
        self.assert_narrow_view(
            zero_narrowed,
            zero_source,
            start=0,
            shape=(0, 2),
            stride=(2, 1),
            offset=0,
            values=[],
        )

    def test_top_level_call_forms_and_aliases_return_first_axis_views(self):
        source = contiguous_source()
        expected_values = [
            [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
            [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
        ]
        calls = (
            ("positional", lambda: torch.narrow(source, 0, 1, 2)),
            ("mixed", lambda: torch.narrow(source, 0, start=1, length=2)),
            ("keywords", lambda: torch.narrow(source, dim=0, start=1, length=2)),
            (
                "all keywords",
                lambda: torch.narrow(input=source, dim=0, start=1, length=2),
            ),
            (
                "reordered keywords",
                lambda: torch.narrow(length=2, start=1, input=source, dim=0),
            ),
            ("input alias x", lambda: torch.narrow(x=source, dim=0, start=1, length=2)),
            ("input alias a", lambda: torch.narrow(a=source, dim=0, start=1, length=2)),
            ("input alias x1", lambda: torch.narrow(x1=source, dim=0, start=1, length=2)),
            ("negative dimension", lambda: torch.narrow(source, -3, 1, 2)),
            ("negative start", lambda: torch.narrow(source, -3, -2, 2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                self.assert_narrow_view(
                    call(),
                    source,
                    start=1,
                    shape=(2, 2, 4),
                    stride=(8, 4, 1),
                    offset=8,
                    values=expected_values,
                )

    def test_bounds_errors_and_unsupported_surface(self):
        tensor = torch.zeros((3, 2, 4))
        cases = (
            (
                lambda: tensor.narrow(0, -4, 0),
                IndexError,
                "start out of range (expected to be in range of [-3, 3], but got -4)",
            ),
            (
                lambda: tensor.narrow(0, 4, 0),
                IndexError,
                "start out of range (expected to be in range of [-3, 3], but got 4)",
            ),
            (
                lambda: torch.zeros((0, 2)).narrow(0, 1, 0),
                IndexError,
                "start out of range (expected to be in range of [0, 0], but got 1)",
            ),
            (
                lambda: tensor.narrow(0, 0, -1),
                RuntimeError,
                "narrow(): length must be non-negative.",
            ),
            (
                lambda: tensor.narrow(0, -1, 2),
                RuntimeError,
                "start (2) + length (2) exceeds dimension size (3).",
            ),
            (
                lambda: torch.tensor(1.0).narrow(0, 0, 1),
                RuntimeError,
                "narrow() cannot be applied to a 0-dim tensor.",
            ),
            (
                lambda: tensor.narrow(3, 0, 1),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: torch.narrow(tensor, 0, 4, 0),
                IndexError,
                "start out of range (expected to be in range of [-3, 3], but got 4)",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for call, operation in (
            (lambda: tensor.narrow(1, 0, 1), "Tensor.narrow"),
            (lambda: tensor.narrow(-2, 0, 1), "Tensor.narrow"),
            (lambda: torch.narrow(tensor, 1, 0, 1), "torch.narrow"),
            (lambda: torch.narrow(tensor, -2, 0, 1), "torch.narrow"),
        ):
            with self.subTest(operation=operation), self.assertRaisesRegex(
                RuntimeError, f"^{re.escape(operation)} only supports dimension 0$"
            ):
                call()

        tensor_start = torch.tensor(0.0)
        for call in (
            lambda: tensor.narrow(0, tensor_start, 1),
            lambda: torch.narrow(tensor, 0, tensor_start, 1),
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^narrow\(\): tensor-valued start is not supported$",
            ):
                call()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode, self.assertRaisesRegex(
            NotImplementedError,
            r"^narrow\(\): __torch_function__ modes are not supported$",
        ):
            tensor.narrow(0, 0, 1)
        self.assertEqual(mode.calls, [])

        mode = RecordingMode()
        with mode, self.assertRaisesRegex(
            NotImplementedError,
            r"^narrow\(\): __torch_function__ modes are not supported$",
        ):
            torch.narrow(tensor, 0, 0, 1)
        self.assertEqual(mode.calls, [])

        class OverrideTensor:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return object()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^narrow\(\): only exact native CPU float32 Tensor inputs are supported$",
        ):
            torch.narrow(OverrideTensor(), 0, 0, 1)

        for owner, name in (
            (torch, "narrow_copy"),
            (torch.Tensor, "narrow_copy"),
            (torch, "slice_scatter"),
            (torch.Tensor, "slice_scatter"),
        ):
            with self.subTest(owner=owner, name=name):
                self.assertFalse(hasattr(owner, name))

    def test_autograd_no_grad_and_backward_through_sum(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = source.narrow(0, 1, 2)
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)
        self.assertEqual(selected.output_nr, 0)
        self.assertEqual(selected.storage_offset(), 28)
        selected.sum().backward()
        expected_gradient = [0.0] * 48
        for index in (
            *range(28, 32),
            *range(40, 44),
            *range(32, 36),
            *range(44, 48),
        ):
            expected_gradient[index] = 2.0
        self.assertEqual(leaf.grad.tolist(), expected_gradient)

        no_grad_source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            untracked = no_grad_source.narrow(dim=0, start=-1, length=1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertEqual(untracked.storage_offset(), 2)
        self.assertEqual(untracked.data_ptr(), no_grad_source[1].data_ptr())

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.narrow(0, 1, 1).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_callable_metadata_imports_exports_reload_copy_and_pickle(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "narrow")
        bound = tensor.narrow

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "narrow")
        self.assertEqual(descriptor.__qualname__, "TensorBase.narrow")
        self.assertEqual(bound.__name__, "narrow")
        self.assertEqual(bound.__qualname__, "Tensor.narrow")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor),
            "<method 'narrow' of 'torch._C.TensorBase' objects>",
        )
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        self.assertIs(copy.copy(bound), bound)
        self.assertIs(copy.deepcopy(bound), bound)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor(tensor, 0, 0, 1).shape, (1, 3))

        function = torch.narrow
        self.assertIs(function, torch._C.narrow)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "narrow")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.narrow")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method narrow of type object at 0x[0-9a-f]+>$",
        )
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.narrow, function)

        from torch_rs import narrow as imported_narrow
        from torch_rs._C import narrow as native_narrow

        self.assertIs(imported_narrow, function)
        self.assertIs(native_narrow, function)
        self.assertEqual(torch.__all__.count("narrow"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["narrow"], function)

        for callable_object in (descriptor, function):
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(callable=callable_object.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(callable_object, protocol=protocol)),
                        callable_object,
                    )

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(inspect.getattr_static(torch.Tensor, "narrow"), descriptor)
        self.assertIs(torch.narrow, function)

    def test_integer_protocol_conversion_order(self):
        tensor = torch.zeros((3, 2))

        class IntegerSubclass(int):
            pass

        self.assertEqual(tensor.narrow(IntegerSubclass(0), np.int64(1), np.int32(2)).shape, (2, 2))

        calls = []

        class StatefulStart:
            def __index__(self):
                calls.append("start")
                return 1

        class StatefulLength:
            def __index__(self):
                calls.append("length")
                return 2

        narrowed = tensor.narrow(0, StatefulStart(), StatefulLength())
        self.assertEqual(calls, ["start", "length", "length", "length", "start", "start"])
        self.assertEqual(narrowed.storage_offset(), 2)


if __name__ == "__main__":
    unittest.main()
