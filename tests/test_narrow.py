import copy
import importlib
import inspect
import pickle
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
    values = [float(value) for value in range(48)]
    return torch.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
        1
    ].transpose(0, 1)


class TensorNarrowTests(unittest.TestCase):
    def assert_narrow_view(self, actual, source, expected):
        self.assertEqual(actual.tolist(), expected["values"])
        self.assertEqual(actual.shape, expected["shape"])
        self.assertEqual(actual.stride(), expected["stride"])
        self.assertEqual(actual.storage_offset(), expected["storage_offset"])
        self.assertEqual(actual.data_ptr(), expected["data_ptr"])
        self.assertIs(actual.dtype, source.dtype)
        self.assertEqual(actual.device, source.device)

    def test_contiguous_offset_noncontiguous_and_empty_views(self):
        contiguous = torch.tensor([float(value) for value in range(12)]).reshape(4, 3)
        self.assert_narrow_view(
            contiguous.narrow(0, 1, 2),
            contiguous,
            {
                "values": [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                "shape": (2, 3),
                "stride": (3, 1),
                "storage_offset": 3,
                "data_ptr": contiguous.select(0, 1).data_ptr(),
            },
        )
        self.assertTrue(contiguous.narrow(0, 0, 4).is_set_to(contiguous))

        offset = torch.tensor([float(value) for value in range(24)]).reshape(2, 3, 4)[
            1
        ]
        self.assert_narrow_view(
            offset.narrow(dim=0, start=1, length=2),
            offset,
            {
                "values": [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
                "shape": (2, 4),
                "stride": (4, 1),
                "storage_offset": 16,
                "data_ptr": offset.select(0, 1).data_ptr(),
            },
        )

        noncontiguous = offset_noncontiguous_source()
        narrowed = noncontiguous.narrow(-3, -2, 2)
        self.assert_narrow_view(
            narrowed,
            noncontiguous,
            {
                "values": [
                    [[28.0, 29.0, 30.0, 31.0], [40.0, 41.0, 42.0, 43.0]],
                    [[32.0, 33.0, 34.0, 35.0], [44.0, 45.0, 46.0, 47.0]],
                ],
                "shape": (2, 2, 4),
                "stride": (4, 12, 1),
                "storage_offset": 28,
                "data_ptr": noncontiguous.select(0, 1).data_ptr(),
            },
        )

        empty_length = noncontiguous.narrow(0, 2, 0)
        self.assertEqual(empty_length.tolist(), [])
        self.assertEqual(empty_length.shape, (0, 2, 4))
        self.assertEqual(empty_length.stride(), (4, 12, 1))
        self.assertEqual(empty_length.storage_offset(), 32)
        self.assertEqual(empty_length.data_ptr(), 0)

        zero_sized_source = torch.zeros((0, 2))
        zero_sized_view = zero_sized_source.narrow(0, 0, 0)
        self.assertEqual(zero_sized_view.tolist(), [])
        self.assertEqual(zero_sized_view.shape, (0, 2))
        self.assertEqual(zero_sized_view.stride(), (2, 1))
        self.assertEqual(zero_sized_view.storage_offset(), 0)
        self.assertEqual(zero_sized_view.data_ptr(), 0)
        self.assertTrue(zero_sized_view.is_set_to(zero_sized_source))

        empty_inner = torch.zeros((2, 0, 3)).narrow(0, 1, 1)
        self.assertEqual(empty_inner.tolist(), [[]])
        self.assertEqual(empty_inner.shape, (1, 0, 3))
        self.assertEqual(empty_inner.stride(), (3, 3, 1))
        self.assertEqual(empty_inner.storage_offset(), 3)
        self.assertEqual(empty_inner.data_ptr(), 0)

    def test_top_level_call_forms_reuse_first_axis_narrow_views(self):
        source = offset_noncontiguous_source()
        expected = {
            "values": [
                [[28.0, 29.0, 30.0, 31.0], [40.0, 41.0, 42.0, 43.0]],
                [[32.0, 33.0, 34.0, 35.0], [44.0, 45.0, 46.0, 47.0]],
            ],
            "shape": (2, 2, 4),
            "stride": (4, 12, 1),
            "storage_offset": 28,
            "data_ptr": source.select(0, 1).data_ptr(),
        }
        calls = (
            ("positional", lambda: torch.narrow(source, 0, 1, 2)),
            ("mixed", lambda: torch.narrow(source, 0, 1, length=2)),
            ("keywords", lambda: torch.narrow(source, dim=0, start=1, length=2)),
            (
                "all keywords",
                lambda: torch.narrow(input=source, dim=0, start=1, length=2),
            ),
            (
                "reordered keywords",
                lambda: torch.narrow(length=2, input=source, start=1, dim=0),
            ),
            ("input alias x", lambda: torch.narrow(x=source, dim=0, start=1, length=2)),
            ("input alias a", lambda: torch.narrow(a=source, dim=0, start=1, length=2)),
            ("input alias x1", lambda: torch.narrow(x1=source, dim=0, start=1, length=2)),
            ("negative dimension", lambda: torch.narrow(source, -3, 1, 2)),
            ("negative start", lambda: torch.narrow(source, 0, -2, 2)),
        )
        for case, call in calls:
            with self.subTest(case=case):
                self.assert_narrow_view(call(), source, expected)

    def test_bounds_and_argument_errors_are_explicit(self):
        tensor = torch.zeros((2, 3, 4))
        scalar = torch.tensor(1.0)
        cases = (
            (
                lambda: tensor.narrow(),
                TypeError,
                'narrow() missing 3 required positional argument: "dim", "start", "length"',
            ),
            (
                lambda: tensor.narrow(0),
                TypeError,
                'narrow() missing 2 required positional argument: "start", "length"',
            ),
            (
                lambda: tensor.narrow(0, 0),
                TypeError,
                'narrow() missing 1 required positional arguments: "length"',
            ),
            (
                lambda: tensor.narrow(start=0, length=1),
                TypeError,
                'narrow() missing 3 required positional argument: "dim", "start", "length"',
            ),
            (
                lambda: tensor.narrow(0, 0, 1, 2),
                TypeError,
                "narrow() takes 3 positional arguments but 4 were given",
            ),
            (
                lambda: tensor.narrow(0, 0, 1, dim=0),
                TypeError,
                "narrow() got multiple values for argument 'dim'",
            ),
            (
                lambda: tensor.narrow(0, 0, 1, start=0),
                TypeError,
                "narrow() got multiple values for argument 'start'",
            ),
            (
                lambda: tensor.narrow(0, 0, 1, length=1),
                TypeError,
                "narrow() got multiple values for argument 'length'",
            ),
            (
                lambda: tensor.narrow(0, 0, 1, extra=0),
                TypeError,
                "narrow() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.narrow(None, 0, 1),
                TypeError,
                "narrow(): argument 'dim' (position 1) must be int, not NoneType",
            ),
            (
                lambda: tensor.narrow(dim="0", start=0, length=1),
                TypeError,
                "narrow(): argument 'dim' must be int, not str",
            ),
            (
                lambda: tensor.narrow(0, True, 1),
                TypeError,
                "narrow(): argument 'start' (position 2) must be int, not bool",
            ),
            (
                lambda: tensor.narrow(0, 0, True),
                TypeError,
                "narrow(): argument 'length' (position 3) must be int, not bool",
            ),
            (
                lambda: tensor.narrow(0, 1.0, 1),
                TypeError,
                "narrow(): argument 'start' (position 2) must be int, not float",
            ),
            (
                lambda: tensor.narrow(0, 0, 1.0),
                TypeError,
                "narrow(): argument 'length' (position 3) must be int, not float",
            ),
            (
                lambda: tensor.narrow(0, torch.tensor(1.0), 1),
                NotImplementedError,
                "narrow(): tensor-valued start is not supported",
            ),
            (
                lambda: tensor.narrow(3, 0, 1),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: tensor.narrow(1, 0, 1),
                RuntimeError,
                "Tensor.narrow only supports dimension 0",
            ),
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
                lambda: tensor.narrow(0, -3, 1),
                IndexError,
                "start out of range (expected to be in range of [-2, 2], but got -3)",
            ),
            (
                lambda: tensor.narrow(0, 1, 2),
                RuntimeError,
                "start (1) + length (2) exceeds dimension size (2).",
            ),
            (
                lambda: scalar.narrow(0, 0, 0),
                RuntimeError,
                "narrow() cannot be applied to a 0-dim tensor.",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

        for call in (
            lambda: tensor.narrow(2**100, 0, 1),
            lambda: tensor.narrow(0, 2**100, 1),
            lambda: tensor.narrow(0, 0, 2**100),
        ):
            with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
                call()

        self.assertEqual(tensor.narrow(np.int64(0), np.int32(0), np.uint32(1)).shape, (1, 3, 4))
        self.assertEqual(tensor.narrow(0, 2, 0).shape, (0, 3, 4))
        self.assertEqual(torch.zeros((0, 2)).narrow(0, 0, 0).shape, (0, 2))

    def test_top_level_bounds_and_argument_errors_are_explicit(self):
        tensor = torch.zeros((2, 3, 4))
        cases = (
            (
                lambda: torch.narrow(),
                TypeError,
                'narrow() missing 4 required positional argument: "input", "dim", "start", "length"',
            ),
            (
                lambda: torch.narrow(tensor),
                TypeError,
                'narrow() missing 3 required positional argument: "dim", "start", "length"',
            ),
            (
                lambda: torch.narrow(tensor, 0),
                TypeError,
                'narrow() missing 2 required positional argument: "start", "length"',
            ),
            (
                lambda: torch.narrow(tensor, 0, 0),
                TypeError,
                'narrow() missing 1 required positional arguments: "length"',
            ),
            (
                lambda: torch.narrow(dim=0, start=0, length=1),
                TypeError,
                'narrow() missing 4 required positional argument: "input", "dim", "start", "length"',
            ),
            (
                lambda: torch.narrow(tensor, 0, 0, 1, 2),
                TypeError,
                "narrow() takes 4 positional arguments but 5 were given",
            ),
            (
                lambda: torch.narrow(tensor, 0, 0, 1, input=tensor),
                TypeError,
                "narrow() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.narrow(tensor, 0, 0, 1, dim=0),
                TypeError,
                "narrow() got multiple values for argument 'dim'",
            ),
            (
                lambda: torch.narrow(tensor, 0, 0, 1, start=0),
                TypeError,
                "narrow() got multiple values for argument 'start'",
            ),
            (
                lambda: torch.narrow(tensor, 0, 0, 1, length=1),
                TypeError,
                "narrow() got multiple values for argument 'length'",
            ),
            (
                lambda: torch.narrow(tensor, 0, 0, 1, extra=0),
                TypeError,
                "narrow() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.narrow([], 0, 0, 1),
                TypeError,
                "narrow(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.narrow(tensor, 1, 0, 1),
                RuntimeError,
                "torch.narrow only supports dimension 0",
            ),
            (
                lambda: torch.narrow(tensor, 0, 1, 2),
                RuntimeError,
                "start (1) + length (2) exceeds dimension size (2).",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message), self.assertRaises(error_type) as raised:
                call()
            self.assertEqual(str(raised.exception), message)

    def test_no_grad_and_backward_through_sum(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        narrowed = source.narrow(0, 1, 2)

        self.assertTrue(narrowed.requires_grad)
        self.assertFalse(narrowed.is_leaf)
        self.assertEqual(narrowed.output_nr, 0)

        (narrowed * 3.0).sum().backward()
        expected_gradient = [0.0] * 48
        for index in (*range(28, 36), *range(40, 48)):
            expected_gradient[index] = 6.0
        self.assertEqual(leaf.grad.tolist(), expected_gradient)

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = torch.narrow(no_grad_source, 0, 1, 1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.output_nr, 0)
        self.assertEqual(untracked.tolist(), [[3.0, 4.0]])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.narrow(0, 1, 1).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_callable_metadata_import_wildcard_reload_copy_and_pickle(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.narrow

        from torch_rs import narrow as imported_narrow

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "narrow")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.narrow")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertIs(imported_narrow, function)
        self.assertIs(wildcard_namespace["narrow"], function)
        self.assertEqual(package.__all__.count("narrow"), 1)
        self.assertIs(native.narrow, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.narrow, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.narrow, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.narrow, function)
        self.assertEqual(package.__all__.count("narrow"), 1)

    def test_tensorbase_descriptor_metadata(self):
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
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            repr(descriptor),
            "<method 'narrow' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.narrow, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(descriptor(tensor, 0, 0, 1).shape, (1, 3))
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

    def test_subclasses_modes_narrow_copy_and_mutating_slice_apis_stay_unsupported(self):
        tensor = torch.zeros((2, 3))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        for call in (lambda: tensor.narrow(0, 0, 1), lambda: torch.narrow(tensor, 0, 0, 1)):
            mode = RecordingMode()
            with self.subTest(call=call), self.assertRaisesRegex(
                TypeError, r"^narrow\(\) does not support an active TorchFunctionMode$"
            ):
                with mode:
                    call()
            self.assertEqual(mode.calls, [])

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        with self.assertRaisesRegex(
            TypeError,
            r"^narrow\(\): argument 'input' \(position 1\) must be Tensor, not Override$",
        ):
            torch.narrow(Override(), 0, 0, 1)
        self.assertEqual(Override.calls, 0)

        self.assertFalse(hasattr(torch, "narrow_copy"))
        self.assertFalse(hasattr(torch.Tensor, "narrow_copy"))
        for name in ("slice_scatter", "select_scatter"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(torch.Tensor, name))


if __name__ == "__main__":
    unittest.main()
