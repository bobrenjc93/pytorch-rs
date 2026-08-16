import enum
import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nsize(dim=None) -> torch.Size or int\n\n"
    "Returns the size of the :attr:`self` tensor. If ``dim`` is not specified,\n"
    "the returned value is a :class:`torch.Size`, a subclass of :class:`tuple`.\n"
    "If ``dim`` is specified, returns an int holding the size of that dimension.\n\n"
    "Args:\n"
    "  dim (int, optional): The dimension for which to retrieve the size.\n\n"
    "Example::\n\n"
    "    >>> t = torch.empty(3, 4, 5)\n"
    "    >>> t.size()\n"
    "    torch.Size([3, 4, 5])\n"
    "    >>> t.size(dim=1)\n"
    "    4\n\n"
)


class Dimension(enum.IntEnum):
    FIRST = 0
    SECOND = 1


class IntegerSubclass(int):
    pass


class CustomIndex:
    def __init__(self):
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return 1


class TensorSizeTests(unittest.TestCase):
    def metadata_cases(self):
        base = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ]
        )
        offset = base[1]
        strided = base.transpose(0, 2)
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(strided.is_contiguous())
        return (
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", offset),
            ("strided", strided),
            ("extreme empty", extreme_empty),
        )

    def test_positive_and_negative_dimensions_read_native_shape_metadata(self):
        for case, tensor in self.metadata_cases():
            shape = tuple(tensor.shape)
            metadata = (
                shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.requires_grad,
                tensor.is_leaf,
            )
            for axis, expected in enumerate(shape):
                for dimension in (axis, axis - len(shape)):
                    with self.subTest(case=case, dimension=dimension):
                        result = tensor.size(dimension)
                        keyword_result = tensor.size(dim=dimension)
                        self.assertIs(type(result), int)
                        self.assertIs(type(keyword_result), int)
                        self.assertEqual(result, expected)
                        self.assertEqual(keyword_result, expected)
            self.assertEqual(
                (
                    tuple(tensor.shape),
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.requires_grad,
                    tensor.is_leaf,
                ),
                metadata,
            )

    def test_scalar_and_out_of_range_dimensions_match_pytorch_errors(self):
        scalar = torch.tensor(3.0)
        for dimension in (-1, 0):
            for keyword in (False, True):
                with self.subTest(
                    kind="scalar", dimension=dimension, keyword=keyword
                ):
                    with self.assertRaises(IndexError) as raised:
                        if keyword:
                            scalar.size(dim=dimension)
                        else:
                            scalar.size(dimension)
                    self.assertEqual(
                        str(raised.exception),
                        f"Dimension specified as {dimension} but tensor has no dimensions",
                    )

        tensor = torch.zeros((2, 0, 3))
        for dimension in (-4, 3, -(2**63), 2**63 - 1):
            with self.subTest(kind="range", dimension=dimension):
                with self.assertRaises(IndexError) as raised:
                    tensor.size(dim=dimension)
                self.assertEqual(
                    str(raised.exception),
                    "Dimension out of range (expected to be in range of "
                    f"[-3, 2], but got {dimension})",
                )

    def test_python_and_numpy_integer_scalars_are_accepted(self):
        tensor = torch.zeros((2, 3, 4))
        cases = (
            (0, 2),
            (IntegerSubclass(1), 3),
            (Dimension.FIRST, 2),
            (Dimension.SECOND, 3),
            (np.int8(-1), 4),
            (np.int64(-2), 3),
            (np.uint64(2), 4),
        )
        for dimension, expected in cases:
            for keyword in (False, True):
                with self.subTest(
                    dimension=repr(dimension), keyword=keyword
                ):
                    result = (
                        tensor.size(dim=dimension)
                        if keyword
                        else tensor.size(dimension)
                    )
                    self.assertIs(type(result), int)
                    self.assertEqual(result, expected)

    def test_bool_and_custom_index_objects_are_rejected_without_conversion(self):
        tensor = torch.zeros((2, 3))
        custom = CustomIndex()
        cases = (
            (True, "bool"),
            (np.bool_(False), "numpy.bool"),
            (custom, "CustomIndex"),
            (1.0, "float"),
            ("1", "str"),
        )
        for value, type_name in cases:
            for keyword in (False, True):
                position = "" if keyword else " (position 1)"
                with self.subTest(value=repr(value), keyword=keyword):
                    with self.assertRaises(TypeError) as raised:
                        if keyword:
                            tensor.size(dim=value)
                        else:
                            tensor.size(value)
                    self.assertEqual(
                        str(raised.exception),
                        f"size(): argument 'dim'{position} must be int, "
                        f"not {type_name}",
                    )
        self.assertEqual(custom.calls, 0)

    def test_integer_overflow_uses_pytorch_long_long_error(self):
        tensor = torch.zeros((2, 3))
        for dimension in (2**63, -(2**63) - 1, np.uint64(2**63)):
            for keyword in (False, True):
                with self.subTest(
                    dimension=repr(dimension), keyword=keyword
                ):
                    with self.assertRaises(ValueError) as raised:
                        if keyword:
                            tensor.size(dim=dimension)
                        else:
                            tensor.size(dimension)
                    self.assertEqual(
                        str(raised.exception),
                        "Overflow when unpacking long long",
                    )

    def test_no_argument_overload_stays_out_of_scope(self):
        tensor = torch.zeros((2, 3))
        self.assertFalse(hasattr(torch, "Size"))
        with self.assertRaises(TypeError) as raised:
            tensor.size()
        self.assertEqual(
            str(raised.exception),
            'size() missing 1 required positional arguments: "dim"',
        )
        for call, message in (
            (
                lambda: tensor.size(None),
                "size(): argument 'dim' (position 1) must be int, not NoneType",
            ),
            (
                lambda: tensor.size(dim=None),
                "size(): argument 'dim' must be int, not NoneType",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_tensorbase_descriptor_metadata_and_binding_errors(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "size")
        bound = tensor.size

        self.assertFalse(hasattr(torch, "size"))
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertTrue(callable(descriptor))
        self.assertEqual(descriptor.__name__, "size")
        self.assertEqual(descriptor.__qualname__, "TensorBase.size")
        self.assertEqual(bound.__name__, "size")
        self.assertEqual(bound.__qualname__, "Tensor.size")
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
            "<method 'size' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.size, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(descriptor(tensor, 0), 2)
        self.assertEqual(descriptor(tensor, dim=-1), 3)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        cases = (
            (
                lambda: tensor.size(0, 1),
                "size() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: tensor.size(0, dim=1),
                "size() got multiple values for argument 'dim'",
            ),
            (
                lambda: tensor.size(foo=0),
                "size() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: tensor.size(dim=0, foo=1),
                "size() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: tensor.size(True, dim=0),
                "size(): argument 'dim' (position 1) must be int, not bool",
            ),
            (
                lambda: tensor.size(2**63, foo=1),
                "size() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.size() needs an argument",
            ),
            (
                lambda: descriptor(1, 0),
                "descriptor 'size' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, dim=0),
                "unbound method TensorBase.size() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_original_call_and_forward(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "size")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        positional = RecordingMode(marker)
        with positional:
            positional_result = tensor.size(1)
        self.assertIs(positional_result, marker)
        self.assertEqual(len(positional.calls), 1)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], tensor)
        self.assertEqual(args[1], 1)
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            keyword_result = tensor.size(dim=-1)
        self.assertIs(keyword_result, marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertEqual(kwargs, {"dim": -1})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.size(dim=-1)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(type(forwarded), int)
        self.assertEqual(forwarded, 3)

    def test_mode_dispatch_precedes_conversion_but_follows_type_binding(self):
        tensor = torch.zeros((2, 3))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for dimension in (2**63, 10):
            mode = RecordingMode(marker)
            with mode:
                result = tensor.size(dimension)
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)

        custom = CustomIndex()
        for dimension in (True, custom):
            mode = RecordingMode(marker)
            with self.assertRaises(TypeError):
                with mode:
                    tensor.size(dimension)
            self.assertEqual(mode.calls, [])
        self.assertEqual(custom.calls, 0)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.size(0)
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.size'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])


if __name__ == "__main__":
    unittest.main()
