import inspect
import pickle
import re
import types
import unittest

import numpy as np

import torch_rs as torch


FUNCTION_DOC = (
    "\nreshape(input, shape) -> Tensor\n\n"
    "Returns a tensor with the same data and number of elements as :attr:`input`,\n"
    "but with the specified shape. When possible, the returned tensor will be a view\n"
    "of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs\n"
    "with compatible strides can be reshaped without copying, but you should not\n"
    "depend on the copying vs. viewing behavior.\n\n"
    "See :meth:`torch.Tensor.view` on when it is possible to return a view.\n\n"
    "A single dimension may be -1, in which case it's inferred from the remaining\n"
    "dimensions and the number of elements in :attr:`input`.\n\n"
    "Args:\n"
    "    input (Tensor): the tensor to be reshaped\n"
    "    shape (tuple of int): the new shape\n\n"
    "Example::\n\n"
    "    >>> a = torch.arange(4.)\n"
    "    >>> torch.reshape(a, (2, 2))\n"
    "    tensor([[ 0.,  1.],\n"
    "            [ 2.,  3.]])\n"
    "    >>> b = torch.tensor([[0, 1], [2, 3]])\n"
    "    >>> torch.reshape(b, (-1,))\n"
    "    tensor([ 0,  1,  2,  3])\n"
)


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class TensorReshapeTests(unittest.TestCase):
    def assert_layout(self, tensor, *, shape, stride, offset, values):
        self.assertEqual(tuple(tensor.shape), shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), offset)
        np.testing.assert_array_equal(
            np.asarray(tensor), np.asarray(values, dtype=np.float32)
        )

    def test_top_level_call_forms_and_supported_shape_sequences(self):
        source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        calls = (
            torch.reshape(source, (3, 2)),
            torch.reshape(source, [3, 2]),
            torch.reshape(source, torch.Size([3, 2])),
            torch.reshape(source, shape=(3, 2)),
            torch.reshape(input=source, shape=[3, 2]),
            torch.reshape(shape=torch.Size([3, 2]), input=source),
            torch.reshape(x=source, shape=(3, 2)),
            torch.reshape(a=source, shape=(3, 2)),
            torch.reshape(x1=source, shape=(3, 2)),
        )
        for output in calls:
            self.assertIsNot(output, source)
            self.assert_layout(
                output,
                shape=(3, 2),
                stride=(2, 1),
                offset=0,
                values=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            )

        inferred = torch.reshape(source, (3, -1))
        self.assertEqual(inferred.shape, (3, 2))
        indexed = torch.reshape(source, (np.int64(2), IndexDimension(3)))
        self.assertEqual(indexed.shape, (2, 3))

        scalar = torch.reshape(torch.tensor([7.0]), ())
        self.assert_layout(scalar, shape=(), stride=(), offset=0, values=7.0)

        empty = torch.reshape(torch.zeros((0,)), torch.Size([2, 0, 3]))
        self.assert_layout(
            empty,
            shape=(2, 0, 3),
            stride=(3, 3, 1),
            offset=0,
            values=np.empty((2, 0, 3)),
        )

    def test_top_level_reuses_view_or_copy_reshape_paths(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())

        same_shape = torch.reshape(base, base.shape)
        self.assertIsNot(same_shape, base)
        self.assertTrue(same_shape.is_set_to(base))

        offset_source = base[1]
        view = torch.reshape(offset_source, (2, 6))
        method_view = offset_source.reshape((2, 6))
        self.assert_layout(
            view,
            shape=(2, 6),
            stride=(6, 1),
            offset=12,
            values=values[1].reshape(2, 6),
        )
        self.assertEqual(view.data_ptr(), offset_source.data_ptr())
        self.assertTrue(view.is_set_to(method_view))

        non_contiguous = base.transpose(0, 1)
        copied = torch.reshape(non_contiguous, (6, 4))
        method_copy = non_contiguous.reshape((6, 4))
        self.assert_layout(
            copied,
            shape=(6, 4),
            stride=(4, 1),
            offset=0,
            values=values.transpose(1, 0, 2).reshape(6, 4),
        )
        self.assertNotEqual(copied.data_ptr(), non_contiguous.data_ptr())
        self.assertEqual(copied.tolist(), method_copy.tolist())
        self.assertEqual(copied.stride(), method_copy.stride())

    def test_top_level_autograd_and_no_grad_reuse_tensor_reshape(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = torch.reshape(leaf.transpose(0, 1), (6,))
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, 0)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()
        self.assertEqual(
            leaf.grad.tolist(),
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
        )

        view_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            view = torch.reshape(view_leaf, (2, 1))
            copy = torch.reshape(
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
                ).transpose(0, 1),
                (6,),
            )
        self.assertTrue(view.requires_grad)
        self.assertTrue(view.is_leaf)
        self.assertEqual(view.data_ptr(), view_leaf.data_ptr())
        self.assertFalse(copy.requires_grad)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.reshape(empty, (0, 6)).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_top_level_binding_and_shape_errors(self):
        tensor = torch.zeros((6,))
        cases = (
            (
                lambda: torch.reshape(),
                TypeError,
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor),
                TypeError,
                'reshape() missing 1 required positional arguments: "shape"',
            ),
            (
                lambda: torch.reshape(shape=(2, 3)),
                TypeError,
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor, 2, 3),
                TypeError,
                "reshape() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3), input=tensor),
                TypeError,
                "reshape() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3), shape=(2, 3)),
                TypeError,
                "reshape() got multiple values for argument 'shape'",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3), extra=True),
                TypeError,
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape([], (0,)),
                TypeError,
                "reshape(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.reshape(input=None, shape=()),
                TypeError,
                "reshape(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.reshape(tensor, 6),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, not int",
            ),
            (
                lambda: torch.reshape(tensor, shape=6),
                TypeError,
                "reshape(): argument 'shape' must be tuple of ints, not int",
            ),
            (
                lambda: torch.reshape(tensor, (2.0, 3)),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, but found element of type float at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, shape=(2.0, 3)),
                TypeError,
                "reshape(): argument 'shape' must be tuple of ints, not tuple",
            ),
            (
                lambda: torch.reshape(tensor, (2, 3.0)),
                TypeError,
                "reshape(): argument 'shape' failed to unpack the object at pos 2 with error \"type must be tuple of ints,but got float\"",
            ),
            (
                lambda: torch.reshape(tensor, (True, 6)),
                TypeError,
                "reshape(): argument 'shape' (position 2) must be tuple of ints, but found element of type bool at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, (4, 2)),
                RuntimeError,
                "shape '[4, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.reshape(tensor, (-1, -1)),
                RuntimeError,
                "only one dimension can be inferred",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

        later_bool = torch.reshape(torch.zeros((2,)), (2, True))
        self.assertEqual(later_bool.shape, (2, 1))

    def test_top_level_torch_function_modes_and_overrides(self):
        tensor = torch.zeros((6,))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        cases = (
            (lambda: torch.reshape(tensor, (2, 3)), (tensor, (2, 3)), None),
            (
                lambda: torch.reshape(tensor, shape=[2, 3]),
                (tensor,),
                {"shape": [2, 3]},
            ),
            (
                lambda: torch.reshape(input=tensor, shape=torch.Size([2, 3])),
                (),
                {"input": tensor, "shape": torch.Size([2, 3])},
            ),
            (lambda: torch.reshape(tensor, (4, 2)), (tensor, (4, 2)), None),
            (lambda: torch.reshape(tensor, (2, 3.0)), (tensor, (2, 3.0)), None),
        )
        for call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.reshape)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            torch.reshape(tensor, (2.0, 3))
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.reshape(input=tensor, shape=(2, 3))
        self.assertEqual(forwarded.shape, (2, 3))
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        self.assertTrue(all(entry[1] is torch.reshape for entry in order))

        declining = RecordingMode(NotImplemented)
        with declining, self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.reshape'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.reshape(tensor, (2, 3))
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        for call in (
            lambda: torch.reshape(value, (2, 3)),
            lambda: torch.reshape(input=value, shape=(2, 3)),
            lambda: torch.reshape(x=value, shape=(2, 3)),
            lambda: torch.reshape(value, (4, 2)),
            lambda: torch.reshape(value, (2, 3.0)),
        ):
            self.assertIs(call(), marker)
            function, dispatch_types, _, _ = calls[-1]
            self.assertIs(function, torch.reshape)
            self.assertEqual(dispatch_types, (Override,))

        call_count = len(calls)
        with self.assertRaises(TypeError):
            torch.reshape(value, (2.0, 3))
        self.assertEqual(len(calls), call_count)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.reshape'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.reshape(DecliningOverride(), (2, 3))

    def test_top_level_callable_metadata_exports_and_pickling(self):
        function = torch.reshape
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "reshape")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.reshape")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method reshape of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.reshape, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("reshape"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["reshape"], function)


if __name__ == "__main__":
    unittest.main()
