import gc
import inspect
import pickle
import re
import sys
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


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class TopLevelReshapeTests(unittest.TestCase):
    def assert_reshape_result(self, result, direct, source, *, aliases):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, direct.shape)
        self.assertEqual(result.stride(), direct.stride())
        self.assertEqual(result.storage_offset(), direct.storage_offset())
        self.assertEqual(result.is_contiguous(), direct.is_contiguous())
        self.assertEqual(result.requires_grad, direct.requires_grad)
        self.assertEqual(result.is_leaf, direct.is_leaf)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result.data_ptr() == source.data_ptr(), aliases)
        self.assertEqual(result.is_set_to(direct), aliases)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))

    def reshape_calls(self, source, shape):
        return (
            ("positional tuple", torch.reshape(source, tuple(shape))),
            ("positional list", torch.reshape(source, list(shape))),
            ("positional Size", torch.reshape(source, torch.Size(shape))),
            ("canonical keywords", torch.reshape(input=source, shape=tuple(shape))),
            ("x alias", torch.reshape(x=source, shape=list(shape))),
            ("a alias", torch.reshape(a=source, shape=tuple(shape))),
            ("x1 alias", torch.reshape(x1=source, shape=torch.Size(shape))),
        )

    def test_call_forms_delegate_to_native_reshape_for_supported_layouts(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist(), requires_grad=True)
        noncontiguous = base.transpose(0, 1)
        cases = (
            ("scalar", torch.tensor(-0.0), (), True),
            ("contiguous", base, (6, 4), True),
            ("contiguous-offset", base[1], (2, 6), True),
            ("noncontiguous-same-shape", noncontiguous, (3, 2, 4), True),
            ("noncontiguous-copy", base.transpose(0, 2), (6, 4), False),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                (2, 0),
                True,
            ),
        )

        for case, source, shape, aliases in cases:
            direct = source.reshape(shape)
            for form, result in self.reshape_calls(source, shape):
                with self.subTest(case=case, form=form):
                    self.assert_reshape_result(result, direct, source, aliases=aliases)

        inferred = torch.reshape(base, (2, -1, 2))
        self.assertEqual(inferred.shape, (2, 6, 2))
        self.assertEqual(inferred.stride(), (12, 2, 1))
        self.assertEqual(inferred.data_ptr(), base.data_ptr())

        indexed = torch.reshape(
            torch.zeros((6,)), (IntSubclass(2), np.int64(3))
        )
        self.assertEqual(indexed.shape, (2, 3))

        index_object = torch.reshape(torch.zeros((6,)), [IndexDimension(2), 3])
        self.assertEqual(index_object.shape, (2, 3))

        bool_after_first = torch.reshape(torch.zeros((2, 1)), (2, True))
        self.assertEqual(bool_after_first.shape, (2, 1))

        maximum = sys.maxsize
        extreme = torch.reshape(torch.zeros((0,)), (0, maximum, maximum))
        self.assertEqual(extreme.shape, (0, maximum, maximum))
        self.assertEqual(extreme.stride(), (1, maximum, 1))
        self.assertEqual(extreme.numel(), 0)

        negative_zero = torch.reshape(torch.tensor(-0.0), ())
        self.assertEqual(np.asarray(negative_zero).view(np.uint32).item(), 0x8000_0000)

    def test_view_and_copy_outputs_outlive_their_sources(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            temporary = torch.tensor(values.tolist())
            return torch.reshape(temporary[1], (2, 6))

        def retained_copy():
            temporary = torch.tensor(values.tolist())
            return torch.reshape(temporary.transpose(0, 2), (6, 4))

        view = retained_view()
        copied = retained_copy()
        gc.collect()

        np.testing.assert_array_equal(np.asarray(view), values[1].reshape(2, 6))
        np.testing.assert_array_equal(
            np.asarray(copied), values.transpose(2, 1, 0).reshape(6, 4)
        )

    def test_autograd_empty_backward_and_no_grad_match_tensor_reshape(self):
        for case, transpose, shape, weights, expected_gradient, aliases in (
            (
                "view",
                False,
                (3, 2),
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                True,
            ),
            (
                "copy",
                True,
                (6,),
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
                False,
            ),
        ):
            with self.subTest(case=case):
                leaf = torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
                )
                source = leaf.transpose(0, 1) if transpose else leaf
                result = torch.reshape(source, shape)
                direct = source.reshape(shape)
                self.assert_reshape_result(result, direct, source, aliases=aliases)

                (result * torch.tensor(weights)).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), np.asarray(expected_gradient)
                )

        scalar = torch.tensor(2.0, requires_grad=True)
        (torch.reshape(input=scalar, shape=()) * 7.0).sum().backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.reshape(x=empty, shape=(0, 6)).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))

        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            alias = torch.reshape(leaf, (4,))
            copied = torch.reshape(leaf.transpose(0, 1), (4,))
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertEqual(alias.data_ptr(), leaf.data_ptr())
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)
        self.assertNotEqual(copied.data_ptr(), leaf.data_ptr())

    def test_overrides_and_modes_receive_the_original_top_level_call(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        calls = (
            ("positional", lambda: torch.reshape(value, (1,)), (value, (1,)), None),
            (
                "input",
                lambda: torch.reshape(input=value, shape=(1,)),
                (),
                {"input": value, "shape": (1,)},
            ),
            (
                "x",
                lambda: torch.reshape(x=value, shape=(1,)),
                (),
                {"x": value, "shape": (1,)},
            ),
            (
                "a",
                lambda: torch.reshape(a=value, shape=(1,)),
                (),
                {"a": value, "shape": (1,)},
            ),
            (
                "x1",
                lambda: torch.reshape(x1=value, shape=(1,)),
                (),
                {"x1": value, "shape": (1,)},
            ),
        )
        for form, call, expected_args, expected_kwargs in calls:
            with self.subTest(form=form):
                Override.calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.reshape)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        tensor = torch.tensor([1.0, 2.0])
        shape = Override()
        Override.calls.clear()
        self.assertIs(torch.reshape(tensor, shape), marker)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, torch.reshape)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (tensor, shape))
        self.assertIsNone(kwargs)

        events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                events.append(("left", func, dispatch_types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                events.append(("right", func, dispatch_types, args, kwargs))
                return marker

        self.assertIs(torch.reshape(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.reshape)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.reshape(input=tensor, shape=(2, 1)), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.reshape)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "shape": (2, 1)})

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = torch.reshape(tensor, (2, 1))
        self.assertEqual(forwarded.shape, (2, 1))
        np.testing.assert_array_equal(np.asarray(forwarded), [[1.0], [2.0]])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.reshape'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            "  - tensor subclass <class "
            f"'{DecliningOverride.__module__}.{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.reshape(DecliningOverride(), (1,))

    def test_binding_and_shape_type_errors_match_the_generated_schema(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.reshape(),
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor),
                'reshape() missing 1 required positional arguments: "shape"',
            ),
            (
                lambda: torch.reshape(shape=(1,)),
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor, (1,), None),
                "reshape() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.reshape(tensor, (1,), input=tensor),
                "reshape() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reshape(tensor, (1,), shape=(1,)),
                "reshape() got multiple values for argument 'shape'",
            ),
            (
                lambda: torch.reshape(tensor, (1,), extra=True),
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape(x=tensor, shape=(1,), extra=True),
                "reshape() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.reshape(1, (1,)),
                "reshape(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.reshape(input=1, shape=(1,)),
                "reshape(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.reshape(tensor, 1),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, not int",
            ),
            (
                lambda: torch.reshape(input=tensor, shape=1),
                "reshape(): argument 'shape' must be tuple of ints, not int",
            ),
            (
                lambda: torch.reshape(tensor, (1.0,)),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type float at pos 0",
            ),
            (
                lambda: torch.reshape(input=tensor, shape=(1.0,)),
                "reshape(): argument 'shape' must be tuple of ints, not tuple",
            ),
            (
                lambda: torch.reshape(torch.zeros((2,)), (1, 1.0)),
                "reshape(): argument 'shape' failed to unpack the object at pos 2 "
                'with error "type must be tuple of ints,but got float"',
            ),
            (
                lambda: torch.reshape(torch.zeros((2,)), (True, 2)),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type bool at pos 0",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_callable_metadata_documentation_ownership_exports_and_pickling(self):
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
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("reshape"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["reshape"], function)

        message = (
            "cannot set 'reshape' attribute of immutable type "
            "'torch_rs._C._VariableFunctionsClass'"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            owner.reshape = None
        self.assertIs(owner.reshape, function)


if __name__ == "__main__":
    unittest.main()
