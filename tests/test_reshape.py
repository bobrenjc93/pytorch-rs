import gc
import inspect
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


FUNCTION_DOC = (
    "\nreshape(input, shape) -> Tensor\n\n"
    "Returns a tensor with the same data and number of elements as :attr:`input`,\n"
    "but with the specified shape. When possible, the returned tensor will be a view\n"
    "of :attr:`input`. Otherwise, it will be a copy. Contiguous inputs and inputs\n"
    "with compatible strides can be reshaped without copying, but you should not\n"
    "depend on the copying vs. viewing behavior.\n"
    "\n"
    "See :meth:`torch.Tensor.view` on when it is possible to return a view.\n"
    "\n"
    "A single dimension may be -1, in which case it's inferred from the remaining\n"
    "dimensions and the number of elements in :attr:`input`.\n"
    "\n"
    "Args:\n"
    "    input (Tensor): the tensor to be reshaped\n"
    "    shape (tuple of int): the new shape\n"
    "\n"
    "Example::\n"
    "\n"
    "    >>> a = torch.arange(4.)\n"
    "    >>> torch.reshape(a, (2, 2))\n"
    "    tensor([[ 0.,  1.],\n"
    "            [ 2.,  3.]])\n"
    "    >>> b = torch.tensor([[0, 1], [2, 3]])\n"
    "    >>> torch.reshape(b, (-1,))\n"
    "    tensor([ 0,  1,  2,  3])\n"
)


class TopLevelReshapeTests(unittest.TestCase):
    def assert_matches_method(self, output, source, method_output, *, aliases):
        self.assertIsNot(output, source)
        self.assertEqual(output.shape, method_output.shape)
        self.assertEqual(output.stride(), method_output.stride())
        self.assertEqual(output.storage_offset(), method_output.storage_offset())
        self.assertEqual(output.is_contiguous(), method_output.is_contiguous())
        self.assertEqual(output.requires_grad, method_output.requires_grad)
        self.assertEqual(output.is_leaf, method_output.is_leaf)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertEqual(output.data_ptr() == source.data_ptr(), aliases)
        self.assertEqual(output.is_set_to(method_output), aliases)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(method_output))

    def reshape_calls(self, source, shape):
        return (
            ("positional-tuple", lambda: torch.reshape(source, tuple(shape))),
            ("positional-list", lambda: torch.reshape(source, list(shape))),
            ("positional-size", lambda: torch.reshape(source, torch.Size(shape))),
            ("keyword-tuple", lambda: torch.reshape(input=source, shape=tuple(shape))),
            ("keyword-list", lambda: torch.reshape(input=source, shape=list(shape))),
            ("keyword-size", lambda: torch.reshape(input=source, shape=torch.Size(shape))),
            ("x-alias", lambda: torch.reshape(x=source, shape=tuple(shape))),
            ("a-alias", lambda: torch.reshape(a=source, shape=tuple(shape))),
            ("x1-alias", lambda: torch.reshape(x1=source, shape=tuple(shape))),
        )

    def test_sequence_forms_delegate_to_the_native_reshape_engine(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist(), requires_grad=True)
        cases = (
            ("scalar", torch.tensor(-0.0), (), True),
            ("empty-offset", torch.zeros((2, 0, 3)).transpose(0, 2)[1], (2, 0), True),
            ("contiguous", base, (6, 4), True),
            ("contiguous-offset", base[1], (2, 6), True),
            ("transposed-copy", base.transpose(0, 2), (6, 4), False),
        )

        for case, source, shape, aliases in cases:
            method_output = source.reshape(shape)
            for form, call in self.reshape_calls(source, shape):
                with self.subTest(case=case, form=form):
                    self.assert_matches_method(
                        call(), source, method_output, aliases=aliases
                    )

        negative_zero = torch.reshape(torch.tensor(-0.0), ())
        self.assertEqual(np.asarray(negative_zero).view(np.uint32).item(), 0x8000_0000)

    def test_variadic_dimensions_delegate_to_the_native_reshape_engine(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist(), requires_grad=True)
        noncontiguous = base.transpose(0, 1)
        cases = (
            ("single-inferred", base, (-1,), True),
            ("contiguous", base, (6, 4), True),
            ("inferred", base, (2, -1, 2), True),
            ("offset", base[1], (IntSubclass(2), np.int64(6)), True),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (IndexDimension(2), 0),
                True,
            ),
            ("noncontiguous-compatible", noncontiguous, (3, 2, 2, 2), True),
            ("noncontiguous-copy", base.transpose(0, 2), (6, 4), False),
        )

        for case, source, dimensions, aliases in cases:
            with self.subTest(case=case):
                method_output = source.reshape(dimensions)
                self.assert_matches_method(
                    torch.reshape(source, *dimensions),
                    source,
                    method_output,
                    aliases=aliases,
                )

    def test_inferred_empty_offset_and_shape_errors_match_tensor_reshape(self):
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        inferred = torch.reshape(source, (2, -1, 2))
        direct = source.reshape((2, -1, 2))
        self.assert_matches_method(inferred, source, direct, aliases=True)

        maximum = sys.maxsize
        empty = torch.zeros((0,))
        result = torch.reshape(empty, (0, maximum, maximum))
        direct = empty.reshape((0, maximum, maximum))
        self.assertIsNot(result, empty)
        self.assertEqual(result.shape, direct.shape)
        self.assertEqual(result.stride(), direct.stride())
        self.assertEqual(result.storage_offset(), direct.storage_offset())
        self.assertEqual(result.requires_grad, direct.requires_grad)
        self.assertEqual(result.is_leaf, direct.is_leaf)
        self.assertEqual(result.data_ptr(), empty.data_ptr())
        self.assertTrue(result.is_set_to(direct))
        self.assertEqual(result.tolist(), [])

        for shape in ((2, 2), (-1, -1), (2, -2), (0, -1)):
            with self.subTest(shape=shape):
                with self.assertRaises(RuntimeError) as top_level_raised:
                    torch.reshape(torch.zeros((6,)), shape)
                with self.assertRaises(RuntimeError) as method_raised:
                    torch.zeros((6,)).reshape(shape)
                self.assertEqual(
                    str(top_level_raised.exception), str(method_raised.exception)
                )

        for shape in ((2, 2), (-1, -1), (2, -2)):
            with self.subTest(variadic_shape=shape):
                with self.assertRaises(RuntimeError) as top_level_raised:
                    torch.reshape(torch.zeros((6,)), *shape)
                with self.assertRaises(RuntimeError) as method_raised:
                    torch.zeros((6,)).reshape(shape)
                self.assertEqual(
                    str(top_level_raised.exception), str(method_raised.exception)
                )

        with self.assertRaises(RuntimeError) as top_level_raised:
            torch.reshape(torch.zeros((0,)), 0, -1)
        with self.assertRaises(RuntimeError) as method_raised:
            torch.zeros((0,)).reshape((0, -1))
        self.assertEqual(str(top_level_raised.exception), str(method_raised.exception))

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

    def test_autograd_repeated_backward_and_no_grad_match_tensor_reshape(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        view = torch.reshape(leaf, (3, 2))
        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertEqual(view.data_ptr(), leaf.data_ptr())
        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        (view * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
        )

        copy_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        copied = torch.reshape(copy_leaf.transpose(0, 1), [6])
        self.assertTrue(copied.requires_grad)
        self.assertFalse(copied.is_leaf)
        self.assertNotEqual(copied.data_ptr(), copy_leaf.data_ptr())
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (copied * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(copy_leaf.grad),
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
        )

        repeated_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        loss = torch.reshape(repeated_leaf.transpose(0, 1), (3, 2)).sum()
        loss.backward()
        loss.backward()
        np.testing.assert_array_equal(
            np.asarray(repeated_leaf.grad),
            np.full((2, 3), 2.0, dtype=np.float32),
        )

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            alias = torch.reshape(no_grad_source, (4,))
            copied = torch.reshape(no_grad_source.transpose(0, 1), (4,))
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertEqual(alias.data_ptr(), no_grad_source.data_ptr())
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)

    def test_variadic_full_sum_backward_matches_tensor_reshape(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        loss = torch.reshape(leaf, 3, 2).sum()
        loss.backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.ones((2, 3), dtype=np.float32),
        )

        copy_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        copied_loss = torch.reshape(copy_leaf.transpose(0, 1), 6).sum()
        copied_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(copy_leaf.grad),
            np.ones((2, 3), dtype=np.float32),
        )

    def test_overrides_and_modes_receive_the_original_top_level_call(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        value = Override()
        for form, call, expected_args, expected_kwargs in (
            ("positional", lambda: torch.reshape(value, (2, 2)), (value, (2, 2)), None),
            (
                "positional-variadic",
                lambda: torch.reshape(value, 2, 2),
                (value, 2, 2),
                None,
            ),
            (
                "keyword",
                lambda: torch.reshape(input=value, shape=(2, 2)),
                (),
                {"input": value, "shape": (2, 2)},
            ),
            (
                "alias",
                lambda: torch.reshape(x=value, shape=(2, 2)),
                (),
                {"x": value, "shape": (2, 2)},
            ),
        ):
            with self.subTest(form=form):
                Override.calls.clear()
                self.assertIs(call(), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.reshape)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

        for form, shape, call in (
            (
                "shape object",
                Override(),
                lambda shape: torch.reshape(tensor, shape),
            ),
            (
                "shape tuple element",
                (Override(), 2),
                lambda shape: torch.reshape(tensor, shape),
            ),
            (
                "shape list element",
                [Override(), 2],
                lambda shape: torch.reshape(input=tensor, shape=shape),
            ),
            (
                "variadic shape element",
                Override(),
                lambda shape: torch.reshape(tensor, 2, shape),
            ),
        ):
            with self.subTest(form=form):
                Override.calls.clear()
                self.assertIs(call(shape), marker)
                self.assertEqual(len(Override.calls), 1)
                function, dispatch_types, args, kwargs = Override.calls[0]
                self.assertIs(function, torch.reshape)
                self.assertEqual(dispatch_types, (Override,))
                if form == "shape list element":
                    self.assertEqual(args, ())
                    self.assertEqual(kwargs, {"input": tensor, "shape": shape})
                elif form == "variadic shape element":
                    self.assertEqual(args, (tensor, 2, shape))
                    self.assertIsNone(kwargs)
                else:
                    self.assertEqual(args, (tensor, shape))
                    self.assertIsNone(kwargs)

        class InputOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        class ShapeOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        self.assertIs(torch.reshape(InputOverride(), ShapeOverride()), marker)
        self.assertEqual(len(InputOverride.calls), 1)
        self.assertEqual(ShapeOverride.calls, [])
        function, dispatch_types, args, kwargs = InputOverride.calls[0]
        self.assertIs(function, torch.reshape)
        self.assertEqual(dispatch_types, (InputOverride, ShapeOverride))
        self.assertEqual(len(args), 2)
        self.assertIsNone(kwargs)

        input_value = InputOverride()
        InputOverride.calls.clear()
        self.assertIs(torch.reshape(input_value, (1, 2.0)), marker)
        self.assertEqual(len(InputOverride.calls), 1)
        function, dispatch_types, args, kwargs = InputOverride.calls[0]
        self.assertIs(function, torch.reshape)
        self.assertEqual(dispatch_types, (InputOverride,))
        self.assertEqual(args, (input_value, (1, 2.0)))
        self.assertIsNone(kwargs)

        class BaseShape:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append(("base", dispatch_types))
                return marker

        class DerivedShape(BaseShape):
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append(("derived", dispatch_types))
                return marker

        self.assertIs(torch.reshape(tensor, (BaseShape(), DerivedShape())), marker)
        self.assertEqual(
            BaseShape.calls, [("derived", (DerivedShape, BaseShape))]
        )

        shape = (1, 2.0, ShapeOverride())
        ShapeOverride.calls.clear()
        self.assertIs(torch.reshape(tensor, shape), marker)
        self.assertEqual(len(ShapeOverride.calls), 1)
        function, dispatch_types, args, kwargs = ShapeOverride.calls[0]
        self.assertIs(function, torch.reshape)
        self.assertEqual(dispatch_types, (ShapeOverride,))
        self.assertEqual(args, (tensor, shape))
        self.assertIsNone(kwargs)

        Override.calls.clear()
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' \(position 2\) must be tuple of ints, "
            r"but found element of type float at pos 0$",
        ):
            torch.reshape(tensor, (2.0, Override()))
        self.assertEqual(Override.calls, [])

        InputOverride.calls.clear()
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' \(position 2\) must be tuple of ints, "
            r"but found element of type float at pos 0$",
        ):
            torch.reshape(InputOverride(), (2.0, 2))
        self.assertEqual(InputOverride.calls, [])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.reshape(input=tensor, shape=(2, 2)), marker)
        self.assertEqual(
            mode.calls,
            [(torch.reshape, (), (), {"input": tensor, "shape": (2, 2)})],
        )

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.reshape(tensor, (1, 2.0)), marker)
        self.assertEqual(
            mode.calls,
            [(torch.reshape, (), (tensor, (1, 2.0)), None)],
        )

        mode = RecordingMode(marker)
        with mode:
            self.assertIs(torch.reshape(tensor, 2, 2), marker)
        self.assertEqual(
            mode.calls,
            [(torch.reshape, (), (tensor, 2, 2), None)],
        )

        mode = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' \(position 2\) must be tuple of ints, "
            r"but found element of type float at pos 0$",
        ):
            with mode:
                torch.reshape(tensor, (2.0, 1))
        self.assertEqual(mode.calls, [])

        mode = RecordingMode(marker)
        shape = Override()
        Override.calls.clear()
        with mode:
            self.assertIs(torch.reshape(tensor, shape), marker)
        self.assertEqual(
            mode.calls,
            [(torch.reshape, (Override,), (tensor, shape), None)],
        )
        self.assertEqual(Override.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.reshape(tensor, 2, 2)
        self.assertEqual([call[0] for call in order], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.reshape for call in order))
        self.assertTrue(all(call[2] == () for call in order))
        self.assertEqual(order[0][3], (tensor, 2, 2))
        self.assertIsNone(order[0][4])
        self.assertEqual(order[1][3], (tensor, 2, 2))
        self.assertEqual(order[1][4], {})
        self.assertEqual(forwarded.shape, (2, 2))

        declining_mode = RecordingMode(NotImplemented)
        Override.calls.clear()
        with declining_mode:
            self.assertIs(torch.reshape(value, (2, 2)), marker)
        self.assertEqual(len(declining_mode.calls), 1)
        self.assertEqual(len(Override.calls), 1)

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
            torch.reshape(DecliningOverride(), (2, 2))

        for shape in (DecliningOverride(), (DecliningOverride(), 2)):
            with self.subTest(declining_shape=type(shape).__name__):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    torch.reshape(tensor, shape)

        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.reshape(tensor, 2, DecliningOverride())

    def test_disabled_torch_function_shape_handlers_are_ignored_without_crashing(self):
        source = r"""
import sys
try:
    import torch
except ImportError:
    raise SystemExit(77)
import torch_rs

class Disabled:
    __torch_function__ = torch._C._disabled_torch_function_impl

tensor = torch_rs.tensor([1.0, 2.0, 3.0, 4.0])
cases = (
    lambda: torch_rs.reshape(tensor, (Disabled(), 2)),
    lambda: torch_rs.reshape(tensor, (1, Disabled())),
    lambda: torch_rs.reshape(tensor, Disabled()),
)
for call in cases:
    try:
        call()
    except TypeError as error:
        print(str(error).splitlines()[0])
    else:
        raise SystemExit("expected TypeError")
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            text=True,
            capture_output=True,
        )
        if completed.returncode == 77:
            self.skipTest("install the reference dependency group")
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type Disabled at pos 0",
                "reshape(): argument 'shape' failed to unpack the object at pos 2 "
                'with error "type must be tuple of ints,but got Disabled"',
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "not Disabled",
            ],
        )

    def test_binding_and_type_error_precedence_matches_pytorch_schema(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
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
                lambda: torch.reshape(tensor, (2, 2), (4,)),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type tuple at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, (2, 2), input=tensor),
                "reshape() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.reshape(tensor, (2, 2), shape=(4,)),
                "reshape() got multiple values for argument 'shape'",
            ),
            (
                lambda: torch.reshape(input=tensor, size=(2, 2)),
                'reshape() missing 1 required positional arguments: "shape"',
            ),
            (
                lambda: torch.reshape(shape=(2, 2)),
                'reshape() missing 2 required positional argument: "input", "shape"',
            ),
            (
                lambda: torch.reshape(tensor, torch.float32),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, not torch.dtype",
            ),
            (
                lambda: torch.reshape(tensor, [True]),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type bool at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, [2.0, 2]),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type float at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, True, 4),
                "reshape(): argument 'shape' (position 2) must be tuple of ints, "
                "but found element of type bool at pos 0",
            ),
            (
                lambda: torch.reshape(tensor, 2, 2.0),
                "reshape(): argument 'shape' failed to unpack the object at pos 2 "
                'with error "type must be tuple of ints,but got float"',
            ),
            (
                lambda: torch.reshape(input=tensor, shape=[True]),
                "reshape(): argument 'shape' must be tuple of ints, not list",
            ),
            (
                lambda: torch.reshape([1.0], (1,)),
                "reshape(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.reshape(input=[1.0], shape=(1,)),
                "reshape(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.reshape(tensor, (2, 2), extra=True),
                "reshape() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.reshape(tensor, 2, 2, shape=(4,)),
                "reshape() got multiple values for argument 'shape'",
            ),
            (
                lambda: torch.reshape(tensor, 2, 2, dtype=torch.float32),
                "reshape() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.reshape(tensor, 2, 2, device=torch.device("cpu")),
                "reshape() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.reshape(x=tensor, a=tensor, shape=(2, 2)),
                "reshape() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_oversized_shape_dimensions_preserve_unpack_error(self):
        tensor = torch.tensor([1.0])
        too_large = 2**63
        too_small = -(2**63) - 1
        cases = (
            (
                "positional-tuple",
                lambda: torch.reshape(tensor, (too_large,)),
                1,
            ),
            (
                "positional-list",
                lambda: torch.reshape(tensor, [too_large]),
                1,
            ),
            (
                "keyword-tuple",
                lambda: torch.reshape(input=tensor, shape=(too_large,)),
                1,
            ),
            (
                "keyword-list",
                lambda: torch.reshape(input=tensor, shape=[too_large]),
                1,
            ),
            (
                "second-dimension",
                lambda: torch.reshape(tensor, (1, too_large)),
                2,
            ),
            (
                "negative-overflow",
                lambda: torch.reshape(tensor, (too_small,)),
                1,
            ),
            (
                "numpy-uint64",
                lambda: torch.reshape(tensor, (np.uint64(too_large),)),
                1,
            ),
            (
                "variadic-first",
                lambda: torch.reshape(tensor, too_large),
                1,
            ),
            (
                "variadic-second",
                lambda: torch.reshape(tensor, 1, too_large),
                2,
            ),
        )
        for name, call, position in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^reshape\(\): argument 'shape' failed to unpack the object "
                    rf"at pos {position} with error "
                    r'"Overflow when unpacking long long',
                ):
                    call()

    def test_stateful_index_dimensions_are_converted_again_by_native_reshape(self):
        class StatefulIndexDimension:
            def __init__(self, values):
                self.values = list(values)
                self.calls = 0

            def __index__(self):
                value = self.values[self.calls]
                self.calls += 1
                return value

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        dimension = StatefulIndexDimension((1, 2))
        result = torch.reshape(tensor, (dimension, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(dimension.calls, 2)

        dimension = StatefulIndexDimension((1, 2))
        result = torch.reshape(input=tensor, shape=[dimension, 2])
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(dimension.calls, 2)

        dimension = StatefulIndexDimension((2**63, 2))
        result = torch.reshape(tensor, (dimension, 2))
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(dimension.calls, 2)

        dimension = StatefulIndexDimension((2, 1))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^shape '\[1, 2\]' is invalid for input of size 4$",
        ):
            torch.reshape(tensor, (dimension, 2))
        self.assertEqual(dimension.calls, 2)

        dimension = StatefulIndexDimension((2, 2**63))
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' failed to unpack the object at pos 1 "
            r'with error "Overflow when unpacking long long',
        ):
            torch.reshape(tensor, (dimension, 2))
        self.assertEqual(dimension.calls, 2)

        dimension = StatefulIndexDimension((2, 2.0))
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' failed to unpack the object at pos 1 "
            r'with error "type must be tuple of ints,but got StatefulIndexDimension"$',
        ):
            torch.reshape(tensor, (dimension, 2))
        self.assertEqual(dimension.calls, 2)

    def test_variadic_dimension_conversion_matches_sequence_overload(self):
        class StatefulIndexDimension:
            def __init__(self, values):
                self.values = list(values)
                self.calls = 0

            def __index__(self):
                value = self.values[self.calls]
                self.calls += 1
                return value

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

        first = StatefulIndexDimension((1, 2))
        second = StatefulIndexDimension((2,))
        result = torch.reshape(tensor, first, second)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual((first.calls, second.calls), (2, 1))

        dimension = StatefulIndexDimension((1, 4))
        result = torch.reshape(tensor, dimension)
        self.assertEqual(result.shape, (4,))
        self.assertEqual(dimension.calls, 2)

        first = StatefulIndexDimension((2, 1))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^shape '\[1, 2\]' is invalid for input of size 4$",
        ):
            torch.reshape(tensor, first, 2)
        self.assertEqual(first.calls, 2)

        first = StatefulIndexDimension((2, 2**63))
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' failed to unpack the object at pos 1 "
            r'with error "Overflow when unpacking long long',
        ):
            torch.reshape(tensor, first, 2)
        self.assertEqual(first.calls, 2)

        second = StatefulIndexDimension((2.0,))
        with self.assertRaisesRegex(
            TypeError,
            r"^reshape\(\): argument 'shape' failed to unpack the object at pos 2 "
            r'with error "type must be tuple of ints,but got StatefulIndexDimension"$',
        ):
            torch.reshape(tensor, 2, second)
        self.assertEqual(second.calls, 1)

    def test_variadic_shape_does_not_expand_sequence_subclasses(self):
        class ShapeTuple(tuple):
            pass

        class ShapeList(list):
            pass

        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        for shape in (ShapeTuple((2, 2)), ShapeList([2, 2])):
            with self.subTest(shape_type=type(shape).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^reshape\(\): argument 'shape' \(position 2\) must be tuple "
                    rf"of ints, but found element of type {type(shape).__name__} at pos 0$",
                ):
                    torch.reshape(tensor, shape, 1)

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
