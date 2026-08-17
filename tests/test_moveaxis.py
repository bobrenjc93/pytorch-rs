import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nmoveaxis(source, destination) -> Tensor\n\n"
    "See :func:`torch.moveaxis`\n"
)
FUNCTION_DOC = (
    "\nmoveaxis(input, source, destination) -> Tensor\n\n"
    "Alias for :func:`torch.movedim`.\n\n"
    "This function is equivalent to NumPy's moveaxis function.\n\n"
    "Examples::\n\n"
    "    >>> t = torch.randn(3,2,1)\n"
    "    >>> t\n"
    "    tensor([[[-0.3362],\n"
    "            [-0.8437]],\n\n"
    "            [[-0.9627],\n"
    "            [ 0.1727]],\n\n"
    "            [[ 0.5173],\n"
    "            [-0.1398]]])\n"
    "    >>> torch.moveaxis(t, 1, 0).shape\n"
    "    torch.Size([2, 3, 1])\n"
    "    >>> torch.moveaxis(t, 1, 0)\n"
    "    tensor([[[-0.3362],\n"
    "            [-0.9627],\n"
    "            [ 0.5173]],\n\n"
    "            [[-0.8437],\n"
    "            [ 0.1727],\n"
    "            [-0.1398]]])\n"
    "    >>> torch.moveaxis(t, (1, 2), (0, 1)).shape\n"
    "    torch.Size([2, 1, 3])\n"
    "    >>> torch.moveaxis(t, (1, 2), (0, 1))\n"
    "    tensor([[[-0.3362, -0.9627,  0.5173]],\n\n"
    "            [[-0.8437,  0.1727, -0.1398]]])\n"
)


class MoveaxisTests(unittest.TestCase):
    def assert_view_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, source.dtype)
            self.assertEqual(actual.device, source.device)
            self.assertEqual(actual.data_ptr(), source.data_ptr())
            self.assertEqual(expected.data_ptr(), source.data_ptr())
            self.assertIsNot(actual, source)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assert_error_matches_movedim(self, moveaxis_call, movedim_call):
        with self.assertRaises(Exception) as actual_raised:
            moveaxis_call()
        with self.assertRaises(Exception) as expected_raised:
            movedim_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            str(actual_raised.exception),
            str(expected_raised.exception).replace("movedim", "moveaxis"),
        )

    def test_integer_forms_reuse_movedim_views_and_normalize_negative_axes(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        cases = (
            ("scalar", torch.tensor([1.5, 2.5])[1], 0, -1),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2), -1, 0),
            ("offset", base.transpose(0, 2)[1], -1, 0),
            ("noncontiguous", base.transpose(0, 2), 0, -1),
        )
        for name, source, source_axis, destination_axis in cases:
            expected = torch.movedim(source, source_axis, destination_axis)
            calls = (
                torch.moveaxis(source, source_axis, destination_axis),
                torch.moveaxis(
                    source,
                    source=source_axis,
                    destination=destination_axis,
                ),
                torch.moveaxis(
                    input=source,
                    source=source_axis,
                    destination=destination_axis,
                ),
                torch.moveaxis(
                    destination=destination_axis,
                    input=source,
                    source=source_axis,
                ),
            )
            for style, actual in enumerate(calls):
                self.assert_view_matches(
                    actual,
                    expected,
                    source,
                    case=(name, style),
                )

        for alias in ("input", "x", "a", "x1"):
            actual = torch.moveaxis(
                **{alias: base, "source": np.int64(0), "destination": -1}
            )
            self.assert_view_matches(
                actual,
                torch.movedim(base, 0, -1),
                base,
                case=("input-alias", alias),
            )

    def test_autograd_empty_backward_and_no_grad_reuse_movedim_policy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(
            4, 2, 3
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = torch.tensor(values.tolist(), requires_grad=True)
        actual = torch.moveaxis(actual_leaf, -1, 0)
        expected = torch.movedim(expected_leaf, -1, 0)
        self.assertEqual(
            (actual.requires_grad, actual.is_leaf),
            (expected.requires_grad, expected.is_leaf),
        )
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), np.asarray(expected_leaf.grad)
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.moveaxis(actual_empty, 0, -1).sum().backward()
        torch.movedim(expected_empty, 0, -1).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), np.asarray(expected_empty.grad)
        )

        with torch.no_grad():
            untracked = torch.moveaxis(
                input=actual_leaf,
                source=0,
                destination=1,
            )
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(torch.moveaxis(actual_leaf, 0, 1).requires_grad)

    def test_integer_binding_errors_and_deliberate_unsupported_surface(self):
        class IntegerSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 0

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            torch.moveaxis(tensor, IntegerSubclass(0), np.uint64(2)).shape,
            (3, 4, 2),
        )

        cases = (
            (lambda: torch.moveaxis(), lambda: torch.movedim()),
            (lambda: torch.moveaxis(tensor), lambda: torch.movedim(tensor)),
            (
                lambda: torch.moveaxis(tensor, 0),
                lambda: torch.movedim(tensor, 0),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, 1, 2),
                lambda: torch.movedim(tensor, 0, 1, 2),
            ),
            (
                lambda: torch.moveaxis(source=0, destination=1),
                lambda: torch.movedim(source=0, destination=1),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, source=1),
                lambda: torch.movedim(tensor, 0, source=1),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, 1, extra=True),
                lambda: torch.movedim(tensor, 0, 1, extra=True),
            ),
            (lambda: torch.moveaxis(1, 0, 1), lambda: torch.movedim(1, 0, 1)),
            (
                lambda: torch.moveaxis(tensor, True, 0),
                lambda: torch.movedim(tensor, True, 0),
            ),
            (
                lambda: torch.moveaxis(tensor, IndexOnly(), 0),
                lambda: torch.movedim(tensor, IndexOnly(), 0),
            ),
            (
                lambda: torch.moveaxis(tensor, 1.5, 0),
                lambda: torch.movedim(tensor, 1.5, 0),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, "1"),
                lambda: torch.movedim(tensor, 0, "1"),
            ),
            (
                lambda: torch.moveaxis(tensor, 2**100, 0),
                lambda: torch.movedim(tensor, 2**100, 0),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, np.uint64(2**63)),
                lambda: torch.movedim(tensor, 0, np.uint64(2**63)),
            ),
            (
                lambda: torch.moveaxis(tensor, 3, 0),
                lambda: torch.movedim(tensor, 3, 0),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, -4),
                lambda: torch.movedim(tensor, 0, -4),
            ),
            (
                lambda: torch.moveaxis(tensor, (0, 1), (1, 2)),
                lambda: torch.movedim(tensor, (0, 1), (1, 2)),
            ),
            (
                lambda: torch.moveaxis(tensor, 0, **{"bad\0tail": 1}),
                lambda: torch.movedim(tensor, 0, **{"bad\0tail": 1}),
            ),
        )
        for case, (moveaxis_call, movedim_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches_movedim(moveaxis_call, movedim_call)

        self.assertTrue(hasattr(torch.Tensor, "moveaxis"))
        self.assertTrue(hasattr(torch.Tensor, "movedim"))

    def test_torch_function_modes_and_overrides_receive_moveaxis(self):
        tensor = torch.zeros((2, 3, 4))
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
            self.assertIs(torch.moveaxis(tensor, 0, -1), marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, torch.moveaxis)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, -1))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            self.assertIs(
                torch.moveaxis(destination=-1, input=tensor, source=0),
                marker,
            )
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, torch.moveaxis)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(
            kwargs,
            {"destination": -1, "input": tensor, "source": 0},
        )

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.moveaxis(tensor, source=0, destination=-1)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.shape, (3, 4, 2))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        deferred = RecordingMode(marker)
        with deferred:
            self.assertIs(torch.moveaxis(tensor, 2**100, -4), marker)
        self.assertEqual(len(deferred.calls), 1)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                torch.moveaxis(tensor, True, 0)
        self.assertEqual(invalid.calls, [])

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.moveaxis'; all ",
        ):
            with lower:
                with declining:
                    torch.moveaxis(tensor, 0, 1)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])

        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.moveaxis(value, 0, 1), marker)
        function, dispatch_types, args, kwargs = calls[0]
        self.assertIs(function, torch.moveaxis)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value, 0, 1))
        self.assertIsNone(kwargs)

    def test_builtin_documentation_ownership_exports_and_pickling(self):
        function = torch.moveaxis
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "moveaxis")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.moveaxis")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method moveaxis of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.moveaxis, function)
        self.assertIsNot(function, torch.movedim)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("moveaxis"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["moveaxis"], function)


class TensorMoveaxisTests(unittest.TestCase):
    def assert_view_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, source.dtype)
            self.assertEqual(actual.device, source.device)
            self.assertEqual(actual.data_ptr(), source.data_ptr())
            self.assertEqual(expected.data_ptr(), source.data_ptr())
            self.assertIsNot(actual, source)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def assert_error_matches_movedim(self, moveaxis_call, movedim_call):
        with self.assertRaises(Exception) as actual_raised:
            moveaxis_call()
        with self.assertRaises(Exception) as expected_raised:
            movedim_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            str(actual_raised.exception),
            str(expected_raised.exception).replace("movedim", "moveaxis"),
        )

    def test_integer_forms_reuse_movedim_views_and_normalize_negative_axes(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        cases = (
            ("scalar", torch.tensor([1.5, 2.5])[1], 0, -1),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2), -1, 0),
            ("offset", base.transpose(0, 2)[1], -1, 0),
            ("noncontiguous", base.transpose(0, 2), 0, -1),
        )
        for name, source, source_axis, destination_axis in cases:
            expected = source.movedim(source_axis, destination_axis)
            calls = (
                source.moveaxis(source_axis, destination_axis),
                source.moveaxis(
                    source=source_axis,
                    destination=destination_axis,
                ),
                source.moveaxis(
                    destination=destination_axis,
                    source=source_axis,
                ),
            )
            for style, actual in enumerate(calls):
                self.assert_view_matches(
                    actual,
                    expected,
                    source,
                    case=(name, style),
                )

    def test_autograd_empty_backward_and_no_grad_reuse_movedim_policy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(
            4, 2, 3
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = torch.tensor(values.tolist(), requires_grad=True)
        actual = actual_leaf.moveaxis(-1, 0)
        expected = expected_leaf.movedim(-1, 0)
        self.assertEqual(
            (actual.requires_grad, actual.is_leaf),
            (expected.requires_grad, expected.is_leaf),
        )
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), np.asarray(expected_leaf.grad)
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.moveaxis(0, -1).sum().backward()
        expected_empty.movedim(0, -1).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), np.asarray(expected_empty.grad)
        )

        with torch.no_grad():
            untracked = actual_leaf.moveaxis(source=0, destination=1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.data_ptr(), actual_leaf.data_ptr())

    def test_integer_binding_errors_and_sequence_axes(self):
        class IntegerSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 0

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            tensor.moveaxis(IntegerSubclass(0), np.uint64(2)).shape,
            (3, 4, 2),
        )

        cases = (
            (lambda: tensor.moveaxis(), lambda: tensor.movedim()),
            (lambda: tensor.moveaxis(0), lambda: tensor.movedim(0)),
            (
                lambda: tensor.moveaxis(0, 1, 2),
                lambda: tensor.movedim(0, 1, 2),
            ),
            (
                lambda: tensor.moveaxis(source=0),
                lambda: tensor.movedim(source=0),
            ),
            (
                lambda: tensor.moveaxis(0, source=1),
                lambda: tensor.movedim(0, source=1),
            ),
            (
                lambda: tensor.moveaxis(0, 1, extra=True),
                lambda: tensor.movedim(0, 1, extra=True),
            ),
            (lambda: tensor.moveaxis(True, 0), lambda: tensor.movedim(True, 0)),
            (
                lambda: tensor.moveaxis(IndexOnly(), 0),
                lambda: tensor.movedim(IndexOnly(), 0),
            ),
            (lambda: tensor.moveaxis(1.5, 0), lambda: tensor.movedim(1.5, 0)),
            (lambda: tensor.moveaxis(0, "1"), lambda: tensor.movedim(0, "1")),
            (
                lambda: tensor.moveaxis(2**100, 0),
                lambda: tensor.movedim(2**100, 0),
            ),
            (
                lambda: tensor.moveaxis(0, np.uint64(2**63)),
                lambda: tensor.movedim(0, np.uint64(2**63)),
            ),
            (lambda: tensor.moveaxis(3, 0), lambda: tensor.movedim(3, 0)),
            (lambda: tensor.moveaxis(0, -4), lambda: tensor.movedim(0, -4)),
            (
                lambda: tensor.moveaxis((0, 1), (1, 2)),
                lambda: tensor.movedim((0, 1), (1, 2)),
            ),
            (
                lambda: tensor.moveaxis(0, **{"bad\0tail": 1}),
                lambda: tensor.movedim(0, **{"bad\0tail": 1}),
            ),
        )
        for case, (moveaxis_call, movedim_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches_movedim(moveaxis_call, movedim_call)

        for source, destination in (
            ((0, 2), (2, 0)),
            ([0, 2], [2, 0]),
            ((), ()),
        ):
            with self.subTest(source=source, destination=destination):
                with self.assertRaises(TypeError):
                    tensor.moveaxis(source, destination)

    def test_torch_function_modes_receive_the_moveaxis_descriptor(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "moveaxis")
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
            self.assertIs(tensor.moveaxis(0, -1), marker)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor, 0, -1))
        self.assertIsNone(kwargs)

        keyword = RecordingMode(marker)
        with keyword:
            self.assertIs(tensor.moveaxis(destination=-1, source=0), marker)
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (tensor,))
        self.assertEqual(kwargs, {"destination": -1, "source": 0})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.moveaxis(source=0, destination=-1)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.shape, (3, 4, 2))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        deferred = RecordingMode(marker)
        with deferred:
            self.assertIs(tensor.moveaxis(2**100, -4), marker)
        self.assertEqual(len(deferred.calls), 1)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                tensor.moveaxis(True, 0)
        self.assertEqual(invalid.calls, [])

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.moveaxis'; all ",
        ):
            with lower:
                with declining:
                    tensor.moveaxis(0, 1)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])

    def test_descriptor_metadata_and_top_level_api_remain_distinct(self):
        tensor = torch.zeros((2, 3, 4))
        descriptor = inspect.getattr_static(torch.Tensor, "moveaxis")
        bound = tensor.moveaxis

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "moveaxis")
        self.assertEqual(descriptor.__qualname__, "TensorBase.moveaxis")
        self.assertEqual(bound.__name__, "moveaxis")
        self.assertEqual(bound.__qualname__, "Tensor.moveaxis")
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
            "<method 'moveaxis' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.moveaxis, descriptor)
        self.assertIsNot(torch.Tensor.moveaxis, torch.Tensor.movedim)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor, 0, -1).shape, (3, 4, 2))
        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.moveaxis() needs an argument",
            ),
            (
                lambda: descriptor(1, 0, 1),
                "descriptor 'moveaxis' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, source=0, destination=1),
                "unbound method TensorBase.moveaxis() needs an argument",
            ),
        )
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

        self.assertIsNot(torch.moveaxis, torch.movedim)
        self.assertEqual(torch.__all__.count("moveaxis"), 1)


if __name__ == "__main__":
    unittest.main()
