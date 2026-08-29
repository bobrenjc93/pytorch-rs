import gc
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
unsqueeze(input, dim) -> Tensor

Returns a new tensor with a dimension of size one inserted at the
specified position.
"""


class UnsqueezeTests(unittest.TestCase):
    def assert_view(self, source, result, expected, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_method_and_top_level_preserve_view_metadata(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        source = base.transpose(0, 3)[1]
        expected_source = values.transpose(3, 1, 2, 0)[1]

        cases = (
            ("leading", source.unsqueeze(0), expected_source[None], (1, 2, 3, 2), (24, 12, 4, 24), 1),
            ("middle", torch.unsqueeze(source, 2), expected_source[:, :, None, :], (2, 3, 1, 2), (12, 4, 48, 24), 1),
            ("axis alias", torch.unsqueeze(input=source, axis=-1), expected_source[..., None], (2, 3, 2, 1), (12, 4, 24, 1), 1),
            ("keyword dim", torch.unsqueeze(input=source, dim=-4), expected_source[None], (1, 2, 3, 2), (24, 12, 4, 24), 1),
        )
        for case, result, expected, shape, stride, offset in cases:
            with self.subTest(case=case):
                self.assert_view(source, result, expected, shape, stride, offset)
                self.assertFalse(result.is_contiguous())

    def test_scalar_empty_offset_and_source_lifetime(self):
        scalar = torch.tensor(-0.0)
        for dim in (-1, 0):
            with self.subTest(dim=dim):
                result = scalar.unsqueeze(dim)
                self.assertEqual(result.shape, (1,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), scalar.data_ptr())
                self.assertEqual(np.asarray(result).view(np.uint32).item(), 0x8000_0000)

        empty = torch.zeros((2, 0, 3)).unsqueeze(1)
        self.assertEqual(empty.shape, (2, 1, 0, 3))
        self.assertEqual(empty.stride(), (3, 0, 3, 1))
        self.assertEqual(empty.tolist(), [[[]], [[]]])

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source.unsqueeze(1)

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 1, 2))
        self.assertEqual(surviving.stride(), (4, 24, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), values[:, :, 1].T[:, None, :]
        )

    def test_autograd_and_no_grad_view_semantics(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        source = leaf.transpose(0, 3)[1]
        result = source.unsqueeze(2)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)

        weights = torch.ones(tuple(result.shape))
        (result * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, :, 1] = 1.0
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

        no_grad_leaf = torch.tensor(values.tolist(), requires_grad=True)
        no_grad_source = no_grad_leaf.transpose(0, 3)[1]
        with torch.no_grad():
            no_grad_result = torch.unsqueeze(no_grad_source, dim=2)
        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_invalid_dimensions_bindings_and_unsupported_forms_are_explicit(self):
        source = torch.zeros((2, 3))
        for dim in (-4, 3):
            with self.subTest(dim=dim):
                with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                    source.unsqueeze(dim)

        invalid_calls = (
            lambda: source.unsqueeze(),
            lambda: source.unsqueeze(None),
            lambda: source.unsqueeze(True),
            lambda: source.unsqueeze(1.5),
            lambda: source.unsqueeze(dim=np.float64(1)),
            lambda: torch.unsqueeze(source),
            lambda: torch.unsqueeze(source, None),
            lambda: torch.unsqueeze(source, True),
            lambda: torch.unsqueeze(source, 1.5),
            lambda: torch.unsqueeze(source, 0, 1),
            lambda: torch.unsqueeze(source, 0, dim=1),
            lambda: torch.unsqueeze(source, 0, out=None),
            lambda: source.unsqueeze(0, dim=1),
            lambda: source.unsqueeze(dim=0, axis=1),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        for dim in (2**100, -(2**100)):
            with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                source.unsqueeze(dim)

        with self.assertRaises(TypeError) as raised:
            torch.unsqueeze(np.zeros((2, 3), dtype=np.float32), 0)
        self.assertEqual(
            str(raised.exception),
            "unsqueeze(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
        )

        self.assertFalse(hasattr(torch.Tensor, "unsqueeze_"))
        with self.assertRaises(AttributeError):
            source.unsqueeze_(0)

    def test_top_level_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.zeros((2, 3))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        cases = (
            (lambda: torch.unsqueeze(tensor, 1), (tensor, 1), None),
            (lambda: torch.unsqueeze(tensor, 99), (tensor, 99), None),
            (lambda: torch.unsqueeze(tensor, 2**100), (tensor, 2**100), None),
            (
                lambda: torch.unsqueeze(input=tensor, dim=1),
                (),
                {"input": tensor, "dim": 1},
            ),
            (
                lambda: torch.unsqueeze(input=tensor, axis=-1),
                (),
                {"input": tensor, "axis": -1},
            ),
        )
        for call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(expected_args=expected_args, kwargs=expected_kwargs):
                with mode:
                    result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.unsqueeze)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            torch.unsqueeze(tensor, None)
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
                forwarded = torch.unsqueeze(input=tensor, dim=1)
        self.assertEqual(forwarded.shape, (2, 1, 3))
        self.assertEqual(forwarded.stride(), (3, 3, 1))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, torch.unsqueeze)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, ())
            self.assertEqual(kwargs, {"input": tensor, "dim": 1})

        declining = RecordingMode(NotImplemented)
        with declining, self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.unsqueeze'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.unsqueeze(tensor, 1)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_top_level_tensor_like_overrides_use_public_function(self):
        marker = object()
        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        cases = (
            (lambda: torch.unsqueeze(value, 0), (value, 0), None),
            (lambda: torch.unsqueeze(value, 99), (value, 99), None),
            (lambda: torch.unsqueeze(value, 2**100), (value, 2**100), None),
            (
                lambda: torch.unsqueeze(input=value, dim=0),
                (),
                {"input": value, "dim": 0},
            ),
            (
                lambda: torch.unsqueeze(input=value, axis=0),
                (),
                {"input": value, "axis": 0},
            ),
        )
        for call, expected_args, expected_kwargs in cases:
            self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = calls[-1]
            self.assertIs(function, torch.unsqueeze)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        call_count = len(calls)
        with self.assertRaises(TypeError):
            torch.unsqueeze(value, None)
        self.assertEqual(len(calls), call_count)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch\\.unsqueeze'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            torch.unsqueeze(DecliningOverride(), 0)

    def test_top_level_callable_metadata_documentation_and_exports(self):
        function = torch.unsqueeze
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "unsqueeze")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.unsqueeze")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method unsqueeze of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.unsqueeze, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("unsqueeze"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["unsqueeze"], function)


if __name__ == "__main__":
    unittest.main()
