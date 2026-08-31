import copy
import gc
import importlib
import inspect
import pickle
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\nunsqueeze(dim) -> Tensor\n\nSee :func:`torch.unsqueeze`\n"
FUNCTION_DOC = (
    "\nunsqueeze(input, dim) -> Tensor\n\n"
    "Returns a new tensor with a dimension of size one inserted at the\n"
    "specified position.\n\n"
    "The returned tensor shares the same underlying data with this tensor.\n\n"
    "A :attr:`dim` value within the range ``[-input.dim() - 1, input.dim() + 1)``\n"
    "can be used. Negative :attr:`dim` will correspond to :meth:`unsqueeze`\n"
    "applied at :attr:`dim` = ``dim + input.dim() + 1``.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    dim (int): the index at which to insert the singleton dimension\n\n"
    "Example::\n\n"
    "    >>> x = torch.tensor([1, 2, 3, 4])\n"
    "    >>> torch.unsqueeze(x, 0)\n"
    "    tensor([[ 1,  2,  3,  4]])\n"
    "    >>> torch.unsqueeze(x, 1)\n"
    "    tensor([[ 1],\n"
    "            [ 2],\n"
    "            [ 3],\n"
    "            [ 4]])\n"
)


class TensorNewAxisIndexTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0), (1,), (1,), 0),
            ("empty", torch.zeros((2, 0, 3)), (1, 2, 0, 3), (6, 3, 3, 1), 0),
            (
                "contiguous",
                base,
                (1, 2, 2, 3, 4),
                (48, 24, 12, 4, 1),
                0,
            ),
            (
                "transposed",
                base.transpose(0, 3),
                (1, 4, 2, 3, 2),
                (4, 1, 12, 4, 24),
                0,
            ),
            ("offset", base[1], (1, 2, 3, 4), (24, 12, 4, 1), 24),
            (
                "noncontiguous",
                base.transpose(0, 3)[1],
                (1, 2, 3, 2),
                (24, 12, 4, 24),
                1,
            ),
        )

    def trailing_layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("scalar", torch.tensor(-0.0), (1,), (1,), 0),
            ("empty", torch.zeros((2, 0, 3)), (2, 0, 3, 1), (3, 3, 1, 1), 0),
            (
                "contiguous",
                base,
                (2, 2, 3, 4, 1),
                (24, 12, 4, 1, 1),
                0,
            ),
            (
                "transposed",
                base.transpose(0, 3),
                (4, 2, 3, 2, 1),
                (1, 12, 4, 24, 1),
                0,
            ),
            ("offset", base[1], (2, 3, 4, 1), (12, 4, 1, 1), 24),
            (
                "noncontiguous",
                base.transpose(0, 3)[1],
                (2, 3, 2, 1),
                (12, 4, 24, 1),
                1,
            ),
        )

    def assert_leading_unsqueeze(self, source, result, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.tolist(), [source.tolist()])

    def assert_trailing_unsqueeze(self, source, result, shape, stride, offset):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.stride(), stride)
        self.assertEqual(result.storage_offset(), offset)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_set_to(source))
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(
            result.tolist(), np.expand_dims(np.asarray(source), axis=-1).tolist()
        )

    def test_newaxis_is_the_public_none_alias(self):
        self.assertIsNone(torch.newaxis)
        self.assertEqual(torch.__all__.count("newaxis"), 1)
        self.assertFalse(hasattr(torch._C, "newaxis"))

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIsNone(namespace["newaxis"])

    def test_bare_none_and_torch_newaxis_use_the_leading_unsqueeze_view(self):
        for spelling, index in (("None", None), ("torch.newaxis", torch.newaxis)):
            for case, source, shape, stride, offset in self.layout_cases():
                with self.subTest(spelling=spelling, case=case):
                    result = source[index]
                    self.assert_leading_unsqueeze(
                        source, result, shape, stride, offset
                    )

        scalar = torch.tensor(-0.0)[None]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_exact_trailing_none_and_newaxis_use_the_trailing_unsqueeze_view(self):
        indices = (
            ("None", (Ellipsis, None)),
            ("torch.newaxis", (Ellipsis, torch.newaxis)),
        )
        for spelling, index in indices:
            for case, source, shape, stride, offset in self.trailing_layout_cases():
                with self.subTest(spelling=spelling, case=case):
                    result = source[index]
                    self.assert_trailing_unsqueeze(
                        source, result, shape, stride, offset
                    )

        scalar = torch.tensor(-0.0)[..., None]
        self.assertEqual(np.asarray(scalar).view(np.uint32).item(), 0x8000_0000)

    def test_method_and_top_level_unsqueeze_use_boundary_views(self):
        for case, source, shape, stride, offset in self.layout_cases():
            rank = len(source.shape)
            front_calls = (
                ("method positional", lambda dim: source.unsqueeze(dim)),
                ("method dim", lambda dim: source.unsqueeze(dim=dim)),
                ("method axis", lambda dim: source.unsqueeze(axis=dim)),
                ("top-level positional", lambda dim: torch.unsqueeze(source, dim)),
                (
                    "top-level input dim",
                    lambda dim: torch.unsqueeze(input=source, dim=dim),
                ),
                ("top-level x axis", lambda dim: torch.unsqueeze(x=source, axis=dim)),
            )
            for dimension in (0, -rank - 1):
                for form, call in front_calls:
                    with self.subTest(case=case, dimension=dimension, form=form):
                        result = call(dimension)
                        self.assert_leading_unsqueeze(
                            source, result, shape, stride, offset
                        )

        for case, source, shape, stride, offset in self.trailing_layout_cases():
            rank = len(source.shape)
            back_calls = (
                ("method positional", lambda dim: source.unsqueeze(dim)),
                ("method dim", lambda dim: source.unsqueeze(dim=dim)),
                ("method axis", lambda dim: source.unsqueeze(axis=dim)),
                ("top-level positional", lambda dim: torch.unsqueeze(source, dim)),
                (
                    "top-level input dim",
                    lambda dim: torch.unsqueeze(input=source, dim=dim),
                ),
                ("top-level x axis", lambda dim: torch.unsqueeze(x=source, axis=dim)),
            )
            for dimension in (rank, -1):
                for form, call in back_calls:
                    with self.subTest(case=case, dimension=dimension, form=form):
                        result = call(dimension)
                        self.assert_trailing_unsqueeze(
                            source, result, shape, stride, offset
                        )

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "signed 64-bit stride wrapping requires a 64-bit Python build",
    )
    def test_extreme_empty_leading_stride_uses_signed_wrapping(self):
        non_concrete = torch.zeros((0,)).reshape((1 << 62, 0, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            "SymIntArrayRef expected to contain only concrete integers",
        ):
            non_concrete[None]

        negative_boundary = torch.zeros((0,)).reshape((1 << 62, 0, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^as_strided: Negative strides are not supported at the moment, "
            r"got strides: \[-4611686018427387904, 3, 3, 1\]$",
        ):
            negative_boundary[torch.newaxis]

        wrapped_negative = torch.zeros((0,)).reshape((sys.maxsize, 0, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^as_strided: Negative strides are not supported at the moment, "
            r"got strides: \[-2, 2, 2, 1\]$",
        ):
            wrapped_negative[None]

        wrapped_positive = torch.zeros((0,)).reshape((sys.maxsize, 0, 3))
        result = wrapped_positive[torch.newaxis]
        self.assertEqual(result.shape, (1, sys.maxsize, 0, 3))
        self.assertEqual(result.stride(), (sys.maxsize - 2, 3, 3, 1))
        self.assertEqual(result.storage_offset(), 0)
        self.assertEqual(result.data_ptr(), wrapped_positive.data_ptr())
        self.assertFalse(result.is_set_to(wrapped_positive))

    def make_autograd_case(self, case):
        if case == "scalar":
            leaf = torch.tensor(-2.0, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf

        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        if case == "contiguous":
            return leaf, leaf
        if case == "transposed":
            return leaf, leaf.transpose(0, 3)
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 3)[1]
        raise AssertionError(f"unknown case: {case}")

    def expected_gradient(self, case):
        if case == "scalar":
            return np.asarray(1.0, dtype=np.float32)
        if case == "empty":
            return np.zeros((2, 0, 3), dtype=np.float32)
        if case == "contiguous":
            return np.ones((2, 2, 3, 4), dtype=np.float32)
        if case == "transposed":
            return np.ones((2, 2, 3, 4), dtype=np.float32)
        if case == "offset":
            expected = np.zeros((2, 2, 3, 4), dtype=np.float32)
            expected[1] = 1.0
            return expected
        if case == "noncontiguous":
            expected = np.zeros((2, 2, 3, 4), dtype=np.float32)
            expected[:, :, :, 1] = 1.0
            return expected
        raise AssertionError(f"unknown case: {case}")

    def test_autograd_and_no_grad_cover_every_supported_layout(self):
        for case, _, shape, stride, offset in self.layout_cases():
            with self.subTest(case=case, mode="autograd"):
                leaf, source = self.make_autograd_case(case)
                result = source[None]
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                weights = torch.ones(tuple(result.shape))
                (result * weights).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), self.expected_gradient(case)
                )

            with self.subTest(case=case, mode="no_grad"):
                leaf, source = self.make_autograd_case(case)
                with torch.no_grad():
                    result = source[torch.newaxis]
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(leaf.grad)

        diagnostic_leaf = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[\[2\.\]\], grad_fn=<UnsqueezeBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[None], training=False
            )

    def test_trailing_autograd_and_no_grad_cover_every_supported_layout(self):
        for case, _, shape, stride, offset in self.trailing_layout_cases():
            with self.subTest(case=case, mode="autograd"):
                leaf, source = self.make_autograd_case(case)
                result = source[..., None]
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                weights = torch.ones(tuple(result.shape))
                (result * weights).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), self.expected_gradient(case)
                )

            with self.subTest(case=case, mode="no_grad"):
                leaf, source = self.make_autograd_case(case)
                with torch.no_grad():
                    result = source[..., torch.newaxis]
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(leaf.grad)

        diagnostic_leaf = torch.tensor([2.0], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            r"tensor\(\[\[2\.\]\], grad_fn=<UnsqueezeBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[..., None], training=False
            )

    def test_public_unsqueeze_backward_through_full_sum_and_no_grad(self):
        for case, _, shape, stride, offset in self.layout_cases():
            with self.subTest(case=case, side="front", mode="autograd"):
                leaf, source = self.make_autograd_case(case)
                result = source.unsqueeze(0)
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                result.sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), self.expected_gradient(case)
                )

            with self.subTest(case=case, side="front", mode="no_grad"):
                leaf, source = self.make_autograd_case(case)
                with torch.no_grad():
                    result = torch.unsqueeze(source, -len(source.shape) - 1)
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(leaf.grad)

        for case, _, shape, stride, offset in self.trailing_layout_cases():
            with self.subTest(case=case, side="back", mode="autograd"):
                leaf, source = self.make_autograd_case(case)
                result = torch.unsqueeze(source, len(source.shape))
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                result.sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), self.expected_gradient(case)
                )

            with self.subTest(case=case, side="back", mode="no_grad"):
                leaf, source = self.make_autograd_case(case)
                with torch.no_grad():
                    result = source.unsqueeze(-1)
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )
                self.assertTrue(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(leaf.grad)

    def test_storage_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[None]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (1, 3, 2))
        self.assertEqual(surviving.stride(), (12, 4, 12))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), values[:, :, 1].T[None]
        )

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[torch.newaxis]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights)[0].T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_trailing_storage_and_autograd_survive_source_lifetime(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, 2)[1]
            return source[..., None]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, (3, 2, 1))
        self.assertEqual(surviving.stride(), (4, 12, 1))
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(
            np.asarray(surviving), values[:, :, 1].T[..., None]
        )

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, 2)[1]
            return source[..., torch.newaxis]

        tracked = retained_autograd_view()
        gc.collect()
        weights = torch.tensor(
            [[[1.0], [2.0]], [[3.0], [4.0]], [[5.0], [6.0]]]
        )
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, :, 1] = 2.0 * np.asarray(weights)[..., 0].T
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_torch_function_mode_observes_none_before_indexing(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        mode = RecordingMode()
        for case, source, _, _, _ in self.layout_cases():
            with self.subTest(case=case, mode="recording"):
                mode.calls.clear()
                with mode:
                    result = source[torch.newaxis]
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
                self.assertIs(type(function), types.WrapperDescriptorType)
                self.assertIs(function, descriptor)
                self.assertEqual(function.__qualname__, "TensorBase.__getitem__")
                self.assertEqual(function.__objclass__.__name__, "TensorBase")
                self.assertEqual(function.__objclass__.__module__, "torch._C")
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], source)
                self.assertIsNone(args[1])
                self.assertIsNone(kwargs)
                self.assertEqual(handler_stack, ())
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), []
                )

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        for case, source, shape, stride, offset in self.layout_cases():
            with self.subTest(case=case, mode="forwarding"):
                with ForwardingMode():
                    result = source[None]
                self.assert_leading_unsqueeze(
                    source, result, shape, stride, offset
                )

    def test_torch_function_mode_observes_the_exact_trailing_tuple(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        mode = RecordingMode()
        for case, source, _, _, _ in self.trailing_layout_cases():
            with self.subTest(case=case, mode="recording"):
                index = (Ellipsis, torch.newaxis)
                mode.calls.clear()
                with mode:
                    result = source[index]
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
                self.assertIs(type(function), types.WrapperDescriptorType)
                self.assertIs(function, descriptor)
                self.assertEqual(function.__qualname__, "TensorBase.__getitem__")
                self.assertEqual(function.__objclass__.__name__, "TensorBase")
                self.assertEqual(function.__objclass__.__module__, "torch._C")
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 2)
                self.assertIs(args[0], source)
                self.assertIs(args[1], index)
                self.assertIsNone(kwargs)
                self.assertEqual(handler_stack, ())
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), []
                )

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        for case, source, shape, stride, offset in self.trailing_layout_cases():
            with self.subTest(case=case, mode="forwarding"):
                with ForwardingMode():
                    result = source[..., None]
                self.assert_trailing_unsqueeze(
                    source, result, shape, stride, offset
                )

    def test_unsqueeze_callable_metadata_imports_copy_pickle_and_reload(self):
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
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(target="function", protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("unsqueeze"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        imported_namespace = {}
        exec("from torch_rs import unsqueeze as imported_unsqueeze", imported_namespace)
        self.assertIs(imported_namespace["imported_unsqueeze"], function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["unsqueeze"], function)

        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "unsqueeze")
        bound = tensor.unsqueeze
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'unsqueeze' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "unsqueeze")
        self.assertEqual(descriptor.__qualname__, "TensorBase.unsqueeze")
        self.assertEqual(bound.__name__, "unsqueeze")
        self.assertEqual(bound.__qualname__, "Tensor.unsqueeze")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(descriptor(tensor, 0).shape, (1, 1))
        self.assertEqual(bound(-1).shape, (1, 1))
        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        self.assertIs(copy.copy(bound), bound)
        self.assertIs(copy.deepcopy(bound), bound)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(target="descriptor", protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.unsqueeze, function)
        self.assertEqual(torch.__all__.count("unsqueeze"), 1)
        self.assertIs(inspect.getattr_static(torch.Tensor, "unsqueeze"), descriptor)

    def test_unsupported_newaxis_and_unsqueeze_boundaries_are_explicit(self):
        tensor = torch.ones((2, 3))
        leading_mixed = (
            (None,),
            (None, None),
            (None, 0),
            (None, Ellipsis),
            (slice(None), None),
            (0, None),
        )
        for index in leading_mixed:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

        repeated_or_extended = (
            (Ellipsis, None, None),
            (Ellipsis, None, 0),
            (Ellipsis, 0, None),
        )
        for index in repeated_or_extended:
            with self.subTest(index=repr(index)):
                with self.assertRaises(IndexError):
                    tensor[index]

        for call in (
            lambda: tensor.unsqueeze(1),
            lambda: torch.unsqueeze(tensor, 1),
            lambda: torch.unsqueeze(tensor, -2),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^unsqueeze\(\): only front and back dimension insertions are supported$",
                ):
                    call()

        for dimension in (-4, 3):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                    tensor.unsqueeze(dimension)
                with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                    torch.unsqueeze(tensor, dimension)

        invalid_calls = (
            lambda: torch.unsqueeze(),
            lambda: torch.unsqueeze(tensor),
            lambda: torch.unsqueeze(tensor, 0, 0),
            lambda: torch.unsqueeze(1, 0),
            lambda: tensor.unsqueeze(),
            lambda: tensor.unsqueeze(0, 0),
            lambda: tensor.unsqueeze(None),
            lambda: torch.unsqueeze(tensor, None),
            lambda: tensor.unsqueeze(True),
            lambda: tensor.unsqueeze(dim=np.bool_(True)),
            lambda: torch.unsqueeze(tensor, dim="1"),
            lambda: tensor.unsqueeze(torch.float32),
            lambda: torch.unsqueeze(input=tensor, dim=0, axis=0),
            lambda: tensor.unsqueeze(dim=0, axis=0),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        for dimension in (2**100, -(2**100)):
            with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                tensor.unsqueeze(dimension)
            with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                torch.unsqueeze(tensor, dimension)

        self.assertFalse(hasattr(torch, "unsqueeze_"))
        self.assertFalse(hasattr(torch.Tensor, "unsqueeze_"))
        with self.assertRaises(AttributeError):
            tensor.unsqueeze_(0)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return object()

        mode = RecordingMode()
        with mode:
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^unsqueeze\(\): __torch_function__ modes are not supported$",
            ):
                tensor.unsqueeze(0)
        self.assertEqual(mode.calls, [])

        mode.calls.clear()
        with mode:
            with self.assertRaisesRegex(
                NotImplementedError,
                r"^unsqueeze\(\): __torch_function__ modes are not supported$",
            ):
                torch.unsqueeze(tensor, 0)
        self.assertEqual(mode.calls, [])

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return object()

        with self.assertRaisesRegex(TypeError, "must be Tensor"):
            torch.unsqueeze(Override(), 0)


if __name__ == "__main__":
    unittest.main()
