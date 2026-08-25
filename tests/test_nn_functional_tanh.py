import copy
import importlib
import inspect
import pickle
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional

if __package__:
    from .test_tanh import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
    )
else:
    from test_tanh import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
    )


FUNCTION_DOC = r"""tanh(input) -> Tensor

    Applies element-wise,
    :math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "tanh(input) -> Tensor\n\n"
        "Applies element-wise,\n"
        r":math:`\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`"
        "\n\n"
        "See :class:`~torch.nn.Tanh` for more details.\n"
    )


class FunctionalTanhTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    def assert_tensor_matches(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertFalse(actual.is_set_to(source))
            self.assertFalse(actual.is_set_to(expected))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_imports_signature_documentation_copy_and_pickle(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import tanh

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(tanh, functional.tanh)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))

        function = functional.tanh
        signature = inspect.signature(function)
        parameter = tuple(signature.parameters.values())[0]
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "tanh")
        self.assertEqual(function.__qualname__, "tanh")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__annotations__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(str(signature), "(input)")
        self.assertEqual(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(parameter.annotation, inspect.Parameter.empty)
        self.assertIs(signature.return_annotation, inspect.Signature.empty)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_delegates_values_layout_and_storage_to_tensor_tanh(self):
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        channels_last = torch.tensor(
            np.linspace(-2.0, 2.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels_last", channels_last),
        )

        for case, source in cases:
            expected = source.tanh()
            actual = functional.tanh(input=source)
            self.assert_tensor_matches(actual, expected, source, case=case)

    def test_direct_receiver_and_subclass_method_semantics(self):
        marker = object()
        calls = []

        class BaseReceiver:
            def tanh(self):
                calls.append(("base", self))
                return object()

        class DerivedReceiver(BaseReceiver):
            def tanh(self):
                calls.append(("derived", self))
                return marker

        receiver = DerivedReceiver()
        self.assertIs(functional.tanh(receiver), marker)
        self.assertEqual(calls, [("derived", receiver)])

        class TorchFunctionReceiver:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise AssertionError("functional.tanh must delegate to the method")

            def tanh(self):
                return marker

        self.assertIs(functional.tanh(TorchFunctionReceiver()), marker)

    def test_modes_observe_the_tensorbase_method_descriptor(self):
        source = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "tanh")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = functional.tanh(input=source)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )
        self.assertIs(result, marker)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (source,))
        self.assertIsNone(kwargs)

    def test_rank_three_or_lower_autograd_preserves_unsupported_boundaries(self):
        scalar = torch.tensor(0.5, requires_grad=True)
        scalar_output = functional.tanh(input=scalar)
        self.assertTrue(scalar_output.requires_grad)
        self.assertFalse(scalar_output.is_leaf)
        scalar_output.backward()
        self.assertEqual(self.tensor_bits(scalar.grad).item(), 0x3F49_54A3)

        vector = torch.tensor(
            AUTOGRAD_INPUT_BITS.view(np.float32).tolist(), requires_grad=True
        )
        weights = torch.tensor(AUTOGRAD_WEIGHTS.tolist())
        vector_output = functional.tanh(input=vector)
        self.assertTrue(vector_output.requires_grad)
        self.assertFalse(vector_output.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(vector_output),
            ", grad_fn=<TanhBackward0>",
        )
        (vector_output * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(vector.grad), AUTOGRAD_GRADIENT_BITS
        )

        matrix = torch.tensor(
            AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist(),
            requires_grad=True,
        )
        matrix_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 4).tolist())
        matrix_output = functional.tanh(input=matrix)
        self.assertTrue(matrix_output.requires_grad)
        self.assertFalse(matrix_output.is_leaf)
        self.assertEqual(matrix_output.shape, (2, 4))
        self.assertEqual(matrix_output.stride(), (4, 1))
        self.assertEqual(matrix_output.storage_offset(), 0)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(matrix_output),
            ", grad_fn=<TanhBackward0>",
        )
        np.testing.assert_array_equal(
            self.tensor_bits(matrix_output), AUTOGRAD_OUTPUT_BITS
        )
        (matrix_output * matrix_weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(matrix.grad), AUTOGRAD_GRADIENT_BITS
        )

        rank_three_values = (
            AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4).tolist()
        )
        rank_three_weights = torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist()
        )
        rank_three = torch.tensor(rank_three_values, requires_grad=True)
        rank_three_output = functional.tanh(input=rank_three)
        self.assertTrue(rank_three_output.requires_grad)
        self.assertFalse(rank_three_output.is_leaf)
        self.assertEqual(rank_three_output.shape, (2, 1, 4))
        self.assertEqual(rank_three_output.stride(), (4, 4, 1))
        self.assertEqual(rank_three_output.storage_offset(), 0)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                rank_three_output
            ),
            ", grad_fn=<TanhBackward0>",
        )
        np.testing.assert_array_equal(
            self.tensor_bits(rank_three_output), AUTOGRAD_OUTPUT_BITS
        )
        rank_three_loss = (rank_three_output * rank_three_weights).sum()
        rank_three_loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(rank_three.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(rank_three.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            rank_three_loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(rank_three.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(rank_three_values, requires_grad=True)
        for _ in range(2):
            (functional.tanh(accumulated) * rank_three_weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        empty = torch.tensor([], requires_grad=True)
        empty_output = functional.tanh(empty)
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, (0,))
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_output),
            ", grad_fn=<TanhBackward0>",
        )
        empty_output.sum().backward()
        self.assertEqual(empty.grad.tolist(), [])

        for shape, expected_stride in (
            ((0, 0), (1, 1)),
            ((0, 3), (3, 1)),
            ((2, 0), (1, 1)),
            ((0, 1, 4), (4, 4, 1)),
            ((2, 0, 4), (4, 4, 1)),
            ((2, 1, 0), (1, 1, 1)),
            ((1, 0, 1), (1, 1, 1)),
        ):
            with self.subTest(empty_shape=shape):
                empty_tensor = torch.zeros(shape, requires_grad=True)
                empty_output = functional.tanh(empty_tensor)
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty_tensor.grad.shape, shape)
                self.assertEqual(empty_tensor.grad.stride(), expected_stride)
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[[0.25, -0.25]]], requires_grad=True)
        higher_order_loss = functional.tanh(higher_order).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

        rank_four = torch.tensor([[[[0.5, -1.0]]]], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(rank_four)
        self.assertIsNone(rank_four.grad)
        rank_four.sum().backward()
        self.assertEqual(rank_four.grad.tolist(), [[[[1.0, 1.0]]]])

        nonfinite = torch.tensor([[[0.5, float("inf")]]], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(nonfinite)
        self.assertIsNone(nonfinite.grad)

        rank_three_view_base = torch.tensor(
            [[[[0.5, -1.0]]], [[[2.0, -3.0]]]], requires_grad=True
        )
        rank_three_view = rank_three_view_base[0]
        self.assertEqual(rank_three_view.shape, (1, 1, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(rank_three_view)
        rank_three_view.sum().backward()
        self.assertEqual(
            rank_three_view_base.grad.tolist(),
            [[[[1.0, 1.0]]], [[[0.0, 0.0]]]],
        )

        nonleaf_base = torch.tensor([[[0.5, -1.0]]], requires_grad=True)
        nonleaf = nonleaf_base.sin()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(nonleaf)
        nonleaf.sum().backward()
        self.assertIsNotNone(nonleaf_base.grad)

        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        for call in (source.tanh, lambda: functional.tanh(source)):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): autograd recording is not supported$",
                ):
                    call()

        with torch.no_grad():
            actual = functional.tanh(source)
            expected = source.tanh()
        self.assert_tensor_matches(actual, expected, source, case="no_grad")

        detached = source.detach()
        self.assert_tensor_matches(
            functional.tanh(detached),
            detached.tanh(),
            detached,
            case="detached",
        )

    def test_argument_receiver_and_scope_errors(self):
        source = torch.tensor([0.5])
        cases = (
            (
                lambda: functional.tanh(),
                TypeError,
                "tanh() missing 1 required positional argument: 'input'",
            ),
            (
                lambda: functional.tanh(source, source),
                TypeError,
                "tanh() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: functional.tanh(source, input=source),
                TypeError,
                "tanh() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.tanh(source, out=None),
                TypeError,
                "tanh() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: functional.tanh(1),
                AttributeError,
                "'int' object has no attribute 'tanh'",
            ),
        )
        for case, (call, error_type, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        expected_error = ValueError("receiver failed")

        class RaisingReceiver:
            def tanh(self):
                raise expected_error

        with self.assertRaises(ValueError) as raised:
            functional.tanh(RaisingReceiver())
        self.assertIs(raised.exception, expected_error)

        class NonCallableReceiver:
            tanh = 1

        with self.assertRaisesRegex(TypeError, "^'int' object is not callable$"):
            functional.tanh(NonCallableReceiver())

        self.assertFalse(hasattr(nn, "Tanh"))
        self.assertFalse(hasattr(torch.Tensor, "tanh_"))
        self.assertFalse(hasattr(functional, "tanh_"))


if __name__ == "__main__":
    unittest.main()
