import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


SIGMOID_DOC = """
sigmoid() -> Tensor

See :func:`torch.sigmoid`
"""


SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
        0xBF7F_FFFF,
        0xBF80_0000,
        0xBFC0_0000,
        0x3FC0_0000,
        0x42B0_0000,
        0x42B2_0000,
        0xC2B0_0000,
        0xC2B2_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7F81_2345,
        0xFF81_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)

SPECIAL_OUTPUT_BITS = np.asarray(
    (
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F1F_597F,
        0x3F1F_597F,
        0x3F3B_26A8,
        0x3F3B_26A8,
        0x3EC1_4D03,
        0x3E89_B2B1,
        0x3E89_B2B1,
        0x3E3A_CDC2,
        0x3F51_4C8F,
        0x3F80_0000,
        0x3F80_0000,
        0x0041_EDC4,
        0x0000_0000,
        0x3F80_0000,
        0x0000_0000,
        0x3F80_0000,
        0x0000_0000,
        0xFFC1_2345,
        0x7FC1_2345,
        0xFFC1_2345,
        0x7FC5_4321,
    ),
    dtype=np.uint32,
)

AUTOGRAD_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x4185_1591,
        0x4185_1592,
        0xC2B1_7217,
        0xC2B1_7218,
    ),
    dtype=np.uint32,
)
AUTOGRAD_WEIGHTS = np.asarray(
    (1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0), dtype=np.float32
)
AUTOGRAD_OUTPUT_BITS = np.asarray(
    (
        0x3F00_0000,
        0x3F00_0000,
        0x3F1F_597F,
        0x3EC1_4D03,
        0x3F7F_FFFE,
        0x3F80_0000,
        0x0020_0010,
        0x0000_0000,
    ),
    dtype=np.uint32,
)
AUTOGRAD_GRADIENT_BITS = np.asarray(
    (
        0x3E80_0000,
        0xBF00_0000,
        0x3DF0_A4D0,
        0xBD70_A4D0,
        0x34BF_FFFE,
        0x8000_0000,
        0x00A0_0050,
        0x8000_0000,
    ),
    dtype=np.uint32,
)
AUTOGRAD_ACCUMULATED_GRADIENT_BITS = np.asarray(
    (
        0x3F00_0000,
        0xBF80_0000,
        0x3E70_A4D0,
        0xBDF0_A4D0,
        0x353F_FFFE,
        0x8000_0000,
        0x0120_0050,
        0x8000_0000,
    ),
    dtype=np.uint32,
)


class TensorSigmoidTests(unittest.TestCase):
    @staticmethod
    def tensor_values(tensor):
        return np.asarray(tensor, dtype=np.float32)

    @classmethod
    def tensor_bits(cls, tensor):
        return cls.tensor_values(tensor).reshape(-1).view(np.uint32)

    def assert_result(self, actual, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.stride(), expected_stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        channels_last = torch.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist()
        ).contiguous(memory_format=torch.channels_last_3d)
        return (
            ("scalar", torch.tensor(-0.0), ()),
            (
                "empty offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 1),
            ),
            ("empty singleton trailing", torch.zeros((0, 1)), (1, 1)),
            ("empty singleton middle", torch.zeros((0, 1, 2)), (2, 2, 1)),
            ("empty singleton surrounding", torch.zeros((1, 0, 1)), (1, 1, 1)),
            ("offset", strided[1], (1, 3)),
            ("noncontiguous", strided, (1, 4, 12)),
            ("channels last", channels_last, channels_last.stride()),
            (
                "channels last 3d",
                channels_last_3d,
                channels_last_3d.stride(),
            ),
            (
                "numerical edges",
                torch.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
                (1,),
            ),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))
        for case, source, expected_stride in self.make_cases():
            output = source.sigmoid()
            self.assert_result(output, source, expected_stride, case=case)
            actual = self.tensor_values(output).reshape(-1)
            if case == "numerical edges":
                np.testing.assert_array_equal(actual.view(np.uint32), SPECIAL_OUTPUT_BITS)
            else:
                values = self.tensor_values(source).reshape(-1)
                with np.errstate(over="ignore", invalid="ignore"):
                    expected = np.float32(1.0) / (
                        np.float32(1.0) + np.exp(-values, dtype=np.float32)
                    )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        return scalar, empty, strided[1], strided

    def test_finite_owned_scalar_autograd_matches_signed_zero_and_saturation(self):
        cases = (
            (0x0000_0000, 0x3F00_0000, 0x3E80_0000),
            (0x8000_0000, 0x3F00_0000, 0x3E80_0000),
            (0x0000_0001, 0x3F00_0000, 0x3E80_0000),
            (0x8000_0001, 0x3F00_0000, 0x3E80_0000),
            (0x3F00_0000, 0x3F1F_597F, 0x3E70_A4D0),
            (0xBF00_0000, 0x3EC1_4D03, 0x3E70_A4D0),
            (0x4185_1591, 0x3F7F_FFFE, 0x33FF_FFFE),
            (0x4185_1592, 0x3F80_0000, 0x0000_0000),
            (0xC2B1_7217, 0x0020_0010, 0x0020_0010),
            (0xC2B1_7218, 0x0000_0000, 0x0000_0000),
        )
        for input_bits, output_bits, gradient_bits in cases:
            with self.subTest(input_bits=f"0x{input_bits:08x}"):
                value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
                leaf = torch.tensor(value, requires_grad=True)
                output = leaf.sigmoid()

                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertEqual(output.shape, ())
                self.assertEqual(output.stride(), ())
                self.assertEqual(output.storage_offset(), 0)
                self.assertFalse(output.is_set_to(leaf))
                self.assertEqual(self.tensor_bits(output).item(), output_bits)

                output.backward()
                self.assertEqual(self.tensor_bits(leaf.grad).item(), gradient_bits)
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    output.backward()

        output = torch.tensor(0.5, requires_grad=True).sigmoid()
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

    def test_finite_owned_vectors_support_weighted_autograd_and_empty_inputs(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (8,))
        self.assertEqual(output.stride(), (1,))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(values, requires_grad=True)
        for _ in range(2):
            (accumulated.sigmoid() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad), AUTOGRAD_ACCUMULATED_GRADIENT_BITS
        )

        empty = torch.tensor([], requires_grad=True)
        empty_output = empty.sigmoid()
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, (0,))
        self.assertEqual(empty_output.stride(), (1,))
        self.assertFalse(empty_output.is_set_to(empty))
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_output),
            ", grad_fn=<SigmoidBackward0>",
        )
        empty_loss = empty_output.sum()
        empty_loss.backward()
        self.assertEqual(empty.grad.shape, (0,))
        self.assertEqual(empty.grad.stride(), (1,))
        self.assertEqual(empty.grad.tolist(), [])
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            empty_loss.backward()

        higher_order = torch.tensor([0.25, -0.25], requires_grad=True)
        higher_order_loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_matrices_support_weighted_autograd_and_empty_shapes(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 4).tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.stride(), (4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (2, 4))
        self.assertEqual(leaf.grad.stride(), (4, 1))
        self.assertEqual(leaf.grad.storage_offset(), 0)
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(values, requires_grad=True)
        for _ in range(2):
            (accumulated.sigmoid() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        empty_cases = (
            ((0, 3), (3, 1)),
            ((2, 0), (1, 1)),
            ((0, 0), (1, 1)),
        )
        for shape, expected_stride in empty_cases:
            with self.subTest(shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.sigmoid()
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                self.assertEqual(empty_output.storage_offset(), 0)
                self.assertFalse(empty_output.is_set_to(empty))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        empty_output
                    ),
                    ", grad_fn=<SigmoidBackward0>",
                )

                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.storage_offset(), 0)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

    def test_scalar_autograd_composes_accumulates_and_obeys_grad_mode(self):
        composed = torch.tensor(0.5, requires_grad=True)
        composed.sigmoid().sin().backward()
        sigmoid = np.float32(1.0) / (
            np.float32(1.0) + np.exp(np.float32(-0.5), dtype=np.float32)
        )
        expected = np.cos(sigmoid) * (np.float32(1.0) - sigmoid) * sigmoid
        np.testing.assert_allclose(
            np.asarray(composed.grad), expected, rtol=2.0e-6, atol=0.0
        )

        accumulated = torch.tensor(-0.5, requires_grad=True)
        accumulated.sigmoid().backward()
        first = np.asarray(accumulated.grad).copy()
        accumulated.sigmoid().backward()
        np.testing.assert_array_equal(np.asarray(accumulated.grad), first * 2.0)

        higher_order = torch.tensor(0.25, requires_grad=True)
        loss = higher_order.sigmoid()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        loss.backward()
        self.assertIsNotNone(higher_order.grad)

        tracked = torch.tensor(-0.5, requires_grad=True)
        with torch.no_grad():
            no_grad_output = tracked.sigmoid()
        self.assert_result(no_grad_output, tracked, (), case="scalar no_grad")
        self.assertTrue(tracked.sigmoid().requires_grad)

        detached = tracked.detach()
        detached_output = detached.sigmoid()
        self.assert_result(detached_output, detached, (), case="scalar detached")
        np.testing.assert_array_equal(
            self.tensor_bits(no_grad_output), self.tensor_bits(detached_output)
        )

    def test_unsupported_tracked_inputs_fail_before_existing_graphs_or_layouts_change(self):
        message = r"^sigmoid\(\): autograd recording is not supported$"

        for bits in (0x7F80_0000, 0xFF80_0000, 0x7FC1_2345, 0xFFC5_4321):
            with self.subTest(nonfinite=f"0x{bits:08x}"):
                value = np.asarray(bits, dtype=np.uint32).view(np.float32).item()
                leaf = torch.tensor(value, requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    leaf.sigmoid()
                self.assertIsNone(leaf.grad)
                leaf.sum().backward()
                self.assertEqual(leaf.grad.item(), 1.0)

                vector = torch.tensor([0.5, value], requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    vector.sigmoid()
                self.assertIsNone(vector.grad)
                vector.sum().backward()
                self.assertEqual(vector.grad.tolist(), [1.0, 1.0])

                matrix = torch.tensor(
                    [[0.5, value], [-1.0, 2.0]], requires_grad=True
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    matrix.sigmoid()
                self.assertIsNone(matrix.grad)
                matrix.sum().backward()
                self.assertEqual(matrix.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

        rank_three = torch.tensor([[[0.5, -1.0]]], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_three.sigmoid()
        self.assertIsNone(rank_three.grad)
        rank_three.sum().backward()
        self.assertEqual(rank_three.grad.tolist(), [[[1.0, 1.0]]])

        view_base = torch.tensor([0.5], requires_grad=True)
        scalar_view = view_base[0]
        self.assertFalse(scalar_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            scalar_view.sigmoid()
        scalar_view.backward()
        self.assertEqual(view_base.grad.tolist(), [1.0])

        vector_view_base = torch.tensor(
            [[0.5, -1.0], [2.0, -3.0]], requires_grad=True
        )
        vector_view = vector_view_base[0]
        self.assertEqual(vector_view.shape, (2,))
        self.assertFalse(vector_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            vector_view.sigmoid()
        vector_view.sum().backward()
        self.assertEqual(vector_view_base.grad.tolist(), [[1.0, 1.0], [0.0, 0.0]])

        matrix_view_base = torch.tensor(
            [
                [[0.5, -1.0], [2.0, -3.0]],
                [[4.0, -5.0], [6.0, -7.0]],
            ],
            requires_grad=True,
        )
        matrix_view = matrix_view_base[0]
        self.assertEqual(matrix_view.shape, (2, 2))
        self.assertFalse(matrix_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            matrix_view.sigmoid()
        matrix_view.sum().backward()
        self.assertEqual(
            matrix_view_base.grad.tolist(),
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        )

        nonleaf_base = torch.tensor([0.5, -0.5], requires_grad=True)
        nonleaf = nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            nonleaf.sigmoid()
        nonleaf.sum().backward()
        np.testing.assert_allclose(
            np.asarray(nonleaf_base.grad),
            np.cos(np.asarray([0.5, -0.5], dtype=np.float32)),
        )

        with torch.no_grad():
            no_grad_view = vector_view_base[0]
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        self.assertEqual(no_grad_view.shape, (2,))
        with self.assertRaisesRegex(RuntimeError, message):
            no_grad_view.sigmoid()

        empty_view_base = torch.zeros((1, 0), requires_grad=True)
        with torch.no_grad():
            empty_view = empty_view_base[0]
        self.assertTrue(empty_view.requires_grad)
        self.assertTrue(empty_view.is_leaf)
        self.assertEqual(empty_view.shape, (0,))
        with self.assertRaisesRegex(RuntimeError, message):
            empty_view.sigmoid()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.sigmoid()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.sigmoid()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.sigmoid()
            with torch.no_grad():
                actual = source.sigmoid()
            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertFalse(actual.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                )
            with self.subTest(case=case, mode="detached"):
                self.assertFalse(expected.is_set_to(detached))
                if detached.numel():
                    self.assertNotEqual(expected.data_ptr(), detached.data_ptr())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        bound = tensor.sigmoid

        self.assertIs(torch.Tensor.sigmoid, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'sigmoid' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "sigmoid")
        self.assertEqual(descriptor.__qualname__, "TensorBase.sigmoid")
        self.assertEqual(bound.__name__, "sigmoid")
        self.assertEqual(bound.__qualname__, "Tensor.sigmoid")
        self.assertEqual(descriptor.__doc__, SIGMOID_DOC)
        self.assertEqual(bound.__doc__, SIGMOID_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.sigmoid(1), "TensorBase.sigmoid() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.sigmoid() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.sigmoid() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.sigmoid(1, 2),
                "TensorBase.sigmoid() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.sigmoid(input=tensor),
                (
                    "Tensor.sigmoid() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.sigmoid() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.sigmoid() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.sigmoid() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.sigmoid() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'sigmoid' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.sigmoid() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tracked.sigmoid()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.sigmoid()
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_allclose(forwarded.tolist(), [0.7772999], rtol=1.0e-6)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_output = tracked.sigmoid()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_output.requires_grad)
        self.assertFalse(tracked_output.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(tracked_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.sigmoid()
                self.assertEqual(
                    len(torch.overrides._get_current_function_mode_stack()), 1
                )
        finally:
            sys.setrecursionlimit(old_recursion_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.sigmoid(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_and_inplace_forms_remain_unsupported_without_mutation(self):
        tensor = torch.tensor([1.25])
        self.assertFalse(hasattr(torch, "sigmoid"))
        self.assertNotIn("sigmoid", torch.__all__)
        self.assertTrue(hasattr(torch.nn.functional, "sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(tensor, "sigmoid_"))
        self.assertFalse(hasattr(torch, "sigmoid_"))
        self.assertNotIn("sigmoid_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.sigmoid(out=None)

        tracked = torch.tensor(0.5, requires_grad=True)
        before = self.tensor_bits(tracked).copy()
        with self.assertRaises(AttributeError):
            tracked.sigmoid_()
        np.testing.assert_array_equal(self.tensor_bits(tracked), before)
        self.assertIsNone(tracked.grad)
        tracked.sigmoid().backward()
        self.assertEqual(self.tensor_bits(tracked.grad).item(), 0x3E70_A4D0)


if __name__ == "__main__":
    unittest.main()
