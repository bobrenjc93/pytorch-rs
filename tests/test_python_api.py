import math
import operator
import sys
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch


class PythonApiBaselineTests(unittest.TestCase):
    def assert_tensor_values(self, actual, expected, shape):
        self.assertEqual(actual.shape, shape)
        actual_values = np.asarray(actual.tolist(), dtype=np.float32).reshape(-1)
        expected_values = np.asarray(expected, dtype=np.float32).reshape(-1)
        self.assertEqual(actual_values.size, expected_values.size)
        for actual_value, expected_value in zip(actual_values, expected_values):
            if np.isnan(expected_value):
                self.assertTrue(np.isnan(actual_value))
            else:
                actual_bits = actual_value.view(np.uint32).item()
                expected_bits = expected_value.view(np.uint32).item()
                self.assertEqual(actual_bits, expected_bits)

    def test_readme_style_tensor_expression(self):
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        y = torch.ones([2, 2])
        result = (x + y).relu()

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[0.0, 3.0], [4.0, 0.0]])

    def test_float32_descriptor_identity_type_and_repr(self):
        self.assertIs(torch.float, torch.float32)
        self.assertIsInstance(torch.float32, torch.dtype)
        self.assertEqual(repr(torch.float32), "torch.float32")
        self.assertEqual(str(torch.float32), "torch.float32")
        self.assertEqual(hash(torch.float), hash(torch.float32))

        with self.assertRaises(TypeError):
            torch.dtype()

    def test_cpu_device_constructor_value_repr_and_equality(self):
        cpu = torch.device("cpu")
        copied = torch.device(cpu)

        self.assertIsInstance(cpu, torch.device)
        self.assertEqual(cpu, copied)
        self.assertEqual(hash(cpu), hash(copied))
        self.assertNotEqual(cpu, "cpu")
        self.assertEqual(cpu.type, "cpu")
        self.assertIsNone(cpu.index)
        self.assertEqual(str(cpu), "cpu")
        self.assertEqual(repr(cpu), "device(type='cpu')")
        self.assertEqual(torch.device(type="cpu"), cpu)

    def test_device_constructor_rejects_unsupported_values_and_types(self):
        for specification in ("cuda", "meta", "cpu:0", "CPU", ""):
            with self.subTest(specification=specification):
                with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
                    torch.device(specification)

        for specification in (object(), 0, b"cpu", torch.float32):
            with self.subTest(specification=specification):
                with self.assertRaises(TypeError):
                    torch.device(specification)

        with self.assertRaises(TypeError):
            torch.device()

    def test_creation_metadata_keywords_preserve_values_for_all_shapes(self):
        creators = (
            ("tensor scalar", lambda **kw: torch.tensor(-2.5, **kw), (), -2.5),
            ("zeros empty", lambda **kw: torch.zeros((2, 0, 3), **kw), (2, 0, 3), [[], []]),
            ("ones ordinary", lambda **kw: torch.ones((2, 2), **kw), (2, 2), [[1.0, 1.0], [1.0, 1.0]]),
            ("full ordinary", lambda **kw: torch.full((2,), 3.25, **kw), (2,), [3.25, 3.25]),
        )
        metadata = (
            (None, None),
            (torch.float32, None),
            (torch.float, "cpu"),
            (None, torch.device("cpu")),
            (torch.float32, torch.device("cpu")),
        )

        for name, create, shape, values in creators:
            for dtype, device in metadata:
                with self.subTest(name=name, dtype=dtype, device=device):
                    tensor = create(dtype=dtype, device=device)
                    self.assertEqual(tensor.shape, shape)
                    self.assertEqual(tensor.tolist(), values)
                    self.assertIs(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, torch.device("cpu"))

        self.assertEqual(torch.zeros(size=(2,), dtype=torch.float32).tolist(), [0.0, 0.0])
        self.assertEqual(torch.ones(size=(2,), device="cpu").tolist(), [1.0, 1.0])

    def test_zeros_and_ones_accept_size_and_legacy_shape_keywords(self):
        for name, create, expected in (
            ("zeros", torch.zeros, [[0.0, 0.0], [0.0, 0.0]]),
            ("ones", torch.ones, [[1.0, 1.0], [1.0, 1.0]]),
        ):
            for keyword in ("size", "shape"):
                with self.subTest(function=name, keyword=keyword):
                    tensor = create(
                        **{keyword: (2, 2)},
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )
                    self.assertEqual(tensor.tolist(), expected)
                    self.assertIs(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, torch.device("cpu"))

            with self.subTest(function=name, error="conflicting aliases"):
                with self.assertRaises(TypeError):
                    create(size=(1,), shape=(1,))

            with self.subTest(function=name, error="missing size"):
                with self.assertRaises(TypeError):
                    create()

    def test_metadata_survives_views_and_native_kernels(self):
        source = torch.tensor([[-1.0, 2.0], [3.0, -4.0]], dtype=torch.float32, device="cpu")
        outputs = (
            source.reshape(4),
            source + torch.ones((2, 2)),
            source * 2.0,
            source.relu(),
            source @ torch.ones((2, 2)),
            source.sum(),
        )
        for output in outputs:
            with self.subTest(shape=output.shape):
                self.assertIs(output.dtype, torch.float32)
                self.assertEqual(output.device, torch.device("cpu"))

    def test_creation_rejects_invalid_dtype_and_device_types(self):
        creators = (
            lambda **kw: torch.tensor([1.0], **kw),
            lambda **kw: torch.zeros((1,), **kw),
            lambda **kw: torch.ones((1,), **kw),
            lambda **kw: torch.full((1,), 2.0, **kw),
        )
        invalid_dtypes = ("float32", np.dtype("float32"), np.float32, float, object(), torch.device("cpu"))
        invalid_devices = (object(), 0, b"cpu", torch.float32)

        for create in creators:
            for dtype in invalid_dtypes:
                with self.subTest(argument="dtype", value=dtype):
                    with self.assertRaises(TypeError):
                        create(dtype=dtype)
            for device in invalid_devices:
                with self.subTest(argument="device", value=device):
                    with self.assertRaises(TypeError):
                        create(device=device)

    def test_creation_rejects_every_unimplemented_device(self):
        creators = (
            lambda **kw: torch.tensor(1.0, **kw),
            lambda **kw: torch.zeros((), **kw),
            lambda **kw: torch.ones((), **kw),
            lambda **kw: torch.full((), 2.0, **kw),
        )
        for create in creators:
            for device in ("cuda", "meta", "mps", "cpu:0"):
                with self.subTest(device=device):
                    with self.assertRaises(RuntimeError):
                        create(device=device)

    def test_creation_metadata_parameters_are_keyword_only(self):
        with self.assertRaises(TypeError):
            torch.tensor([1.0], torch.float32)
        with self.assertRaises(TypeError):
            torch.zeros((1,), torch.float32)
        with self.assertRaises(TypeError):
            torch.ones((1,), torch.float32)
        with self.assertRaises(TypeError):
            torch.full((1,), 2.0, torch.float32)

    def test_scalar_reduction_and_item(self):
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).sum()
        self.assertEqual(value.shape, ())
        self.assertEqual(value.item(), 10.0)

    def test_stride_reports_contiguous_row_major_layout(self):
        cases = (
            (torch.tensor(1.0), ()),
            (torch.zeros((2, 3, 4)), (12, 4, 1)),
            (torch.zeros((2, 0, 3)), (3, 3, 1)),
            (torch.zeros((1, 0, 1)), (1, 1, 1)),
        )
        for tensor, expected in cases:
            with self.subTest(shape=tensor.shape):
                self.assertEqual(tensor.stride(), expected)

    def test_stride_accepts_positive_and_negative_dimensions(self):
        class IntSubclass(int):
            pass

        class IndexLike:
            def __index__(self):
                return 0

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(tensor.stride(0), 12)
        self.assertEqual(tensor.stride(1), 4)
        self.assertEqual(tensor.stride(-1), 1)
        self.assertEqual(tensor.stride(dim=-3), 12)
        self.assertEqual(tensor.stride(IntSubclass(1)), 4)
        self.assertEqual(tensor.stride(np.int64(-1)), 1)
        self.assertEqual(tensor.stride(np.uint64(1)), 4)

        for dimension in (3, -4):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    r"Dimension out of range \(expected to be in range of \[-3, 2\]",
                ):
                    tensor.stride(dimension)

        scalar = torch.tensor(1.0)
        for dimension in (0, -1):
            with self.subTest(scalar_dimension=dimension):
                with self.assertRaisesRegex(IndexError, "tensor has no dimensions"):
                    scalar.stride(dimension)

        for dimension in (True, np.bool_(False), IndexLike()):
            with self.subTest(invalid_type=type(dimension).__name__):
                with self.assertRaises(TypeError):
                    tensor.stride(dimension)

        for dimension in (1 << 100, -(1 << 100), np.uint64(2**64 - 1)):
            with self.subTest(overflow=dimension):
                with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                    tensor.stride(dimension)

    def test_empty_elementwise_results_match_pytorch_strides(self):
        scalar_cases = (
            ((1, 0), (1, 1)),
            ((0, 1), (1, 0)),
            ((1, 0, 1), (0, 1, 0)),
            ((2, 0, 3), (3, 3, 1)),
        )
        for shape, expected in scalar_cases:
            with self.subTest(operation="scalar", shape=shape):
                self.assertEqual((torch.zeros(shape) + 1).stride(), expected)

        empty = torch.zeros((1, 0, 1))
        self.assertEqual((empty + torch.ones((1, 0, 1))).stride(), (1, 1, 1))

        broadcast = empty + torch.ones((2, 1, 3))
        self.assertEqual(broadcast.shape, (2, 0, 3))
        self.assertEqual(broadcast.stride(), (3, 3, 1))

        compatible = torch.zeros((0, 1)) + torch.ones((1, 1))
        self.assertEqual(compatible.stride(), (1, 0))

        chained = torch.zeros((0, 1)) + 1
        self.assertEqual(chained.stride(), (1, 0))
        self.assertEqual(chained.relu().stride(), (1, 1))

    def test_extreme_empty_pointwise_outputs_match_pytorch_stride_boundaries(self):
        tensor = torch.zeros((0,)).reshape((0, sys.maxsize, 3))

        scalar_output = tensor + 1
        self.assertEqual(scalar_output.shape, (0, sys.maxsize, 3))
        self.assertEqual(scalar_output.stride(), (1, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            tensor.relu()

        wrapped_shape = torch.zeros((0,)).reshape(
            (0, 2, sys.maxsize, sys.maxsize)
        )
        wrapped_output = wrapped_shape + 1
        self.assertEqual(wrapped_output.shape, wrapped_shape.shape)
        self.assertEqual(wrapped_output.stride(), (2, sys.maxsize, 1, 1))

        zeroed_byte_stride = torch.zeros((0,)).reshape((0, 1, 2, 1 << 61))
        self.assertEqual((zeroed_byte_stride + 1).stride(), (0, 0, 1, 2))

    def test_empty_reshape_preserves_compatible_source_strides(self):
        source = torch.zeros((0, 1)) + 1
        view = source.reshape((0, 1))

        self.assertEqual(source.stride(), (1, 0))
        self.assertEqual(view.stride(), (1, 0))
        self.assertEqual(view.shape, source.shape)
        self.assertEqual(view.tolist(), source.tolist())

    def test_reshape_accepts_variadic_and_sequence_signatures(self):
        source = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variadic = source.reshape(3, 2)
        tuple_shape = source.reshape((1, 6))
        list_shape = source.reshape([6, 1])
        keyword_shape = source.reshape(shape=(2, 3))

        self.assertEqual(variadic.shape, (3, 2))
        self.assertEqual(variadic.stride(), (2, 1))
        self.assertEqual(variadic.tolist(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(tuple_shape.tolist(), [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        self.assertEqual(list_shape.tolist(), [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        self.assertEqual(keyword_shape.tolist(), source.tolist())
        self.assertEqual(source.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with self.assertRaises(TypeError):
            source.reshape(shape=-1)

    def test_reshape_inference_scalar_and_empty_cases(self):
        source = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertEqual(source.reshape(2, -1).shape, (2, 3))
        self.assertEqual(source.reshape(-1).shape, (6,))

        scalar = torch.tensor([7.0]).reshape(())
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.stride(), ())
        self.assertEqual(scalar.item(), 7.0)
        self.assertEqual(scalar.reshape([1]).tolist(), [7.0])

        empty = torch.zeros((0,))
        inferred = empty.reshape(2, -1, 3)
        self.assertEqual(inferred.shape, (2, 0, 3))
        self.assertEqual(inferred.stride(), (3, 3, 1))
        self.assertEqual(inferred.tolist(), [[], []])
        self.assertEqual(empty.reshape((0, 2)).shape, (0, 2))

        large = 2**32
        large_empty = empty.reshape((0, large, large))
        self.assertEqual(large_empty.shape, (0, large, large))
        self.assertEqual(large_empty.stride(), (0, large, 1))
        self.assertEqual(large_empty.numel(), 0)

        maximum = sys.maxsize
        wrapped_inference = empty.reshape(-1, maximum, maximum)
        self.assertEqual(wrapped_inference.shape, (0, maximum, maximum))
        self.assertEqual(wrapped_inference.stride(), (1, maximum, 1))
        self.assertEqual(wrapped_inference.tolist(), [])

        with self.assertRaisesRegex(RuntimeError, "element count overflowed"):
            torch.tensor([1.0]).reshape(maximum, maximum, -1)
        with self.assertRaisesRegex(RuntimeError, "element count overflowed"):
            empty.reshape(3, maximum, -1)

        with self.assertRaisesRegex(RuntimeError, "is invalid for input of size 0"):
            empty.reshape(2, -1, 1 << 62)

        self.assertEqual(empty.reshape((0, maximum, maximum)).tolist(), [])

        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            empty.reshape((0, 1 << 62, 3))

    def test_reshape_reports_pytorch_compatible_errors(self):
        tensor = torch.zeros((6,))
        invalid = (
            ((4, 2), "shape '\\[4, 2\\]' is invalid for input of size 6"),
            ((-1, -1), "only one dimension can be inferred"),
            ((-2, 3), "invalid shape dimension -2 at index 0 of shape \\[-2, 3\\]"),
        )
        for shape, message in invalid:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, message):
                    tensor.reshape(shape)

        with self.assertRaisesRegex(RuntimeError, "unspecified dimension size -1"):
            torch.zeros((0,)).reshape(0, -1)

        large = 2**62
        with self.assertRaisesRegex(RuntimeError, "invalid shape dimension -2"):
            tensor.reshape((large, 4, -2))
        with self.assertRaisesRegex(RuntimeError, "only one dimension can be inferred"):
            tensor.reshape((large, 4, -1, -1))

        for shape in ((2.0, 3), (True, 6), [[2, 3]]):
            with self.subTest(shape=shape):
                with self.assertRaises(TypeError):
                    tensor.reshape(shape)

        with self.assertRaises(TypeError):
            torch.tensor(1.0).reshape()

    def test_reshape_observables_survive_source_lifetime_and_numpy_mutation(self):
        source = torch.tensor([1.0, 2.0, 3.0, 4.0])
        view = source.reshape(2, 2)
        del source

        copied = np.asarray(view)
        copied[0, 0] = 99.0
        self.assertEqual(view.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual((view + 1.0).tolist(), [[2.0, 3.0], [4.0, 5.0]])

    def test_matrix_multiplication_operator(self):
        left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        output = left @ right
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(output.tolist(), [[58.0, 64.0], [139.0, 154.0]])

    def test_binary_arithmetic_broadcasts_trailing_dimensions(self):
        left = torch.tensor([[[1.0, 2.0, 4.0]], [[8.0, 16.0, 32.0]]])
        right = torch.tensor([[1.0], [2.0], [4.0]])
        cases = (
            (
                operator.add,
                [
                    [[2.0, 3.0, 5.0], [3.0, 4.0, 6.0], [5.0, 6.0, 8.0]],
                    [
                        [9.0, 17.0, 33.0],
                        [10.0, 18.0, 34.0],
                        [12.0, 20.0, 36.0],
                    ],
                ],
            ),
            (
                operator.sub,
                [
                    [[0.0, 1.0, 3.0], [-1.0, 0.0, 2.0], [-3.0, -2.0, 0.0]],
                    [
                        [7.0, 15.0, 31.0],
                        [6.0, 14.0, 30.0],
                        [4.0, 12.0, 28.0],
                    ],
                ],
            ),
            (
                operator.mul,
                [
                    [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]],
                    [
                        [8.0, 16.0, 32.0],
                        [16.0, 32.0, 64.0],
                        [32.0, 64.0, 128.0],
                    ],
                ],
            ),
            (
                operator.truediv,
                [
                    [[1.0, 2.0, 4.0], [0.5, 1.0, 2.0], [0.25, 0.5, 1.0]],
                    [
                        [8.0, 16.0, 32.0],
                        [4.0, 8.0, 16.0],
                        [2.0, 4.0, 8.0],
                    ],
                ],
            ),
        )

        for operation, expected in cases:
            with self.subTest(operation=operation):
                self.assert_tensor_values(operation(left, right), expected, (2, 3, 3))

    def test_binary_arithmetic_broadcasts_scalars_and_zero_dimensions(self):
        scalar = torch.tensor(2.0)
        matrix = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        self.assert_tensor_values(matrix + scalar, [[3.0, 5.0], [7.0, 9.0]], (2, 2))
        self.assert_tensor_values(scalar - matrix, [[1.0, -1.0], [-3.0, -5.0]], (2, 2))

        empty = torch.zeros((2, 0, 3))
        row = torch.ones((1, 1, 3))
        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation):
                self.assert_tensor_values(operation(empty, row), [[], []], (2, 0, 3))

        self.assertEqual((torch.zeros((0,)) + torch.ones((1,))).shape, (0,))

        large_empty = torch.full((sys.maxsize, 0), 1.0)
        large_output = large_empty + torch.tensor(2.0)
        self.assertEqual(large_output.shape, (sys.maxsize, 0))
        self.assertEqual(large_output.numel(), 0)

        large = sys.maxsize // 2 + 1
        left = torch.full((0, large, 1), 1.0)
        right = torch.tensor([[[1.0, 2.0]]])
        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation, shape=(0, large, 2)):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    operation(left, right)

    def test_python_real_scalar_and_reverse_arithmetic(self):
        tensor = torch.tensor([1.0, -2.0, 4.0])
        cases = (
            (tensor + 2, [3.0, 0.0, 6.0]),
            (2 + tensor, [3.0, 0.0, 6.0]),
            (tensor - 2.0, [-1.0, -4.0, 2.0]),
            (2.0 - tensor, [1.0, 4.0, -2.0]),
            (tensor * np.float32(2.0), [2.0, -4.0, 8.0]),
            (np.float32(2.0) * tensor, [2.0, -4.0, 8.0]),
            (tensor / 2, [0.5, -1.0, 2.0]),
            (2 / tensor, [2.0, -1.0, 0.5]),
            (tensor + True, [2.0, -1.0, 5.0]),
        )
        for actual, expected in cases:
            with self.subTest(expected=expected):
                self.assert_tensor_values(actual, expected, (3,))

        zero = torch.tensor(0.0)
        self.assertEqual((zero + (-(2**63))).item(), -9223372036854775808.0)
        self.assertEqual((zero + (2**64 - 1)).item(), 18446744073709551616.0)
        self.assertEqual(
            (zero + np.uint64(2**63 - 1)).item(),
            9223372036854775808.0,
        )

    def test_wide_numpy_unsigned_scalars_delegate_to_numpy(self):
        tensor = torch.tensor([0.0])
        value = np.uint64(2**63 + 2048)

        result = tensor + value
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertEqual(result.shape, (1,))
        self.assertEqual(result[0], np.float64(2**63 + 2048))
        self.assertNotEqual(result[0], np.float64(2**63))

        for operation in (operator.add, operator.sub, operator.mul):
            with self.subTest(operation=operation):
                with self.assertRaises(TypeError):
                    operation(value, tensor)

        denominator = torch.tensor([2.0])
        for numerator in (
            np.uint64(2**63),
            np.uint64(2**63 + 2048),
            np.uint64(2**64 - 1),
        ):
            with self.subTest(numerator=numerator, operation=operator.truediv):
                result = numerator / denominator
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.dtype, np.dtype(np.float64))
                self.assertEqual(result.shape, (1,))
                self.assertEqual(result[0], np.float64(numerator) / np.float64(2.0))

    def test_numpy_array_conversion_rejects_requests_prohibiting_a_copy(self):
        tensor = torch.tensor([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "non-copying NumPy view"):
            np.array(tensor, copy=False)

        copied = np.array(tensor, copy=True)
        self.assertEqual(copied.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(copied, np.array([1.0, 2.0], dtype=np.float32))
        copied[0] = 9.0
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_python_bool_subtraction_matches_pytorch_errors(self):
        tensor = torch.tensor([1.0, 2.0])
        for operation in (
            lambda: tensor - True,
            lambda: False - tensor,
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "bool tensor is not supported"):
                    operation()

        numpy_bool = np.bool_(True)
        self.assert_tensor_values(tensor - numpy_bool, [0.0, 1.0], (2,))
        self.assert_tensor_values(numpy_bool - tensor, [0.0, -1.0], (2,))

    def test_unsupported_operands_use_python_reflected_dispatch(self):
        class ReflectedArithmetic:
            def __init__(self):
                self.calls = []

            def reflected(self, name, tensor):
                self.calls.append(name)
                return name, tensor

            def __radd__(self, tensor):
                return self.reflected("add", tensor)

            def __rsub__(self, tensor):
                return self.reflected("sub", tensor)

            def __rmul__(self, tensor):
                return self.reflected("mul", tensor)

            def __rtruediv__(self, tensor):
                return self.reflected("truediv", tensor)

        tensor = torch.tensor([1.0])
        value = ReflectedArithmetic()
        for operation, expected_name in (
            (operator.add, "add"),
            (operator.sub, "sub"),
            (operator.mul, "mul"),
            (operator.truediv, "truediv"),
        ):
            with self.subTest(operation=operation):
                name, reflected_tensor = operation(tensor, value)
                self.assertEqual(name, expected_name)
                self.assertIs(reflected_tensor, tensor)
        self.assertEqual(value.calls, ["add", "sub", "mul", "truediv"])

    def test_recognized_scalar_errors_do_not_fall_back_to_reflection(self):
        class OverflowingInteger(int):
            def __new__(cls):
                instance = super().__new__(cls, 2**64)
                instance.reflected = False
                return instance

            def __rmul__(self, tensor):
                self.reflected = True
                return tensor

        value = OverflowingInteger()
        with self.assertRaises(OverflowError):
            torch.ones((1,)) * value
        self.assertFalse(value.reflected)

    def test_scalar_division_preserves_non_finite_and_signed_zero_results(self):
        tensor = torch.tensor([1.0, -1.0, 0.0, -0.0])
        self.assert_tensor_values(
            tensor / -0.0,
            [-math.inf, math.inf, math.nan, math.nan],
            (4,),
        )
        self.assert_tensor_values(
            -0.0 / tensor,
            [-0.0, 0.0, math.nan, math.nan],
            (4,),
        )
        self.assert_tensor_values(
            tensor + math.nan,
            [math.nan, math.nan, math.nan, math.nan],
            (4,),
        )

        self.assert_tensor_values(
            1.0e-38 / torch.tensor([1.0e-39]),
            [math.inf],
            (1,),
        )
        self.assert_tensor_values(
            0.0 / torch.tensor([1.0e-39]),
            [math.nan],
            (1,),
        )

        scalar = np.array([0xC25FB64C], dtype=np.uint32).view(np.float32)[0].item()
        denominator = (
            np.array([0xC27C80A7], dtype=np.uint32).view(np.float32)[0].item()
        )
        expected = np.array([0x3F62CF8F], dtype=np.uint32).view(np.float32)
        self.assert_tensor_values(
            scalar / torch.tensor([denominator]),
            expected,
            (1,),
        )

    def test_scalar_arithmetic_rejects_non_real_and_out_of_range_values(self):
        tensor = torch.ones((2,))
        for value in (object(), Decimal("1.0"), 1 + 2j, [1.0]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    operator.add(tensor, value)
                with self.assertRaises(TypeError):
                    operator.add(value, tensor)

        for value in (-(2**63) - 1, 2**64):
            with self.subTest(value=value):
                with self.assertRaises(OverflowError):
                    tensor * value

    def test_subtraction_and_division_cover_general_same_shapes(self):
        cases = (
            (torch.tensor(7.0), torch.tensor(2.0), (), 5.0, 3.5),
            (
                torch.tensor([[[12.0, -8.0]], [[3.0, 0.5]]]),
                torch.tensor([[[3.0, 2.0]], [[-1.5, 0.25]]]),
                (2, 1, 2),
                [[[9.0, -10.0]], [[4.5, 0.25]]],
                [[[4.0, -4.0]], [[-2.0, 2.0]]],
            ),
            (
                torch.full((2, 0, 3), 1.0),
                torch.full((2, 0, 3), 2.0),
                (2, 0, 3),
                [[], []],
                [[], []],
            ),
        )

        for left, right, shape, expected_sub, expected_div in cases:
            with self.subTest(shape=shape):
                self.assert_tensor_values(left - right, expected_sub, shape)
                self.assert_tensor_values(left / right, expected_div, shape)

    def test_subtraction_and_division_match_pytorch_special_values(self):
        cases = (
            (
                operator.sub,
                [math.nan, math.inf, -math.inf, math.inf, -math.inf, -0.0, 0.0],
                [1.0, math.inf, -math.inf, -math.inf, math.inf, 0.0, -0.0],
            ),
            (
                operator.truediv,
                [
                    math.nan,
                    math.inf,
                    -math.inf,
                    math.inf,
                    -math.inf,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    0.0,
                    -0.0,
                ],
                [
                    1.0,
                    math.inf,
                    -math.inf,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.0,
                    2.0,
                    2.0,
                    -2.0,
                    -2.0,
                ],
            ),
        )

        expected = (
            [math.nan, math.nan, math.nan, math.inf, -math.inf, -0.0, 0.0],
            [
                math.nan,
                math.nan,
                math.nan,
                math.inf,
                -math.inf,
                math.inf,
                -math.inf,
                -math.inf,
                math.inf,
                0.0,
                -0.0,
                -0.0,
                0.0,
            ],
        )
        for (operation, left, right), expected_values in zip(cases, expected):
            with self.subTest(operation=operation):
                self.assert_tensor_values(
                    operation(torch.tensor(left), torch.tensor(right)),
                    expected_values,
                    (len(expected_values),),
                )

    def test_binary_arithmetic_rejects_incompatible_shapes(self):
        left = torch.zeros([2, 2])
        right = torch.zeros([3])

        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation):
                with self.assertRaises(RuntimeError):
                    operation(left, right)

    def test_ragged_input_is_rejected(self):
        with self.assertRaises(ValueError):
            torch.tensor([[1.0], [2.0, 3.0]])

    def test_full_handles_scalar_empty_and_multidimensional_shapes(self):
        scalar = torch.full([], -2.5)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.numel(), 1)
        self.assertEqual(scalar.item(), -2.5)

        empty = torch.full([2, 0, 3], 7.0)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.numel(), 0)
        self.assertEqual(empty.tolist(), [[], []])

        matrix = torch.full((2, 3), 1.25)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix.tolist(), [[1.25] * 3] * 2)

    def test_tolist_maps_zero_element_list_capacity_overflow_to_memory_error(self):
        tensor = torch.full((sys.maxsize, 0), 1.0)
        self.assertEqual(tensor.numel(), 0)

        with self.assertRaises(MemoryError):
            tensor.tolist()

    def test_full_preserves_nan_and_infinities(self):
        nan_values = torch.full([2], math.nan).tolist()
        self.assertTrue(all(math.isnan(value) for value in nan_values))
        self.assertEqual(torch.full([2], math.inf).tolist(), [math.inf, math.inf])
        self.assertEqual(torch.full([2], -math.inf).tolist(), [-math.inf, -math.inf])

    def test_full_accepts_pytorch_keyword_names(self):
        result = torch.full(size=[2], fill_value=3.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [3.0, 3.0])

    def test_full_rejects_negative_sizes_as_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "negative dimension -1"):
            torch.full([-1], 3.0)

    def test_full_rejects_storage_capacity_overflow(self):
        oversized = sys.maxsize // 4 + 1
        with self.assertRaisesRegex(RuntimeError, "exceeds the platform capacity"):
            torch.full([oversized], 1.0)

    def test_full_rejects_finite_fill_value_overflow(self):
        for fill_value in (1e40, -1e40):
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(RuntimeError, "float32 without overflow"):
                    torch.full((2,), fill_value)

    def test_full_maps_shape_product_overflow_to_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.full((2**62, 4), 1.0)

    def test_full_rejects_invalid_size_arguments(self):
        for size in ([True], (False,), range(2)):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    torch.full(size, 3.0)

    def test_full_accepts_index_protocol_dimensions(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        dimension = IndexDimension(2)
        result = torch.full([dimension], 3.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [3.0, 3.0])
        self.assertEqual(dimension.calls, 1)

    def test_full_normalizes_invalid_index_dimensions_to_type_error(self):
        class FailingIndex:
            def __index__(self):
                raise RuntimeError("index conversion failed")

        for dimension in (2**63, -(2**63) - 1, FailingIndex()):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(TypeError, "size element at index 0"):
                    torch.full([dimension], 3.0)

    def test_full_accepts_scalar_tensor_fill_value(self):
        result = torch.full((2,), torch.tensor(3.0))
        self.assertEqual(result.tolist(), [3.0, 3.0])

        with self.assertRaises(TypeError):
            torch.full((2,), torch.tensor([3.0]))

    def test_full_accepts_real_numpy_scalar_fill_values(self):
        cases = (
            (np.longdouble(1.25), [1.25, 1.25]),
            (np.float32(1.25), [1.25, 1.25]),
            (np.int64(3), [3.0, 3.0]),
            (np.bool_(True), [1.0, 1.0]),
        )
        for fill_value, expected in cases:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((2,), fill_value).tolist(), expected)

    def test_full_rejects_zero_dimensional_buffer_fill_values(self):
        array = np.array(3.0)
        for fill_value in (array, memoryview(array)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((2,), fill_value)

    def test_full_enforces_numpy_integer_signed_boundary(self):
        accepted = (
            np.int64(-(2**63)),
            np.int64(2**63 - 1),
            np.uint64(2**63 - 1),
        )
        for fill_value in accepted:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((1,), fill_value).numel(), 1)

        for fill_value in (np.uint64(2**63), np.uint64(2**64 - 1)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((1,), fill_value)

    def test_full_rejects_non_scalar_numeric_coercions(self):
        class FloatLike:
            def __init__(self):
                self.calls = 0

            def __float__(self):
                self.calls += 1
                return 3.0

        float_like = FloatLike()
        for fill_value in (Decimal("3.0"), float_like):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((2,), fill_value)
        self.assertEqual(float_like.calls, 0)

    def test_full_converts_integer_fill_values_without_double_rounding(self):
        class IntWithFloat(int):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.float_calls = 0
                return instance

            def __float__(self):
                self.float_calls += 1
                return 0.0

        fill_value = IntWithFloat(9007199791611905)
        result = torch.full((1,), fill_value)
        self.assertEqual(result.item(), 9007200328482816.0)
        self.assertEqual(fill_value.float_calls, 0)

    def test_full_enforces_python_integer_scalar_boundaries(self):
        accepted = (
            (-(2**63), -9223372036854775808.0),
            (2**64 - 1, 18446744073709551616.0),
        )
        for fill_value, expected in accepted:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((1,), fill_value).item(), expected)

        for fill_value in (-(2**63) - 1, 2**64):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(OverflowError):
                    torch.full((1,), fill_value)

    def test_full_matches_pytorch_validation_order(self):
        with self.assertRaises(TypeError):
            torch.full([-1], object())

        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.full((2**62, 4), 1e40)

    def test_full_validates_strides_for_empty_shapes(self):
        large = 2**62
        for size in ((0, large, 2), (2, 0, large, 2), (1, large, 2, 0)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    torch.full(size, 1.0)


if __name__ == "__main__":
    unittest.main()
