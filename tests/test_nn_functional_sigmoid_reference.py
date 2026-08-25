import copy
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional

if __package__:
    from .test_sigmoid import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
    )
else:
    from test_sigmoid import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
    )

try:
    import torch as reference_torch
    import torch.nn.functional as reference_functional
except ImportError:
    reference_torch = None
    reference_functional = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FunctionalSigmoidReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.sigmoid differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def error(call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        return None

    @staticmethod
    def make_case(module, case):
        if case == "scalar":
            return module.tensor(-0.0, dtype=module.float32)
        if case == "empty":
            return module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1]

        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        source = module.tensor(values.tolist(), dtype=module.float32)
        if case == "offset":
            return source[1]
        if case == "noncontiguous":
            return source.transpose(0, 2)[1]
        if case == "channels_last":
            values = np.linspace(-15.0, 15.0, 120, dtype=np.float32).reshape(
                2, 3, 4, 5
            )
            return module.tensor(values.tolist(), dtype=module.float32).contiguous(
                memory_format=module.channels_last
            )

        values = np.linspace(-90.0, 90.0, 720, dtype=np.float32).reshape(
            2, 3, 4, 5, 6
        )
        return module.tensor(values.tolist(), dtype=module.float32).contiguous(
            memory_format=module.channels_last_3d
        )

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

        np.testing.assert_allclose(
            self.tensor_values(actual),
            self.tensor_values(expected),
            rtol=2.0e-6,
            atol=np.nextafter(np.float32(0), np.float32(1)),
            equal_nan=True,
        )

    def test_metadata_documentation_copy_and_pickle_match_pytorch_2_13(self):
        actual = functional.sigmoid
        expected = reference_functional.sigmoid
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__module__, "torch_rs.nn.functional")
        self.assertEqual(expected.__module__, "torch.nn.functional")
        self.assertFalse(hasattr(actual, "__text_signature__"))
        self.assertFalse(hasattr(expected, "__text_signature__"))

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=function.__module__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

    def test_values_layouts_and_storage_match_pytorch_2_13(self):
        cases = (
            "scalar",
            "empty",
            "offset",
            "noncontiguous",
            "channels_last",
            "channels_last_3d",
        )
        for case in cases:
            actual_input = self.make_case(torch, case)
            expected_input = self.make_case(reference_torch, case)
            actual = functional.sigmoid(input=actual_input)
            expected = reference_functional.sigmoid(input=expected_input)
            self.assert_tensor_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def receiver_contract(self, function):
        marker = object()
        calls = []

        class BaseReceiver:
            def sigmoid(self):
                calls.append(("base", self))
                return object()

        class DerivedReceiver(BaseReceiver):
            def sigmoid(self):
                calls.append(("derived", self))
                return marker

        receiver = DerivedReceiver()
        result = function(receiver)

        class TorchFunctionReceiver:
            torch_function_calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.torch_function_calls += 1
                return object()

            def sigmoid(self):
                return marker

        torch_function_receiver = TorchFunctionReceiver()
        torch_function_result = function(torch_function_receiver)
        return (
            result is marker,
            tuple(label for label, _ in calls),
            calls[0][1] is receiver,
            torch_function_result is marker,
            TorchFunctionReceiver.torch_function_calls,
        )

    def test_receiver_and_subclass_method_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.receiver_contract(functional.sigmoid),
            self.receiver_contract(reference_functional.sigmoid),
        )

    def mode_contract(self, module):
        function = module.nn.functional.sigmoid
        descriptor = inspect.getattr_static(module.Tensor, "sigmoid")
        source = module.tensor([0.5], dtype=module.float32, requires_grad=True)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        observations = []
        for call in (lambda: function(source), lambda: function(input=source)):
            mode = RecordingMode()
            with mode:
                result = call()
            dispatched, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    dispatched is descriptor,
                    dispatched is function,
                    type(dispatched).__name__,
                    dispatched.__name__,
                    dispatched.__qualname__,
                    dispatched.__objclass__.__name__,
                    dispatched.__objclass__.__module__,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args) == 1 and args[0] is source,
                    kwargs,
                )
            )
        return tuple(observations)

    def test_modes_observe_tensorbase_descriptor_like_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_argument_and_receiver_errors_match_pytorch_2_13(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5])
        cases = (
            (lambda: functional.sigmoid(), lambda: reference_functional.sigmoid()),
            (
                lambda: functional.sigmoid(actual, actual),
                lambda: reference_functional.sigmoid(expected, expected),
            ),
            (
                lambda: functional.sigmoid(actual, input=actual),
                lambda: reference_functional.sigmoid(expected, input=expected),
            ),
            (
                lambda: functional.sigmoid(actual, out=None),
                lambda: reference_functional.sigmoid(expected, out=None),
            ),
            (
                lambda: functional.sigmoid(1),
                lambda: reference_functional.sigmoid(1),
            ),
            (
                lambda: functional.sigmoid(None),
                lambda: reference_functional.sigmoid(None),
            ),
            (
                lambda: functional.sigmoid([]),
                lambda: reference_functional.sigmoid([]),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

        class NonCallableReceiver:
            sigmoid = 1

        self.assertEqual(
            self.error(lambda: functional.sigmoid(NonCallableReceiver())),
            self.error(lambda: reference_functional.sigmoid(NonCallableReceiver())),
        )

    def test_rank_two_autograd_and_boundaries_match_pytorch_2_13(self):
        matrix_values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist()
        actual_matrix = torch.tensor(matrix_values, requires_grad=True)
        expected_matrix = reference_torch.tensor(
            matrix_values,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 4).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(2, 4), dtype=reference_torch.float32
        )
        actual_output = functional.sigmoid(actual_matrix)
        expected_output = reference_functional.sigmoid(expected_matrix)
        self.assert_tensor_matches(
            actual_output, expected_output, case="rank-two forward"
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            self.tensor_values(expected_output).reshape(-1).view(np.uint32),
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_matrix.grad,
            expected_matrix.grad,
            case="rank-two weighted gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_matrix.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_matrix.grad).reshape(-1).view(np.uint32),
            self.tensor_values(expected_matrix.grad).reshape(-1).view(np.uint32),
        )
        actual_gradient_before = self.tensor_values(actual_matrix.grad).copy()
        expected_gradient_before = self.tensor_values(expected_matrix.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_matrix.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_matrix.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(matrix_values, requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            matrix_values,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        for _ in range(2):
            (functional.sigmoid(actual_accumulated) * actual_weights).sum().backward()
            (
                reference_functional.sigmoid(expected_accumulated)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-two accumulated gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape in ((0, 0), (0, 3), (2, 0)):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = functional.sigmoid(actual_empty)
            expected_empty_output = reference_functional.sigmoid(expected_empty)
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-two forward", shape),
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-two gradient", shape),
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

        higher_order = torch.tensor([[0.25, -0.25]], requires_grad=True)
        higher_order_loss = functional.sigmoid(higher_order).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

        message = r"^sigmoid\(\): autograd recording is not supported$"
        nonfinite = torch.tensor([[[0.5, float("inf")]]], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(nonfinite)
        self.assertIsNone(nonfinite.grad)
        nonfinite.sum().backward()
        self.assertEqual(nonfinite.grad.tolist(), [[[1.0, 1.0]]])

        rank_four_nonfinite = torch.tensor(
            [[[[0.5, float("inf")]]]], requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_four_nonfinite)
        self.assertIsNone(rank_four_nonfinite.grad)
        rank_four_nonfinite.sum().backward()
        self.assertEqual(rank_four_nonfinite.grad.tolist(), [[[[1.0, 1.0]]]])

        rank_five_nonfinite = torch.tensor(
            [[[[[0.5, float("inf")]]]]], requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_five_nonfinite)
        self.assertIsNone(rank_five_nonfinite.grad)
        rank_five_nonfinite.sum().backward()
        self.assertEqual(
            rank_five_nonfinite.grad.tolist(), [[[[[1.0, 1.0]]]]]
        )

        rank_six_nonfinite = torch.tensor(
            [[[[[[0.5, float("inf")]]]]]], requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_six_nonfinite)
        self.assertIsNone(rank_six_nonfinite.grad)
        rank_six_nonfinite.sum().backward()
        self.assertEqual(rank_six_nonfinite.grad.tolist(), [[[[[[1.0, 1.0]]]]]])

        high_rank_nonfinite = torch.full(
            (1,) * 65, float("inf"), requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(high_rank_nonfinite)
        self.assertIsNone(high_rank_nonfinite.grad)
        high_rank_nonfinite.sum().backward()
        self.assertEqual(high_rank_nonfinite.grad.item(), 1.0)

        matrix_view_base = torch.tensor(
            [
                [[0.5, -1.0], [2.0, -3.0]],
                [[4.0, -5.0], [6.0, -7.0]],
            ],
            requires_grad=True,
        )
        matrix_view = matrix_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(matrix_view)
        matrix_view.sum().backward()
        self.assertEqual(
            matrix_view_base.grad.tolist(),
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        )

        rank_three_nonleaf_base = torch.tensor(
            [[[0.5, -0.5], [1.0, -1.0]]], requires_grad=True
        )
        rank_three_nonleaf = rank_three_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_three_nonleaf)
        rank_three_nonleaf.sum().backward()
        self.assertIsNotNone(rank_three_nonleaf_base.grad)

        rank_four_view_base = torch.tensor(
            [[[[[0.5, -1.0]]]], [[[[2.0, -3.0]]]]], requires_grad=True
        )
        rank_four_view = rank_four_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_four_view)
        rank_four_view.sum().backward()
        self.assertEqual(
            rank_four_view_base.grad.tolist(),
            [[[[[1.0, 1.0]]]], [[[[0.0, 0.0]]]]],
        )

        rank_four_nonleaf_base = torch.tensor(
            [[[[0.5, -0.5]]]], requires_grad=True
        )
        rank_four_nonleaf = rank_four_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_four_nonleaf)
        rank_four_nonleaf.sum().backward()
        self.assertIsNotNone(rank_four_nonleaf_base.grad)

        rank_five_view_base = torch.tensor(
            [[[[[[0.5, -1.0]]]]], [[[[[2.0, -3.0]]]]]], requires_grad=True
        )
        rank_five_view = rank_five_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_five_view)
        rank_five_view.sum().backward()
        self.assertEqual(
            rank_five_view_base.grad.tolist(),
            [[[[[[1.0, 1.0]]]]], [[[[[0.0, 0.0]]]]]],
        )

        rank_five_nonleaf_base = torch.tensor(
            [[[[[0.5, -0.5]]]]], requires_grad=True
        )
        rank_five_nonleaf = rank_five_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_five_nonleaf)
        rank_five_nonleaf.sum().backward()
        self.assertIsNotNone(rank_five_nonleaf_base.grad)

        rank_six_view_base = torch.full(
            (2,) + (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        rank_six_view = rank_six_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_six_view)
        rank_six_view.sum().backward()
        self.assertEqual(rank_six_view_base.grad.sum().item(), 2.0)

        high_rank_view_base = torch.full(
            (2,) + (1,) * 65, 0.5, requires_grad=True
        )
        high_rank_view = high_rank_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(high_rank_view)
        high_rank_view.backward()
        self.assertEqual(high_rank_view_base.grad.sum().item(), 1.0)

        rank_six_nonleaf_base = torch.full(
            (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        rank_six_nonleaf = rank_six_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(rank_six_nonleaf)
        rank_six_nonleaf.sum().backward()
        self.assertIsNotNone(rank_six_nonleaf_base.grad)

        high_rank_nonleaf_base = torch.full(
            (1,) * 65, 0.5, requires_grad=True
        )
        high_rank_nonleaf = high_rank_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            functional.sigmoid(high_rank_nonleaf)
        high_rank_nonleaf.backward()
        self.assertIsNotNone(high_rank_nonleaf_base.grad)

    def test_rank_three_autograd_matches_pytorch_2_13(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(2, 1, 4), dtype=reference_torch.float32
        )
        actual_output = functional.sigmoid(actual_leaf)
        expected_output = reference_functional.sigmoid(expected_leaf)

        self.assert_tensor_matches(
            actual_output, expected_output, case="rank-three singleton forward"
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            self.tensor_values(expected_output).reshape(-1).view(np.uint32),
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-three weighted gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            self.tensor_values(expected_leaf.grad).reshape(-1).view(np.uint32),
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (functional.sigmoid(actual_accumulated) * actual_weights).sum().backward()
            (
                reference_functional.sigmoid(expected_accumulated)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-three accumulated gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        functional.sigmoid(actual_composed).sin().sum().backward()
        reference_functional.sigmoid(expected_composed).sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-three composition gradient",
        )

        for shape in ((0, 1, 3), (1, 0, 3), (2, 3, 0), (0, 0, 0)):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = functional.sigmoid(actual_empty)
            expected_empty_output = reference_functional.sigmoid(expected_empty)
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-three forward", shape),
            )
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-three gradient", shape),
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

        higher_order = torch.tensor([[[0.25, -0.25]]], requires_grad=True)
        higher_order_loss = functional.sigmoid(higher_order).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_rank_four_autograd_matches_pytorch_2_13(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4), dtype=reference_torch.float32
        )
        actual_output = functional.sigmoid(actual_leaf)
        expected_output = reference_functional.sigmoid(expected_leaf)

        self.assert_tensor_matches(
            actual_output, expected_output, case="rank-four singleton forward"
        )
        self.assertFalse(actual_output.is_set_to(actual_leaf))
        self.assertFalse(expected_output.is_set_to(expected_leaf))
        self.assertNotEqual(actual_output.data_ptr(), actual_leaf.data_ptr())
        self.assertNotEqual(expected_output.data_ptr(), expected_leaf.data_ptr())
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            self.tensor_values(expected_output).reshape(-1).view(np.uint32),
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-four weighted gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            self.tensor_values(expected_leaf.grad).reshape(-1).view(np.uint32),
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (functional.sigmoid(actual_accumulated) * actual_weights).sum().backward()
            (
                reference_functional.sigmoid(expected_accumulated)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-four accumulated gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        functional.sigmoid(actual_composed).sin().sum().backward()
        reference_functional.sigmoid(expected_composed).sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-four composition gradient",
        )

        for shape in (
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (1, 2, 0, 3),
            (1, 2, 3, 0),
            (0, 0, 0, 0),
        ):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = functional.sigmoid(actual_empty)
            expected_empty_output = reference_functional.sigmoid(expected_empty)
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-four forward", shape),
            )
            self.assertFalse(actual_empty_output.is_set_to(actual_empty))
            self.assertFalse(expected_empty_output.is_set_to(expected_empty))
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-four gradient", shape),
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

        higher_order = torch.tensor([[[[0.25, -0.25]]]], requires_grad=True)
        higher_order_loss = functional.sigmoid(higher_order).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_rank_five_autograd_matches_pytorch_2_13(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 1, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 1, 4).tolist()
        )
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 1, 4),
            dtype=reference_torch.float32,
        )
        actual_output = functional.sigmoid(actual_leaf)
        expected_output = reference_functional.sigmoid(expected_leaf)

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-five NCDHW singleton forward",
        )
        self.assertFalse(actual_output.is_set_to(actual_leaf))
        self.assertFalse(expected_output.is_set_to(expected_leaf))
        self.assertNotEqual(actual_output.data_ptr(), actual_leaf.data_ptr())
        self.assertNotEqual(expected_output.data_ptr(), expected_leaf.data_ptr())
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            self.tensor_values(expected_output).reshape(-1).view(np.uint32),
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-five weighted gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            self.tensor_values(expected_leaf.grad).reshape(-1).view(np.uint32),
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (functional.sigmoid(actual_accumulated) * actual_weights).sum().backward()
            (
                reference_functional.sigmoid(expected_accumulated)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-five accumulated gradient",
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        functional.sigmoid(actual_composed).sin().sum().backward()
        reference_functional.sigmoid(expected_composed).sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-five composition gradient",
        )

        for shape in (
            (0, 1, 2, 3, 4),
            (1, 0, 2, 3, 4),
            (1, 2, 0, 3, 4),
            (1, 2, 3, 0, 4),
            (1, 2, 3, 4, 0),
            (0, 0, 0, 0, 0),
        ):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = functional.sigmoid(actual_empty)
            expected_empty_output = reference_functional.sigmoid(expected_empty)
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-five forward", shape),
            )
            self.assertFalse(actual_empty_output.is_set_to(actual_empty))
            self.assertFalse(expected_empty_output.is_set_to(expected_empty))
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-five gradient", shape),
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

        higher_order = torch.tensor([[[[[0.25, -0.25]]]]], requires_grad=True)
        higher_order_loss = functional.sigmoid(higher_order).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_rank_six_and_high_rank_autograd_match_pytorch_2_13(self):
        shape = (1, 2, 1, 1, 1, 4)
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(shape)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(shape).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(shape), dtype=reference_torch.float32
        )
        actual_output = functional.sigmoid(actual_leaf)
        expected_output = reference_functional.sigmoid(expected_leaf)

        self.assert_tensor_matches(
            actual_output, expected_output, case="rank-six singleton forward"
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            self.tensor_values(expected_output).reshape(-1).view(np.uint32),
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad, expected_leaf.grad, case="rank-six weighted gradient"
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            self.tensor_values(expected_leaf.grad).reshape(-1).view(np.uint32),
        )
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (functional.sigmoid(actual_accumulated) * actual_weights).sum().backward()
            (
                reference_functional.sigmoid(expected_accumulated)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-six accumulated gradient",
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        functional.sigmoid(actual_composed).sin().sum().backward()
        reference_functional.sigmoid(expected_composed).sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-six composition gradient",
        )

        actual_empty = torch.zeros((1, 2, 0, 1, 1, 4), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (1, 2, 0, 1, 1, 4),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_empty_output = functional.sigmoid(actual_empty)
        expected_empty_output = reference_functional.sigmoid(expected_empty)
        self.assert_tensor_matches(
            actual_empty_output, expected_empty_output, case="empty rank-six forward"
        )
        self.assertEqual(
            type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
        )
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                actual_empty_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        actual_empty_loss = actual_empty_output.sum()
        expected_empty_loss = expected_empty_output.sum()
        actual_empty_loss.backward()
        expected_empty_loss.backward()
        self.assert_tensor_matches(
            actual_empty.grad, expected_empty.grad, case="empty rank-six gradient"
        )
        self.assertEqual(
            self.error(actual_empty_loss.backward),
            self.error(expected_empty_loss.backward),
        )

        high_rank_shape = (1,) * 65
        actual_high_rank = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        expected_high_rank = reference_torch.full(
            high_rank_shape,
            0.5,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_high_rank_output = functional.sigmoid(actual_high_rank)
        expected_high_rank_output = reference_functional.sigmoid(expected_high_rank)
        self.assertEqual(actual_high_rank_output.shape, expected_high_rank_output.shape)
        self.assertEqual(
            actual_high_rank_output.stride(), expected_high_rank_output.stride()
        )
        self.assertEqual(
            np.float32(actual_high_rank_output.item()).view(np.uint32).item(),
            np.float32(expected_high_rank_output.item()).view(np.uint32).item(),
        )
        self.assertEqual(
            type(expected_high_rank_output.grad_fn).__name__, "SigmoidBackward0"
        )
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                actual_high_rank_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        actual_high_rank_output.backward()
        expected_high_rank_output.backward()
        self.assertEqual(
            np.float32(actual_high_rank.grad.item()).view(np.uint32).item(),
            np.float32(expected_high_rank.grad.item()).view(np.uint32).item(),
        )
        self.assertEqual(
            self.error(actual_high_rank_output.backward),
            self.error(expected_high_rank_output.backward),
        )

        actual_high_rank_accumulated = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        expected_high_rank_accumulated = reference_torch.full(
            high_rank_shape,
            0.5,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        for _ in range(2):
            functional.sigmoid(actual_high_rank_accumulated).backward()
            reference_functional.sigmoid(expected_high_rank_accumulated).backward()
        self.assertEqual(
            np.float32(actual_high_rank_accumulated.grad.item())
            .view(np.uint32)
            .item(),
            np.float32(expected_high_rank_accumulated.grad.item())
            .view(np.uint32)
            .item(),
        )

        actual_high_rank_composed = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        expected_high_rank_composed = reference_torch.full(
            high_rank_shape,
            0.5,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        functional.sigmoid(actual_high_rank_composed).sin().backward()
        reference_functional.sigmoid(expected_high_rank_composed).sin().backward()
        np.testing.assert_allclose(
            np.float32(actual_high_rank_composed.grad.item()),
            np.float32(expected_high_rank_composed.grad.item()),
            rtol=2.0e-6,
            atol=0.0,
        )

        high_rank_empty_shape = (1,) * 32 + (0,) + (1,) * 32
        actual_high_rank_empty = torch.zeros(
            high_rank_empty_shape, requires_grad=True
        )
        expected_high_rank_empty = reference_torch.zeros(
            high_rank_empty_shape,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_high_rank_empty_output = functional.sigmoid(actual_high_rank_empty)
        expected_high_rank_empty_output = reference_functional.sigmoid(
            expected_high_rank_empty
        )
        self.assertEqual(
            actual_high_rank_empty_output.shape,
            expected_high_rank_empty_output.shape,
        )
        self.assertEqual(
            actual_high_rank_empty_output.stride(),
            expected_high_rank_empty_output.stride(),
        )
        self.assertEqual(actual_high_rank_empty_output.numel(), 0)
        self.assertEqual(expected_high_rank_empty_output.numel(), 0)
        self.assertEqual(
            type(expected_high_rank_empty_output.grad_fn).__name__,
            "SigmoidBackward0",
        )
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                actual_high_rank_empty_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        actual_high_rank_empty_loss = actual_high_rank_empty_output.sum()
        expected_high_rank_empty_loss = expected_high_rank_empty_output.sum()
        actual_high_rank_empty_loss.backward()
        expected_high_rank_empty_loss.backward()
        self.assertEqual(
            actual_high_rank_empty.grad.shape, expected_high_rank_empty.grad.shape
        )
        self.assertEqual(
            actual_high_rank_empty.grad.stride(),
            expected_high_rank_empty.grad.stride(),
        )
        self.assertEqual(actual_high_rank_empty.grad.numel(), 0)
        self.assertEqual(expected_high_rank_empty.grad.numel(), 0)
        self.assertEqual(
            self.error(actual_high_rank_empty_loss.backward),
            self.error(expected_high_rank_empty_loss.backward),
        )

    def test_supported_and_unsupported_boundaries_are_explicit(self):
        actual_scalar = torch.tensor(0.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(
            0.5, dtype=reference_torch.float32, requires_grad=True
        )
        actual_scalar_output = functional.sigmoid(actual_scalar)
        expected_scalar_output = reference_functional.sigmoid(expected_scalar)
        actual_scalar_output.backward()
        expected_scalar_output.backward()
        self.assert_tensor_matches(
            actual_scalar_output,
            expected_scalar_output,
            case="scalar forward",
        )
        self.assert_tensor_matches(
            actual_scalar.grad,
            expected_scalar.grad,
            case="scalar gradient",
        )

        actual_leaf = torch.tensor([0.5, -1.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [0.5, -1.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor([2.0, -3.0])
        expected_weights = reference_torch.tensor(
            [2.0, -3.0], dtype=reference_torch.float32
        )
        actual_output = functional.sigmoid(actual_leaf)
        expected_output = reference_functional.sigmoid(expected_leaf)
        self.assert_tensor_matches(
            actual_output, expected_output, case="rank-one forward"
        )
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_tensor_matches(
            actual_leaf.grad, expected_leaf.grad, case="rank-one gradient"
        )

        actual_empty = torch.tensor([], requires_grad=True)
        expected_empty = reference_torch.tensor(
            [], dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_output = functional.sigmoid(actual_empty)
        expected_empty_output = reference_functional.sigmoid(expected_empty)
        self.assert_tensor_matches(
            actual_empty_output, expected_empty_output, case="empty forward"
        )
        actual_empty_output.sum().backward()
        expected_empty_output.sum().backward()
        self.assert_tensor_matches(
            actual_empty.grad, expected_empty.grad, case="empty gradient"
        )

        actual_rank_six_nonfinite = torch.tensor(
            [[[[[[0.5, float("inf")]]]]]], requires_grad=True
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.sigmoid(actual_rank_six_nonfinite)
        self.assertIsNone(actual_rank_six_nonfinite.grad)
        actual_rank_six_nonfinite.sum().backward()
        self.assertEqual(
            actual_rank_six_nonfinite.grad.tolist(), [[[[[[1.0, 1.0]]]]]]
        )

        actual_high_rank_nonfinite = torch.full(
            (1,) * 65, float("inf"), requires_grad=True
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.sigmoid(actual_high_rank_nonfinite)
        self.assertIsNone(actual_high_rank_nonfinite.grad)
        actual_high_rank_nonfinite.sum().backward()
        self.assertEqual(actual_high_rank_nonfinite.grad.item(), 1.0)

        actual_view_base = torch.tensor([[0.5, -1.0]], requires_grad=True)
        actual_view = actual_view_base[0]
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.sigmoid(actual_view)
        self.assertIsNone(actual_view_base.grad)
        actual_view.sum().backward()
        self.assertEqual(actual_view_base.grad.tolist(), [[1.0, 1.0]])

        actual_rank_six_view_base = torch.full(
            (2,) + (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        actual_rank_six_view = actual_rank_six_view_base[0]
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.sigmoid(actual_rank_six_view)
        actual_rank_six_view.sum().backward()
        self.assertEqual(actual_rank_six_view_base.grad.sum().item(), 2.0)

        actual_high_rank_nonleaf_base = torch.full(
            (1,) * 65, 0.5, requires_grad=True
        )
        actual_high_rank_nonleaf = actual_high_rank_nonleaf_base.sin()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.sigmoid(actual_high_rank_nonleaf)
        actual_high_rank_nonleaf.backward()
        self.assertIsNotNone(actual_high_rank_nonleaf_base.grad)

        with torch.no_grad():
            actual_no_grad = functional.sigmoid(actual_leaf)
        with reference_torch.no_grad():
            expected_no_grad = reference_functional.sigmoid(expected_leaf)
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        actual_detached = functional.sigmoid(actual_leaf.detach())
        expected_detached = reference_functional.sigmoid(expected_leaf.detach())
        self.assert_tensor_matches(
            actual_detached, expected_detached, case="detached"
        )

        self.assertFalse(hasattr(torch, "sigmoid"))
        self.assertTrue(hasattr(reference_torch, "sigmoid"))
        self.assertTrue(hasattr(torch.nn.functional, "sigmoid"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "sigmoid"))
        self.assertFalse(hasattr(torch.nn, "Sigmoid"))
        self.assertTrue(hasattr(reference_torch.nn, "Sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(functional, "sigmoid_"))
        self.assertFalse(hasattr(reference_functional, "sigmoid_"))


if __name__ == "__main__":
    unittest.main()
