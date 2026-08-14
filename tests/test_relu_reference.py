import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ReluReferenceTests(unittest.TestCase):
    @staticmethod
    def call_relu(module, tensor, form):
        if form == "method":
            return tensor.relu()
        if form == "positional":
            return module.relu(tensor)
        return module.relu(**{form: tensor})

    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_matches(self, actual_input, expected_input, expected_bits, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(
                actual_input.storage_offset(), expected_input.storage_offset()
            )

        for form in ("method", "positional", "input", "x", "a"):
            actual_output = self.call_relu(torch, actual_input, form)
            expected_output = self.call_relu(reference_torch, expected_input, form)
            invocation = (case, form)

            with self.subTest(case=invocation):
                self.assertFalse(actual_output.is_set_to(actual_input))
                self.assertFalse(expected_output.is_set_to(expected_input))

            self.assert_metadata_matches(actual_output, expected_output, case=invocation)
            with self.subTest(case=invocation):
                actual_bits = (
                    np.asarray(actual_output, dtype=np.float32)
                    .reshape(-1)
                    .view(np.uint32)
                )
                reference_bits = (
                    expected_output.cpu().numpy().reshape(-1).view(np.uint32)
                )
                np.testing.assert_array_equal(actual_bits, reference_bits)
                np.testing.assert_array_equal(
                    actual_bits,
                    np.asarray(expected_bits, dtype=np.uint32).reshape(-1),
                )

    def assert_backward_error_matches(self, actual_loss, expected_loss):
        with self.assertRaises(RuntimeError) as actual_raised:
            actual_loss.backward()
        with self.assertRaises(RuntimeError) as expected_raised:
            expected_loss.backward()
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_signed_zero_values_layouts_and_copies_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        for zero_bits in (0x8000_0000, 0x0000_0000):
            zero = np.asarray(zero_bits, dtype=np.uint32).view(np.float32).item()
            self.assert_matches(
                torch.tensor(zero),
                reference_torch.tensor(zero),
                [zero_bits],
                case=("scalar", hex(zero_bits)),
            )

        input_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x8000_0001,
                0x0000_0001,
                0xFF7F_FFFF,
                0x7F7F_FFFF,
                0xBF00_0000,
                0x3F00_0000,
            ),
            dtype=np.uint32,
        )
        expected_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0x0000_0000,
                0x3F80_0000,
                0x0000_0000,
                0x7F80_0000,
                0x0000_0000,
                0x0000_0001,
                0x0000_0000,
                0x7F7F_FFFF,
                0x0000_0000,
                0x3F00_0000,
            ),
            dtype=np.uint32,
        )
        storage_bits = np.concatenate(
            (np.full(input_bits.size, 0x3F80_0000, dtype=np.uint32), input_bits)
        )
        storage_values = memoryview(storage_bits.view(np.float32))
        actual_offset = torch.tensor(storage_values).reshape(2, 3, 4)[1]
        expected_offset = reference_torch.tensor(storage_values).reshape(2, 3, 4)[1]

        self.assert_matches(
            actual_offset,
            expected_offset,
            expected_bits,
            case="offset contiguous",
        )
        self.assert_matches(
            actual_offset.transpose(0, 1),
            expected_offset.transpose(0, 1),
            expected_bits.reshape(3, 4).transpose(1, 0).reshape(-1),
            case="offset strided",
        )

    def test_vjp_matches_pytorch_for_threshold_edges_and_nans(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        input_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x0000_0001,
                0x8000_0001,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x8000_0000,
                0x8000_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0xC040_0000,
                0x4040_0000,
                0x4000_0000,
                0x7FC0_1234,
            ),
            dtype=np.uint32,
        )
        expected_gradient_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x7F80_0000,
                0x0000_0000,
                0x3F00_0000,
                0xBF00_0000,
                0xC040_0000,
                0x0000_0000,
                0x4000_0000,
                0x0000_0000,
            ),
            dtype=np.uint32,
        )
        actual_leaf = torch.tensor(
            memoryview(input_bits.view(np.float32)), requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            input_bits.view(np.float32), requires_grad=True
        )
        actual_weights = torch.tensor(memoryview(weight_bits.view(np.float32)))
        expected_weights = reference_torch.tensor(weight_bits.view(np.float32))
        actual_output = actual_leaf.relu()
        expected_output = expected_leaf.relu()

        self.assert_metadata_matches(
            actual_output, expected_output, case="special-value output"
        )
        self.assertFalse(actual_output.is_set_to(actual_leaf))
        self.assertFalse(expected_output.is_set_to(expected_leaf))
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()

        actual_gradient_bits = np.asarray(actual_leaf.grad).view(np.uint32)
        reference_gradient_bits = expected_leaf.grad.detach().numpy().view(np.uint32)
        np.testing.assert_array_equal(reference_gradient_bits, expected_gradient_bits)
        np.testing.assert_array_equal(actual_gradient_bits, reference_gradient_bits)

    def test_scalar_empty_offset_strided_and_accumulation_match_pytorch(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_scalar = torch.tensor(2.0, requires_grad=True)
        expected_scalar = reference_torch.tensor(2.0, requires_grad=True)
        actual_scalar_output = actual_scalar.relu()
        expected_scalar_output = expected_scalar.relu()
        self.assert_metadata_matches(
            actual_scalar_output, expected_scalar_output, case="scalar output"
        )
        actual_scalar_output.backward()
        expected_scalar_output.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_scalar.grad).view(np.uint32),
            expected_scalar.grad.detach().numpy().view(np.uint32),
        )
        self.assert_backward_error_matches(
            actual_scalar_output, expected_scalar_output
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty_output = actual_empty.relu()
        expected_empty_output = expected_empty.relu()
        self.assert_metadata_matches(
            actual_empty_output, expected_empty_output, case="empty output"
        )
        actual_empty_loss = actual_empty_output.sum()
        expected_empty_loss = expected_empty_output.sum()
        actual_empty_loss.backward()
        expected_empty_loss.backward()
        self.assert_metadata_matches(
            actual_empty.grad, expected_empty.grad, case="empty gradient"
        )
        self.assert_backward_error_matches(actual_empty_loss, expected_empty_loss)

        storage = np.asarray(
            [9.0] * 12
            + [
                -1.0,
                2.0,
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                0.5,
                3.0,
                -4.0,
                5.0,
                -6.0,
                7.0,
            ],
            dtype=np.float32,
        ).reshape(2, 3, 4)
        actual_source = torch.tensor(storage.tolist(), requires_grad=True)
        expected_source = reference_torch.tensor(storage, requires_grad=True)
        actual_offset = actual_source[1]
        expected_offset = expected_source[1]
        actual_offset_output = actual_offset.relu()
        expected_offset_output = expected_offset.relu()
        self.assert_metadata_matches(
            actual_offset_output, expected_offset_output, case="offset output"
        )
        self.assertFalse(actual_offset_output.is_set_to(actual_offset))
        self.assertFalse(expected_offset_output.is_set_to(expected_offset))
        actual_offset_output.sum().backward()
        expected_offset_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_source.grad).view(np.uint32),
            expected_source.grad.detach().numpy().view(np.uint32),
        )

        actual_strided = actual_offset.transpose(0, 1)
        expected_strided = expected_offset.transpose(0, 1)
        actual_strided_output = actual_strided.relu()
        expected_strided_output = expected_strided.relu()
        self.assert_metadata_matches(
            actual_strided_output, expected_strided_output, case="strided output"
        )
        self.assertFalse(actual_strided_output.is_set_to(actual_strided))
        self.assertFalse(expected_strided_output.is_set_to(expected_strided))
        weights = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
        (actual_strided_output * torch.tensor(weights.tolist())).sum().backward()
        (
            expected_strided_output * reference_torch.tensor(weights)
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_source.grad).view(np.uint32),
            expected_source.grad.detach().numpy().view(np.uint32),
        )

    def test_detach_no_grad_and_freed_graph_match_pytorch(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[-1.0, 2.0], [0.0, float("nan")]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)

        actual_detached_input = actual_leaf.detach().relu()
        expected_detached_input = expected_leaf.detach().relu()
        self.assert_metadata_matches(
            actual_detached_input,
            expected_detached_input,
            case="detached input",
        )

        actual_tracked = actual_leaf.relu()
        expected_tracked = expected_leaf.relu()
        actual_detached_output = actual_tracked.detach()
        expected_detached_output = expected_tracked.detach()
        self.assert_metadata_matches(
            actual_detached_output,
            expected_detached_output,
            case="detached output",
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.transpose(0, 1).relu()
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.transpose(0, 1).relu()
        self.assert_metadata_matches(
            actual_untracked, expected_untracked, case="no_grad output"
        )

        with torch.no_grad():
            actual_no_grad_view = actual_leaf.transpose(0, 1)
        with reference_torch.no_grad():
            expected_no_grad_view = expected_leaf.transpose(0, 1)
        actual_boundary_output = actual_no_grad_view.relu()
        expected_boundary_output = expected_no_grad_view.relu()
        self.assert_metadata_matches(
            actual_boundary_output,
            expected_boundary_output,
            case="operation after no_grad view",
        )
        actual_boundary_loss = actual_boundary_output.sum()
        expected_boundary_loss = expected_boundary_output.sum()
        actual_boundary_loss.backward()
        expected_boundary_loss.backward()
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)
        self.assert_backward_error_matches(
            actual_boundary_loss, expected_boundary_loss
        )

        actual_loss = actual_tracked.sum()
        expected_loss = expected_tracked.sum()
        actual_loss.backward()
        expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_leaf.grad.detach().numpy().view(np.uint32),
        )
        self.assert_backward_error_matches(actual_loss, expected_loss)

    def test_top_level_empty_call_forms_match_the_method_and_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_input = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        expected_input = reference_torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        actual_method = actual_input.relu()

        for form in ("positional", "input", "x", "a"):
            actual_output = self.call_relu(torch, actual_input, form)
            expected_output = self.call_relu(reference_torch, expected_input, form)
            self.assert_metadata_matches(actual_output, expected_output, case=form)
            with self.subTest(form=form):
                self.assertEqual(actual_output.shape, actual_method.shape)
                self.assertEqual(actual_output.stride(), actual_method.stride())
                self.assertEqual(
                    actual_output.storage_offset(), actual_method.storage_offset()
                )
                self.assertFalse(actual_output.is_set_to(actual_input))
                self.assertFalse(expected_output.is_set_to(expected_input))
                np.testing.assert_array_equal(
                    np.asarray(actual_output), np.asarray(actual_method)
                )

    def test_top_level_autograd_repeated_backward_and_no_grad_match_the_method(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def autograd_outcome(module, form):
            values = np.asarray(
                [9.0] * 12
                + [
                    -1.0,
                    2.0,
                    0.0,
                    -0.0,
                    np.inf,
                    -np.inf,
                    0.5,
                    3.0,
                    -4.0,
                    5.0,
                    -6.0,
                    7.0,
                ],
                dtype=np.float32,
            ).reshape(2, 3, 4)
            leaf = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
            )
            source = leaf[1].transpose(0, 1)
            output = self.call_relu(module, source, form)
            weights = module.tensor(
                np.arange(1, 13, dtype=np.float32).reshape(4, 3).tolist(),
                dtype=module.float32,
            )
            loss = (output * weights).sum()
            state = (
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.is_contiguous(),
                output.requires_grad,
                output.is_leaf,
            )
            output_values = np.asarray(output.detach()).copy()
            loss.backward()
            gradient = np.asarray(leaf.grad).copy()
            try:
                loss.backward()
            except RuntimeError as error:
                repeated_backward_error = (type(error).__name__, str(error))
            else:
                raise AssertionError("a second backward through ReLU must fail")
            return state, output_values, gradient, repeated_backward_error

        def no_grad_outcome(module, form):
            leaf = module.tensor(
                [[-1.0, 2.0], [0.0, 3.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = leaf.transpose(0, 1)
            with module.no_grad():
                output = self.call_relu(module, source, form)
            return (
                tuple(output.shape),
                output.stride(),
                output.storage_offset(),
                output.requires_grad,
                output.is_leaf,
                np.asarray(output).copy(),
            )

        actual_method = autograd_outcome(torch, "method")
        actual_method_no_grad = no_grad_outcome(torch, "method")
        for form in ("positional", "input", "x", "a"):
            actual = autograd_outcome(torch, form)
            expected = autograd_outcome(reference_torch, form)
            with self.subTest(form=form, behavior="autograd-state"):
                self.assertEqual(actual[0], expected[0])
                self.assertEqual(actual[0], actual_method[0])
                self.assertEqual(actual[3], expected[3])
                self.assertEqual(actual[3], actual_method[3])
            for index, behavior in ((1, "values"), (2, "gradient")):
                with self.subTest(form=form, behavior=behavior):
                    np.testing.assert_array_equal(actual[index], expected[index])
                    np.testing.assert_array_equal(actual[index], actual_method[index])

            actual_no_grad = no_grad_outcome(torch, form)
            expected_no_grad = no_grad_outcome(reference_torch, form)
            with self.subTest(form=form, behavior="no-grad-state"):
                self.assertEqual(actual_no_grad[:5], expected_no_grad[:5])
                self.assertEqual(actual_no_grad[:5], actual_method_no_grad[:5])
                np.testing.assert_array_equal(
                    actual_no_grad[5], expected_no_grad[5]
                )
                np.testing.assert_array_equal(
                    actual_no_grad[5], actual_method_no_grad[5]
                )

    def test_top_level_callable_metadata_and_export_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.relu
        expected = reference_torch.relu

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__module__, torch.tensor.__module__)
        self.assertEqual(expected.__module__, reference_torch.tensor.__module__)
        self.assertEqual(torch.__all__.count("relu"), 1)
        self.assertEqual(reference_torch.__all__.count("relu"), 1)
        for function in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def test_top_level_binding_and_error_precedence_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.relu(), lambda: reference_torch.relu()),
            (
                lambda: torch.relu(actual, actual),
                lambda: reference_torch.relu(expected, expected),
            ),
            (
                lambda: torch.relu(actual, input=actual),
                lambda: reference_torch.relu(expected, input=expected),
            ),
            (
                lambda: torch.relu(actual, extra=True, input=actual),
                lambda: reference_torch.relu(expected, extra=True, input=expected),
            ),
            (
                lambda: torch.relu(actual, input=actual, extra=True),
                lambda: reference_torch.relu(expected, input=expected, extra=True),
            ),
            (
                lambda: torch.relu(extra=actual),
                lambda: reference_torch.relu(extra=expected),
            ),
            (
                lambda: torch.relu(1, extra=True),
                lambda: reference_torch.relu(1, extra=True),
            ),
            (lambda: torch.relu(input=[]), lambda: reference_torch.relu(input=[])),
            (lambda: torch.relu(a=1), lambda: reference_torch.relu(a=1)),
            (lambda: torch.relu(x=[]), lambda: reference_torch.relu(x=[])),
            (
                lambda: torch.relu(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.relu(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
            (
                lambda: torch.relu(input=actual, a=actual),
                lambda: reference_torch.relu(input=expected, a=expected),
            ),
            (
                lambda: torch.relu(a=actual, x=actual),
                lambda: reference_torch.relu(a=expected, x=expected),
            ),
            (
                lambda: torch.relu(x=actual, a=actual),
                lambda: reference_torch.relu(x=expected, a=expected),
            ),
            (
                lambda: torch.relu(input=1, a=actual),
                lambda: reference_torch.relu(input=1, a=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(Exception) as actual_raised:
                    actual_call()
                with self.assertRaises(Exception) as expected_raised:
                    expected_call()
                self.assertIs(
                    type(actual_raised.exception), type(expected_raised.exception)
                )
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )


if __name__ == "__main__":
    unittest.main()
