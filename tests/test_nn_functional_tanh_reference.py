import copy
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional

if __package__:
    from .test_tanh import AUTOGRAD_INPUT_BITS, AUTOGRAD_WEIGHTS
else:
    from test_tanh import AUTOGRAD_INPUT_BITS, AUTOGRAD_WEIGHTS

try:
    import torch as reference_torch
    import torch.nn.functional as reference_functional
except ImportError:
    reference_torch = None
    reference_functional = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FunctionalTanhReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.tanh differentials require pinned PyTorch 2.13.0"
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
        return source.transpose(0, 2)[1]

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

        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
        )

    def test_metadata_documentation_copy_and_pickle_match_pytorch_2_13(self):
        actual = functional.tanh
        expected = reference_functional.tanh
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
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_input = self.make_case(torch, case)
            expected_input = self.make_case(reference_torch, case)
            actual = functional.tanh(input=actual_input)
            expected = reference_functional.tanh(input=expected_input)
            self.assert_tensor_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))

    def receiver_contract(self, function):
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
        result = function(receiver)

        class TorchFunctionReceiver:
            torch_function_calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.torch_function_calls += 1
                return object()

            def tanh(self):
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
            self.receiver_contract(functional.tanh),
            self.receiver_contract(reference_functional.tanh),
        )

    def mode_contract(self, module):
        function = module.nn.functional.tanh
        descriptor = inspect.getattr_static(module.Tensor, "tanh")
        source = module.tensor(
            [0.5], dtype=module.float32, requires_grad=True
        )
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
            (lambda: functional.tanh(), lambda: reference_functional.tanh()),
            (
                lambda: functional.tanh(actual, actual),
                lambda: reference_functional.tanh(expected, expected),
            ),
            (
                lambda: functional.tanh(actual, input=actual),
                lambda: reference_functional.tanh(expected, input=expected),
            ),
            (
                lambda: functional.tanh(actual, out=None),
                lambda: reference_functional.tanh(expected, out=None),
            ),
            (lambda: functional.tanh(1), lambda: reference_functional.tanh(1)),
            (
                lambda: functional.tanh(None),
                lambda: reference_functional.tanh(None),
            ),
            (
                lambda: functional.tanh([]),
                lambda: reference_functional.tanh([]),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

        class NonCallableReceiver:
            tanh = 1

        self.assertEqual(
            self.error(lambda: functional.tanh(NonCallableReceiver())),
            self.error(lambda: reference_functional.tanh(NonCallableReceiver())),
        )

    def test_supported_and_unsupported_boundaries_are_explicit(self):
        actual_scalar = torch.tensor(0.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(
            0.5, dtype=reference_torch.float32, requires_grad=True
        )
        actual_scalar_output = functional.tanh(actual_scalar)
        expected_scalar_output = reference_functional.tanh(expected_scalar)
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

        values = AUTOGRAD_INPUT_BITS.view(np.float32).tolist()
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.tolist(), dtype=reference_torch.float32
        )
        actual_output = functional.tanh(actual_leaf)
        expected_output = reference_functional.tanh(expected_leaf)
        self.assert_tensor_matches(
            actual_output, expected_output, case="rank-one forward"
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "TanhBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<TanhBackward0>",
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
        actual_empty_output = functional.tanh(actual_empty)
        expected_empty_output = reference_functional.tanh(expected_empty)
        self.assert_tensor_matches(
            actual_empty_output, expected_empty_output, case="empty forward"
        )
        actual_empty_output.sum().backward()
        expected_empty_output.sum().backward()
        self.assert_tensor_matches(
            actual_empty.grad, expected_empty.grad, case="empty gradient"
        )

        higher_order = torch.tensor([0.25, -0.25], requires_grad=True)
        higher_order_loss = functional.tanh(higher_order).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

        actual_matrix = torch.tensor([[0.5, -1.0]], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(actual_matrix)
        self.assertIsNone(actual_matrix.grad)
        actual_matrix.sum().backward()
        self.assertEqual(actual_matrix.grad.tolist(), [[1.0, 1.0]])

        actual_view_base = torch.tensor([[0.5, -1.0]], requires_grad=True)
        actual_view = actual_view_base[0]
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(actual_view)
        self.assertIsNone(actual_view_base.grad)
        actual_view.sum().backward()
        self.assertEqual(actual_view_base.grad.tolist(), [[1.0, 1.0]])

        actual_nonfinite = torch.tensor([0.5, float("inf")], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): autograd recording is not supported$",
        ):
            functional.tanh(actual_nonfinite)
        self.assertIsNone(actual_nonfinite.grad)
        actual_nonfinite.sum().backward()
        self.assertEqual(actual_nonfinite.grad.tolist(), [1.0, 1.0])

        with torch.no_grad():
            actual_no_grad = functional.tanh(actual_leaf)
        with reference_torch.no_grad():
            expected_no_grad = reference_functional.tanh(expected_leaf)
        self.assert_tensor_matches(
            actual_no_grad, expected_no_grad, case="no_grad"
        )

        self.assertTrue(hasattr(torch.nn.functional, "tanh"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "tanh"))
        self.assertFalse(hasattr(torch.nn, "Tanh"))
        self.assertTrue(hasattr(reference_torch.nn, "Tanh"))
        self.assertFalse(hasattr(torch.Tensor, "tanh_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "tanh_"))
        self.assertFalse(hasattr(functional, "tanh_"))
        self.assertFalse(hasattr(reference_functional, "tanh_"))


if __name__ == "__main__":
    unittest.main()
