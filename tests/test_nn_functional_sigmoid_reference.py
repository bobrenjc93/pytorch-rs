import copy
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional

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
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.sigmoid(actual_leaf)
        self.assertTrue(reference_functional.sigmoid(expected_leaf).requires_grad)

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
