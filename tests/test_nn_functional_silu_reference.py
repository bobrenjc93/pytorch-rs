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
class FunctionalSiluReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.silu differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def error(call):
        try:
            result = call()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        return "OK", repr(result)

    @staticmethod
    def make_case(module, case, *, requires_grad=False):
        if case == "scalar":
            return module.tensor(
                -0.0, dtype=module.float32, requires_grad=requires_grad
            )
        if case == "empty":
            return module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=requires_grad
            ).transpose(0, 2)[1]

        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        source = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        if case == "offset":
            return source[1]
        if case == "noncontiguous":
            return source.transpose(0, 2)[1]
        if case == "channels_last":
            values = np.linspace(-15.0, 15.0, 120, dtype=np.float32).reshape(
                2, 3, 4, 5
            )
            return module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=requires_grad
            ).contiguous(memory_format=module.channels_last)

        values = np.linspace(-90.0, 90.0, 720, dtype=np.float32).reshape(
            2, 3, 4, 5, 6
        )
        return module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=requires_grad
        ).contiguous(memory_format=module.channels_last_3d)

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor.detach() if tensor.requires_grad else tensor)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(
                actual.is_contiguous(memory_format=torch.channels_last),
                expected.is_contiguous(memory_format=reference_torch.channels_last),
            )
            self.assertEqual(
                actual.is_contiguous(memory_format=torch.channels_last_3d),
                expected.is_contiguous(memory_format=reference_torch.channels_last_3d),
            )
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
        actual = functional.silu
        expected = reference_functional.silu
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__module__, "torch_rs.nn.functional")
        self.assertEqual(expected.__module__, "torch.nn.functional")
        self.assertFalse(hasattr(actual, "__text_signature__"))
        self.assertFalse(hasattr(expected, "__text_signature__"))

        actual_signature = inspect.signature(actual)
        expected_signature = inspect.signature(expected)
        actual_parameters = tuple(actual_signature.parameters.values())
        expected_parameters = tuple(expected_signature.parameters.values())
        for actual_parameter, expected_parameter in zip(
            actual_parameters, expected_parameters, strict=True
        ):
            self.assertEqual(actual_parameter.name, expected_parameter.name)
            self.assertEqual(actual_parameter.kind, expected_parameter.kind)
            self.assertEqual(actual_parameter.default, expected_parameter.default)
        self.assertIs(actual_parameters[0].annotation, torch.Tensor)
        self.assertIs(expected_parameters[0].annotation, reference_torch.Tensor)
        self.assertIs(actual_parameters[1].annotation, bool)
        self.assertIs(expected_parameters[1].annotation, bool)
        self.assertIs(actual_signature.return_annotation, torch.Tensor)
        self.assertIs(expected_signature.return_annotation, reference_torch.Tensor)

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
        for case in (
            "scalar",
            "empty",
            "offset",
            "noncontiguous",
            "channels_last",
            "channels_last_3d",
        ):
            actual_input = self.make_case(torch, case)
            expected_input = self.make_case(reference_torch, case)
            actual = functional.silu(input=actual_input)
            expected = reference_functional.silu(input=expected_input)
            self.assert_tensor_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def test_numerical_edges_match_pytorch_2_13_with_float32_tolerance(self):
        special_bits = np.asarray(
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
        actual_input = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected_input = reference_torch.tensor(
            special_bits.view(np.float32), dtype=reference_torch.float32
        )
        self.assert_tensor_matches(
            functional.silu(actual_input),
            reference_functional.silu(expected_input),
            case="special",
        )

    def mode_contract(self, module):
        function = module.nn.functional.silu
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
                    dispatched is function,
                    dispatched.__name__,
                    dispatched.__qualname__,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args) == 1 and args[0] is source,
                    kwargs,
                )
            )
        return tuple(observations)

    def override_contract(self, module):
        function = module.nn.functional.silu
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        result = function(value)
        called_function, dispatch_types, args, kwargs = Override.calls[0]
        return (
            result is marker,
            called_function is function,
            tuple(item.__name__ for item in dispatch_types),
            len(args) == 1 and args[0] is value,
            kwargs,
        )

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(self.mode_contract(torch), self.mode_contract(reference_torch))
        self.assertEqual(
            self.override_contract(torch),
            self.override_contract(reference_torch),
        )

    def test_argument_and_receiver_errors_match_pytorch_2_13(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5])
        cases = (
            (lambda: functional.silu(), lambda: reference_functional.silu()),
            (
                lambda: functional.silu(actual, False, None),
                lambda: reference_functional.silu(expected, False, None),
            ),
            (
                lambda: functional.silu(actual, input=actual),
                lambda: reference_functional.silu(expected, input=expected),
            ),
            (
                lambda: functional.silu(actual, out=None),
                lambda: reference_functional.silu(expected, out=None),
            ),
            (lambda: functional.silu(1), lambda: reference_functional.silu(1)),
            (
                lambda: functional.silu(None),
                lambda: reference_functional.silu(None),
            ),
            (lambda: functional.silu([]), lambda: reference_functional.silu([])),
            (
                lambda: functional.silu(1, inplace=True),
                lambda: reference_functional.silu(1, inplace=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_supported_autograd_matches_pytorch_2_13(self):
        values = np.asarray(
            [-3.0, -1.0, -0.0, 0.5, 2.0, 4.0, -6.0, 8.0],
            dtype=np.float32,
        ).reshape(2, 4)
        actual_input = torch.tensor(values.tolist(), requires_grad=True)
        expected_input = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        weights = np.asarray(
            [1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0],
            dtype=np.float32,
        ).reshape(2, 4)
        actual_weights = torch.tensor(weights.tolist())
        expected_weights = reference_torch.tensor(
            weights, dtype=reference_torch.float32
        )

        actual_output = functional.silu(actual_input)
        expected_output = reference_functional.silu(expected_input)
        self.assert_tensor_matches(actual_output, expected_output, case="forward")

        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_tensor_matches(
            actual_input.grad,
            expected_input.grad,
            case="weighted gradient",
        )

        actual_parent = torch.tensor(values.tolist(), requires_grad=True)
        expected_parent = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_nonleaf = actual_parent.sin()
        expected_nonleaf = expected_parent.sin()
        actual_nonleaf_output = functional.silu(actual_nonleaf)
        expected_nonleaf_output = reference_functional.silu(expected_nonleaf)
        self.assert_tensor_matches(
            actual_nonleaf_output,
            expected_nonleaf_output,
            case="nonleaf forward",
        )
        actual_nonleaf_output.sum().backward()
        expected_nonleaf_output.sum().backward()
        self.assert_tensor_matches(
            actual_parent.grad,
            expected_parent.grad,
            case="nonleaf gradient",
        )


if __name__ == "__main__":
    unittest.main()
