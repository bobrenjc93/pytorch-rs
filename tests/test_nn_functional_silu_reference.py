import copy
import inspect
import pickle
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
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            cls.tensor_values(tensor).reshape(-1).view(np.uint32).copy(),
        )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        channels_last = module.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
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
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels_last", channels_last),
            ("channels_last_3d", channels_last_3d),
            (
                "numerical_edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
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
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)

        np.testing.assert_allclose(
            self.tensor_values(actual),
            self.tensor_values(expected),
            rtol=2.0e-6,
            atol=np.nextafter(np.float32(0), np.float32(1)),
            equal_nan=True,
        )

    def test_metadata_copy_and_pickle_match_pytorch_2_13(self):
        actual = functional.silu
        expected = reference_functional.silu
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        actual_signature = inspect.signature(actual)
        expected_signature = inspect.signature(expected)
        actual_parameters = tuple(actual_signature.parameters.values())
        expected_parameters = tuple(expected_signature.parameters.values())
        for actual_parameter, expected_parameter in zip(
            actual_parameters,
            expected_parameters,
            strict=True,
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
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertFalse(hasattr(actual, "__text_signature__"))
        self.assertFalse(hasattr(expected, "__text_signature__"))
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)),
                    actual,
                )

    def test_default_values_layouts_storage_and_nonmutation_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual_before = self.tensor_state(actual_input)
            expected_before = self.tensor_state(expected_input)
            actual = functional.silu(input=actual_input)
            expected = reference_functional.silu(input=expected_input)
            self.assert_tensor_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            actual_after = self.tensor_state(actual_input)
            expected_after = self.tensor_state(expected_input)
            self.assertEqual(actual_after[:-1], actual_before[:-1])
            self.assertEqual(expected_after[:-1], expected_before[:-1])
            np.testing.assert_array_equal(actual_after[-1], actual_before[-1])
            np.testing.assert_array_equal(expected_after[-1], expected_before[-1])

    @staticmethod
    def dispatch_observation(module, module_functional):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module_functional.silu(input=value)
        override_function, override_types, override_args, override_kwargs = Override.calls[0]

        tensor = module.tensor([0.5], dtype=module.float32, requires_grad=True)

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        observations = []
        for call in (
            lambda: module_functional.silu(tensor),
            lambda: module_functional.silu(input=tensor),
            lambda: module_functional.silu(tensor, False),
            lambda: module_functional.silu(input=tensor, inplace=True),
        ):
            mode = Mode()
            with mode:
                mode_result = call()
            mode_function, mode_types, mode_args, mode_kwargs = mode.calls[0]
            observations.append(
                (
                    mode_result is marker,
                    mode_function is module_functional.silu,
                    tuple(item.__name__ for item in mode_types),
                    mode_args == (tensor,),
                    mode_kwargs,
                )
            )

        return (
            override_result is marker,
            override_function is module_functional.silu,
            tuple(item.__name__ for item in override_types),
            override_args == (value,),
            override_kwargs,
            tuple(observations),
        )

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch, functional),
            self.dispatch_observation(reference_torch, reference_functional),
        )

    def test_supported_autograd_matches_pytorch_2_13(self):
        actual_leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_weights = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 3.0, -4.0]])
        expected_weights = reference_torch.tensor(
            [[1.0, -2.0, 0.5], [-0.25, 3.0, -4.0]],
            dtype=reference_torch.float32,
        )

        actual_output = functional.silu(actual_leaf)
        expected_output = reference_functional.silu(expected_leaf)
        self.assert_tensor_matches(actual_output, expected_output, case="forward")

        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_tensor_matches(actual_leaf.grad, expected_leaf.grad, case="grad")

        actual_before = self.tensor_values(actual_leaf.grad).copy()
        expected_before = self.tensor_values(expected_leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            actual_output.sum().backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            expected_output.sum().backward()
        np.testing.assert_array_equal(self.tensor_values(actual_leaf.grad), actual_before)
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad),
            expected_before,
        )

    @staticmethod
    def error(call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        return None

    def test_binding_errors_match_and_unsupported_boundaries_are_explicit(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5], dtype=reference_torch.float32)
        paired_calls = (
            (
                lambda: functional.silu(),
                lambda: reference_functional.silu(),
            ),
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
            (
                lambda: functional.silu(1),
                lambda: reference_functional.silu(1),
            ),
            (
                lambda: functional.silu(None),
                lambda: reference_functional.silu(None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

        before = self.tensor_state(actual)
        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch_rs\\.nn\\.functional\\.silu does not support inplace=True$",
        ):
            functional.silu(actual, inplace=True)
        after = self.tensor_state(actual)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])

        self.assertFalse(hasattr(torch.Tensor, "silu"))
        self.assertFalse(hasattr(torch.nn, "SiLU"))
        self.assertFalse(hasattr(functional, "silu_"))
        self.assertFalse(hasattr(torch, "silu"))
        self.assertTrue(hasattr(reference_torch.nn, "SiLU"))
        self.assertTrue(hasattr(reference_functional, "silu"))


if __name__ == "__main__":
    unittest.main()
