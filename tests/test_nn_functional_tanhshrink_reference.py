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
class FunctionalTanhshrinkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.tanhshrink differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)

        actual_values = self.tensor_values(actual)
        expected_values = self.tensor_values(expected)
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-5,
                atol=4.0e-8,
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                np.signbit(actual_values[expected_values == 0]),
                np.signbit(expected_values[expected_values == 0]),
            )
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32),
                    expected_values.reshape(-1).view(np.uint32),
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
            np.linspace(-2.0, 2.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
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
            ("row major", base),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels last", channels_last),
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    def test_metadata_copy_and_pickle_match_pytorch_2_13(self):
        actual = functional.tanhshrink
        expected = reference_functional.tanhshrink
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)),
                    actual,
                )

    def test_values_layouts_empty_offsets_storage_and_nonmutation_match(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual_before = self.tensor_values(actual_input).copy()
            expected_before = self.tensor_values(expected_input).copy()
            actual = functional.tanhshrink(input=actual_input)
            expected = reference_functional.tanhshrink(input=expected_input)
            self.assert_tensor_matches(
                actual,
                expected,
                case=case,
                exact_bits=case == "numerical edges",
            )
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            np.testing.assert_array_equal(
                self.tensor_values(actual_input), actual_before
            )
            np.testing.assert_array_equal(
                self.tensor_values(expected_input), expected_before
            )

    def test_near_zero_normal_values_match_pytorch_bits(self):
        values = np.asarray(
            (
                1.0e-7,
                -1.0e-7,
                3.0e-7,
                -3.0e-7,
                1.0e-6,
                -1.0e-6,
                1.0e-5,
                -1.0e-5,
                1.0e-4,
                -1.0e-4,
                2.0e-4,
                -2.0e-4,
                5.0e-4,
                -5.0e-4,
                1.0e-3,
                -1.0e-3,
                2.0e-3,
                -2.0e-3,
                5.0e-3,
                -5.0e-3,
                1.0e-2,
                -1.0e-2,
            ),
            dtype=np.float32,
        )
        actual = functional.tanhshrink(torch.tensor(memoryview(values)))
        expected = reference_functional.tanhshrink(
            reference_torch.tensor(
                values.copy(), dtype=reference_torch.float32
            )
        )

        np.testing.assert_array_equal(
            self.tensor_values(actual).view(np.uint32),
            self.tensor_values(expected).view(np.uint32),
        )

    def dispatch_observation(self, module, module_functional):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module_functional.tanhshrink(input=value)
        override_function, override_types, override_args, override_kwargs = (
            Override.calls[0]
        )

        tensor = module.tensor([0.5], requires_grad=True)

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = Mode()
        with mode:
            mode_result = module_functional.tanhshrink(tensor)
        mode_function, mode_types, mode_args, mode_kwargs = mode.calls[0]
        return (
            override_result is marker,
            override_function is module_functional.tanhshrink,
            tuple(item.__name__ for item in override_types),
            override_args == (value,),
            override_kwargs,
            mode_result is marker,
            mode_function is module_functional.tanhshrink,
            tuple(item.__name__ for item in mode_types),
            mode_args == (tensor,),
            mode_kwargs,
        )

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch, functional),
            self.dispatch_observation(reference_torch, reference_functional),
        )

    def test_tracked_inputs_match_inside_no_grad_and_after_detach(self):
        actual_leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]
        with torch.no_grad():
            actual = functional.tanhshrink(actual_input)
        with reference_torch.no_grad():
            expected = reference_functional.tanhshrink(expected_input)
        self.assert_tensor_matches(actual, expected, case="no_grad")

        actual = functional.tanhshrink(actual_input.detach())
        expected = reference_functional.tanhshrink(expected_input.detach())
        self.assert_tensor_matches(actual, expected, case="detached")
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)

    def test_argument_errors_match_and_module_boundary_is_explicit(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5], dtype=reference_torch.float32)
        paired_calls = (
            (
                lambda: functional.tanhshrink(),
                lambda: reference_functional.tanhshrink(),
            ),
            (
                lambda: functional.tanhshrink(actual, actual),
                lambda: reference_functional.tanhshrink(expected, expected),
            ),
            (
                lambda: functional.tanhshrink(actual, input=actual),
                lambda: reference_functional.tanhshrink(expected, input=expected),
            ),
            (
                lambda: functional.tanhshrink(actual, out=None),
                lambda: reference_functional.tanhshrink(expected, out=None),
            ),
            (
                lambda: functional.tanhshrink(1),
                lambda: reference_functional.tanhshrink(1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
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

        self.assertFalse(hasattr(torch.nn, "Tanhshrink"))
        self.assertTrue(hasattr(reference_torch.nn, "Tanhshrink"))
        self.assertFalse(hasattr(torch.Tensor, "tanhshrink"))
        self.assertFalse(hasattr(reference_torch.Tensor, "tanhshrink"))


if __name__ == "__main__":
    unittest.main()
