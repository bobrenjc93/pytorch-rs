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
class FunctionalSoftsignReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.softsign differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_bits(tensor):
        if type(tensor) is torch.Tensor:
            values = np.asarray(tensor, dtype=np.float32)
        else:
            values = tensor.detach().cpu().numpy()
        return values.reshape(-1).view(np.uint32)

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            cls.tensor_bits(tensor).copy(),
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
                0x4000_0000,
                0xC000_0000,
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
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)

        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_metadata_copy_and_pickle_match_pytorch_2_13(self):
        actual = functional.softsign
        expected = reference_functional.softsign
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
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

    def test_values_bits_layouts_storage_and_nonmutation_match(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual_before = self.tensor_state(actual_input)
            expected_before = self.tensor_state(expected_input)
            actual = functional.softsign(input=actual_input)
            expected = reference_functional.softsign(input=expected_input)
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

    def test_every_call_returns_fresh_storage_like_pytorch_2_13(self):
        for (case, actual_input), (_, expected_input) in zip(
            self.make_cases(torch), self.make_cases(reference_torch), strict=True
        ):
            actual_first = functional.softsign(actual_input)
            actual_second = functional.softsign(actual_input)
            expected_first = reference_functional.softsign(expected_input)
            expected_second = reference_functional.softsign(expected_input)
            with self.subTest(case=case):
                self.assertFalse(actual_first.is_set_to(actual_second))
                self.assertFalse(expected_first.is_set_to(expected_second))
                if actual_first.numel():
                    self.assertNotEqual(
                        actual_first.data_ptr(), actual_second.data_ptr()
                    )
                    self.assertNotEqual(
                        expected_first.data_ptr(), expected_second.data_ptr()
                    )

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
        override_result = module_functional.softsign(input=value)
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
            mode_result = module_functional.softsign(tensor)
        mode_function, mode_types, mode_args, mode_kwargs = mode.calls[0]
        return (
            override_result is marker,
            override_function is module_functional.softsign,
            tuple(item.__name__ for item in override_types),
            override_args == (value,),
            override_kwargs,
            mode_result is marker,
            mode_function is module_functional.softsign,
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
            actual = functional.softsign(actual_input)
        with reference_torch.no_grad():
            expected = reference_functional.softsign(expected_input)
        self.assert_tensor_matches(actual, expected, case="no_grad")

        actual = functional.softsign(actual_input.detach())
        expected = reference_functional.softsign(expected_input.detach())
        self.assert_tensor_matches(actual, expected, case="detached")
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)

    @staticmethod
    def error(call):
        try:
            call()
        except Exception as error:
            return type(error), str(error), error.args
        return None

    def test_argument_errors_match_and_module_boundary_is_explicit(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5], dtype=reference_torch.float32)
        paired_calls = (
            (
                lambda: functional.softsign(),
                lambda: reference_functional.softsign(),
            ),
            (
                lambda: functional.softsign(actual, actual),
                lambda: reference_functional.softsign(expected, expected),
            ),
            (
                lambda: functional.softsign(actual, input=actual),
                lambda: reference_functional.softsign(expected, input=expected),
            ),
            (
                lambda: functional.softsign(actual, out=None),
                lambda: reference_functional.softsign(expected, out=None),
            ),
            (
                lambda: functional.softsign(1),
                lambda: reference_functional.softsign(1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

        self.assertFalse(hasattr(torch.nn, "Softsign"))
        self.assertTrue(hasattr(reference_torch.nn, "Softsign"))
        self.assertFalse(hasattr(torch.Tensor, "softsign"))
        self.assertFalse(hasattr(reference_torch.Tensor, "softsign"))


if __name__ == "__main__":
    unittest.main()
