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
class FunctionalRelu6ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.relu6 differentials require pinned PyTorch 2.13.0"
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
            np.linspace(-3.0, 8.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        mixed_singleton = module.tensor(
            np.linspace(-1.0, 7.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist(),
            dtype=module.float32,
        ).permute(2, 1, 0)
        channels_last = module.tensor(
            np.linspace(-4.0, 9.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        edge_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F7F_FFFF,
                0x3F80_0000,
                0x40BF_FFFE,
                0x40BF_FFFF,
                0x40C0_0000,
                0x40C0_0001,
                0x40C0_0002,
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
            ("strided", base.transpose(0, 2)[1]),
            ("mixed singleton", mixed_singleton),
            ("channels last", channels_last),
            (
                "float32 edges",
                module.tensor(memoryview(edge_bits.view(np.float32))),
            ),
        )

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            if len(actual.shape) == 4:
                self.assertEqual(
                    actual.is_contiguous(memory_format=torch.channels_last),
                    expected.is_contiguous(
                        memory_format=reference_torch.channels_last
                    ),
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
        actual = functional.relu6
        expected = reference_functional.relu6
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        actual_signature = inspect.signature(actual)
        expected_signature = inspect.signature(expected)
        self.assertEqual(
            tuple(actual_signature.parameters),
            tuple(expected_signature.parameters),
        )
        for name in actual_signature.parameters:
            with self.subTest(parameter=name):
                self.assertEqual(
                    actual_signature.parameters[name].kind,
                    expected_signature.parameters[name].kind,
                )
                self.assertEqual(
                    actual_signature.parameters[name].default,
                    expected_signature.parameters[name].default,
                )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(
            tuple(actual.__annotations__), tuple(expected.__annotations__)
        )
        self.assertIs(actual.__annotations__["input"], torch.Tensor)
        self.assertIs(expected.__annotations__["input"], reference_torch.Tensor)
        self.assertIs(actual.__annotations__["inplace"], bool)
        self.assertIs(expected.__annotations__["inplace"], bool)
        self.assertIs(actual.__annotations__["return"], torch.Tensor)
        self.assertIs(expected.__annotations__["return"], reference_torch.Tensor)
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
            calls = (
                (
                    lambda: functional.relu6(actual_input),
                    lambda: reference_functional.relu6(expected_input),
                ),
                (
                    lambda: functional.relu6(actual_input, False),
                    lambda: reference_functional.relu6(expected_input, False),
                ),
                (
                    lambda: functional.relu6(
                        input=actual_input, inplace=False
                    ),
                    lambda: reference_functional.relu6(
                        input=expected_input, inplace=False
                    ),
                ),
            )
            for form, (actual_call, expected_call) in enumerate(calls):
                actual = actual_call()
                expected = expected_call()
                self.assert_tensor_matches(
                    actual, expected, case=(case, form)
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    if actual_input.numel():
                        self.assertNotEqual(
                            actual.data_ptr(), actual_input.data_ptr()
                        )
                        self.assertNotEqual(
                            expected.data_ptr(), expected_input.data_ptr()
                        )

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
            actual_first = functional.relu6(actual_input)
            actual_second = functional.relu6(actual_input)
            expected_first = reference_functional.relu6(expected_input)
            expected_second = reference_functional.relu6(expected_input)
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

        observations = []
        for inplace in (False, True):
            value = Override()
            result = module_functional.relu6(value, inplace=inplace)
            function, dispatch_types, args, kwargs = Override.calls.pop(0)
            observations.append(
                (
                    result is marker,
                    function is module_functional.relu6,
                    tuple(item.__name__ for item in dispatch_types),
                    args == (value,),
                    kwargs,
                )
            )

        tensor = module.tensor([0.5], requires_grad=True)

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for inplace in (False, True):
            mode = Mode()
            with mode:
                result = module_functional.relu6(tensor, inplace=inplace)
            function, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    function is module_functional.relu6,
                    tuple(item.__name__ for item in dispatch_types),
                    args == (tensor,),
                    kwargs,
                )
            )
        return observations

    def test_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch, functional),
            self.dispatch_observation(reference_torch, reference_functional),
        )

    def test_tracked_inputs_match_inside_no_grad_and_after_detach(self):
        actual_leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 6.0, 8.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 6.0, 8.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]

        with torch.no_grad():
            actual = functional.relu6(actual_input)
        with reference_torch.no_grad():
            expected = reference_functional.relu6(expected_input)
        self.assert_tensor_matches(actual, expected, case="no_grad")

        actual = functional.relu6(actual_input.detach())
        expected = reference_functional.relu6(expected_input.detach())
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

    def test_argument_errors_match_and_unsupported_boundary_is_explicit(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5], dtype=reference_torch.float32)
        paired_calls = (
            (
                lambda: functional.relu6(),
                lambda: reference_functional.relu6(),
            ),
            (
                lambda: functional.relu6(actual, False, None),
                lambda: reference_functional.relu6(expected, False, None),
            ),
            (
                lambda: functional.relu6(actual, input=actual),
                lambda: reference_functional.relu6(expected, input=expected),
            ),
            (
                lambda: functional.relu6(actual, out=None),
                lambda: reference_functional.relu6(expected, out=None),
            ),
            (
                lambda: functional.relu6(1),
                lambda: reference_functional.relu6(1),
            ),
            (
                lambda: functional.relu6(1, inplace=True),
                lambda: reference_functional.relu6(1, inplace=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

        tracked = torch.tensor([0.5], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^relu6\(\): autograd recording is not supported$",
        ):
            functional.relu6(tracked)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.relu6 does not support inplace=True$",
        ):
            functional.relu6(actual, inplace=True)

        self.assertFalse(hasattr(torch.nn, "ReLU6"))
        self.assertTrue(hasattr(reference_torch.nn, "ReLU6"))
        self.assertFalse(hasattr(torch.Tensor, "relu6"))
        self.assertFalse(hasattr(reference_torch.Tensor, "relu6"))


if __name__ == "__main__":
    unittest.main()
