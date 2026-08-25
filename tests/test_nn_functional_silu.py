import copy
import importlib
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


class FunctionalSiluTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            cls.tensor_bits(tensor).copy(),
        )

    @staticmethod
    def layout_cases():
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        channels_last = torch.tensor(
            np.linspace(-2.0, 2.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels_last", channels_last),
        )

    def assert_matches_composition(self, actual, expected, source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(
                actual.is_contiguous(memory_format=torch.channels_last),
                expected.is_contiguous(memory_format=torch.channels_last),
            )
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            self.assertFalse(actual.is_set_to(expected))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_import_signature_documentation_copy_and_pickle(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import silu

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(silu, functional.silu)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))

        function = functional.silu
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "silu")
        self.assertEqual(function.__qualname__, "silu")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (False,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertTrue(
            function.__doc__.startswith(
                "Apply the Sigmoid Linear Unit (SiLU) function, element-wise."
            )
        )
        self.assertEqual(tuple(signature.parameters), ("input", "inplace"))
        self.assertIs(parameters[0].annotation, torch.Tensor)
        self.assertIs(parameters[1].annotation, bool)
        self.assertIs(parameters[1].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["silu"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_supported_layouts_match_the_primitive_composition(self):
        for case, source in self.layout_cases():
            expected = source * source.sigmoid()
            before = self.tensor_state(source)
            calls = (
                lambda: functional.silu(source),
                lambda: functional.silu(source, False),
                lambda: functional.silu(input=source, inplace=False),
            )
            for form, call in enumerate(calls):
                actual = call()
                self.assert_matches_composition(
                    actual,
                    expected,
                    source,
                    case=(case, form),
                )
            after = self.tensor_state(source)
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

    def test_ieee_edges_match_the_primitive_composition_bitwise(self):
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
                0x42B4_0000,
                0xC2B4_0000,
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
        source = torch.tensor(memoryview(special_bits.view(np.float32)))
        before = self.tensor_state(source)
        expected = source * source.sigmoid()
        actual = functional.silu(source)
        self.assert_matches_composition(actual, expected, source, case="IEEE edges")
        after = self.tensor_state(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])

    def test_every_call_has_fresh_independent_storage(self):
        for case, source in self.layout_cases():
            first = functional.silu(source)
            second = functional.silu(source)
            with self.subTest(case=case):
                self.assertIsNot(first, second)
                self.assertFalse(first.is_set_to(second))
                self.assertFalse(first.is_set_to(source))
                if first.numel():
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertNotEqual(first.data_ptr(), source.data_ptr())

    def test_overrides_and_modes_observe_the_public_function_and_inplace(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(functional.silu(input=value), marker)
        self.assertIs(functional.silu(value, inplace=True), marker)
        self.assertEqual(len(Override.calls), 2)
        for index, expected_inplace in enumerate((False, True)):
            function, dispatch_types, args, kwargs = Override.calls[index]
            self.assertIs(function, functional.silu)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(args, (value,))
            self.assertEqual(kwargs, {"inplace": expected_inplace})

        source = torch.tensor([0.5], requires_grad=True)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = functional.silu(source)
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, functional.silu)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (source,))
        self.assertEqual(kwargs, {"inplace": False})

    def test_active_autograd_is_rejected_before_composition(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        for case, source in (
            ("scalar", torch.tensor(0.5, requires_grad=True)),
            ("noncontiguous view", leaf.transpose(0, 1)[1]),
        ):
            before = self.tensor_state(source)
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^silu\(\): autograd recording is not supported$",
                ):
                    functional.silu(source)
                self.assertIsNone(source.grad)
                after = self.tensor_state(source)
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        with torch.no_grad():
            expected = source * source.sigmoid()
            actual = functional.silu(source)
        self.assert_matches_composition(actual, expected, source, case="no_grad")
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)

        detached = source.detach()
        expected = detached * detached.sigmoid()
        actual = functional.silu(detached)
        self.assert_matches_composition(actual, expected, detached, case="detached")
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        self.assertIsNone(leaf.grad)

    def test_inplace_true_is_rejected_before_computation_or_mutation(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-2.0, -0.0, 1.0]], requires_grad=True
        )
        source = leaf[1]
        before = self.tensor_state(source)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.silu does not support inplace=True$",
        ):
            functional.silu(source, inplace=True)
        after = self.tensor_state(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])
        self.assertIsNone(leaf.grad)

        class ComputationSpy:
            computed = False

            def sigmoid(self):
                self.computed = True
                raise AssertionError("sigmoid must not run")

        spy = ComputationSpy()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.silu does not support inplace=True$",
        ):
            functional.silu(spy, True)
        self.assertFalse(spy.computed)

    def test_argument_errors_and_module_boundary(self):
        source = torch.tensor([0.5])
        cases = (
            (
                lambda: functional.silu(),
                TypeError,
                "silu() missing 1 required positional argument: 'input'",
            ),
            (
                lambda: functional.silu(source, False, False),
                TypeError,
                "silu() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: functional.silu(source, input=source),
                TypeError,
                "silu() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.silu(source, False, inplace=False),
                TypeError,
                "silu() got multiple values for argument 'inplace'",
            ),
            (
                lambda: functional.silu(source, out=None),
                TypeError,
                "silu() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: functional.silu(1),
                TypeError,
                "silu(): argument 'input' (position 1) must be Tensor, not int",
            ),
        )
        for case, (call, error_type, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertFalse(hasattr(nn, "SiLU"))
        self.assertFalse(hasattr(torch.Tensor, "silu"))
        self.assertFalse(hasattr(functional, "silu_"))


if __name__ == "__main__":
    unittest.main()
