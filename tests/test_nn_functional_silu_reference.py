import inspect
import re
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

    def make_case(self, module, case, *, requires_grad):
        if case == "scalar":
            leaf = module.tensor(
                -0.0, dtype=module.float32, requires_grad=requires_grad
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=requires_grad
            )
            return leaf, leaf.transpose(0, 2)[1]

        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "strided":
            return leaf, leaf.transpose(0, 2)[1]

        rank_four = module.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        if case == "channels_last":
            return rank_four, rank_four

        rank_five = module.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        if case == "channels_last_3d":
            return rank_five, rank_five

        raise AssertionError(f"unknown SiLU case: {case}")

    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
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

    def assert_values_match(self, actual, expected, *, case):
        with self.subTest(case=case):
            np.testing.assert_allclose(
                np.asarray(actual.detach()),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=0.0,
            )

    def test_import_signature_and_documentation_match_pytorch_2_13(self):
        self.assertIs(torch.nn.functional, functional)
        self.assertIs(type(functional.silu), types.FunctionType)
        self.assertIs(type(reference_functional.silu), types.FunctionType)
        self.assertEqual(functional.silu.__name__, reference_functional.silu.__name__)
        self.assertEqual(
            functional.silu.__qualname__, reference_functional.silu.__qualname__
        )
        self.assertEqual(functional.silu.__doc__, reference_functional.silu.__doc__)
        self.assertEqual(
            functional.silu.__defaults__, reference_functional.silu.__defaults__
        )
        self.assertEqual(
            functional.silu.__kwdefaults__, reference_functional.silu.__kwdefaults__
        )

        actual_signature = inspect.signature(functional.silu)
        expected_signature = inspect.signature(reference_functional.silu)
        actual_parameters = tuple(actual_signature.parameters.values())
        expected_parameters = tuple(expected_signature.parameters.values())
        for actual, expected in zip(
            actual_parameters, expected_parameters, strict=True
        ):
            self.assertEqual(actual.name, expected.name)
            self.assertEqual(actual.kind, expected.kind)
            self.assertEqual(actual.default, expected.default)
        self.assertIs(actual_parameters[0].annotation, torch.Tensor)
        self.assertIs(expected_parameters[0].annotation, reference_torch.Tensor)
        self.assertIs(actual_parameters[1].annotation, bool)
        self.assertIs(expected_parameters[1].annotation, bool)
        self.assertIs(actual_signature.return_annotation, torch.Tensor)
        self.assertIs(
            expected_signature.return_annotation, reference_torch.Tensor
        )

    def test_values_layouts_and_out_of_place_calls_match_pytorch_2_13(self):
        for case in (
            "scalar",
            "empty",
            "offset",
            "strided",
            "channels_last",
            "channels_last_3d",
        ):
            _, actual_input = self.make_case(torch, case, requires_grad=False)
            _, expected_input = self.make_case(
                reference_torch, case, requires_grad=False
            )
            self.assert_metadata_matches(actual_input, expected_input, case=(case, "input"))

            calls = (
                (
                    lambda: functional.silu(actual_input),
                    lambda: reference_functional.silu(expected_input),
                ),
                (
                    lambda: functional.silu(actual_input, False),
                    lambda: reference_functional.silu(expected_input, False),
                ),
                (
                    lambda: functional.silu(input=actual_input, inplace=False),
                    lambda: reference_functional.silu(
                        input=expected_input, inplace=False
                    ),
                ),
            )
            for form, (actual_call, expected_call) in enumerate(calls):
                actual = actual_call()
                expected = expected_call()
                invocation = (case, form)
                self.assert_metadata_matches(actual, expected, case=invocation)
                self.assert_values_match(actual, expected, case=invocation)
                with self.subTest(case=invocation):
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))

    def test_autograd_matches_pytorch_2_13_for_owned_finite_inputs(self):
        values = np.asarray(
            [[-3.0, -1.0, -0.0, 0.5], [1.0, 2.0, 4.0, 8.0]],
            dtype=np.float32,
        )
        weights = np.asarray(
            [[1.0, -2.0, 0.25, -0.5], [3.0, 0.125, -1.5, 2.0]],
            dtype=np.float32,
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values.tolist(), dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(weights.tolist())
        expected_weights = reference_torch.tensor(
            weights.tolist(), dtype=reference_torch.float32
        )

        actual_output = functional.silu(actual_leaf)
        expected_output = reference_functional.silu(expected_leaf)
        self.assert_metadata_matches(
            actual_output, expected_output, case="autograd output"
        )
        self.assert_values_match(actual_output, expected_output, case="autograd output")

        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_metadata_matches(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.detach().cpu().numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )

        for shape in ((2, 0, 3), (1,) * 65):
            with self.subTest(shape=shape):
                actual = torch.full(shape, 0.5, requires_grad=True)
                expected = reference_torch.full(
                    shape,
                    0.5,
                    dtype=reference_torch.float32,
                    requires_grad=True,
                )
                actual_out = functional.silu(actual)
                expected_out = reference_functional.silu(expected)
                self.assert_metadata_matches(actual_out, expected_out, case=shape)
                actual_out.sum().backward()
                expected_out.sum().backward()
                if len(shape) > 64:
                    self.assertAlmostEqual(
                        actual.grad.item(),
                        expected.grad.item(),
                        delta=2.0e-6,
                    )
                else:
                    np.testing.assert_allclose(
                        np.asarray(actual.grad),
                        expected.grad.detach().cpu().numpy(),
                        rtol=2.0e-6,
                        atol=0.0,
                    )

    def test_modes_and_overrides_match_pytorch_2_13_observations(self):
        def observation(module, function):
            source = module.tensor([0.5], dtype=module.float32)
            marker = object()

            class RecordingMode(module.overrides.TorchFunctionMode):
                def __init__(self):
                    self.calls = []

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    self.calls.append((func, types, args, kwargs))
                    return marker

            mode = RecordingMode()
            with mode:
                result = function(source)
            return result is marker, mode.calls

        actual_result, actual_calls = observation(torch, functional.silu)
        expected_result, expected_calls = observation(
            reference_torch, reference_functional.silu
        )
        self.assertEqual(actual_result, expected_result)
        self.assertEqual(len(actual_calls), len(expected_calls))
        actual_function, actual_types, actual_args, actual_kwargs = actual_calls[0]
        expected_function, expected_types, expected_args, expected_kwargs = expected_calls[0]
        self.assertIs(actual_function, functional.silu)
        self.assertIs(expected_function, reference_functional.silu)
        self.assertEqual(
            tuple(type_.__name__ for type_ in actual_types),
            tuple(type_.__name__ for type_ in expected_types),
        )
        self.assertEqual(len(actual_args), len(expected_args))
        self.assertEqual(actual_kwargs, expected_kwargs)

        class ActualOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return (func, types, args, kwargs)

        class ExpectedOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return (func, types, args, kwargs)

        actual = functional.silu(ActualOverride(), inplace=True)
        expected = reference_functional.silu(ExpectedOverride(), inplace=True)
        self.assertIs(actual[0], functional.silu)
        self.assertIs(expected[0], reference_functional.silu)
        self.assertEqual(tuple(type_.__name__ for type_ in actual[1]), ("ActualOverride",))
        self.assertEqual(tuple(type_.__name__ for type_ in expected[1]), ("ExpectedOverride",))
        self.assertEqual(actual[3], expected[3])

    def test_binding_errors_match_pytorch_2_13_for_default_form(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5], dtype=reference_torch.float32)
        cases = (
            (lambda: functional.silu(), lambda: reference_functional.silu()),
            (
                lambda: functional.silu(actual, False, False),
                lambda: reference_functional.silu(expected, False, False),
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
                lambda: functional.silu(input=[]),
                lambda: reference_functional.silu(input=[]),
            ),
            (
                lambda: functional.silu(1, inplace=True),
                lambda: reference_functional.silu(1, inplace=True),
            ),
            (
                lambda: functional.silu(input=[], inplace=True),
                lambda: reference_functional.silu(input=[], inplace=True),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(index=index):
                with self.assertRaises(Exception) as actual_raised:
                    actual_call()
                with self.assertRaises(Exception) as expected_raised:
                    expected_call()
                self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
                self.assertEqual(
                    str(actual_raised.exception),
                    str(expected_raised.exception).replace("torch", "torch_rs"),
                )

        before = np.asarray(actual).copy()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.nn\.functional\.silu does not support inplace=True$",
        ):
            functional.silu(actual, inplace=True)
        np.testing.assert_array_equal(np.asarray(actual), before)
        with self.assertRaisesRegex(TypeError, r"^silu\(\):"):
            functional.silu(np.zeros((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
