import inspect
import pickle
import types
import unittest
import warnings
from decimal import Decimal
from fractions import Fraction

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
class FunctionalDropout3dReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.dropout3d differentials require pinned "
                "PyTorch 2.13.0"
            )

    def make_case(self, module, case, *, requires_grad):
        if case == "contiguous":
            leaf = module.tensor(
                [[[[[1.0, -2.0], [3.0, -4.0]]]]],
                dtype=module.float32,
                requires_grad=requires_grad,
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 1, 0, 3, 4, 5),
                dtype=module.float32,
                requires_grad=requires_grad,
            )
            return leaf, leaf[1].transpose(3, 4)
        if case == "channels_last_3d":
            leaf = module.tensor(
                [
                    [
                        [[[-1.0, 2.0], [-0.0, 3.0]]],
                        [[[4.0, -5.0], [6.0, -7.0]]],
                    ]
                ],
                dtype=module.float32,
                requires_grad=requires_grad,
            )
            return leaf, leaf.contiguous(memory_format=module.channels_last_3d)

        leaf = module.zeros(
            (2, 1, 2, 3, 4, 5),
            dtype=module.float32,
            requires_grad=requires_grad,
        )
        offset = leaf[1]
        if case == "offset":
            return leaf, offset
        return leaf, offset.transpose(3, 4)

    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(actual.output_nr, expected.output_nr)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_values_match(self, actual, expected, *, case):
        with self.subTest(case=case):
            np.testing.assert_array_equal(
                np.asarray(actual.detach()).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def capture_error(self, call):
        try:
            call()
        except Exception as error:
            return type(error), str(error)
        self.fail("expected call to raise")

    def test_imports_signature_documentation_and_pickling_match(self):
        self.assertIs(type(functional.dropout3d), types.FunctionType)
        self.assertIs(type(reference_functional.dropout3d), types.FunctionType)
        self.assertEqual(functional.__doc__, reference_functional.__doc__)
        self.assertEqual(
            functional.dropout3d.__name__, reference_functional.dropout3d.__name__
        )
        self.assertEqual(
            functional.dropout3d.__qualname__,
            reference_functional.dropout3d.__qualname__,
        )
        self.assertEqual(
            functional.dropout3d.__doc__, reference_functional.dropout3d.__doc__
        )
        self.assertEqual(
            functional.dropout3d.__defaults__,
            reference_functional.dropout3d.__defaults__,
        )
        self.assertEqual(
            functional.dropout3d.__kwdefaults__,
            reference_functional.dropout3d.__kwdefaults__,
        )

        actual_signature = inspect.signature(functional.dropout3d)
        expected_signature = inspect.signature(reference_functional.dropout3d)
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
        for actual, expected in zip(
            actual_parameters[1:], expected_parameters[1:], strict=True
        ):
            self.assertIs(actual.annotation, expected.annotation)
        self.assertIs(actual_signature.return_annotation, torch.Tensor)
        self.assertIs(expected_signature.return_annotation, reference_torch.Tensor)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.nn.functional import *", actual_wildcard)
        exec("from torch.nn.functional import *", expected_wildcard)
        self.assertIs(actual_wildcard["dropout3d"], functional.dropout3d)
        self.assertIs(
            expected_wildcard["dropout3d"], reference_functional.dropout3d
        )

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(functional.dropout3d, protocol=protocol)
                    ),
                    functional.dropout3d,
                )
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(
                            reference_functional.dropout3d, protocol=protocol
                        )
                    ),
                    reference_functional.dropout3d,
                )

    def test_rank_five_identity_values_layouts_and_metadata_match(self):
        calls = (
            (
                lambda function, input, module: function(
                    input, p=0.75, training=False
                ),
                "evaluation",
            ),
            (
                lambda function, input, module: function(
                    input, p=1.0, training=False, inplace=True
                ),
                "evaluation_inplace",
            ),
            (
                lambda function, input, module: function(
                    input, p=0, training=True
                ),
                "zero_probability",
            ),
            (
                lambda function, input, module: function(
                    input, p=module.tensor(0.0), training=True, inplace=True
                ),
                "tensor_zero_probability_inplace",
            ),
        )

        for requires_grad in (False, True):
            for case in ("contiguous", "offset", "strided"):
                _, actual_input = self.make_case(
                    torch, case, requires_grad=requires_grad
                )
                _, expected_input = self.make_case(
                    reference_torch, case, requires_grad=requires_grad
                )
                self.assert_metadata_matches(
                    actual_input,
                    expected_input,
                    case=(requires_grad, case, "input"),
                )

                for call, label in calls:
                    actual = call(functional.dropout3d, actual_input, torch)
                    expected = call(
                        reference_functional.dropout3d,
                        expected_input,
                        reference_torch,
                    )
                    invocation = (requires_grad, case, label)
                    with self.subTest(case=invocation):
                        self.assertIs(actual, actual_input)
                        self.assertIs(expected, expected_input)
                        self.assertTrue(actual.is_set_to(actual_input))
                        self.assertTrue(expected.is_set_to(expected_input))
                    self.assert_metadata_matches(actual, expected, case=invocation)
                    self.assert_values_match(actual, expected, case=invocation)

    def test_training_probability_one_matches_native_zero_multiplication(self):
        for requires_grad in (False, True):
            for case in ("contiguous", "offset", "strided", "channels_last_3d"):
                _, actual_input = self.make_case(
                    torch, case, requires_grad=requires_grad
                )
                _, expected_input = self.make_case(
                    reference_torch, case, requires_grad=requires_grad
                )
                probabilities = (
                    (1.0, 1.0),
                    (1, 1),
                    (True, True),
                    (np.float32(1.0), np.float32(1.0)),
                    (
                        torch.tensor(1.0),
                        reference_torch.tensor(
                            1.0, dtype=reference_torch.float32
                        ),
                    ),
                )
                for probability_case, (
                    actual_probability,
                    expected_probability,
                ) in enumerate(probabilities):
                    expected_rng = reference_torch.get_rng_state().clone()
                    actual = functional.dropout3d(
                        actual_input,
                        p=actual_probability,
                        training=True,
                        inplace=False,
                    )
                    expected = reference_functional.dropout3d(
                        expected_input,
                        p=expected_probability,
                        training=True,
                        inplace=False,
                    )
                    invocation = (requires_grad, case, probability_case)
                    with self.subTest(case=invocation):
                        self.assertIsNot(actual, actual_input)
                        self.assertIsNot(expected, expected_input)
                        self.assertFalse(actual.is_set_to(actual_input))
                        self.assertFalse(expected.is_set_to(expected_input))
                        self.assertNotEqual(
                            actual.data_ptr(), actual_input.data_ptr()
                        )
                        self.assertNotEqual(
                            expected.data_ptr(), expected_input.data_ptr()
                        )
                        self.assertTrue(
                            reference_torch.equal(
                                expected_rng,
                                reference_torch.get_rng_state(),
                            )
                        )
                    self.assert_metadata_matches(actual, expected, case=invocation)
                    self.assert_values_match(actual, expected, case=invocation)

    def test_probability_one_signed_zero_nan_and_infinity_bits_match(self):
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0xBF80_0000,
                0x3F80_0000,
            ),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        actual_input = torch.tensor(values).reshape(1, 2, 1, 2, 2)
        expected_input = reference_torch.tensor(values).reshape(1, 2, 1, 2, 2)

        for case, actual, expected in (
            ("contiguous", actual_input, expected_input),
            (
                "strided",
                actual_input.transpose(3, 4),
                expected_input.transpose(3, 4),
            ),
            (
                "channels_last_3d",
                actual_input.contiguous(memory_format=torch.channels_last_3d),
                expected_input.contiguous(
                    memory_format=reference_torch.channels_last_3d
                ),
            ),
        ):
            actual_output = functional.dropout3d(
                actual, p=1, training=True, inplace=False
            )
            expected_output = reference_functional.dropout3d(
                expected, p=1, training=True, inplace=False
            )
            self.assert_metadata_matches(actual_output, expected_output, case=case)
            self.assert_values_match(actual_output, expected_output, case=case)

    def test_probability_one_backward_and_no_grad_match(self):
        actual_leaf = torch.tensor(
            [[[[[-1.0, 2.0], [-0.0, 3.0]]]]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[[[[-1.0, 2.0], [-0.0, 3.0]]]]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(3, 4)
        expected_input = expected_leaf.transpose(3, 4)
        actual_output = functional.dropout3d(
            actual_input,
            p=torch.tensor(1.0),
            training=True,
            inplace=False,
        )
        expected_output = reference_functional.dropout3d(
            expected_input,
            p=reference_torch.tensor(1.0),
            training=True,
            inplace=False,
        )
        self.assert_metadata_matches(
            actual_output, expected_output, case="probability-one output"
        )
        self.assert_values_match(
            actual_output, expected_output, case="probability-one output"
        )
        actual_weights = torch.tensor([[[[[2.0, -3.0], [-5.0, 7.0]]]]])
        expected_weights = reference_torch.tensor(
            [[[[[2.0, -3.0], [-5.0, 7.0]]]]]
        )
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_metadata_matches(
            actual_leaf.grad, expected_leaf.grad, case="probability-one gradient"
        )
        self.assert_values_match(
            actual_leaf.grad, expected_leaf.grad, case="probability-one gradient"
        )

        actual_leaf = torch.tensor(
            [[[[[-1.0, 2.0], [-0.0, 3.0]]]]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[[[[-1.0, 2.0], [-0.0, 3.0]]]]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(3, 4)
        expected_input = expected_leaf.transpose(3, 4)
        with torch.no_grad():
            actual_output = functional.dropout3d(
                actual_input, p=1, training=True
            )
        with reference_torch.no_grad():
            expected_output = reference_functional.dropout3d(
                expected_input, p=1, training=True
            )
        self.assertIsNot(actual_output, actual_input)
        self.assertIsNot(expected_output, expected_input)
        self.assert_metadata_matches(
            actual_output, expected_output, case="probability-one no_grad"
        )
        self.assert_values_match(
            actual_output, expected_output, case="probability-one no_grad"
        )

    def test_backward_no_grad_empty_training_and_rng_state_match(self):
        actual_leaf = torch.tensor(
            [[[[[1.0, 2.0], [3.0, 4.0]]]]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[[[[1.0, 2.0], [3.0, 4.0]]]]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(3, 4)
        expected_input = expected_leaf.transpose(3, 4)
        actual_output = functional.dropout3d(
            actual_input, p=0, training=True, inplace=True
        )
        expected_output = reference_functional.dropout3d(
            expected_input, p=0, training=True, inplace=True
        )
        self.assertIs(actual_output, actual_input)
        self.assertIs(expected_output, expected_input)

        actual_weights = torch.tensor([[[[[2.0, 3.0], [5.0, 7.0]]]]])
        expected_weights = reference_torch.tensor(
            [[[[[2.0, 3.0], [5.0, 7.0]]]]]
        )
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_metadata_matches(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )
        self.assert_values_match(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )

        actual_leaf = torch.zeros((1, 2, 3, 4, 5), requires_grad=True)
        expected_leaf = reference_torch.zeros(
            (1, 2, 3, 4, 5), requires_grad=True
        )
        with torch.no_grad():
            actual_output = functional.dropout3d(
                actual_leaf, p=0.75, training=False
            )
        with reference_torch.no_grad():
            expected_output = reference_functional.dropout3d(
                expected_leaf, p=0.75, training=False
            )
        self.assertIs(actual_output, actual_leaf)
        self.assertIs(expected_output, expected_leaf)
        self.assert_metadata_matches(actual_output, expected_output, case="no_grad")

        for requires_grad in (False, True):
            _, actual_empty = self.make_case(
                torch, "empty", requires_grad=requires_grad
            )
            _, expected_empty = self.make_case(
                reference_torch, "empty", requires_grad=requires_grad
            )
            for probability in (0.25, 1.0):
                for inplace in (False, True):
                    expected_rng = reference_torch.get_rng_state().clone()
                    actual = functional.dropout3d(
                        actual_empty,
                        p=probability,
                        training=True,
                        inplace=inplace,
                    )
                    expected = reference_functional.dropout3d(
                        expected_empty,
                        p=probability,
                        training=True,
                        inplace=inplace,
                    )
                    invocation = (requires_grad, probability, inplace)
                    with self.subTest(case=invocation):
                        self.assertIs(actual, actual_empty)
                        self.assertIs(expected, expected_empty)
                        self.assertTrue(
                            reference_torch.equal(
                                expected_rng, reference_torch.get_rng_state()
                            )
                        )
                    self.assert_metadata_matches(actual, expected, case=invocation)

    def test_supported_probability_forms_and_errors_match(self):
        actual_input = torch.zeros((1, 2, 3, 4, 5))
        expected_input = reference_torch.zeros((1, 2, 3, 4, 5))

        scalar_cases = (
            (False, True),
            (0, True),
            (-0.0, True),
            (np.bool_(False), True),
            (np.int64(0), True),
            (np.float32(0), True),
            (True, False),
            (1, False),
            (np.float64(0.75), False),
            (np.complex64(0), True),
        )
        for probability, training in scalar_cases:
            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always")
                actual = functional.dropout3d(
                    actual_input, p=probability, training=training
                )
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter("always")
                expected = reference_functional.dropout3d(
                    expected_input, p=probability, training=training
                )
            with self.subTest(probability=type(probability)):
                self.assertIs(actual, actual_input)
                self.assertIs(expected, expected_input)
                self.assertEqual(
                    [type(item.message) for item in actual_warnings],
                    [type(item.message) for item in expected_warnings],
                )
                self.assertEqual(
                    [str(item.message) for item in actual_warnings],
                    [str(item.message) for item in expected_warnings],
                )

        for value, training in ((0.0, True), (0.5, False), (1.0, False)):
            for inplace in (False, True):
                actual_probability = torch.tensor(value)
                expected_probability = reference_torch.tensor(
                    value, dtype=reference_torch.float32
                )
                actual = functional.dropout3d(
                    actual_input,
                    p=actual_probability,
                    training=training,
                    inplace=inplace,
                )
                expected = reference_functional.dropout3d(
                    expected_input,
                    p=expected_probability,
                    training=training,
                    inplace=inplace,
                )
                with self.subTest(tensor_probability=value, inplace=inplace):
                    self.assertIs(actual, actual_input)
                    self.assertIs(expected, expected_input)

        paired_calls = (
            (
                lambda: functional.dropout3d(
                    None, p=-0.1, training="invalid", inplace=True
                ),
                lambda: reference_functional.dropout3d(
                    None, p=-0.1, training="invalid", inplace=True
                ),
            ),
            (
                lambda: functional.dropout3d(None, p=0),
                lambda: reference_functional.dropout3d(None, p=0),
            ),
            (
                lambda: functional.dropout3d(
                    None, p=0, training=1, inplace=True
                ),
                lambda: reference_functional.dropout3d(
                    None, p=0, training=1, inplace=True
                ),
            ),
            (
                lambda: functional.dropout3d(actual_input, p=-0.1),
                lambda: reference_functional.dropout3d(
                    expected_input, p=-0.1
                ),
            ),
            (
                lambda: functional.dropout3d(actual_input, p=float("nan")),
                lambda: reference_functional.dropout3d(
                    expected_input, p=float("nan")
                ),
            ),
            (
                lambda: functional.dropout3d(actual_input, p=None),
                lambda: reference_functional.dropout3d(expected_input, p=None),
            ),
            (
                lambda: functional.dropout3d(actual_input, p=Decimal("0")),
                lambda: reference_functional.dropout3d(
                    expected_input, p=Decimal("0")
                ),
            ),
            (
                lambda: functional.dropout3d(
                    actual_input, p=Fraction(0, 1), inplace=True
                ),
                lambda: reference_functional.dropout3d(
                    expected_input, p=Fraction(0, 1), inplace=True
                ),
            ),
            (
                lambda: functional.dropout3d(
                    actual_input, p=0, training=np.bool_(False)
                ),
                lambda: reference_functional.dropout3d(
                    expected_input, p=0, training=np.bool_(False)
                ),
            ),
            (
                lambda: functional.dropout3d(),
                lambda: reference_functional.dropout3d(),
            ),
            (
                lambda: functional.dropout3d(actual_input, 0, p=0),
                lambda: reference_functional.dropout3d(
                    expected_input, 0, p=0
                ),
            ),
            (
                lambda: functional.dropout3d(actual_input, unknown=True),
                lambda: reference_functional.dropout3d(
                    expected_input, unknown=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
            actual_error = self.capture_error(actual_call)
            expected_error = self.capture_error(expected_call)
            with self.subTest(case=case):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

        tensor_cases = (
            (torch.tensor([0.0]), reference_torch.tensor([0.0])),
            (
                torch.tensor(0.0, requires_grad=True),
                reference_torch.tensor(0.0, requires_grad=True),
            ),
            (torch.zeros((0,)), reference_torch.zeros((0,))),
            (torch.zeros((2,)), reference_torch.zeros((2,))),
        )
        for case, (actual_probability, expected_probability) in enumerate(
            tensor_cases
        ):
            actual_error = self.capture_error(
                lambda: functional.dropout3d(
                    actual_input, p=actual_probability, training=False
                )
            )
            expected_error = self.capture_error(
                lambda: reference_functional.dropout3d(
                    expected_input, p=expected_probability, training=False
                )
            )
            with self.subTest(tensor_case=case):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

    def run_override_case(self, function):
        replacement = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return replacement

        input = Override()
        output = function(input, p=-1, training="invalid", inplace=True)
        func, dispatch_types, args, kwargs = Override.calls[0]
        return {
            "output": output is replacement,
            "function": func is function,
            "types": tuple(item.__name__ for item in dispatch_types),
            "args": args == (input,),
            "kwargs": kwargs,
        }

    def test_overrides_and_torch_function_modes_match(self):
        self.assertEqual(
            self.run_override_case(functional.dropout3d),
            self.run_override_case(reference_functional.dropout3d),
        )

        class ActualMode(torch.overrides.TorchFunctionMode):
            def __init__(self, *, forward=False):
                self.forward = forward
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **kwargs)
                return "mode-result"

        class ExpectedMode(reference_torch.overrides.TorchFunctionMode):
            def __init__(self, *, forward=False):
                self.forward = forward
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **kwargs)
                return "mode-result"

        actual_input = torch.zeros((1, 2, 3, 4, 5))
        expected_input = reference_torch.zeros((1, 2, 3, 4, 5))
        actual_mode = ActualMode()
        expected_mode = ExpectedMode()
        with actual_mode:
            actual_output = functional.dropout3d(
                actual_input, p=-1, training="invalid", inplace=True
            )
        with expected_mode:
            expected_output = reference_functional.dropout3d(
                expected_input, p=-1, training="invalid", inplace=True
            )
        self.assertEqual(actual_output, expected_output)
        actual_call = actual_mode.calls[0]
        expected_call = expected_mode.calls[0]
        self.assertIs(actual_call[0], functional.dropout3d)
        self.assertIs(expected_call[0], reference_functional.dropout3d)
        self.assertEqual(
            tuple(item.__name__ for item in actual_call[1]),
            tuple(item.__name__ for item in expected_call[1]),
        )
        self.assertEqual(actual_call[3], expected_call[3])

        actual_mode = ActualMode(forward=True)
        expected_mode = ExpectedMode(forward=True)
        with actual_mode:
            actual_output = functional.dropout3d(
                actual_input, p=0, training=True, inplace=True
            )
        with expected_mode:
            expected_output = reference_functional.dropout3d(
                expected_input, p=0, training=True, inplace=True
            )
        self.assertIs(actual_output, actual_input)
        self.assertIs(expected_output, expected_input)
        self.assertEqual(len(actual_mode.calls), len(expected_mode.calls))

        actual_mode = ActualMode(forward=True)
        expected_mode = ExpectedMode(forward=True)
        with actual_mode:
            actual_output = functional.dropout3d(
                actual_input, p=1, training=True, inplace=False
            )
        with expected_mode:
            expected_output = reference_functional.dropout3d(
                expected_input, p=1, training=True, inplace=False
            )
        self.assertEqual(len(actual_mode.calls), len(expected_mode.calls))
        self.assert_metadata_matches(
            actual_output, expected_output, case="probability-one mode"
        )
        self.assert_values_match(
            actual_output, expected_output, case="probability-one mode"
        )


if __name__ == "__main__":
    unittest.main()
