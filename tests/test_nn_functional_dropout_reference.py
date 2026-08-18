import inspect
import sys
import types
import unittest
import warnings
from decimal import Decimal

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
class FunctionalDropoutReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.dropout differentials require pinned PyTorch 2.13.0"
            )

    def make_case(self, module, case, *, requires_grad):
        if case == "scalar":
            leaf = module.tensor(
                -0.0, dtype=module.float32, requires_grad=requires_grad
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3),
                dtype=module.float32,
                requires_grad=requires_grad,
            )
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            [
                [9.0, 9.0, 9.0, 9.0],
                [-1.0, 2.0, -0.0, 3.0],
                [4.0, -5.0, 6.0, -7.0],
            ],
            dtype=module.float32,
            requires_grad=requires_grad,
        )
        offset = leaf[1]
        if case == "offset":
            return leaf, offset
        return leaf, offset.reshape(2, 2).transpose(0, 1)

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

    def test_signature_documentation_and_callable_metadata_match(self):
        self.assertIs(type(functional.dropout), types.FunctionType)
        self.assertIs(type(reference_functional.dropout), types.FunctionType)
        self.assertEqual(functional.__doc__, reference_functional.__doc__)
        self.assertEqual(
            functional.dropout.__name__, reference_functional.dropout.__name__
        )
        self.assertEqual(
            functional.dropout.__qualname__,
            reference_functional.dropout.__qualname__,
        )
        self.assertEqual(
            functional.dropout.__doc__, reference_functional.dropout.__doc__
        )
        self.assertEqual(
            functional.dropout.__defaults__,
            reference_functional.dropout.__defaults__,
        )
        self.assertEqual(
            functional.dropout.__kwdefaults__,
            reference_functional.dropout.__kwdefaults__,
        )

        actual_signature = inspect.signature(functional.dropout)
        expected_signature = inspect.signature(reference_functional.dropout)
        actual_parameters = tuple(actual_signature.parameters.values())
        expected_parameters = tuple(expected_signature.parameters.values())
        for actual, expected in zip(
            actual_parameters, expected_parameters, strict=True
        ):
            self.assertEqual(actual.name, expected.name)
            self.assertEqual(actual.kind, expected.kind)
            self.assertEqual(actual.default, expected.default)

        self.assertIs(actual_parameters[0].annotation, torch.Tensor)
        self.assertIs(
            expected_parameters[0].annotation, reference_torch.Tensor
        )
        for actual, expected in zip(
            actual_parameters[1:], expected_parameters[1:], strict=True
        ):
            self.assertIs(actual.annotation, expected.annotation)
        self.assertIs(actual_signature.return_annotation, torch.Tensor)
        self.assertIs(
            expected_signature.return_annotation, reference_torch.Tensor
        )

    def test_identity_calls_match_for_values_layouts_and_autograd_metadata(self):
        calls = (
            (
                lambda function, input: function(
                    input, p=0.5, training=False
                ),
                "evaluation",
            ),
            (
                lambda function, input: function(input, 1.0, False, True),
                "evaluation_inplace",
            ),
            (
                lambda function, input: function(input, p=0, training=True),
                "zero_probability",
            ),
            (
                lambda function, input: function(
                    input, p=-0.0, training=True, inplace=True
                ),
                "zero_probability_inplace",
            ),
        )

        for requires_grad in (False, True):
            for case in ("scalar", "empty", "offset", "strided"):
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
                    actual = call(functional.dropout, actual_input)
                    expected = call(reference_functional.dropout, expected_input)
                    invocation = (requires_grad, case, label)
                    with self.subTest(case=invocation):
                        self.assertIs(actual, actual_input)
                        self.assertIs(expected, expected_input)
                        self.assertTrue(actual.is_set_to(actual_input))
                        self.assertTrue(expected.is_set_to(expected_input))
                    self.assert_metadata_matches(
                        actual, expected, case=invocation
                    )
                    self.assert_values_match(actual, expected, case=invocation)

    def test_backward_and_no_grad_identity_match(self):
        actual_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 1)
        expected_input = expected_leaf.transpose(0, 1)
        actual_output = functional.dropout(
            actual_input, p=0, training=True, inplace=True
        )
        expected_output = reference_functional.dropout(
            expected_input, p=0, training=True, inplace=True
        )
        self.assertIs(actual_output, actual_input)
        self.assertIs(expected_output, expected_input)

        actual_weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        expected_weights = reference_torch.tensor(
            [[2.0, 3.0], [5.0, 7.0]]
        )
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        self.assert_metadata_matches(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )
        self.assert_values_match(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )

        actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            actual_output = functional.dropout(
                actual_leaf, p=0.75, training=False
            )
        with reference_torch.no_grad():
            expected_output = reference_functional.dropout(
                expected_leaf, p=0.75, training=False
            )
        self.assertIs(actual_output, actual_leaf)
        self.assertIs(expected_output, expected_leaf)
        self.assert_metadata_matches(
            actual_output, expected_output, case="no_grad"
        )

    def test_empty_training_identity_and_rng_state_match(self):
        for requires_grad in (False, True):
            actual_sources = self.make_case(
                torch, "empty", requires_grad=requires_grad
            )
            expected_sources = self.make_case(
                reference_torch, "empty", requires_grad=requires_grad
            )
            for source_kind, (actual_input, expected_input) in enumerate(
                zip(actual_sources, expected_sources, strict=True)
            ):
                for probability in (0.25, 1.0):
                    for inplace in (False, True):
                        expected_rng = reference_torch.get_rng_state().clone()
                        actual = functional.dropout(
                            actual_input,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                        expected = reference_functional.dropout(
                            expected_input,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                        case = (
                            requires_grad,
                            source_kind,
                            probability,
                            inplace,
                        )
                        with self.subTest(case=case):
                            self.assertIs(actual, actual_input)
                            self.assertIs(expected, expected_input)
                            self.assertTrue(
                                reference_torch.equal(
                                    expected_rng,
                                    reference_torch.get_rng_state(),
                                )
                            )
                        self.assert_metadata_matches(
                            actual, expected, case=case
                        )
                        self.assert_values_match(
                            actual, expected, case=case
                        )

    def test_scalar_tensor_probability_schema_matches(self):
        actual_input = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_input = reference_torch.tensor(
            [1.0, 2.0], requires_grad=True
        )

        for value, training in (
            (0.0, True),
            (0.5, False),
            (1.0, False),
        ):
            for inplace in (False, True):
                actual_probability = torch.tensor(value)
                expected_probability = reference_torch.tensor(
                    value, dtype=reference_torch.float32
                )
                with self.subTest(
                    value=value, training=training, inplace=inplace
                ):
                    actual = functional.dropout(
                        actual_input,
                        p=actual_probability,
                        training=training,
                        inplace=inplace,
                    )
                    expected = reference_functional.dropout(
                        expected_input,
                        p=expected_probability,
                        training=training,
                        inplace=inplace,
                    )
                    self.assertIs(actual, actual_input)
                    self.assertIs(expected, expected_input)

        for value in (-0.1, 1.1, float("inf"), float("nan")):
            actual_probability = torch.tensor(value)
            expected_probability = reference_torch.tensor(
                value, dtype=reference_torch.float32
            )
            actual_error = self.capture_error(
                lambda: functional.dropout(
                    actual_input,
                    p=actual_probability,
                    training=False,
                )
            )
            expected_error = self.capture_error(
                lambda: reference_functional.dropout(
                    expected_input,
                    p=expected_probability,
                    training=False,
                )
            )
            with self.subTest(invalid_value=value):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

        for inplace in (False, True):
            actual_probability = torch.tensor(0.5, requires_grad=True)
            expected_probability = reference_torch.tensor(
                0.5, dtype=reference_torch.float32, requires_grad=True
            )
            actual_error = self.capture_error(
                lambda: functional.dropout(
                    actual_input,
                    p=actual_probability,
                    training=False,
                    inplace=inplace,
                )
            )
            expected_error = self.capture_error(
                lambda: reference_functional.dropout(
                    expected_input,
                    p=expected_probability,
                    training=False,
                    inplace=inplace,
                )
            )
            with self.subTest(grad_probability_inplace=inplace):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

    def test_non_scalar_tensor_probability_validation_matches(self):
        actual_input = torch.tensor([1.0, 2.0])
        expected_input = reference_torch.tensor([1.0, 2.0])
        actual_leaf_probability = torch.tensor([2.0], requires_grad=True)
        expected_leaf_probability = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_reflected_leaf = torch.tensor([1.0], requires_grad=True)
        expected_reflected_leaf = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_copy_leaf = torch.tensor([[[[2.0]]]], requires_grad=True)
        expected_copy_leaf = reference_torch.tensor(
            [[[[2.0]]]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )

        probability_cases = (
            (
                torch.tensor([2.0]),
                reference_torch.tensor(
                    [2.0], dtype=reference_torch.float32
                ),
            ),
            (
                torch.tensor([-0.1]),
                reference_torch.tensor(
                    [-0.1], dtype=reference_torch.float32
                ),
            ),
            (
                torch.tensor([[1.0e9]]),
                reference_torch.tensor(
                    [[1.0e9]], dtype=reference_torch.float32
                ),
            ),
            (torch.zeros((0,)), reference_torch.zeros((0,))),
            (
                torch.tensor([0.0, 2.0]),
                reference_torch.tensor(
                    [0.0, 2.0], dtype=reference_torch.float32
                ),
            ),
            (
                torch.tensor([0.5]),
                reference_torch.tensor(
                    [0.5], dtype=reference_torch.float32
                ),
            ),
            (
                torch.tensor([float("nan")]),
                reference_torch.tensor(
                    [float("nan")], dtype=reference_torch.float32
                ),
            ),
            (actual_leaf_probability, expected_leaf_probability),
            (
                actual_leaf_probability * 2,
                expected_leaf_probability * 2,
            ),
            (
                actual_leaf_probability + 1,
                expected_leaf_probability + 1,
            ),
            (
                actual_leaf_probability.reshape(1, 1),
                expected_leaf_probability.reshape(1, 1),
            ),
            (
                torch.tensor([-2.0], requires_grad=True).ravel(),
                reference_torch.tensor(
                    [-2.0],
                    dtype=reference_torch.float32,
                    requires_grad=True,
                ).ravel(),
            ),
            (3 - actual_reflected_leaf, 3 - expected_reflected_leaf),
            (
                actual_copy_leaf.cpu(memory_format=torch.channels_last),
                expected_copy_leaf.cpu(
                    memory_format=reference_torch.channels_last
                ),
            ),
            (
                actual_copy_leaf.float(memory_format=torch.channels_last),
                expected_copy_leaf.float(
                    memory_format=reference_torch.channels_last
                ),
            ),
        )
        for case, (actual_probability, expected_probability) in enumerate(
            probability_cases
        ):
            actual_error = self.capture_error(
                lambda: functional.dropout(
                    actual_input, p=actual_probability, training=False
                )
            )
            expected_error = self.capture_error(
                lambda: reference_functional.dropout(
                    expected_input, p=expected_probability, training=False
                )
            )
            with self.subTest(case=case):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

    def test_probability_validation_schema_and_binding_errors_match(self):
        actual_input = torch.tensor([1.0])
        expected_input = reference_torch.tensor([1.0])

        paired_calls = (
            (
                lambda: functional.dropout(None, p=-0.1, training="bad"),
                lambda: reference_functional.dropout(
                    None, p=-0.1, training="bad"
                ),
            ),
            (
                lambda: functional.dropout(None, p=1.1, inplace=True),
                lambda: reference_functional.dropout(
                    None, p=1.1, inplace=True
                ),
            ),
            (
                lambda: functional.dropout(
                    actual_input, p=float("nan"), training=False
                ),
                lambda: reference_functional.dropout(
                    expected_input, p=float("nan"), training=False
                ),
            ),
            (
                lambda: functional.dropout(actual_input, p=None),
                lambda: reference_functional.dropout(expected_input, p=None),
            ),
            (
                lambda: functional.dropout(
                    actual_input, p=Decimal("0"), training=False
                ),
                lambda: reference_functional.dropout(
                    expected_input, p=Decimal("0"), training=False
                ),
            ),
            (
                lambda: functional.dropout(None, p=0),
                lambda: reference_functional.dropout(None, p=0),
            ),
            (
                lambda: functional.dropout(None, p=0, inplace=True),
                lambda: reference_functional.dropout(
                    None, p=0, inplace=True
                ),
            ),
            (
                lambda: functional.dropout(
                    actual_input, p=0, training=np.bool_(False)
                ),
                lambda: reference_functional.dropout(
                    expected_input, p=0, training=np.bool_(False)
                ),
            ),
            (
                lambda: functional.dropout(),
                lambda: reference_functional.dropout(),
            ),
            (
                lambda: functional.dropout(actual_input, 0, p=0),
                lambda: reference_functional.dropout(
                    expected_input, 0, p=0
                ),
            ),
            (
                lambda: functional.dropout(actual_input, unknown=True),
                lambda: reference_functional.dropout(
                    expected_input, unknown=True
                ),
            ),
            (
                lambda: functional.dropout(
                    actual_input, 0, False, False, "extra"
                ),
                lambda: reference_functional.dropout(
                    expected_input, 0, False, False, "extra"
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
            with self.subTest(case=case):
                actual_error = self.capture_error(actual_call)
                expected_error = self.capture_error(expected_call)
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

        class BoolFailure:
            def __bool__(self):
                raise RuntimeError("inplace truthiness failed")

        actual_error = self.capture_error(
            lambda: functional.dropout(
                actual_input, p=0, inplace=BoolFailure()
            )
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(
                expected_input, p=0, inplace=BoolFailure()
            )
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])

        for probability in (
            np.bool_(False),
            np.int64(0),
            np.float32(0),
            np.complex64(0),
        ):
            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always")
                actual = functional.dropout(
                    actual_input, p=probability, training=True
                )
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter("always")
                expected = reference_functional.dropout(
                    expected_input, p=probability, training=True
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

        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            actual_error = self.capture_error(
                lambda: functional.dropout(
                    actual_input, p=np.complex64(0), training=False
                )
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            expected_error = self.capture_error(
                lambda: reference_functional.dropout(
                    expected_input, p=np.complex64(0), training=False
                )
            )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])

        class MemoryFailure(int):
            def __float__(self):
                raise MemoryError("probability conversion failed")

            def __lt__(self, other):
                return False

            def __gt__(self, other):
                return False

        actual_error = self.capture_error(
            lambda: functional.dropout(
                actual_input, p=MemoryFailure(0), training=False
            )
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(
                expected_input, p=MemoryFailure(0), training=False
            )
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])

        class SneakyProbability(float):
            def __lt__(self, other):
                return False

            def __gt__(self, other):
                return False

        for probability in (
            -float("nan"),
            SneakyProbability(1.0000000000001),
            SneakyProbability(1.23456789),
            SneakyProbability(999999.9),
            SneakyProbability(-1.23456789e-7),
            SneakyProbability(-5e-324),
        ):
            actual_error = self.capture_error(
                lambda: functional.dropout(
                    actual_input, p=probability, training=False
                )
            )
            expected_error = self.capture_error(
                lambda: reference_functional.dropout(
                    expected_input, p=probability, training=False
                )
            )
            with self.subTest(native_probability=probability):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

    def test_tensor_probability_recursion_limit_matches(self):
        actual_probability = torch.tensor([-2.0]).reshape((1,) * 72)
        expected_probability = reference_torch.tensor(
            [-2.0], dtype=reference_torch.float32
        ).reshape((1,) * 72)

        def capture_with_limit(call):
            previous_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(80)
                return self.capture_error(call)
            finally:
                sys.setrecursionlimit(previous_limit)

        actual_error = capture_with_limit(
            lambda: functional.dropout(
                torch.tensor([0.0]),
                p=actual_probability,
                training=False,
            )
        )
        expected_error = capture_with_limit(
            lambda: reference_functional.dropout(
                reference_torch.tensor([0.0]),
                p=expected_probability,
                training=False,
            )
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])

    def run_override_case(self, function):
        replacement = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(
                cls, func, types, args=(), kwargs=None
            ):
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
            self.run_override_case(functional.dropout),
            self.run_override_case(reference_functional.dropout),
        )

        class ClassInput:
            @classmethod
            def __torch_function__(
                cls, func, types, args=(), kwargs=None
            ):
                return "unreachable"

        actual_error = self.capture_error(
            lambda: functional.dropout(ClassInput, p=0)
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(ClassInput, p=0)
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(
            actual_error[1].replace("torch_rs.nn", "torch.nn"),
            expected_error[1],
        )

        class BrokenProbe:
            def __getattribute__(self, name):
                if name == "__torch_function__":
                    raise RuntimeError("probe failed")
                return object.__getattribute__(self, name)

        actual_error = self.capture_error(
            lambda: functional.dropout(BrokenProbe(), p=-1)
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(BrokenProbe(), p=-1)
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])

        def stateful_override():
            class StatefulDescriptor:
                def __init__(self):
                    self.type_accesses = 0

                def __get__(self, instance, owner):
                    if instance is None:
                        self.type_accesses += 1
                        if self.type_accesses == 2:
                            raise RuntimeError("second type lookup")

                    def override(func, types, args=(), kwargs=None):
                        return "override"

                    return override

            descriptor = StatefulDescriptor()

            class StatefulOverride:
                __torch_function__ = descriptor

            return StatefulOverride(), descriptor

        actual_override, actual_descriptor = stateful_override()
        expected_override, expected_descriptor = stateful_override()
        actual_error = self.capture_error(
            lambda: functional.dropout(actual_override, training=False)
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(
                expected_override, training=False
            )
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])
        self.assertEqual(
            actual_descriptor.type_accesses,
            expected_descriptor.type_accesses,
        )

        def disabled_input():
            return type(
                "DisabledInput",
                (),
                {
                    "__torch_function__": (
                        reference_torch._C._disabled_torch_function_impl
                    )
                },
            )()

        actual_error = self.capture_error(
            lambda: functional.dropout(
                disabled_input(), p=0, training=False
            )
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(
                disabled_input(), p=0, training=False
            )
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(actual_error[1], expected_error[1])

        actual_error = self.capture_error(
            lambda: functional.dropout(torch.Tensor, p=0)
        )
        expected_error = self.capture_error(
            lambda: reference_functional.dropout(
                reference_torch.Tensor, p=0
            )
        )
        self.assertIs(actual_error[0], expected_error[0])
        self.assertEqual(
            actual_error[1].replace("torch_rs.nn", "torch.nn"),
            expected_error[1],
        )

        class ActualMode(torch.overrides.TorchFunctionMode):
            def __init__(self, *, forward=False):
                self.forward = forward
                self.calls = []

            def __torch_function__(
                self, func, types, args=(), kwargs=None
            ):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **kwargs)
                return "mode-result"

        class ExpectedMode(reference_torch.overrides.TorchFunctionMode):
            def __init__(self, *, forward=False):
                self.forward = forward
                self.calls = []

            def __torch_function__(
                self, func, types, args=(), kwargs=None
            ):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **kwargs)
                return "mode-result"

        actual_input = torch.tensor([1.0])
        expected_input = reference_torch.tensor([1.0])
        actual_mode = ActualMode()
        expected_mode = ExpectedMode()
        with actual_mode:
            actual_output = functional.dropout(
                actual_input, p=-1, training="invalid", inplace=True
            )
        with expected_mode:
            expected_output = reference_functional.dropout(
                expected_input, p=-1, training="invalid", inplace=True
            )
        self.assertEqual(actual_output, expected_output)
        actual_call = actual_mode.calls[0]
        expected_call = expected_mode.calls[0]
        self.assertIs(actual_call[0], functional.dropout)
        self.assertIs(expected_call[0], reference_functional.dropout)
        self.assertEqual(
            tuple(item.__name__ for item in actual_call[1]),
            tuple(item.__name__ for item in expected_call[1]),
        )
        self.assertIs(actual_call[2][0], actual_input)
        self.assertIs(expected_call[2][0], expected_input)
        self.assertEqual(actual_call[3], expected_call[3])

        actual_mode = ActualMode(forward=True)
        expected_mode = ExpectedMode(forward=True)
        with actual_mode:
            actual_output = functional.dropout(
                actual_input, p=0, training=True, inplace=True
            )
        with expected_mode:
            expected_output = reference_functional.dropout(
                expected_input, p=0, training=True, inplace=True
            )
        self.assertIs(actual_output, actual_input)
        self.assertIs(expected_output, expected_input)
        self.assertEqual(len(actual_mode.calls), len(expected_mode.calls))

    def test_sampling_boundary_remains_deliberately_unsupported(self):
        self.assertFalse(hasattr(torch, "dropout"))
        self.assertTrue(hasattr(reference_torch, "dropout"))

        source = torch.tensor([1.0, 2.0], requires_grad=True)
        before = np.asarray(source.detach()).copy().view(np.uint32)
        for probability in (0.25, 1.0):
            for inplace in (False, True):
                with self.subTest(probability=probability, inplace=inplace):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        "^torch_rs.nn.functional.dropout does not support sampling$",
                    ):
                        functional.dropout(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                    np.testing.assert_array_equal(
                        np.asarray(source.detach()).view(np.uint32), before
                    )
                    self.assertIsNone(source.grad)


if __name__ == "__main__":
    unittest.main()
