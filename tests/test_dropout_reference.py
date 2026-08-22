import inspect
import pickle
import types
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DropoutReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.dropout differentials require pinned PyTorch 2.13.0"
            )

    def make_case(self, module, case, *, requires_grad):
        if case == "scalar":
            leaf = module.tensor(-0.0, requires_grad=requires_grad)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros((2, 0, 3), requires_grad=requires_grad)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = module.tensor(
            [
                [9.0, 9.0, 9.0, 9.0],
                [-1.0, 2.0, -0.0, 3.0],
                [4.0, -5.0, 6.0, -7.0],
            ],
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

    def assert_reference_rng_unchanged(self, state):
        self.assertTrue(
            reference_torch.equal(state, reference_torch.get_rng_state())
        )

    def test_deterministic_values_storage_strides_and_rng_match(self):
        calls = (
            ("evaluation", 0.25, False, True),
            ("zero", -0.0, True, True),
            ("probability_one", 1.0, True, False),
        )
        for requires_grad in (False, True):
            for case in ("scalar", "offset", "strided"):
                _, actual_input = self.make_case(
                    torch, case, requires_grad=requires_grad
                )
                _, expected_input = self.make_case(
                    reference_torch, case, requires_grad=requires_grad
                )
                for label, probability, train, identity in calls:
                    rng = reference_torch.get_rng_state().clone()
                    actual = torch.dropout(actual_input, probability, train)
                    self.assert_reference_rng_unchanged(rng)
                    expected = reference_torch.dropout(
                        expected_input, probability, train
                    )
                    self.assert_reference_rng_unchanged(rng)
                    invocation = (requires_grad, case, label)
                    with self.subTest(case=invocation):
                        self.assertEqual(actual is actual_input, identity)
                        self.assertEqual(expected is expected_input, identity)
                        self.assertEqual(actual.is_set_to(actual_input), identity)
                        self.assertEqual(expected.is_set_to(expected_input), identity)
                        if not identity:
                            self.assertNotEqual(
                                actual.data_ptr(), actual_input.data_ptr()
                            )
                            self.assertNotEqual(
                                expected.data_ptr(), expected_input.data_ptr()
                            )
                    self.assert_metadata_matches(
                        actual, expected, case=invocation
                    )
                    self.assert_values_match(actual, expected, case=invocation)

        _, actual_empty = self.make_case(
            torch, "empty", requires_grad=True
        )
        _, expected_empty = self.make_case(
            reference_torch, "empty", requires_grad=True
        )
        for probability in (0.25, 1.0):
            rng = reference_torch.get_rng_state().clone()
            actual = torch.dropout(actual_empty, probability, True)
            self.assert_reference_rng_unchanged(rng)
            expected = reference_torch.dropout(
                expected_empty, probability, True
            )
            self.assert_reference_rng_unchanged(rng)
            with self.subTest(empty_probability=probability):
                self.assertIs(actual, actual_empty)
                self.assertIs(expected, expected_empty)
            self.assert_metadata_matches(
                actual, expected, case=("empty", probability)
            )
            self.assert_values_match(
                actual, expected, case=("empty", probability)
            )

    def test_scalar_tensor_probability_autograd_and_no_grad_match(self):
        actual_leaf = torch.tensor(
            [[-1.0, 2.0], [-0.0, 3.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-1.0, 2.0], [-0.0, 3.0]], requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 1)
        expected_input = expected_leaf.transpose(0, 1)
        rng = reference_torch.get_rng_state().clone()
        actual = torch.dropout(actual_input, torch.tensor(1.0), True)
        self.assert_reference_rng_unchanged(rng)
        expected = reference_torch.dropout(
            expected_input, reference_torch.tensor(1.0), True
        )
        self.assert_reference_rng_unchanged(rng)
        self.assert_metadata_matches(actual, expected, case="autograd output")
        self.assert_values_match(actual, expected, case="autograd output")

        actual_weights = torch.tensor([[2.0, -3.0], [-5.0, 7.0]])
        expected_weights = reference_torch.tensor(
            [[2.0, -3.0], [-5.0, 7.0]]
        )
        (actual * actual_weights).sum().backward()
        (expected * expected_weights).sum().backward()
        self.assert_metadata_matches(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )
        self.assert_values_match(
            actual_leaf.grad, expected_leaf.grad, case="gradient"
        )

        actual_leaf = torch.tensor([-1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [-1.0, 2.0], requires_grad=True
        )
        with torch.no_grad():
            actual = torch.dropout(actual_leaf, 1, True)
        with reference_torch.no_grad():
            expected = reference_torch.dropout(expected_leaf, 1, True)
        self.assert_metadata_matches(actual, expected, case="no_grad")
        self.assert_values_match(actual, expected, case="no_grad")
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)

    def test_builtin_metadata_exports_and_argument_errors_match(self):
        actual = torch.dropout
        expected = reference_torch.dropout
        self.assertIs(type(actual), type(expected))
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__self__, expected.__self__)
        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)

        actual_owner = actual.__reduce__()[1][0]
        expected_owner = expected.__reduce__()[1][0]
        self.assertEqual(actual_owner.__name__, expected_owner.__name__)
        self.assertEqual(actual_owner.__qualname__, expected_owner.__qualname__)
        self.assertEqual(actual_owner.__module__, "torch_rs._C")
        self.assertEqual(expected_owner.__module__, "torch._C")
        self.assertIs(actual_owner, torch._C._VariableFunctionsClass)
        self.assertIs(expected_owner, reference_torch._C._VariableFunctionsClass)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected, protocol=protocol)),
                    expected,
                )
        self.assertEqual(torch.__all__.count("dropout"), 1)
        self.assertEqual(reference_torch.__all__.count("dropout"), 1)

        actual_input = torch.tensor([1.0])
        expected_input = reference_torch.tensor([1.0])
        paired_calls = (
            (lambda: torch.dropout(), lambda: reference_torch.dropout()),
            (
                lambda: torch.dropout(actual_input),
                lambda: reference_torch.dropout(expected_input),
            ),
            (
                lambda: torch.dropout(actual_input, 0),
                lambda: reference_torch.dropout(expected_input, 0),
            ),
            (
                lambda: torch.dropout(actual_input, 0, False, None),
                lambda: reference_torch.dropout(
                    expected_input, 0, False, None
                ),
            ),
            (
                lambda: torch.dropout(None, 0, False),
                lambda: reference_torch.dropout(None, 0, False),
            ),
            (
                lambda: torch.dropout(input=None, p=0, train=False),
                lambda: reference_torch.dropout(
                    input=None, p=0, train=False
                ),
            ),
            (
                lambda: torch.dropout(actual_input, None, False),
                lambda: reference_torch.dropout(expected_input, None, False),
            ),
            (
                lambda: torch.dropout(
                    input=actual_input, p=Decimal("0"), train=False
                ),
                lambda: reference_torch.dropout(
                    input=expected_input, p=Decimal("0"), train=False
                ),
            ),
            (
                lambda: torch.dropout(actual_input, 0, 1),
                lambda: reference_torch.dropout(expected_input, 0, 1),
            ),
            (
                lambda: torch.dropout(
                    input=actual_input, p=0, train=np.bool_(False)
                ),
                lambda: reference_torch.dropout(
                    input=expected_input, p=0, train=np.bool_(False)
                ),
            ),
            (
                lambda: torch.dropout(
                    actual_input, 0, False, input=actual_input
                ),
                lambda: reference_torch.dropout(
                    expected_input, 0, False, input=expected_input
                ),
            ),
            (
                lambda: torch.dropout(actual_input, 0, False, extra=True),
                lambda: reference_torch.dropout(
                    expected_input, 0, False, extra=True
                ),
            ),
            (
                lambda: torch.dropout(actual_input, -0.1, False),
                lambda: reference_torch.dropout(expected_input, -0.1, False),
            ),
            (
                lambda: torch.dropout(actual_input, float("nan"), False),
                lambda: reference_torch.dropout(
                    expected_input, float("nan"), False
                ),
            ),
            (
                lambda: torch.dropout(actual_input, torch.tensor([0.0]), False),
                lambda: reference_torch.dropout(
                    expected_input, reference_torch.tensor([0.0]), False
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(paired_calls):
            actual_error = self.capture_error(actual_call)
            expected_error = self.capture_error(expected_call)
            with self.subTest(case=case):
                self.assertIs(actual_error[0], expected_error[0])
                self.assertEqual(actual_error[1], expected_error[1])

        for actual_call, expected_call in (
            (
                lambda: torch.dropout(x=actual_input, p=0, train=False),
                lambda: reference_torch.dropout(
                    x=expected_input, p=0, train=False
                ),
            ),
            (
                lambda: torch.dropout(a=actual_input, p=0, train=False),
                lambda: reference_torch.dropout(
                    a=expected_input, p=0, train=False
                ),
            ),
            (
                lambda: torch.dropout(x1=actual_input, p=0, train=False),
                lambda: reference_torch.dropout(
                    x1=expected_input, p=0, train=False
                ),
            ),
        ):
            self.assertIs(actual_call(), actual_input)
            self.assertIs(expected_call(), expected_input)

    def test_torch_function_mode_dispatch_matches(self):
        def observe(module):
            source = module.tensor([1.0])
            calls = []

            class RecordingMode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    calls.append((func, types, args, kwargs))
                    return "mode-result"

            with RecordingMode():
                positional = module.dropout(source, -1, False)
            with RecordingMode():
                keyword = module.dropout(
                    input=source, p=0.25, train=True, **{}
                )

            normalized = []
            for func, dispatch_types, args, kwargs in calls:
                normalized.append(
                    (
                        func is module.dropout,
                        tuple(item.__name__ for item in dispatch_types),
                        len(args),
                        bool(args) and args[0] is source,
                        args[1:] if args else (),
                        None
                        if kwargs is None
                        else (
                            tuple(kwargs),
                            kwargs.get("input") is source,
                            kwargs.get("p"),
                            kwargs.get("train"),
                        ),
                    )
                )
            return positional, keyword, tuple(normalized)

        self.assertEqual(observe(torch), observe(reference_torch))

        class ActualForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        class ExpectedForwardingMode(
            reference_torch.overrides.TorchFunctionMode
        ):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        actual_input = torch.tensor([-1.0, 2.0])
        expected_input = reference_torch.tensor([-1.0, 2.0])
        with ActualForwardingMode():
            actual = torch.dropout(actual_input, 1, True)
        with ExpectedForwardingMode():
            expected = reference_torch.dropout(expected_input, 1, True)
        self.assert_metadata_matches(actual, expected, case="forwarding mode")
        self.assert_values_match(actual, expected, case="forwarding mode")

    def test_probability_and_training_overrides_match(self):
        def observe(module):
            source = module.tensor([1.0])

            class ProbabilityOverride:
                calls = []

                @classmethod
                def __torch_function__(
                    cls, func, types, args=(), kwargs=None
                ):
                    cls.calls.append((func, types, args, kwargs))
                    return "probability-result"

            probability_argument = ProbabilityOverride()
            probability_result = module.dropout(
                source, probability_argument, False
            )
            probability_call = ProbabilityOverride.calls[0]

            class TrainingOverride:
                calls = []

                @classmethod
                def __torch_function__(
                    cls, func, types, args=(), kwargs=None
                ):
                    cls.calls.append((func, types, args, kwargs))
                    return "training-result"

            training_argument = TrainingOverride()
            training_result = module.dropout(
                input=source, p=0, train=training_argument
            )
            training_call = TrainingOverride.calls[0]

            ordered_events = []

            class BaseOverride:
                @classmethod
                def __torch_function__(
                    cls, func, types, args=(), kwargs=None
                ):
                    ordered_events.append(
                        ("base", tuple(item.__name__ for item in types))
                    )
                    return NotImplemented

            class DerivedOverride(BaseOverride):
                @classmethod
                def __torch_function__(
                    cls, func, types, args=(), kwargs=None
                ):
                    ordered_events.append(
                        ("derived", tuple(item.__name__ for item in types))
                    )
                    return "derived-result"

            class OtherOverride:
                @classmethod
                def __torch_function__(
                    cls, func, types, args=(), kwargs=None
                ):
                    ordered_events.append(
                        ("other", tuple(item.__name__ for item in types))
                    )
                    return NotImplemented

            ordered_result = module.dropout(
                OtherOverride(), BaseOverride(), DerivedOverride()
            )

            class SharedOverride:
                calls = []

                @classmethod
                def __torch_function__(
                    cls, func, types, args=(), kwargs=None
                ):
                    cls.calls.append(tuple(item.__name__ for item in types))
                    return "shared-result"

            shared_result = module.dropout(
                source, SharedOverride(), SharedOverride()
            )

            mode_calls = []

            class RecordingMode(module.overrides.TorchFunctionMode):
                def __torch_function__(
                    self, func, types, args=(), kwargs=None
                ):
                    mode_calls.append((func, types, args, kwargs))
                    return "mode-result"

            ProbabilityOverride.calls.clear()
            TrainingOverride.calls.clear()
            mode_probability = ProbabilityOverride()
            mode_training = TrainingOverride()
            with RecordingMode():
                mode_result = module.dropout(
                    source, mode_probability, mode_training
                )
            mode_call = mode_calls[0]
            mode_probability_calls = tuple(ProbabilityOverride.calls)
            mode_training_calls = tuple(TrainingOverride.calls)

            declining_mode_calls = []

            class DecliningMode(module.overrides.TorchFunctionMode):
                def __torch_function__(
                    self, func, types, args=(), kwargs=None
                ):
                    declining_mode_calls.append(
                        (func is module.dropout, tuple(item.__name__ for item in types))
                    )
                    return NotImplemented

            ProbabilityOverride.calls.clear()
            declining_probability = ProbabilityOverride()
            with DecliningMode():
                declining_result = module.dropout(
                    source, declining_probability, False
                )
            declining_override_call = ProbabilityOverride.calls[0]

            one_shot_results = []
            for slot in (1, 2):
                events = []

                class Descriptor:
                    def __init__(self):
                        self.lookups = 0

                    def __get__(self, instance, owner):
                        self.lookups += 1
                        events.append(self.lookups)
                        if self.lookups == 1:
                            raise AttributeError("transient probe failure")

                        def handler(func, types, args=(), kwargs=None):
                            return "unexpected"

                        return handler

                descriptor = Descriptor()

                class OneShot:
                    __torch_function__ = descriptor

                arguments = [source, 0, False]
                arguments[slot] = OneShot()
                error_type, message = self.capture_error(
                    lambda arguments=arguments: module.dropout(*arguments)
                )
                one_shot_results.append(
                    (slot, error_type.__name__, message, tuple(events))
                )

            def normalize(call, overridden, *, keyword):
                func, dispatch_types, args, kwargs = call
                return (
                    func is module.dropout,
                    tuple(item.__name__ for item in dispatch_types),
                    ()
                    if keyword
                    else (
                        args[0] is source,
                        args[1] is overridden,
                        args[2] is False,
                    ),
                    None
                    if kwargs is None
                    else (
                        tuple(kwargs),
                        kwargs.get("input") is source,
                        kwargs.get("p"),
                        kwargs.get("train") is overridden,
                    ),
                )

            mode_func, mode_types, mode_args, mode_kwargs = mode_call
            return {
                "probability": (
                    probability_result,
                    normalize(
                        probability_call,
                        probability_argument,
                        keyword=False,
                    ),
                ),
                "training": (
                    training_result,
                    normalize(training_call, training_argument, keyword=True),
                ),
                "ordered": (ordered_result, tuple(ordered_events)),
                "shared": (
                    shared_result,
                    tuple(SharedOverride.calls),
                ),
                "mode": (
                    mode_result,
                    mode_func is module.dropout,
                    tuple(item.__name__ for item in mode_types),
                    mode_args[0] is source,
                    mode_args[1] is mode_probability,
                    mode_args[2] is mode_training,
                    mode_kwargs,
                    mode_probability_calls,
                    mode_training_calls,
                ),
                "declining_mode": (
                    declining_result,
                    tuple(declining_mode_calls),
                    normalize(
                        declining_override_call,
                        declining_probability,
                        keyword=False,
                    ),
                ),
                "one_shot": tuple(one_shot_results),
            }

        self.assertEqual(observe(torch), observe(reference_torch))

    def test_stochastic_rejection_preserves_reference_rng_and_input(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        before = np.asarray(source.detach()).copy().view(np.uint32)

        for probability in (0.25, torch.tensor(0.25)):
            rng = reference_torch.get_rng_state().clone()
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.dropout does not support sampling$",
                ):
                    torch.dropout(source, probability, True)
                self.assert_reference_rng_unchanged(rng)
                np.testing.assert_array_equal(
                    np.asarray(source.detach()).view(np.uint32), before
                )
                self.assertIsNone(leaf.grad)


if __name__ == "__main__":
    unittest.main()
