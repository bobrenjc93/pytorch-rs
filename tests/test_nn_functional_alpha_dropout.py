import importlib
import inspect
import pickle
import sys
import types
import unittest
import warnings
from decimal import Decimal
from fractions import Fraction

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


FUNCTION_DOC = """Apply alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = """Apply alpha dropout to the input.

See :class:`~torch.nn.AlphaDropout` for details.
"""


class FunctionalAlphaDropoutTests(unittest.TestCase):
    def make_cases(self, *, requires_grad):
        scalar = torch.tensor(
            -0.0, dtype=torch.float32, requires_grad=requires_grad
        )
        empty_leaf = torch.zeros(
            (2, 0, 3), dtype=torch.float32, requires_grad=requires_grad
        )
        values = torch.tensor(
            [
                [9.0, 9.0, 9.0, 9.0],
                [-1.0, 2.0, -0.0, 3.0],
                [4.0, -5.0, 6.0, -7.0],
            ],
            requires_grad=requires_grad,
        )
        offset = values[1]
        return (
            scalar,
            empty_leaf.transpose(0, 2)[1],
            offset,
            offset.reshape(2, 2).transpose(0, 1),
        )

    def snapshot(self, tensor):
        return (
            id(tensor),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
            np.asarray(tensor.detach()).copy().reshape(-1).view(np.uint32),
        )

    def assert_unchanged_identity(self, output, source, before):
        self.assertIs(output, source)
        self.assertTrue(output.is_set_to(source))
        after = self.snapshot(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])

    def test_imports_signature_documentation_and_pickling(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module(
            "torch_rs.nn.functional"
        )
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import alpha_dropout

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(alpha_dropout, functional.alpha_dropout)
        self.assertNotIn("nn", torch.__all__)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))
        wildcard_namespace = {}
        exec(
            "from torch_rs.nn.functional import *", wildcard_namespace
        )
        self.assertIs(
            wildcard_namespace["alpha_dropout"], functional.alpha_dropout
        )
        self.assertIsNone(nn.__doc__)
        self.assertEqual(functional.__doc__, "Functional interface.")
        self.assertFalse(hasattr(torch, "alpha_dropout"))
        self.assertNotIn("alpha_dropout", torch.__all__)
        self.assertFalse(hasattr(torch, "_nn_functional_alpha_dropout"))

        function = functional.alpha_dropout
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "alpha_dropout")
        self.assertEqual(function.__qualname__, "alpha_dropout")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (0.5, False, False))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(
            tuple(signature.parameters),
            ("input", "p", "training", "inplace"),
        )
        self.assertIs(parameters[0].annotation, torch.Tensor)
        self.assertIs(parameters[1].annotation, float)
        self.assertIs(parameters[2].annotation, bool)
        self.assertIs(parameters[3].annotation, bool)
        self.assertEqual(parameters[1].default, 0.5)
        self.assertIs(parameters[2].default, False)
        self.assertIs(parameters[3].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_evaluation_and_zero_probability_return_the_exact_input(self):
        evaluation_probabilities = (
            0.0,
            0.25,
            1,
            True,
            np.float32(0.75),
        )
        zero_probabilities = (
            0,
            0.0,
            -0.0,
            False,
            np.bool_(False),
            np.int64(0),
            np.float32(0.0),
        )

        for case, source in enumerate(self.make_cases(requires_grad=True)):
            before = self.snapshot(source)
            with self.subTest(case=case, default=True):
                output = functional.alpha_dropout(source)
                self.assert_unchanged_identity(output, source, before)

            for probability in evaluation_probabilities:
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=probability,
                        training=False,
                        inplace=inplace,
                    ):
                        output = functional.alpha_dropout(
                            source,
                            p=probability,
                            training=False,
                            inplace=inplace,
                        )
                        self.assert_unchanged_identity(output, source, before)

            for probability in zero_probabilities:
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=probability,
                        training=True,
                        inplace=inplace,
                    ):
                        output = functional.alpha_dropout(
                            source,
                            probability,
                            True,
                            inplace,
                        )
                        self.assert_unchanged_identity(output, source, before)

    def test_identity_does_not_change_autograd_state(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        output = functional.alpha_dropout(
            source, p=0, training=True, inplace=True
        )
        self.assertIs(output, source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, source.output_nr)

        weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 5.0], [3.0, 7.0]])

        untracked_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        untracked_source = untracked_leaf.transpose(0, 1)
        with torch.no_grad():
            unchanged = functional.alpha_dropout(
                untracked_source, p=0.75, training=False
            )
        self.assertIs(unchanged, untracked_source)
        self.assertTrue(unchanged.requires_grad)
        self.assertFalse(unchanged.is_leaf)
        self.assertIsNone(untracked_leaf.grad)

    def test_empty_training_inputs_return_the_exact_input(self):
        leaf = torch.zeros((2, 0, 3), requires_grad=True)
        sources = (leaf, leaf.transpose(0, 2)[1])

        for case, source in enumerate(sources):
            for probability in (0.25, 1.0, np.float32(0.25)):
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=probability,
                        inplace=inplace,
                    ):
                        output = functional.alpha_dropout(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                        self.assert_unchanged_identity(
                            output, source, before
                        )

    def test_probability_validation_and_native_schema_errors(self):
        source = torch.tensor([1.0, 2.0])

        for probability in (-0.1, 1.1, float("inf"), -float("inf")):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError) as raised:
                    functional.alpha_dropout(
                        None,
                        p=probability,
                        training="invalid",
                        inplace=True,
                    )
                self.assertEqual(
                    str(raised.exception),
                    "dropout probability has to be between 0 and 1, but got "
                    f"{probability}",
                )

        for probability in (float("nan"), np.float32(np.nan)):
            with self.subTest(probability=type(probability)):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^dropout probability has to be between 0 and 1, but got nan$",
                ):
                    functional.alpha_dropout(
                        source, p=probability, training=False
                    )

        error_cases = (
            (
                lambda: functional.alpha_dropout(None, p=0),
                TypeError,
                "alpha_dropout(): argument 'input' (position 1) must be "
                "Tensor, not NoneType",
            ),
            (
                lambda: functional.alpha_dropout(None, p=0, inplace=True),
                TypeError,
                "alpha_dropout_(): argument 'input' (position 1) must be "
                "Tensor, not NoneType",
            ),
            (
                lambda: functional.alpha_dropout(source, p=None),
                TypeError,
                "'<' not supported between instances of 'NoneType' and 'float'",
            ),
            (
                lambda: functional.alpha_dropout(
                    source, p=Decimal("0"), training=False
                ),
                TypeError,
                "alpha_dropout(): argument 'p' (position 2) must be float, "
                "not decimal.Decimal",
            ),
            (
                lambda: functional.alpha_dropout(
                    source, p=Fraction(0, 1), training=False, inplace=True
                ),
                TypeError,
                "alpha_dropout_(): argument 'p' (position 2) must be float, "
                "not Fraction",
            ),
            (
                lambda: functional.alpha_dropout(source, p=0, training=1),
                TypeError,
                "alpha_dropout(): argument 'train' (position 3) must be "
                "bool, not int",
            ),
            (
                lambda: functional.alpha_dropout(
                    source, p=0, training=np.bool_(False)
                ),
                TypeError,
                "alpha_dropout(): argument 'train' (position 3) must be "
                "bool, not numpy.bool",
            ),
        )
        for case, (call, error_type, message) in enumerate(error_cases):
            with self.subTest(case=case):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        for inplace in (0, 1, None, "", "inplace", [], [1]):
            with self.subTest(inplace=inplace):
                self.assertIs(
                    functional.alpha_dropout(
                        source, p=0.5, training=False, inplace=inplace
                    ),
                    source,
                )

        class BoolFailure:
            def __bool__(self):
                raise RuntimeError("inplace truthiness failed")

        with self.assertRaisesRegex(
            RuntimeError, "^inplace truthiness failed$"
        ):
            functional.alpha_dropout(
                source, p=0, training=True, inplace=BoolFailure()
            )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = functional.alpha_dropout(
                source, p=np.complex64(0), training=True
            )
        self.assertIs(output, source)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, np.exceptions.ComplexWarning)
        self.assertEqual(
            str(caught[0].message),
            "Casting complex values to real discards the imaginary part",
        )

    def test_native_probability_formatting_matches_dropout_operators(self):
        source = torch.tensor([1.0])

        class SneakyProbability(float):
            def __lt__(self, other):
                return False

            def __gt__(self, other):
                return False

        cases = (
            (-float("nan"), "-nan"),
            (SneakyProbability(1.0000000000001), "1"),
            (SneakyProbability(1.23456789), "1.23457"),
            (SneakyProbability(999999.9), "1e+06"),
            (SneakyProbability(-1.23456789e-7), "-1.23457e-07"),
            (SneakyProbability(-5e-324), "-4.94066e-324"),
        )
        for probability, formatted in cases:
            with self.subTest(probability=formatted):
                with self.assertRaises(RuntimeError) as raised:
                    functional.alpha_dropout(
                        source, p=probability, training=False
                    )
                self.assertEqual(
                    str(raised.exception),
                    "dropout probability has to be between 0 and 1, but got "
                    f"{formatted}",
                )

    def test_tensor_probabilities_remain_unsupported(self):
        source = torch.tensor([1.0, 2.0])
        probabilities = (
            torch.tensor(0.0),
            torch.tensor([0.0]),
            torch.tensor(0.0, requires_grad=True),
        )

        for probability in probabilities:
            for inplace, operation in (
                (False, "alpha_dropout"),
                (True, "alpha_dropout_"),
            ):
                with self.subTest(
                    shape=probability.shape, inplace=inplace
                ):
                    with self.assertRaises(TypeError) as raised:
                        functional.alpha_dropout(
                            source,
                            p=probability,
                            training=False,
                            inplace=inplace,
                        )
                    self.assertEqual(
                        str(raised.exception),
                        f"{operation}(): argument 'p' (position 2) must be "
                        "float, not Tensor",
                    )

        for probability, message in (
            (
                torch.zeros((0,)),
                "Boolean value of Tensor with no values is ambiguous",
            ),
            (
                torch.tensor([0.0, 1.0]),
                "Boolean value of Tensor with more than one value is ambiguous",
            ),
        ):
            with self.subTest(shape=probability.shape):
                with self.assertRaises(RuntimeError) as raised:
                    functional.alpha_dropout(source, p=probability)
                self.assertEqual(str(raised.exception), message)

    def test_overrides_and_torch_function_modes_use_the_public_function(self):
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
        inplace = object()
        output = functional.alpha_dropout(
            input, p=-1, training="invalid", inplace=inplace
        )
        self.assertIs(output, replacement)
        self.assertEqual(len(Override.calls), 1)
        func, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(func, functional.alpha_dropout)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (input,))
        self.assertEqual(
            kwargs,
            {"p": -1, "training": "invalid", "inplace": inplace},
        )

        class PlainOverride:
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return replacement

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(
                functional.alpha_dropout(PlainOverride()), replacement
            )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)

        class DecliningOverride:
            @classmethod
            def __torch_function__(
                cls, func, types, args=(), kwargs=None
            ):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^no implementation found for "
            "'torch_rs.nn.functional.alpha_dropout' ",
        ):
            functional.alpha_dropout(DecliningOverride(), p=0)

        source = torch.tensor([1.0, 2.0])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=replacement, *, forward=False):
                self.result = result
                self.forward = forward
                self.calls = []

            def __torch_function__(
                self, func, types, args=(), kwargs=None
            ):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **kwargs)
                return self.result

        recording = RecordingMode()
        with recording:
            output = functional.alpha_dropout(
                source, p=-1, training="invalid", inplace=True
            )
        self.assertIs(output, replacement)
        self.assertEqual(len(recording.calls), 1)
        self.assertIs(recording.calls[0][0], functional.alpha_dropout)
        self.assertEqual(recording.calls[0][1], (torch.Tensor,))

        forwarding = RecordingMode(forward=True)
        with forwarding:
            forwarded = functional.alpha_dropout(
                source, p=0, training=True, inplace=True
            )
        self.assertIs(forwarded, source)
        self.assertEqual(len(forwarding.calls), 1)

    def test_sampling_paths_are_explicitly_unsupported_and_non_mutating(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]

        for probability in (0.25, 1.0):
            for inplace in (False, True):
                before = self.snapshot(source)
                with self.subTest(probability=probability, inplace=inplace):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        "^torch_rs.nn.functional.alpha_dropout does not "
                        "support sampling$",
                    ):
                        functional.alpha_dropout(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                    after = self.snapshot(source)
                    self.assertEqual(after[:-1], before[:-1])
                    np.testing.assert_array_equal(after[-1], before[-1])
                    self.assertIsNone(leaf.grad)


if __name__ == "__main__":
    unittest.main()
