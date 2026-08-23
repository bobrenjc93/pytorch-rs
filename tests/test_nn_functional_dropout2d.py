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


FUNCTION_DOC = r"""Randomly zero out entire channels (a channel is a 2D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = r"""Randomly zero out entire channels (a channel is a 2D feature map).

For example, the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 2D tensor :math:`\text{input}[i, j]` of the input tensor.
Each channel will be zeroed out independently on every forward call with
probability :attr:`p` using samples from a Bernoulli distribution.

See :class:`~torch.nn.Dropout2d` for details.

Args:
    p: probability of a channel to be zeroed. Default: 0.5
    training: apply dropout if is ``True``. Default: ``True``
    inplace: If set to ``True``, will do this operation in-place. Default: ``False``
"""

DROPOUT2D_3D_WARNING = (
    "dropout2d: Received a 3D input to dropout2d and assuming that channel-wise "
    "1D dropout behavior is desired - input is interpreted as shape (N, C, L), "
    "where C is the channel dim. This behavior will change in a future release "
    "to interpret the input as one without a batch dimension, i.e. shape (C, H, "
    "W). To maintain the 1D channel-wise dropout behavior, please switch to using "
    "dropout1d instead."
)

DROPOUT2D_2D_WARNING = (
    "dropout2d: Received a 2-D input to dropout2d, which is deprecated and will "
    "result in an error in a future release. To retain the behavior and silence "
    "this warning, please use dropout instead. Note that dropout2d exists to "
    "provide channel-wise dropout on inputs with 2 spatial dimensions, a channel "
    "dimension, and an optional batch dimension (i.e. 3D or 4D inputs)."
)


class FunctionalDropout2dTests(unittest.TestCase):
    def make_nonempty_cases(self, *, requires_grad):
        contiguous = torch.tensor(
            [
                [
                    [[1.0, -2.0], [3.0, -4.0]],
                    [[5.0, -6.0], [7.0, -8.0]],
                ]
            ],
            requires_grad=requires_grad,
        )
        backing = torch.tensor(
            [
                [
                    [
                        [[9.0, 10.0], [11.0, 12.0]],
                        [[13.0, 14.0], [15.0, 16.0]],
                    ]
                ],
                [
                    [
                        [[-1.0, 2.0], [-3.0, 4.0]],
                        [[-5.0, 6.0], [-7.0, 8.0]],
                    ]
                ],
            ],
            requires_grad=requires_grad,
        )
        offset = backing[1]
        return contiguous, offset, offset.transpose(2, 3)

    def make_rank_three_nonempty_cases(self, *, requires_grad):
        contiguous = torch.tensor(
            [
                [[1.0, -2.0], [3.0, -4.0]],
                [[5.0, -6.0], [7.0, -8.0]],
            ],
            requires_grad=requires_grad,
        )
        backing = torch.tensor(
            [
                [
                    [[9.0, 10.0], [11.0, 12.0]],
                    [[13.0, 14.0], [15.0, 16.0]],
                ],
                [
                    [[-1.0, 2.0], [-3.0, 4.0]],
                    [[-5.0, 6.0], [-7.0, 8.0]],
                ],
            ],
            requires_grad=requires_grad,
        )
        offset = backing[1]
        return contiguous, offset, offset.transpose(1, 2)

    def make_rank_two_nonempty_cases(self, *, requires_grad):
        contiguous = torch.tensor(
            [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
            requires_grad=requires_grad,
        )
        backing = torch.tensor(
            [
                [
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                    [17.0, 18.0, 19.0, 20.0],
                ],
                [
                    [-1.0, 2.0, -3.0, 4.0],
                    [-5.0, 6.0, -7.0, 8.0],
                    [-9.0, 10.0, -11.0, 12.0],
                ],
            ],
            requires_grad=requires_grad,
        )
        offset = backing[1]
        return contiguous, offset, offset.transpose(0, 1)

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

    def assert_legacy_rank_three_warning(self, caught):
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(str(caught[0].message), DROPOUT2D_3D_WARNING)

    def assert_deprecated_rank_two_warning(self, caught):
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(str(caught[0].message), DROPOUT2D_2D_WARNING)

    def test_imports_signature_documentation_and_pickling(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import dropout2d

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(dropout2d, functional.dropout2d)
        wildcard_namespace = {}
        exec("from torch_rs.nn.functional import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["dropout2d"], functional.dropout2d)
        self.assertFalse(hasattr(torch, "dropout2d"))
        self.assertNotIn("dropout2d", torch.__all__)
        self.assertFalse(hasattr(torch, "_nn_functional_dropout2d"))

        function = functional.dropout2d
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "dropout2d")
        self.assertEqual(function.__qualname__, "dropout2d")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (0.5, True, False))
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
        self.assertIs(parameters[2].default, True)
        self.assertIs(parameters[3].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_evaluation_and_zero_probability_return_exact_rank_four_inputs(self):
        evaluation_probabilities = (
            0.25,
            1,
            True,
            np.float32(0.75),
            torch.tensor(0.5),
        )
        zero_probabilities = (
            0,
            -0.0,
            False,
            np.bool_(False),
            np.float32(0.0),
            torch.tensor(0.0),
        )

        for case, source in enumerate(
            self.make_nonempty_cases(requires_grad=True)
        ):
            for probability in evaluation_probabilities:
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=type(probability),
                        training=False,
                        inplace=inplace,
                    ):
                        output = functional.dropout2d(
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
                        probability=type(probability),
                        training=True,
                        inplace=inplace,
                    ):
                        output = functional.dropout2d(
                            source, probability, True, inplace
                        )
                        self.assert_unchanged_identity(output, source, before)

    def test_empty_rank_four_training_inputs_return_the_exact_input(self):
        backing = torch.zeros((2, 1, 0, 3, 4), requires_grad=True)
        sources = (
            torch.zeros((2, 0, 3, 4), requires_grad=True),
            backing[1],
            backing[1].transpose(2, 3),
        )

        for case, source in enumerate(sources):
            for probability in (0.25, 1.0, np.float32(0.5), torch.tensor(0.5)):
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=type(probability),
                        inplace=inplace,
                    ):
                        output = functional.dropout2d(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                        self.assert_unchanged_identity(output, source, before)

    def test_rank_three_identity_paths_warn_and_preserve_exact_inputs(self):
        calls = (
            (0.75, False, False),
            (1.0, False, True),
            (0, True, False),
            (torch.tensor(0.0), True, True),
        )
        for case, source in enumerate(
            self.make_rank_three_nonempty_cases(requires_grad=True)
        ):
            for probability, training, inplace in calls:
                before = self.snapshot(source)
                with self.subTest(
                    case=case,
                    probability=type(probability),
                    training=training,
                    inplace=inplace,
                ):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        output = functional.dropout2d(
                            source,
                            p=probability,
                            training=training,
                            inplace=inplace,
                        )
                    self.assert_legacy_rank_three_warning(caught)
                    self.assert_unchanged_identity(output, source, before)

        backing = torch.zeros((2, 1, 0, 4), requires_grad=True)
        empty_sources = (
            torch.zeros((2, 0, 4), requires_grad=True),
            backing[1],
            backing[1].transpose(1, 2),
        )
        for case, source in enumerate(empty_sources):
            for probability in (0.25, 1.0, torch.tensor(0.5)):
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        empty_case=case,
                        probability=type(probability),
                        inplace=inplace,
                    ):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            output = functional.dropout2d(
                                source,
                                p=probability,
                                training=True,
                                inplace=inplace,
                            )
                        self.assert_legacy_rank_three_warning(caught)
                        self.assert_unchanged_identity(output, source, before)

    def test_rank_two_identity_paths_warn_and_preserve_exact_inputs(self):
        calls = (
            (0.75, False, False),
            (1.0, False, True),
            (0, True, False),
            (torch.tensor(0.0), True, True),
        )
        for case, source in enumerate(
            self.make_rank_two_nonempty_cases(requires_grad=True)
        ):
            for probability, training, inplace in calls:
                before = self.snapshot(source)
                with self.subTest(
                    case=case,
                    probability=type(probability),
                    training=training,
                    inplace=inplace,
                ):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        output = functional.dropout2d(
                            source,
                            p=probability,
                            training=training,
                            inplace=inplace,
                        )
                    self.assert_deprecated_rank_two_warning(caught)
                    self.assert_unchanged_identity(output, source, before)

        empty_sources = (
            torch.zeros((2, 0), requires_grad=True),
            torch.zeros((0, 3), requires_grad=True).transpose(0, 1),
        )
        for case, source in enumerate(empty_sources):
            for probability in (0.25, 1.0, torch.tensor(0.5)):
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        empty_case=case,
                        probability=type(probability),
                        inplace=inplace,
                    ):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            output = functional.dropout2d(
                                source,
                                p=probability,
                                training=True,
                                inplace=inplace,
                            )
                        self.assert_deprecated_rank_two_warning(caught)
                        self.assert_unchanged_identity(output, source, before)

    def test_rank_three_identity_preserves_autograd_and_no_grad_state(self):
        leaf = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
        )
        source = leaf.transpose(1, 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = functional.dropout2d(
                source, p=0, training=True, inplace=True
            )
        self.assert_legacy_rank_three_warning(caught)
        self.assertIs(output, source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, source.output_nr)

        weights = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[[2.0, 5.0], [3.0, 7.0]]])

        untracked_leaf = torch.zeros((1, 2, 3), requires_grad=True)
        untracked_source = untracked_leaf.transpose(1, 2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                unchanged = functional.dropout2d(
                    untracked_source, p=0.75, training=False
                )
        self.assert_legacy_rank_three_warning(caught)
        self.assertIs(unchanged, untracked_source)
        self.assertTrue(unchanged.requires_grad)
        self.assertFalse(unchanged.is_leaf)
        self.assertIsNone(untracked_leaf.grad)

    def test_rank_two_identity_preserves_autograd_and_no_grad_state(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = functional.dropout2d(
                source, p=0, training=True, inplace=True
            )
        self.assert_deprecated_rank_two_warning(caught)
        self.assertIs(output, source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, source.output_nr)

        weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 5.0], [3.0, 7.0]])

        untracked_leaf = torch.zeros((2, 3), requires_grad=True)
        untracked_source = untracked_leaf.transpose(0, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                unchanged = functional.dropout2d(
                    untracked_source, p=0.75, training=False, inplace=True
                )
        self.assert_deprecated_rank_two_warning(caught)
        self.assertIs(unchanged, untracked_source)
        self.assertTrue(unchanged.requires_grad)
        self.assertFalse(unchanged.is_leaf)
        self.assertIsNone(untracked_leaf.grad)

    def test_identity_preserves_autograd_and_no_grad_state(self):
        leaf = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True
        )
        source = leaf.transpose(2, 3)
        output = functional.dropout2d(
            source, p=0, training=True, inplace=True
        )
        self.assertIs(output, source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, source.output_nr)

        weights = torch.tensor([[[[2.0, 3.0], [5.0, 7.0]]]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[[[2.0, 5.0], [3.0, 7.0]]]])

        untracked_leaf = torch.zeros((1, 2, 3, 4), requires_grad=True)
        untracked_source = untracked_leaf.transpose(2, 3)
        with torch.no_grad():
            unchanged = functional.dropout2d(
                untracked_source, p=0.75, training=False
            )
        self.assertIs(unchanged, untracked_source)
        self.assertTrue(unchanged.requires_grad)
        self.assertFalse(unchanged.is_leaf)
        self.assertIsNone(untracked_leaf.grad)

    def test_probability_validation_and_feature_dropout_schema(self):
        source = torch.zeros((1, 2, 3, 4))

        for probability in (-0.1, 1.1, float("inf"), -float("inf")):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError) as raised:
                    functional.dropout2d(
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
                    functional.dropout2d(
                        source, p=probability, training=False
                    )

        for inplace in (False, True):
            with self.subTest(none_input=True, inplace=inplace):
                with self.assertRaisesRegex(
                    AttributeError,
                    "^'NoneType' object has no attribute 'dim'$",
                ):
                    functional.dropout2d(None, p=0, inplace=inplace)

        error_cases = (
            (
                lambda: functional.dropout2d(source, p=Decimal("0")),
                "feature_dropout(): argument 'p' (position 2) must be float, "
                "not decimal.Decimal",
            ),
            (
                lambda: functional.dropout2d(
                    source, p=Fraction(0, 1), inplace=True
                ),
                "feature_dropout_(): argument 'p' (position 2) must be float, "
                "not Fraction",
            ),
            (
                lambda: functional.dropout2d(source, p=0, training=1),
                "feature_dropout(): argument 'train' (position 3) must be "
                "bool, not int",
            ),
            (
                lambda: functional.dropout2d(
                    source, p=0, training=np.bool_(False), inplace=True
                ),
                "feature_dropout_(): argument 'train' (position 3) must be "
                "bool, not numpy.bool",
            ),
        )
        for case, (call, message) in enumerate(error_cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        for value, training in ((0.0, True), (0.5, False), (1.0, False)):
            for inplace in (False, True):
                probability = torch.tensor(value)
                with self.subTest(value=value, inplace=inplace):
                    self.assertIs(
                        functional.dropout2d(
                            source,
                            p=probability,
                            training=training,
                            inplace=inplace,
                        ),
                        source,
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
                    functional.dropout2d(source, p=probability)
                self.assertEqual(str(raised.exception), message)

        for probability in (
            torch.tensor([0.0]),
            torch.tensor(0.0, requires_grad=True),
        ):
            for inplace, operation in (
                (False, "feature_dropout"),
                (True, "feature_dropout_"),
            ):
                with self.subTest(shape=probability.shape, inplace=inplace):
                    with self.assertRaises(TypeError) as raised:
                        functional.dropout2d(
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

        for inplace in (0, 1, None, "", "inplace", [], [1]):
            with self.subTest(inplace=inplace):
                self.assertIs(
                    functional.dropout2d(
                        source, p=0.5, training=False, inplace=inplace
                    ),
                    source,
                )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = functional.dropout2d(
                source, p=np.complex64(0), training=True
            )
        self.assertIs(output, source)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, np.exceptions.ComplexWarning)

    def test_rank_two_warning_validation_order(self):
        source = torch.zeros((2, 3))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(
                ValueError,
                "^dropout probability has to be between 0 and 1, but got -0.1$",
            ):
                functional.dropout2d(
                    source, p=-0.1, training="invalid", inplace=True
                )
        self.assertEqual(caught, [])

        error_cases = (
            (
                lambda: functional.dropout2d(
                    source, p=Decimal("0"), training=False
                ),
                "feature_dropout(): argument 'p' (position 2) must be float, "
                "not decimal.Decimal",
            ),
            (
                lambda: functional.dropout2d(
                    source, p=0, training=np.bool_(False), inplace=True
                ),
                "feature_dropout_(): argument 'train' (position 3) must be "
                "bool, not numpy.bool",
            ),
        )
        for case, (call, message) in enumerate(error_cases):
            with self.subTest(case=case):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaises(TypeError) as raised:
                        call()
                self.assertEqual(str(raised.exception), message)
                self.assert_deprecated_rank_two_warning(caught)

    def test_overrides_and_modes_receive_the_public_function(self):
        replacement = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return replacement

        input = Override()
        output = functional.dropout2d(
            input, p=-1, training="invalid", inplace=True
        )
        self.assertIs(output, replacement)
        func, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(func, functional.dropout2d)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (input,))
        self.assertEqual(
            kwargs, {"p": -1, "training": "invalid", "inplace": True}
        )

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, *, forward=False):
                self.forward = forward
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **kwargs)
                return "mode-result"

        source = torch.zeros((1, 2, 3, 4))
        mode = RecordingMode()
        with mode:
            self.assertEqual(
                functional.dropout2d(
                    source, p=-1, training="invalid", inplace=True
                ),
                "mode-result",
            )
        self.assertIs(mode.calls[0][0], functional.dropout2d)
        self.assertEqual(mode.calls[0][1], (torch.Tensor,))

        mode = RecordingMode(forward=True)
        with mode:
            output = functional.dropout2d(
                source, p=0, training=True, inplace=True
            )
        self.assertIs(output, source)
        self.assertEqual(len(mode.calls), 1)

        source = torch.zeros((2, 3, 4))
        mode = RecordingMode()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mode:
                output = functional.dropout2d(
                    source, p=0.5, training=False, inplace=True
                )
        self.assertEqual(output, "mode-result")
        self.assertEqual(caught, [])
        self.assertEqual(len(mode.calls), 1)

        mode = RecordingMode(forward=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mode:
                output = functional.dropout2d(
                    source, p=0, training=True, inplace=True
                )
        self.assertIs(output, source)
        self.assertEqual(len(mode.calls), 1)
        self.assert_legacy_rank_three_warning(caught)

        source = torch.zeros((2, 3))
        mode = RecordingMode()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mode:
                output = functional.dropout2d(
                    source, p=0.5, training=False, inplace=True
                )
        self.assertEqual(output, "mode-result")
        self.assertEqual(caught, [])
        self.assertEqual(len(mode.calls), 1)
        self.assertIs(mode.calls[0][0], functional.dropout2d)

        mode = RecordingMode(forward=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mode:
                output = functional.dropout2d(
                    source, p=0, training=True, inplace=True
                )
        self.assertIs(output, source)
        self.assertEqual(len(mode.calls), 1)
        self.assertIs(mode.calls[0][0], functional.dropout2d)
        self.assert_deprecated_rank_two_warning(caught)

    def test_sampling_and_unsupported_ranks_are_explicitly_rejected(self):
        sources = (
            torch.tensor(
                [[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True
            ),
            torch.tensor(
                [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True
            ),
            torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            ),
        )
        for source in sources:
            before = self.snapshot(source)
            for probability in (0.25, 1.0):
                for inplace in (False, True):
                    with self.subTest(
                        rank=len(source.shape),
                        sampling=probability,
                        inplace=inplace,
                    ):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            with self.assertRaisesRegex(
                                NotImplementedError,
                                "^torch_rs.nn.functional.dropout2d does not "
                                "support sampling$",
                            ):
                                functional.dropout2d(
                                    source,
                                    p=probability,
                                    training=True,
                                    inplace=inplace,
                                )
                        if len(source.shape) == 2:
                            self.assert_deprecated_rank_two_warning(caught)
                        elif len(source.shape) == 3:
                            self.assert_legacy_rank_three_warning(caught)
                        else:
                            self.assertEqual(caught, [])
                        self.assertEqual(
                            self.snapshot(source)[:-1], before[:-1]
                        )
                        np.testing.assert_array_equal(
                            self.snapshot(source)[-1], before[-1]
                        )
                        self.assertIsNone(source.grad)

        unsupported_ranks = (
            torch.tensor(-0.0),
            torch.zeros((2,)),
            torch.zeros((2, 3, 4, 5, 6)),
            torch.zeros((2, 0, 3, 4, 5)),
        )
        identity_modes = ((0.5, False), (0.0, True), (0.5, True))
        for source in unsupported_ranks:
            for probability, training in identity_modes:
                for inplace in (False, True):
                    with self.subTest(
                        rank=len(source.shape),
                        empty=source.numel() == 0,
                        probability=probability,
                        training=training,
                        inplace=inplace,
                    ):
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            with self.assertRaisesRegex(
                                NotImplementedError,
                                "^torch_rs.nn.functional.dropout2d only "
                                "supports rank-2 or rank-3 or rank-4 inputs$",
                            ):
                                functional.dropout2d(
                                    source,
                                    p=probability,
                                    training=training,
                                    inplace=inplace,
                                )
                        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
