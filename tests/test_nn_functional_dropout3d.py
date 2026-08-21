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


FUNCTION_DOC = r"""Randomly zero out entire channels (a channel is a 3D feature map).

    For example, the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]` of the input tensor.
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = r"""Randomly zero out entire channels (a channel is a 3D feature map).

For example, the :math:`j`-th channel of the :math:`i`-th sample in the
batched input is a 3D tensor :math:`\text{input}[i, j]` of the input tensor.
Each channel will be zeroed out independently on every forward call with
probability :attr:`p` using samples from a Bernoulli distribution.

See :class:`~torch.nn.Dropout3d` for details.

Args:
    p: probability of a channel to be zeroed. Default: 0.5
    training: apply dropout if is ``True``. Default: ``True``
    inplace: If set to ``True``, will do this operation in-place. Default: ``False``
"""


class FunctionalDropout3dTests(unittest.TestCase):
    def make_probability_one_cases(self):
        values = [
            -1.0,
            2.0,
            -0.0,
            0.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            -3.0,
        ] * 2
        contiguous = torch.tensor(values).reshape((1, 2, 2, 2, 2))
        backing = torch.tensor(([9.0] * 16) + values).reshape(
            (2, 1, 2, 2, 2, 2)
        )
        return (
            contiguous,
            backing[1],
            contiguous.transpose(2, 4),
            contiguous.contiguous(memory_format=torch.channels_last_3d),
        )

    def make_rank_four_probability_one_cases(self):
        return tuple(source[0] for source in self.make_probability_one_cases())

    def make_rank_four_empty_cases(self, *, requires_grad=False):
        backing = torch.zeros(
            (2, 2, 0, 3, 4), requires_grad=requires_grad
        )
        offset = backing[1]
        return (
            torch.zeros((2, 0, 3, 4), requires_grad=requires_grad),
            offset,
            offset.transpose(2, 3),
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

    def assert_optional_batch_identity(self, output, source, before):
        self.assertIsNot(output, source)
        self.assertTrue(output.is_set_to(source))
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.stride(), source.stride())
        self.assertEqual(output.storage_offset(), source.storage_offset())
        self.assertEqual(output.data_ptr(), source.data_ptr())
        self.assertIs(output.dtype, source.dtype)
        self.assertEqual(output.device, source.device)
        after = self.snapshot(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])

    def test_imports_signature_documentation_and_pickling(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import dropout3d

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(dropout3d, functional.dropout3d)
        wildcard_namespace = {}
        exec("from torch_rs.nn.functional import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["dropout3d"], functional.dropout3d)
        self.assertFalse(hasattr(torch, "dropout3d"))
        self.assertNotIn("dropout3d", torch.__all__)
        self.assertFalse(hasattr(nn, "Dropout3d"))
        self.assertFalse(hasattr(torch, "_nn_functional_dropout3d"))

        function = functional.dropout3d
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "dropout3d")
        self.assertEqual(function.__qualname__, "dropout3d")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (0.5, True, False))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(
            tuple(signature.parameters), ("input", "p", "training", "inplace")
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

    def test_identity_modes_return_exact_rank_five_inputs(self):
        contiguous = torch.tensor(
            [[[[[1.0, -2.0], [3.0, -4.0]]]]], requires_grad=True
        )
        backing = torch.zeros((2, 1, 2, 3, 4, 5), requires_grad=True)
        offset = backing[1]
        sources = (contiguous, offset, offset.transpose(3, 4))

        calls = (
            {"p": 0.75, "training": False, "inplace": False},
            {"p": 1, "training": False, "inplace": True},
            {"p": 0, "training": True, "inplace": False},
            {"p": torch.tensor(0.0), "training": True, "inplace": True},
        )
        for case, source in enumerate(sources):
            for kwargs in calls:
                before = self.snapshot(source)
                with self.subTest(case=case, kwargs=kwargs):
                    output = functional.dropout3d(source, **kwargs)
                    self.assert_unchanged_identity(output, source, before)

    def test_empty_rank_five_training_inputs_return_exact_input(self):
        backing = torch.zeros((2, 1, 0, 3, 4, 5), requires_grad=True)
        sources = (
            torch.zeros((2, 0, 3, 4, 5), requires_grad=True),
            backing[1],
            backing[1].transpose(3, 4),
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
                        output = functional.dropout3d(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                        self.assert_unchanged_identity(output, source, before)

    def test_rank_four_identity_modes_use_optional_batch_views(self):
        calls = (
            {"p": 0.75, "training": False},
            {"p": 1, "training": False},
            {"p": 0, "training": True},
            {"p": torch.tensor(0.0), "training": True},
        )
        sources = tuple(source[0] for source in self.make_probability_one_cases())
        for case, source in enumerate(sources):
            for kwargs in calls:
                before = self.snapshot(source)
                with self.subTest(case=case, kwargs=kwargs):
                    output = functional.dropout3d(source, **kwargs)
                    self.assert_optional_batch_identity(output, source, before)

        for case, source in enumerate(sources):
            for kwargs in calls:
                before = self.snapshot(source)
                with self.subTest(case=case, kwargs=kwargs, inplace=True):
                    output = functional.dropout3d(source, inplace=True, **kwargs)
                    self.assert_unchanged_identity(output, source, before)

    def test_empty_rank_four_inputs_use_optional_batch_views(self):
        for requires_grad in (False, True):
            for case, source in enumerate(
                self.make_rank_four_empty_cases(requires_grad=requires_grad)
            ):
                for probability in (
                    0.25,
                    1.0,
                    np.float32(0.5),
                    torch.tensor(0.5),
                ):
                    before = self.snapshot(source)
                    with self.subTest(
                        requires_grad=requires_grad,
                        case=case,
                        probability=type(probability),
                    ):
                        output = functional.dropout3d(
                            source,
                            p=probability,
                            training=True,
                            inplace=False,
                        )
                        self.assert_optional_batch_identity(
                            output, source, before
                        )
                        self.assertEqual(output.requires_grad, requires_grad)
                        self.assertEqual(output.is_leaf, not requires_grad)

        for case, source in enumerate(self.make_rank_four_empty_cases()):
            for probability in (0.25, 1.0):
                before = self.snapshot(source)
                with self.subTest(case=case, probability=probability, inplace=True):
                    output = functional.dropout3d(
                        source,
                        p=probability,
                        training=True,
                        inplace=True,
                    )
                    self.assert_unchanged_identity(output, source, before)

    def test_training_probability_one_returns_a_new_signed_zero_product(self):
        for case, source in enumerate(self.make_probability_one_cases()):
            for probability in (
                1.0,
                1,
                True,
                np.float32(1.0),
                torch.tensor(1.0),
            ):
                before = self.snapshot(source)
                output = functional.dropout3d(
                    source,
                    p=probability,
                    training=True,
                    inplace=False,
                )
                with self.subTest(case=case, probability=type(probability)):
                    self.assertIsNot(output, source)
                    self.assertFalse(output.is_set_to(source))
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.shape, source.shape)
                    self.assertEqual(output.stride(), source.stride())
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertIs(output.dtype, source.dtype)
                    self.assertEqual(output.device, source.device)
                    self.assertFalse(output.requires_grad)
                    self.assertTrue(output.is_leaf)
                    self.assertEqual(output.output_nr, 0)
                    source_values = np.asarray(source).reshape(-1)
                    output_values = np.asarray(output).reshape(-1)
                    finite = np.isfinite(source_values)
                    np.testing.assert_array_equal(
                        output_values[finite].view(np.uint32),
                        np.where(
                            np.signbit(source_values[finite]),
                            np.uint32(0x80000000),
                            np.uint32(0x00000000),
                        ),
                    )
                    self.assertTrue(np.isnan(output_values[~finite]).all())
                    self.assertEqual(self.snapshot(source)[:-1], before[:-1])
                    np.testing.assert_array_equal(
                        self.snapshot(source)[-1], before[-1]
                    )

    def test_rank_four_training_probability_one_uses_optional_batch_views(self):
        for case, source in enumerate(self.make_rank_four_probability_one_cases()):
            for probability in (
                1.0,
                1,
                True,
                np.float32(1.0),
                torch.tensor(1.0),
            ):
                before = self.snapshot(source)
                output = functional.dropout3d(
                    source,
                    p=probability,
                    training=True,
                    inplace=False,
                )
                with self.subTest(case=case, probability=type(probability)):
                    self.assertIsNot(output, source)
                    self.assertFalse(output.is_set_to(source))
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.shape, source.shape)
                    self.assertEqual(output.stride(), source.stride())
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertIs(output.dtype, source.dtype)
                    self.assertEqual(output.device, source.device)
                    self.assertFalse(output.requires_grad)
                    self.assertTrue(output.is_leaf)
                    self.assertEqual(output.output_nr, 0)
                    source_values = np.asarray(source).reshape(-1)
                    output_values = np.asarray(output).reshape(-1)
                    finite = np.isfinite(source_values)
                    np.testing.assert_array_equal(
                        output_values[finite].view(np.uint32),
                        np.where(
                            np.signbit(source_values[finite]),
                            np.uint32(0x80000000),
                            np.uint32(0x00000000),
                        ),
                    )
                    self.assertTrue(np.isnan(output_values[~finite]).all())
                    self.assertEqual(self.snapshot(source)[:-1], before[:-1])
                    np.testing.assert_array_equal(
                        self.snapshot(source)[-1], before[-1]
                    )

    def test_training_probability_one_autograd_and_no_grad(self):
        leaf = torch.tensor(
            [[[[[-1.0, 2.0], [-0.0, 3.0]]]]], requires_grad=True
        )
        source = leaf.transpose(3, 4)
        before = np.asarray(source.detach()).copy().view(np.uint32)
        output = functional.dropout3d(
            source,
            p=torch.tensor(1.0),
            training=True,
            inplace=False,
        )
        self.assertIsNot(output, source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, 0)
        np.testing.assert_array_equal(
            np.asarray(output.detach()).reshape(-1).view(np.uint32),
            [0x80000000, 0x80000000, 0x00000000, 0x00000000],
        )
        weights = torch.tensor([[[[[2.0, -3.0], [-5.0, 7.0]]]]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[[[[0.0, 0.0], [0.0, 0.0]]]]])
        np.testing.assert_array_equal(
            np.asarray(source.detach()).view(np.uint32), before
        )

        no_grad_leaf = torch.tensor(
            [[[[[-1.0, 2.0], [-0.0, 3.0]]]]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(3, 4)
        with torch.no_grad():
            untracked = functional.dropout3d(
                no_grad_source, p=1, training=True
            )
        self.assertIsNot(untracked, no_grad_source)
        self.assertFalse(untracked.is_set_to(no_grad_source))
        self.assertEqual(untracked.stride(), no_grad_source.stride())
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_rank_four_autograd_and_no_grad_follow_optional_batch_views(self):
        identity_leaf = torch.tensor(
            [[[[-1.0, 2.0], [-0.0, 3.0]]]], requires_grad=True
        )
        identity_source = identity_leaf.transpose(2, 3)
        identity_output = functional.dropout3d(
            identity_source,
            p=0,
            training=True,
            inplace=False,
        )
        self.assertIsNot(identity_output, identity_source)
        self.assertTrue(identity_output.is_set_to(identity_source))
        self.assertTrue(identity_output.requires_grad)
        self.assertFalse(identity_output.is_leaf)
        identity_weights = torch.tensor(
            [[[[2.0, -3.0], [-5.0, 7.0]]]]
        ).transpose(2, 3)
        (identity_output * identity_weights).sum().backward()
        self.assertEqual(
            identity_leaf.grad.tolist(),
            [[[[2.0, -3.0], [-5.0, 7.0]]]],
        )

        zero_leaf = torch.tensor(
            [[[[-1.0, 2.0], [-0.0, 3.0]]]], requires_grad=True
        )
        zero_source = zero_leaf.transpose(2, 3)
        zero_output = functional.dropout3d(
            zero_source,
            p=1,
            training=True,
            inplace=False,
        )
        self.assertIsNot(zero_output, zero_source)
        self.assertFalse(zero_output.is_set_to(zero_source))
        self.assertTrue(zero_output.requires_grad)
        self.assertFalse(zero_output.is_leaf)
        (zero_output * identity_weights).sum().backward()
        self.assertEqual(
            zero_leaf.grad.tolist(),
            [[[[0.0, 0.0], [0.0, 0.0]]]],
        )

        no_grad_leaf = torch.tensor(
            [[[[-1.0, 2.0], [-0.0, 3.0]]]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(2, 3)
        with torch.no_grad():
            identity = functional.dropout3d(
                no_grad_source, p=0.75, training=False
            )
            zeros = functional.dropout3d(
                no_grad_source, p=1, training=True
            )
        self.assertIsNot(identity, no_grad_source)
        self.assertTrue(identity.is_set_to(no_grad_source))
        self.assertTrue(identity.requires_grad)
        self.assertTrue(identity.is_leaf)
        self.assertIsNot(zeros, no_grad_source)
        self.assertFalse(zeros.is_set_to(no_grad_source))
        self.assertFalse(zeros.requires_grad)
        self.assertTrue(zeros.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_identity_preserves_autograd_and_no_grad_state(self):
        leaf = torch.tensor(
            [[[[[1.0, 2.0], [3.0, 4.0]]]]], requires_grad=True
        )
        source = leaf.transpose(3, 4)
        output = functional.dropout3d(source, p=0, training=True, inplace=True)
        self.assertIs(output, source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.output_nr, source.output_nr)

        weights = torch.tensor([[[[[2.0, 3.0], [5.0, 7.0]]]]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[[[[2.0, 5.0], [3.0, 7.0]]]]])

        untracked_leaf = torch.zeros((1, 2, 3, 4, 5), requires_grad=True)
        untracked_source = untracked_leaf.transpose(3, 4)
        with torch.no_grad():
            unchanged = functional.dropout3d(
                untracked_source, p=0.75, training=False
            )
        self.assertIs(unchanged, untracked_source)
        self.assertTrue(unchanged.requires_grad)
        self.assertFalse(unchanged.is_leaf)
        self.assertIsNone(untracked_leaf.grad)

    def test_probability_validation_and_feature_dropout_schema(self):
        source = torch.zeros((1, 2, 3, 4, 5))

        for probability in (-0.1, 1.1, float("inf"), -float("inf")):
            with self.subTest(probability=probability):
                with self.assertRaises(ValueError) as raised:
                    functional.dropout3d(
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
                    functional.dropout3d(source, p=probability, training=False)

        for inplace in (False, True):
            with self.subTest(none_input=True, inplace=inplace):
                with self.assertRaisesRegex(
                    AttributeError, "^'NoneType' object has no attribute 'dim'$"
                ):
                    functional.dropout3d(None, p=0, inplace=inplace)

        error_cases = (
            (
                lambda: functional.dropout3d(source, p=Decimal("0")),
                "feature_dropout(): argument 'p' (position 2) must be float, "
                "not decimal.Decimal",
            ),
            (
                lambda: functional.dropout3d(
                    source, p=Fraction(0, 1), inplace=True
                ),
                "feature_dropout_(): argument 'p' (position 2) must be float, "
                "not Fraction",
            ),
            (
                lambda: functional.dropout3d(source, p=0, training=1),
                "feature_dropout(): argument 'train' (position 3) must be "
                "bool, not int",
            ),
            (
                lambda: functional.dropout3d(
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
                        functional.dropout3d(
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
                    functional.dropout3d(source, p=probability)
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
                        functional.dropout3d(
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
                    functional.dropout3d(
                        source, p=0.5, training=False, inplace=inplace
                    ),
                    source,
                )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = functional.dropout3d(
                source, p=np.complex64(0), training=True
            )
        self.assertIs(output, source)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, np.exceptions.ComplexWarning)

    def test_overrides_and_modes_receive_the_public_function(self):
        replacement = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return replacement

        input = Override()
        output = functional.dropout3d(
            input, p=-1, training="invalid", inplace=True
        )
        self.assertIs(output, replacement)
        func, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(func, functional.dropout3d)
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

        source = torch.tensor([[[[[-1.0, 2.0]]]]])
        mode = RecordingMode()
        with mode:
            self.assertEqual(
                functional.dropout3d(
                    source, p=-1, training="invalid", inplace=True
                ),
                "mode-result",
            )
        self.assertIs(mode.calls[0][0], functional.dropout3d)
        self.assertEqual(mode.calls[0][1], (torch.Tensor,))

        mode = RecordingMode(forward=True)
        with mode:
            output = functional.dropout3d(
                source, p=0, training=True, inplace=True
            )
        self.assertIs(output, source)
        self.assertEqual(len(mode.calls), 1)

        probability_one_mode = RecordingMode(forward=True)
        with probability_one_mode:
            output = functional.dropout3d(
                source, p=1, training=True, inplace=False
            )
        self.assertIsNot(output, source)
        np.testing.assert_array_equal(
            np.asarray(output).view(np.uint32),
            [[[[[0x80000000, 0x00000000]]]]],
        )
        self.assertEqual(len(probability_one_mode.calls), 1)

        unbatched = source[0]
        unbatched_mode = RecordingMode(forward=True)
        with unbatched_mode:
            output = functional.dropout3d(
                unbatched, p=0, training=True, inplace=False
            )
        self.assertIsNot(output, unbatched)
        self.assertTrue(output.is_set_to(unbatched))
        self.assertEqual(len(unbatched_mode.calls), 1)

        unbatched_probability_one_mode = RecordingMode(forward=True)
        with unbatched_probability_one_mode:
            output = functional.dropout3d(
                unbatched, p=1, training=True, inplace=False
            )
        self.assertIsNot(output, unbatched)
        self.assertFalse(output.is_set_to(unbatched))
        self.assertEqual(len(unbatched_probability_one_mode.calls), 1)

    def test_sampling_and_other_ranks_are_explicitly_unsupported(self):
        sources = (
            torch.tensor(
                [[[[[1.0, 2.0], [3.0, 4.0]]]]], requires_grad=True
            ),
            torch.tensor(
                [[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True
            ),
        )
        unsupported_calls = (
            (0.25, False),
            (0.25, True),
            (torch.tensor(0.25), False),
            (torch.tensor(0.25), True),
            (1.0, True),
            (torch.tensor(1.0), True),
        )
        for source in sources:
            before = self.snapshot(source)
            for probability, inplace in unsupported_calls:
                with self.subTest(
                    rank=len(source.shape),
                    sampling=probability,
                    inplace=inplace,
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        "^torch_rs.nn.functional.dropout3d does not support "
                        "sampling$",
                    ):
                        functional.dropout3d(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                    self.assertEqual(self.snapshot(source)[:-1], before[:-1])
                    np.testing.assert_array_equal(
                        self.snapshot(source)[-1], before[-1]
                    )
                    self.assertIsNone(source.grad)

        unsupported_ranks = (
            torch.tensor(-0.0),
            torch.zeros((2,)),
            torch.zeros((2, 3)),
            torch.zeros((2, 3, 4)),
            torch.zeros((2, 3, 4, 5, 6, 7)),
            torch.zeros((2, 0, 3)),
        )
        identity_modes = ((0.5, False), (0.0, True), (0.5, True), (1.0, True))
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
                                "^torch_rs.nn.functional.dropout3d only "
                                "supports rank-4 or rank-5 inputs$",
                            ):
                                functional.dropout3d(
                                    source,
                                    p=probability,
                                    training=training,
                                    inplace=inplace,
                                )
                        self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
