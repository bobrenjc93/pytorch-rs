import importlib
import inspect
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


FUNCTION_DOC = """During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

    Uses samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = """During training, randomly zeroes some elements of the input tensor with probability :attr:`p`.

Uses samples from a Bernoulli distribution.

See :class:`~torch.nn.Dropout` for details.

Args:
    p: probability of an element to be zeroed. Default: 0.5
    training: apply dropout if is ``True``. Default: ``True``
    inplace: If set to ``True``, will do this operation in-place. Default: ``False``
"""


class FunctionalDropoutTests(unittest.TestCase):
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

    def test_imports_signature_documentation_and_deliberate_surface(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import dropout

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(dropout, functional.dropout)
        self.assertNotIn("nn", torch.__all__)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))
        wildcard_namespace = {}
        exec(
            "from torch_rs.nn.functional import *", wildcard_namespace
        )
        self.assertNotIn("sys", wildcard_namespace)
        self.assertNotIn("types", wildcard_namespace)
        self.assertIsNone(nn.__doc__)
        self.assertEqual(functional.__doc__, "Functional interface.")
        self.assertIsNot(torch.dropout, functional.dropout)
        self.assertEqual(torch.__all__.count("dropout"), 1)
        self.assertFalse(hasattr(torch, "_nn_functional_dropout"))
        self.assertFalse(
            hasattr(
                torch, "_nn_functional_dropout_tensor_autograd_suffix"
            )
        )

        function = functional.dropout
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "dropout")
        self.assertEqual(function.__qualname__, "dropout")
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

    def test_evaluation_and_zero_probability_return_the_exact_input(self):
        evaluation_probabilities = (
            0.0,
            0.25,
            1,
            True,
            np.float32(0.75),
            torch.tensor(0.75),
        )
        zero_probabilities = (
            0,
            0.0,
            -0.0,
            False,
            np.float32(0.0),
            torch.tensor(0.0),
        )

        for case, source in enumerate(self.make_cases(requires_grad=True)):
            for probability in evaluation_probabilities:
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=probability,
                        training=False,
                        inplace=inplace,
                    ):
                        output = functional.dropout(
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
                        output = functional.dropout(
                            source,
                            probability,
                            True,
                            inplace,
                        )
                        self.assert_unchanged_identity(output, source, before)

    def test_identity_does_not_add_or_remove_autograd_state(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        output = functional.dropout(source, p=0, training=True, inplace=True)
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
            unchanged = functional.dropout(
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
            for probability in (0.25, 1.0, torch.tensor(0.25)):
                for inplace in (False, True):
                    before = self.snapshot(source)
                    with self.subTest(
                        case=case,
                        probability=probability,
                        inplace=inplace,
                    ):
                        output = functional.dropout(
                            source,
                            p=probability,
                            training=True,
                            inplace=inplace,
                        )
                        self.assert_unchanged_identity(
                            output, source, before
                        )

    def test_training_probability_one_returns_a_new_signed_zero_product(self):
        cases = self.make_cases(requires_grad=False)
        sources_and_bits = (
            (cases[0], [0x80000000]),
            (
                cases[2],
                [0x80000000, 0x00000000, 0x80000000, 0x00000000],
            ),
            (
                cases[3],
                [0x80000000, 0x80000000, 0x00000000, 0x00000000],
            ),
        )

        for case, (source, expected_bits) in enumerate(sources_and_bits):
            for probability in (
                1.0,
                True,
                np.float32(1.0),
                torch.tensor(1.0),
            ):
                before = self.snapshot(source)
                output = functional.dropout(
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
                    np.testing.assert_array_equal(
                        np.asarray(output).reshape(-1).view(np.uint32),
                        expected_bits,
                    )
                    self.assertEqual(self.snapshot(source)[:-1], before[:-1])
                    np.testing.assert_array_equal(
                        self.snapshot(source)[-1], before[-1]
                    )

    def test_training_probability_one_autograd_and_no_grad(self):
        leaf = torch.tensor(
            [[-1.0, 2.0], [-0.0, 3.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        before = np.asarray(source.detach()).copy().view(np.uint32)
        output = functional.dropout(
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
        weights = torch.tensor([[2.0, -3.0], [-5.0, 7.0]])
        (output * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(
            np.asarray(source.detach()).view(np.uint32), before
        )

        no_grad_leaf = torch.tensor(
            [[-1.0, 2.0], [-0.0, 3.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            untracked = functional.dropout(
                no_grad_source, p=1, training=True
            )
        self.assertIsNot(untracked, no_grad_source)
        self.assertFalse(untracked.is_set_to(no_grad_source))
        self.assertEqual(untracked.stride(), no_grad_source.stride())
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_probability_validation_and_native_schema_errors(self):
        source = torch.tensor([1.0, 2.0])

        for probability in (-0.1, 1.1, float("inf"), -float("inf")):
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    ValueError,
                    "^dropout probability has to be between 0 and 1, but got ",
                ):
                    functional.dropout(
                        None,
                        p=probability,
                        training="invalid",
                        inplace=True,
                    )

        for probability in (float("nan"), np.float32(np.nan)):
            with self.subTest(probability=type(probability)):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^dropout probability has to be between 0 and 1, but got nan$",
                ):
                    functional.dropout(source, p=probability, training=False)

        for value in (-0.1, 1.1, float("inf")):
            probability = torch.tensor(value)
            with self.subTest(tensor_probability=value):
                with self.assertRaises(ValueError) as raised:
                    functional.dropout(
                        None, p=probability, training=False
                    )
                self.assertEqual(
                    str(raised.exception),
                    "dropout probability has to be between 0 and 1, but got "
                    f"{probability.item()}",
                )

        with self.assertRaisesRegex(
            RuntimeError,
            "^dropout probability has to be between 0 and 1, but got nan$",
        ):
            functional.dropout(
                source, p=torch.tensor(float("nan")), training=False
            )

        leaf_probability = torch.tensor([2.0], requires_grad=True)
        copy_probability = torch.tensor([[[[2.0]]]], requires_grad=True)
        tensor_error_cases = (
            (
                torch.tensor([2.0]),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([2.])",
            ),
            (
                leaf_probability,
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([2.], requires_grad=True)",
            ),
            (
                leaf_probability * 2,
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([4.], grad_fn=<MulBackward0>)",
            ),
            (
                leaf_probability + 1,
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([3.], grad_fn=<AddBackward0>)",
            ),
            (
                leaf_probability.reshape(1, 1),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([[2.]], grad_fn=<ViewBackward0>)",
            ),
            (
                torch.tensor([-2.0], requires_grad=True).ravel(),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([-2.], grad_fn=<ViewBackward0>)",
            ),
            (
                3 - torch.tensor([1.0], requires_grad=True),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([2.], grad_fn=<RsubBackward1>)",
            ),
            (
                copy_probability.cpu(memory_format=torch.channels_last),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([[[[2.]]]], grad_fn=<ToCopyBackward0>)",
            ),
            (
                copy_probability.float(memory_format=torch.channels_last),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([[[[2.]]]], grad_fn=<ToCopyBackward0>)",
            ),
            (
                torch.tensor([-0.1]),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([-0.1000])",
            ),
            (
                torch.tensor([[1.0e9]]),
                ValueError,
                "dropout probability has to be between 0 and 1, but got "
                "tensor([[1.0000e+09]])",
            ),
            (
                torch.zeros((0,)),
                RuntimeError,
                "Boolean value of Tensor with no values is ambiguous",
            ),
            (
                torch.tensor([0.0, 2.0]),
                RuntimeError,
                "Boolean value of Tensor with more than one value is ambiguous",
            ),
            (
                torch.tensor([0.5]),
                TypeError,
                "dropout(): argument 'p' (position 2) must be float, not Tensor",
            ),
            (
                torch.tensor([0.5]).reshape((1,) * 1000),
                TypeError,
                "dropout(): argument 'p' (position 2) must be float, not Tensor",
            ),
            (
                torch.tensor([float("nan")]),
                TypeError,
                "dropout(): argument 'p' (position 2) must be float, not Tensor",
            ),
        )
        for probability, error_type, message in tensor_error_cases:
            with self.subTest(tensor_shape=probability.shape, message=message):
                with self.assertRaises(error_type) as raised:
                    functional.dropout(source, p=probability, training=False)
                self.assertEqual(str(raised.exception), message)

        for inplace, operation in ((False, "dropout"), (True, "dropout_")):
            probability = torch.tensor(0.5, requires_grad=True)
            with self.subTest(grad_probability_inplace=inplace):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{operation}\(\): argument 'p' \(position 2\) "
                    "must be float, not Tensor$",
                ):
                    functional.dropout(
                        source,
                        p=probability,
                        training=False,
                        inplace=inplace,
                    )

        class SneakyProbability(float):
            def __lt__(self, other):
                return False

            def __gt__(self, other):
                return False

        with self.assertRaisesRegex(
            RuntimeError,
            "^dropout probability has to be between 0 and 1, but got 2$",
        ):
            functional.dropout(
                source, p=SneakyProbability(2), training=False
            )

        native_format_cases = (
            (-float("nan"), "-nan"),
            (SneakyProbability(1.0000000000001), "1"),
            (SneakyProbability(1.23456789), "1.23457"),
            (SneakyProbability(999999.9), "1e+06"),
            (SneakyProbability(-1.23456789e-7), "-1.23457e-07"),
            (SneakyProbability(-5e-324), "-4.94066e-324"),
        )
        for probability, formatted in native_format_cases:
            with self.subTest(native_probability=formatted):
                with self.assertRaises(RuntimeError) as raised:
                    functional.dropout(
                        source, p=probability, training=False
                    )
                self.assertEqual(
                    str(raised.exception),
                    "dropout probability has to be between 0 and 1, but got "
                    f"{formatted}",
                )

        error_cases = (
            (
                lambda: functional.dropout(None, p=0),
                TypeError,
                r"dropout\(\): argument 'input' \(position 1\) must be Tensor, not NoneType",
            ),
            (
                lambda: functional.dropout(None, p=0, inplace=True),
                TypeError,
                r"dropout_\(\): argument 'input' \(position 1\) must be Tensor, not NoneType",
            ),
            (
                lambda: functional.dropout(source, p=None, training=False),
                TypeError,
                "'<' not supported between instances of 'NoneType' and 'float'",
            ),
            (
                lambda: functional.dropout(source, p=0, training=1),
                TypeError,
                r"dropout\(\): argument 'train' \(position 3\) must be bool, not int",
            ),
            (
                lambda: functional.dropout(
                    source, p=0, training=np.bool_(False)
                ),
                TypeError,
                r"dropout\(\): argument 'train' \(position 3\) must be bool, not numpy.bool",
            ),
        )
        for case, (call, error_type, message) in enumerate(error_cases):
            with self.subTest(case=case):
                with self.assertRaisesRegex(error_type, f"^{message}$"):
                    call()

        for inplace in (0, 1, None, "", "inplace", [], [1]):
            with self.subTest(inplace=inplace):
                self.assertIs(
                    functional.dropout(
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
            functional.dropout(
                source, p=0, training=True, inplace=BoolFailure()
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            with self.assertRaisesRegex(
                np.exceptions.ComplexWarning,
                "^Casting complex values to real discards the imaginary part$",
            ):
                functional.dropout(
                    source, p=np.complex64(0), training=False
                )

        class MemoryFailure(int):
            def __float__(self):
                raise MemoryError("probability conversion failed")

            def __lt__(self, other):
                return False

            def __gt__(self, other):
                return False

        with self.assertRaisesRegex(
            MemoryError, "^probability conversion failed$"
        ):
            functional.dropout(
                source, p=MemoryFailure(0), training=False
            )

    def test_tensor_probability_formatting_obeys_recursion_limit(self):
        probability = torch.tensor([-2.0]).reshape((1,) * 72)
        previous_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(80)
            with self.assertRaises(RecursionError) as raised:
                functional.dropout(
                    torch.tensor([0.0]),
                    p=probability,
                    training=False,
                )
            expected_message = (
                "maximum recursion depth exceeded while calling a Python object"
                if sys.version_info < (3, 12)
                else "maximum recursion depth exceeded"
            )
            self.assertEqual(str(raised.exception), expected_message)
        finally:
            sys.setrecursionlimit(previous_limit)

    def test_overrides_observe_the_public_function_before_validation(self):
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
        output = functional.dropout(
            input, p=-1, training="invalid", inplace=inplace
        )
        self.assertIs(output, replacement)
        self.assertEqual(len(Override.calls), 1)
        func, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(func, functional.dropout)
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
                functional.dropout(PlainOverride(), training=False),
                replacement,
            )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        self.assertEqual(
            str(caught[0].message),
            "Defining your `__torch_function__ as a plain method is "
            "deprecated and will be an error in future, please define it "
            "as a classmethod.",
        )

        class DecliningOverride:
            @classmethod
            def __torch_function__(
                cls, func, types, args=(), kwargs=None
            ):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            "^no implementation found for 'torch_rs.nn.functional.dropout' "
            "on types that implement __torch_function__:",
        ):
            functional.dropout(DecliningOverride(), p=0)

        with self.assertRaisesRegex(
            TypeError,
            "^no implementation found for 'torch_rs.nn.functional.dropout' "
            r"on types that implement __torch_function__: \[\]$",
        ):
            functional.dropout(Override, p=0)

        class BrokenProbe:
            def __getattribute__(self, name):
                if name == "__torch_function__":
                    raise RuntimeError("probe failed")
                return object.__getattribute__(self, name)

        with self.assertRaisesRegex(
            ValueError,
            "^dropout probability has to be between 0 and 1, but got -1$",
        ):
            functional.dropout(BrokenProbe(), p=-1)

        class StatefulDescriptor:
            def __init__(self):
                self.type_accesses = 0

            def __get__(self, instance, owner):
                if instance is None:
                    self.type_accesses += 1
                    if self.type_accesses == 2:
                        raise RuntimeError("second type lookup")

                def override(func, types, args=(), kwargs=None):
                    return replacement

                return override

        descriptor = StatefulDescriptor()

        class StatefulOverride:
            __torch_function__ = descriptor

        with self.assertRaisesRegex(
            RuntimeError, "^second type lookup$"
        ):
            functional.dropout(StatefulOverride(), training=False)
        self.assertEqual(descriptor.type_accesses, 2)

    def test_torch_function_modes_observe_forward_and_decline(self):
        source = torch.tensor([1.0, -2.0])
        replacement = object()

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

        inplace = object()
        recording = RecordingMode()
        with recording:
            output = functional.dropout(
                source, p=-1, training="invalid", inplace=inplace
            )
        self.assertIs(output, replacement)
        self.assertEqual(len(recording.calls), 1)
        func, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(func, functional.dropout)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (source,))
        self.assertEqual(
            kwargs,
            {"p": -1, "training": "invalid", "inplace": inplace},
        )

        forwarding = RecordingMode(forward=True)
        with forwarding:
            forwarded = functional.dropout(
                source, p=1, training=True, inplace=False
            )
        self.assertIsNot(forwarded, source)
        np.testing.assert_array_equal(
            np.asarray(forwarded).view(np.uint32),
            [0x00000000, 0x80000000],
        )
        self.assertEqual(len(forwarding.calls), 1)

        declining = RecordingMode(result=NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            "^no implementation found for 'torch_rs.nn.functional.dropout' "
            r"on types that implement __torch_function__: \[\] nor in mode ",
        ):
            with declining:
                functional.dropout(source, p=0)
        self.assertEqual(
            tuple(call[1] for call in declining.calls),
            ((torch.Tensor,), ()),
        )

    def test_sampling_and_inplace_paths_are_unsupported_and_non_mutating(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]

        cases = (
            (0.25, False),
            (0.25, True),
            (torch.tensor(0.25), False),
            (torch.tensor(0.25), True),
            (1.0, True),
            (torch.tensor(1.0), True),
        )

        for probability, inplace in cases:
            before = self.snapshot(source)
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
                after = self.snapshot(source)
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])
                self.assertIsNone(leaf.grad)


if __name__ == "__main__":
    unittest.main()
