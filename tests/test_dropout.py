import inspect
import pickle
import re
import types
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch


class DropoutTests(unittest.TestCase):
    def assert_error(self, exception_type, message, call):
        with self.assertRaises(exception_type) as raised:
            call()
        self.assertEqual(str(raised.exception), message)

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

    def make_sources(self, *, requires_grad):
        scalar = torch.tensor(-0.0, requires_grad=requires_grad)
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=requires_grad)
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

    def test_evaluation_zero_probability_and_empty_inputs_are_exact_identities(self):
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

        for case, source in enumerate(self.make_sources(requires_grad=True)):
            for probability in evaluation_probabilities:
                before = self.snapshot(source)
                with self.subTest(case=case, probability=probability, train=False):
                    self.assert_unchanged_identity(
                        torch.dropout(source, probability, False), source, before
                    )

            for probability in zero_probabilities:
                before = self.snapshot(source)
                with self.subTest(case=case, probability=probability, train=True):
                    self.assert_unchanged_identity(
                        torch.dropout(source, probability, True), source, before
                    )

        empty = self.make_sources(requires_grad=True)[1]
        for probability in (0.25, 1.0, torch.tensor(0.25)):
            before = self.snapshot(empty)
            with self.subTest(empty_probability=probability):
                self.assert_unchanged_identity(
                    torch.dropout(empty, probability, True), empty, before
                )

    def test_probability_one_returns_fresh_stride_preserving_signed_zeros(self):
        sources = self.make_sources(requires_grad=False)
        sources_and_bits = (
            (sources[0], [0x80000000]),
            (sources[2], [0x80000000, 0x00000000, 0x80000000, 0x00000000]),
            (sources[3], [0x80000000, 0x80000000, 0x00000000, 0x00000000]),
        )

        for case, (source, expected_bits) in enumerate(sources_and_bits):
            for probability in (1, 1.0, True, np.float32(1.0), torch.tensor(1.0)):
                before = self.snapshot(source)
                output = torch.dropout(source, probability, True)
                with self.subTest(case=case, probability=type(probability)):
                    self.assertIsNot(output, source)
                    self.assertFalse(output.is_set_to(source))
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                    self.assertEqual(output.shape, source.shape)
                    self.assertEqual(output.stride(), source.stride())
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertIs(output.dtype, source.dtype)
                    self.assertEqual(output.device, source.device)
                    np.testing.assert_array_equal(
                        np.asarray(output).reshape(-1).view(np.uint32),
                        expected_bits,
                    )
                    after = self.snapshot(source)
                    self.assertEqual(after[:-1], before[:-1])
                    np.testing.assert_array_equal(after[-1], before[-1])

    def test_autograd_and_no_grad_match_the_native_deterministic_paths(self):
        identity_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        identity_source = identity_leaf.transpose(0, 1)
        identity = torch.dropout(identity_source, 0, True)
        self.assertIs(identity, identity_source)
        (identity * torch.tensor([[2.0, 3.0], [5.0, 7.0]])).sum().backward()
        self.assertEqual(identity_leaf.grad.tolist(), [[2.0, 5.0], [3.0, 7.0]])

        zero_leaf = torch.tensor(
            [[-1.0, 2.0], [-0.0, 3.0]], requires_grad=True
        )
        zero_source = zero_leaf.transpose(0, 1)
        zero = torch.dropout(zero_source, torch.tensor(1.0), True)
        self.assertIsNot(zero, zero_source)
        self.assertTrue(zero.requires_grad)
        self.assertFalse(zero.is_leaf)
        self.assertEqual(zero.stride(), zero_source.stride())
        np.testing.assert_array_equal(
            np.asarray(zero.detach()).reshape(-1).view(np.uint32),
            [0x80000000, 0x80000000, 0x00000000, 0x00000000],
        )
        (zero * torch.tensor([[2.0, -3.0], [-5.0, 7.0]])).sum().backward()
        self.assertEqual(zero_leaf.grad.tolist(), [[0.0, 0.0], [0.0, 0.0]])

        no_grad_leaf = torch.tensor([-1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            evaluation = torch.dropout(no_grad_leaf, 0.75, False)
            untracked_zero = torch.dropout(no_grad_leaf, 1, True)
        self.assertIs(evaluation, no_grad_leaf)
        self.assertTrue(evaluation.requires_grad)
        self.assertIsNot(untracked_zero, no_grad_leaf)
        self.assertFalse(untracked_zero.requires_grad)
        self.assertTrue(untracked_zero.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_binding_probability_validation_and_legacy_input_aliases(self):
        source = torch.tensor([1.0])
        for call in (
            lambda: torch.dropout(input=source, p=0, train=True),
            lambda: torch.dropout(x=source, p=0, train=True),
            lambda: torch.dropout(a=source, p=0, train=True),
            lambda: torch.dropout(x1=source, p=0, train=True),
            lambda: torch.dropout(p=0, train=True, x=source),
        ):
            self.assertIs(call(), source)

        cases = (
            (
                lambda: torch.dropout(),
                'dropout() missing 3 required positional argument: "input", "p", "train"',
            ),
            (
                lambda: torch.dropout(source),
                'dropout() missing 2 required positional argument: "p", "train"',
            ),
            (
                lambda: torch.dropout(source, 0),
                'dropout() missing 1 required positional arguments: "train"',
            ),
            (
                lambda: torch.dropout(source, 0, False, None),
                "dropout() takes 3 positional arguments but 4 were given",
            ),
            (
                lambda: torch.dropout(None, 0, False),
                "dropout(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.dropout(input=None, p=0, train=False),
                "dropout(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.dropout(source, None, False),
                "dropout(): argument 'p' (position 2) must be float, not NoneType",
            ),
            (
                lambda: torch.dropout(input=source, p=Decimal("0"), train=False),
                "dropout(): argument 'p' must be float, not decimal.Decimal",
            ),
            (
                lambda: torch.dropout(source, 0, 1),
                "dropout(): argument 'train' (position 3) must be bool, not int",
            ),
            (
                lambda: torch.dropout(input=source, p=0, train=np.bool_(False)),
                "dropout(): argument 'train' must be bool, not numpy.bool",
            ),
            (
                lambda: torch.dropout(source, 0, False, input=source),
                "dropout() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.dropout(source, 0, False, p=0),
                "dropout() got multiple values for argument 'p'",
            ),
            (
                lambda: torch.dropout(source, 0, False, extra=True),
                "dropout() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.dropout(source, 0, False, x=source),
                "dropout() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                self.assert_error(TypeError, message, call)

        for probability, formatted in (
            (-0.1, "-0.1"),
            (1.1, "1.1"),
            (float("inf"), "inf"),
            (float("nan"), "nan"),
            (-float("nan"), "-nan"),
        ):
            with self.subTest(probability=probability):
                self.assert_error(
                    RuntimeError,
                    "dropout probability has to be between 0 and 1, but got "
                    f"{formatted}",
                    lambda probability=probability: torch.dropout(
                        source, probability, False
                    ),
                )

        for probability in (torch.tensor([0.0]), torch.tensor(0.5, requires_grad=True)):
            with self.subTest(probability=probability):
                self.assert_error(
                    TypeError,
                    "dropout(): argument 'p' (position 2) must be float, not Tensor",
                    lambda probability=probability: torch.dropout(
                        source, probability, False
                    ),
                )

    def test_torch_function_mode_dispatches_before_range_and_sampling_checks(self):
        source = torch.tensor([1.0, -2.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker, forward=False):
                self.result = result
                self.forward = forward
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                if self.forward:
                    return func(*args, **(kwargs or {}))
                return self.result

        positional = RecordingMode()
        with positional:
            self.assertIs(torch.dropout(source, -1, False), marker)
        self.assertEqual(len(positional.calls), 1)
        function, dispatch_types, args, kwargs = positional.calls[0]
        self.assertIs(function, torch.dropout)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, (source, -1, False))
        self.assertIsNone(kwargs)

        keyword = RecordingMode()
        with keyword:
            self.assertIs(
                torch.dropout(input=source, p=0.25, train=True), marker
            )
        function, dispatch_types, args, kwargs = keyword.calls[0]
        self.assertIs(function, torch.dropout)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": source, "p": 0.25, "train": True})

        invalid = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^dropout\(\): argument 'p' \(position 2\) must be float, not NoneType$",
        ):
            with invalid:
                torch.dropout(source, None, False)
        self.assertEqual(invalid.calls, [])

        forwarding = RecordingMode(forward=True)
        with forwarding:
            output = torch.dropout(source, 1, True)
        self.assertIsNot(output, source)
        np.testing.assert_array_equal(
            np.asarray(output).view(np.uint32), [0x00000000, 0x80000000]
        )
        self.assertEqual(len(forwarding.calls), 1)

        declining = RecordingMode(result=NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            "^Multiple dispatch failed for 'torch.dropout'; all "
            "__torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                torch.dropout(source, 0, False)
        self.assertEqual(len(declining.calls), 1)

    def test_stochastic_calls_are_rejected_without_mutating_the_input(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        for probability in (0.25, torch.tensor(0.25)):
            before = self.snapshot(source)
            with self.subTest(probability=probability):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.dropout does not support sampling$",
                ):
                    torch.dropout(source, probability, True)
                after = self.snapshot(source)
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])
                self.assertIsNone(leaf.grad)

    def test_builtin_metadata_exports_and_pickle(self):
        function = torch.dropout
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "dropout")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.dropout")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertIsNone(function.__self__)
        self.assertRegex(
            repr(function),
            r"^<built-in method dropout of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.dropout, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertEqual(torch.__all__.count("dropout"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["dropout"], function)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))


if __name__ == "__main__":
    unittest.main()
