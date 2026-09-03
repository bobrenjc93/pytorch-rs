import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


FUNCTION_DOC = r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "Apply the Sigmoid Linear Unit (SiLU) function, element-wise.\n\n"
        "The SiLU function is also known as the swish function.\n\n"
        ".. math::\n"
        r"    \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}"
        "\n\n"
        ".. note::\n"
        "    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_\n"
        "    where the SiLU (Sigmoid Linear Unit) was originally coined, and see\n"
        "    `Sigmoid-Weighted Linear Units for Neural Network Function Approximation\n"
        "    in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:\n"
        "    a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_\n"
        "    where the SiLU was experimented with later.\n\n"
        "See :class:`~torch.nn.SiLU` for more details.\n"
    )


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
            tensor.requires_grad,
            tensor.is_leaf,
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
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist()
        ).contiguous(memory_format=torch.channels_last_3d)
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("contiguous", base),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels_last", channels_last),
            ("channels_last_3d", channels_last_3d),
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
            self.assertEqual(
                actual.is_contiguous(memory_format=torch.channels_last_3d),
                expected.is_contiguous(memory_format=torch.channels_last_3d),
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

    def test_imports_signature_documentation_copy_and_pickle(self):
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
        self.assertEqual(
            function.__annotations__,
            {"input": torch.Tensor, "inplace": bool, "return": torch.Tensor},
        )
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(
            str(signature),
            "(input: torch_rs.Tensor, inplace: bool = False) -> torch_rs.Tensor",
        )
        self.assertEqual(tuple(signature.parameters), ("input", "inplace"))
        self.assertEqual(parameters[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(parameters[1].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIs(parameters[1].default, False)

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

    def test_default_form_matches_sigmoid_multiplication_layout_and_storage(self):
        for case, source in self.layout_cases():
            before = self.tensor_state(source)
            expected = source * source.sigmoid()
            actual = functional.silu(input=source)
            self.assert_matches_composition(actual, expected, source, case=case)
            after = self.tensor_state(source)
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

    def test_numerical_edges_match_sigmoid_multiplication_and_quiet_nans(self):
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EFF_FFFF,
                0x3F00_0000,
                0x3F7F_FFFF,
                0x3F80_0000,
                0xBF00_0000,
                0xBF7F_FFFF,
                0xBF80_0000,
                0xBFC0_0000,
                0x3FC0_0000,
                0x42B0_0000,
                0x42B2_0000,
                0xC2B0_0000,
                0xC2B2_0000,
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
        expected = source * source.sigmoid()
        actual = functional.silu(source)
        actual_bits = self.tensor_bits(actual)
        expected_bits = self.tensor_bits(expected)
        input_bits = self.tensor_bits(source)
        input_values = np.asarray(source, dtype=np.float32)
        nan_mask = np.isnan(input_values)

        np.testing.assert_array_equal(actual_bits[~nan_mask], expected_bits[~nan_mask])
        np.testing.assert_array_equal(
            actual_bits[nan_mask],
            input_bits[nan_mask] | np.uint32(0x0040_0000),
        )

    def test_supported_autograd_matches_explicit_composition(self):
        values = [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]]
        weights = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 3.0, -4.0]])
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = torch.tensor(values, requires_grad=True)

        actual = functional.silu(actual_leaf)
        expected = expected_leaf * expected_leaf.sigmoid()
        self.assert_matches_composition(actual, expected, actual_leaf, case="forward")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual),
            ", grad_fn=<MulBackward0>",
        )

        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(actual_leaf.grad),
            self.tensor_bits(expected_leaf.grad),
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        output = functional.silu(empty)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        output.sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_no_grad_tracked_input_uses_untracked_fresh_output(self):
        source = torch.tensor([[-3.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True)
        before = self.tensor_state(source)
        with torch.no_grad():
            actual = functional.silu(source)
            expected = source * source.sigmoid()
        self.assert_matches_composition(actual, expected, source, case="no_grad")
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        after = self.tensor_state(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])
        self.assertIsNone(source.grad)

    def test_torch_function_overrides_and_modes_observe_the_public_function(self):
        marker = object()

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(functional.silu(input=value), marker)
        self.assertEqual(len(Override.calls), 1)
        function, dispatch_types, args, kwargs = Override.calls[0]
        self.assertIs(function, functional.silu)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, (value,))
        self.assertEqual(kwargs, {"inplace": False})

        source = torch.tensor([0.5], requires_grad=True)

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for case, call, expected_kwargs in (
            ("default", lambda: functional.silu(source), {"inplace": False}),
            ("keyword", lambda: functional.silu(input=source), {"inplace": False}),
            (
                "inplace",
                lambda: functional.silu(input=source, inplace=True),
                {"inplace": True},
            ),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, functional.silu)
                self.assertEqual(dispatch_types, (torch.Tensor,))
                self.assertEqual(args, (source,))
                self.assertEqual(kwargs, expected_kwargs)

    def test_inplace_true_and_unsupported_autograd_fail_before_mutation(self):
        source = torch.tensor([[-1.0, 2.0], [-0.0, 3.0]])
        before = self.tensor_state(source)
        for call in (
            lambda: functional.silu(source, inplace=True),
            lambda: functional.silu(source, True),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "^torch_rs\\.nn\\.functional\\.silu does not support inplace=True$",
                ):
                    call()
                after = self.tensor_state(source)
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

        self.assert_matches_composition(
            functional.silu(source, inplace=[]),
            source * source.sigmoid(),
            source,
            case="falsey inplace",
        )

        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        view = leaf.transpose(0, 1)[1]
        values_before = np.asarray(leaf.detach()).copy().view(np.uint32)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.silu(view)
        np.testing.assert_array_equal(
            np.asarray(leaf.detach()).view(np.uint32),
            values_before,
        )
        self.assertIsNone(leaf.grad)

    def test_argument_errors_and_unsupported_surface(self):
        source = torch.tensor([0.5])
        cases = (
            (
                lambda: functional.silu(),
                TypeError,
                "silu() missing 1 required positional argument: 'input'",
            ),
            (
                lambda: functional.silu(source, False, None),
                TypeError,
                "silu() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: functional.silu(source, input=source),
                TypeError,
                "silu() got multiple values for argument 'input'",
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
            (
                lambda: functional.silu(None),
                TypeError,
                "silu(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: functional.silu(1, inplace=True),
                TypeError,
                "silu_(): argument 'input' (position 1) must be Tensor, not int",
            ),
        )
        for case, (call, error_type, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch.Tensor, "silu"))
        self.assertFalse(hasattr(nn, "SiLU"))
        self.assertFalse(hasattr(functional, "silu_"))


if __name__ == "__main__":
    unittest.main()
