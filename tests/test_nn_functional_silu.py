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
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

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
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 2)[1]),
            ("channels_last", channels_last),
            ("channels_last_3d", channels_last_3d),
        )

    @staticmethod
    def silu_formula(input):
        return input / ((-input).exp() + 1)

    def assert_matches_expected(self, actual, expected, source, *, case):
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

    def test_import_signature_documentation_copy_pickle_and_unsupported_surface(self):
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
        self.assertEqual(tuple(signature.parameters), ("input", "inplace"))
        self.assertIs(parameters[0].annotation, torch.Tensor)
        self.assertIs(parameters[1].annotation, bool)
        self.assertIs(parameters[1].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

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

        self.assertFalse(hasattr(torch.Tensor, "silu"))
        self.assertFalse(hasattr(torch.Tensor, "silu_"))
        self.assertFalse(hasattr(torch, "silu"))
        self.assertFalse(hasattr(nn, "SiLU"))
        self.assertFalse(hasattr(functional, "silu_"))

    def test_values_layout_fresh_storage_and_nonmutation_match_formula(self):
        for case, source in self.layout_cases():
            expected = self.silu_formula(source)
            before = self.tensor_state(source)
            actual = functional.silu(input=source)
            self.assert_matches_expected(actual, expected, source, case=case)
            second = functional.silu(source, False)
            with self.subTest(case=case, fresh_storage=True):
                self.assertIsNot(actual, second)
                self.assertFalse(actual.is_set_to(second))
                if source.numel():
                    self.assertNotEqual(actual.data_ptr(), second.data_ptr())
            after = self.tensor_state(source)
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(after[:-1], before[:-1])
                np.testing.assert_array_equal(after[-1], before[-1])

    def test_numerical_edges_match_the_silu_formula_bitwise(self):
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x4000_0000,
                0xC000_0000,
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
        expected = self.silu_formula(source)
        actual = functional.silu(source)
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_autograd_matches_the_primitive_composition(self):
        values = np.asarray(
            [[-3.0, -1.0, -0.0, 0.5], [1.0, 2.0, 4.0, 8.0]],
            dtype=np.float32,
        )
        weights = torch.tensor(
            np.asarray(
                [[1.0, -2.0, 0.25, -0.5], [3.0, 0.125, -1.5, 2.0]],
                dtype=np.float32,
            ).tolist()
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = torch.tensor(values.tolist(), requires_grad=True)

        actual = functional.silu(actual_leaf)
        expected_graph = expected_leaf * expected_leaf.sigmoid()
        expected_values = self.silu_formula(actual_leaf.detach())
        with self.subTest(case="autograd forward metadata"):
            self.assertEqual(actual.shape, expected_graph.shape)
            self.assertEqual(actual.stride(), expected_graph.stride())
            self.assertEqual(actual.storage_offset(), expected_graph.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected_graph.is_contiguous())
            self.assertEqual(actual.requires_grad, expected_graph.requires_grad)
            self.assertEqual(actual.is_leaf, expected_graph.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(actual_leaf))
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected_values),
        )
        self.assertEqual(actual.requires_grad, expected_graph.requires_grad)
        self.assertEqual(actual.is_leaf, expected_graph.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual),
            ", grad_fn=<MulBackward0>",
        )

        (actual * weights).sum().backward()
        (expected_graph * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(actual_leaf.grad),
            self.tensor_bits(expected_leaf.grad),
        )

        accumulated = torch.tensor(values.tolist(), requires_grad=True)
        for _ in range(2):
            (functional.silu(accumulated) * weights).sum().backward()
        doubled = torch.tensor(values.tolist(), requires_grad=True)
        ((doubled * doubled.sigmoid()) * weights).sum().backward()
        ((doubled * doubled.sigmoid()) * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            self.tensor_bits(doubled.grad),
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_output = functional.silu(empty)
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, empty.shape)
        self.assertEqual(empty_output.stride(), empty.stride())
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_output),
            ", grad_fn=<MulBackward0>",
        )
        empty_output.sum().backward()
        self.assertEqual(empty.grad.tolist(), empty.tolist())

        high_rank_shape = (1,) * 65
        high_rank = torch.full(high_rank_shape, 0.5, requires_grad=True)
        high_rank_output = functional.silu(high_rank)
        self.assertTrue(high_rank_output.requires_grad)
        self.assertFalse(high_rank_output.is_leaf)
        self.assertEqual(high_rank_output.shape, high_rank_shape)
        self.assertEqual(high_rank_output.stride(), (1,) * 65)
        high_rank_output.backward()
        self.assertIsNotNone(high_rank.grad)

    def test_no_grad_and_detached_inputs_use_the_inference_composition(self):
        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        with torch.no_grad():
            expected = self.silu_formula(source)
            actual = functional.silu(source)
        self.assert_matches_expected(actual, expected, source, case="no_grad")
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)

        detached = source.detach()
        expected = self.silu_formula(detached)
        actual = functional.silu(detached)
        self.assert_matches_expected(actual, expected, detached, case="detached")
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        self.assertIsNone(leaf.grad)

    def test_rejects_inplace_true_and_unsupported_autograd_before_mutation(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        before = self.tensor_state(source)
        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch_rs\\.nn\\.functional\\.silu does not support inplace=True$",
        ):
            functional.silu(source, inplace=True)
        after = self.tensor_state(source)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])
        self.assertIsNone(leaf.grad)

        view_base = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        unsupported_view = view_base.transpose(0, 1)[1]
        before = self.tensor_state(unsupported_view)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.silu(unsupported_view)
        after = self.tensor_state(unsupported_view)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])
        self.assertIsNone(view_base.grad)

        nonfinite = torch.tensor([0.5, float("inf")], requires_grad=True)
        before = self.tensor_state(nonfinite)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): autograd recording is not supported$",
        ):
            functional.silu(nonfinite)
        after = self.tensor_state(nonfinite)
        self.assertEqual(after[:-1], before[:-1])
        np.testing.assert_array_equal(after[-1], before[-1])
        self.assertIsNone(nonfinite.grad)

    def test_overrides_and_modes_observe_the_public_function(self):
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

        self.assertIs(functional.silu(value, inplace=True), marker)
        self.assertEqual(Override.calls[-1][3], {"inplace": True})

        source = torch.tensor([0.5])

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            result = functional.silu(source)
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, functional.silu)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(args, (source,))
        self.assertEqual(kwargs, {"inplace": False})

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = functional.silu(input=source)
        self.assert_matches_expected(
            forwarded,
            self.silu_formula(source),
            source,
            case="forwarded mode",
        )

    def test_argument_errors_and_module_boundary(self):
        source = torch.tensor([0.5])
        cases = (
            (
                lambda: functional.silu(),
                TypeError,
                "silu() missing 1 required positional argument: 'input'",
            ),
            (
                lambda: functional.silu(source, False, False),
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
                lambda: functional.silu(input=[]),
                TypeError,
                "silu(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: functional.silu(1, inplace=True),
                TypeError,
                "silu_(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: functional.silu(input=[], inplace=True),
                TypeError,
                "silu_(): argument 'input' (position 1) must be Tensor, not list",
            ),
        )
        for case, (call, error_type, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(TypeError, r"^silu\(\):"):
            functional.silu(np.zeros((2, 3), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
