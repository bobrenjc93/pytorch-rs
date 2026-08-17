import copy
import importlib
import inspect
import math
import pickle
import struct
import types
import unittest
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
from typing import Literal

import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.init as init


NONLINEARITIES = (
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
)

MUTATING_INITIALIZERS = (
    "uniform_",
    "normal_",
    "trunc_normal_",
    "constant_",
    "ones_",
    "zeros_",
    "eye_",
    "dirac_",
    "xavier_uniform_",
    "xavier_normal_",
    "kaiming_uniform_",
    "kaiming_normal_",
    "orthogonal_",
    "sparse_",
    "uniform",
    "normal",
    "constant",
    "eye",
    "dirac",
    "xavier_uniform",
    "xavier_normal",
    "kaiming_uniform",
    "kaiming_normal",
    "orthogonal",
    "sparse",
)


class _IntegerSlope(int):
    pass


class _FloatSlope(float):
    pass


class _EnumeratedSlope(IntEnum):
    TWO = 2


class CalculateGainTests(unittest.TestCase):
    def assert_float_bits_equal(self, actual, expected):
        self.assertIs(type(actual), float)
        self.assertEqual(struct.pack("=d", actual), struct.pack("=d", expected))

    def test_canonical_imports_and_minimal_exports(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_init = importlib.import_module("torch_rs.nn.init")
        from torch_rs.nn import init as from_nn
        from torch_rs.nn.init import calculate_gain

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.init, init)
        self.assertIs(init, imported_init)
        self.assertIs(from_nn, init)
        self.assertIs(calculate_gain, init.calculate_gain)
        self.assertFalse(hasattr(nn, "calculate_gain"))
        self.assertNotIn("nn", torch.__all__)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertEqual(init.__all__, ["calculate_gain"])

        wildcard_namespace = {}
        exec("from torch_rs.nn.init import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["calculate_gain"], init.calculate_gain)
        for name in MUTATING_INITIALIZERS:
            with self.subTest(name=name):
                self.assertFalse(hasattr(init, name))
                self.assertNotIn(name, init.__all__)
                self.assertNotIn(name, wildcard_namespace)

    def test_signature_annotations_metadata_documentation_and_pickle(self):
        function = init.calculate_gain
        expected_nonlinearity_type = Literal[
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "sigmoid",
            "tanh",
            "relu",
            "leaky_relu",
            "selu",
        ]

        self.assertEqual(
            init.__doc__,
            "This file contains utilities for initializing neural network parameters.",
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "calculate_gain")
        self.assertEqual(function.__qualname__, "calculate_gain")
        self.assertEqual(function.__module__, "torch_rs.nn.init")
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIn(
            "Return the recommended gain value for the given nonlinearity function.",
            function.__doc__,
        )
        self.assertIn("Self-Normalizing Neural Networks", function.__doc__)
        self.assertIn('"leaky_relu", 0.2', function.__doc__)

        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertEqual(tuple(signature.parameters), ("nonlinearity", "param"))
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                for parameter in parameters
            )
        )
        self.assertEqual(parameters[0].annotation, expected_nonlinearity_type)
        self.assertEqual(parameters[1].annotation, int | float | None)
        self.assertIs(parameters[1].default, None)
        self.assertIs(signature.return_annotation, float)
        self.assertEqual(
            function.__annotations__,
            {
                "nonlinearity": expected_nonlinearity_type,
                "param": int | float | None,
                "return": float,
            },
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.nn.init", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_every_supported_nonlinearity_and_return_type(self):
        unit_gain_nonlinearities = NONLINEARITIES[:8]
        for nonlinearity in unit_gain_nonlinearities:
            with self.subTest(nonlinearity=nonlinearity):
                result = init.calculate_gain(nonlinearity)
                self.assertIs(type(result), int)
                self.assertEqual(result, 1)

        expected_floats = {
            "tanh": 5.0 / 3,
            "relu": math.sqrt(2.0),
            "leaky_relu": math.sqrt(2.0 / (1 + 0.01**2)),
            "selu": 3.0 / 4,
        }
        for nonlinearity, expected in expected_floats.items():
            with self.subTest(nonlinearity=nonlinearity):
                self.assert_float_bits_equal(
                    init.calculate_gain(nonlinearity), expected
                )

        ignored_parameter = object()
        for nonlinearity in (*unit_gain_nonlinearities, "tanh", "relu", "selu"):
            with self.subTest(nonlinearity=nonlinearity, parameter="ignored"):
                self.assertEqual(
                    init.calculate_gain(nonlinearity, ignored_parameter),
                    init.calculate_gain(nonlinearity),
                )

    def test_leaky_relu_default_explicit_values_and_numeric_subclasses(self):
        cases = (
            (None, 0.01),
            (0, 0),
            (-0.0, -0.0),
            (1, 1),
            (-2, -2),
            (0.2, 0.2),
            (_IntegerSlope(3), 3),
            (_FloatSlope(0.5), 0.5),
            (_EnumeratedSlope.TWO, 2),
            (float("inf"), float("inf")),
            (float("-inf"), float("-inf")),
        )
        for supplied, numeric_value in cases:
            with self.subTest(supplied=supplied):
                expected = math.sqrt(2.0 / (1 + numeric_value**2))
                self.assert_float_bits_equal(
                    init.calculate_gain("leaky_relu", supplied), expected
                )

        result = init.calculate_gain("leaky_relu", float("nan"))
        self.assertIs(type(result), float)
        self.assertTrue(math.isnan(result))

        self.assert_float_bits_equal(
            init.calculate_gain(nonlinearity="leaky_relu", param=0.2),
            math.sqrt(2.0 / (1 + 0.2**2)),
        )

    def test_invalid_leaky_relu_parameters_and_unsupported_nonlinearities(self):
        invalid_parameters = (
            True,
            False,
            "0.2",
            1 + 2j,
            Decimal("0.2"),
            Fraction(1, 5),
            [0.2],
            object(),
        )
        for parameter in invalid_parameters:
            with self.subTest(parameter=parameter):
                with self.assertRaises(ValueError) as raised:
                    init.calculate_gain("leaky_relu", parameter)
                self.assertEqual(
                    str(raised.exception),
                    f"negative_slope {parameter} not a valid number",
                )

        with self.assertRaises(OverflowError):
            init.calculate_gain("leaky_relu", 10**1000)

        unsupported = ("ReLU", "gelu", "", None, True, 1, ["relu"])
        for nonlinearity in unsupported:
            with self.subTest(nonlinearity=nonlinearity):
                with self.assertRaises(ValueError) as raised:
                    init.calculate_gain(nonlinearity)
                self.assertEqual(
                    str(raised.exception),
                    f"Unsupported nonlinearity {nonlinearity}",
                )

    def test_python_call_validation(self):
        invalid_calls = (
            lambda: init.calculate_gain(),
            lambda: init.calculate_gain("relu", None, 1),
            lambda: init.calculate_gain(kind="relu"),
            lambda: init.calculate_gain("relu", nonlinearity="relu"),
            lambda: init.calculate_gain("relu", unsupported=True),
        )
        for case, call in enumerate(invalid_calls):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    call()


if __name__ == "__main__":
    unittest.main()
