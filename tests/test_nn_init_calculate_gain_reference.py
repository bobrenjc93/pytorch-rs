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

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.init as init

try:
    import torch as reference_torch
    import torch.nn as reference_nn
    import torch.nn.init as reference_init
except ImportError:
    reference_torch = None
    reference_nn = None
    reference_init = None


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


class _NonlinearityAlias:
    def __init__(self, target):
        self.target = target

    def __eq__(self, other):
        return other == self.target

    def __str__(self):
        return f"alias:{self.target}"


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CalculateGainReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.init.calculate_gain differentials require pinned PyTorch 2.13.0"
            )

    def outcome(self, call):
        try:
            value = call()
        except Exception as error:
            return ("error", type(error), str(error))
        if type(value) is float:
            return ("return", float, struct.pack("=d", value))
        return ("return", type(value), value)

    def assert_calls_match(self, actual_call, expected_call, *, case):
        with self.subTest(case=case):
            self.assertEqual(self.outcome(actual_call), self.outcome(expected_call))

    def test_imports_exports_and_unsupported_initializers_match_scope(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_init = importlib.import_module("torch_rs.nn.init")
        reference_imported_init = importlib.import_module("torch.nn.init")
        from torch_rs.nn import init as from_nn
        from torch_rs.nn.init import calculate_gain

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.init, init)
        self.assertIs(init, imported_init)
        self.assertIs(from_nn, init)
        self.assertIs(calculate_gain, init.calculate_gain)
        self.assertIs(reference_torch.nn, reference_nn)
        self.assertIs(reference_nn.init, reference_init)
        self.assertIs(reference_init, reference_imported_init)
        self.assertFalse(hasattr(nn, "calculate_gain"))
        self.assertFalse(hasattr(reference_nn, "calculate_gain"))
        self.assertEqual(init.__all__, ["calculate_gain"])
        self.assertIn("calculate_gain", reference_init.__all__)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.nn.init import *", actual_wildcard)
        exec("from torch.nn.init import *", expected_wildcard)
        self.assertIs(actual_wildcard["calculate_gain"], init.calculate_gain)
        self.assertIs(
            expected_wildcard["calculate_gain"], reference_init.calculate_gain
        )
        for name in MUTATING_INITIALIZERS:
            with self.subTest(name=name):
                self.assertFalse(hasattr(init, name))
                self.assertTrue(hasattr(reference_init, name))
                self.assertNotIn(name, init.__all__)
                self.assertNotIn(name, actual_wildcard)

    def test_signature_annotations_metadata_documentation_and_pickle_match(self):
        actual = init.calculate_gain
        expected = reference_init.calculate_gain

        self.assertEqual(init.__doc__, reference_init.__doc__)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertIs(inspect.getmodule(actual), init)
        self.assertIs(inspect.getmodule(expected), reference_init)

        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__):
                self.assertIs(operation(actual), actual)
                self.assertIs(operation(expected), expected)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                actual_payload = pickle.dumps(actual, protocol=protocol)
                expected_payload = pickle.dumps(expected, protocol=protocol)
                self.assertIn(b"torch_rs.nn.init", actual_payload)
                self.assertIn(b"torch.nn.init", expected_payload)
                self.assertIs(pickle.loads(actual_payload), actual)
                self.assertIs(pickle.loads(expected_payload), expected)

    def test_all_numeric_results_argument_forms_and_ignored_parameters_match(self):
        expected_values = {
            "linear": 1,
            "conv1d": 1,
            "conv2d": 1,
            "conv3d": 1,
            "conv_transpose1d": 1,
            "conv_transpose2d": 1,
            "conv_transpose3d": 1,
            "sigmoid": 1,
            "tanh": 5.0 / 3,
            "relu": math.sqrt(2.0),
            "leaky_relu": math.sqrt(2.0 / (1 + 0.01**2)),
            "selu": 3.0 / 4,
        }
        ignored_parameter = object()
        for nonlinearity, value in expected_values.items():
            calls = (
                (
                    lambda n=nonlinearity: init.calculate_gain(n),
                    lambda n=nonlinearity: reference_init.calculate_gain(n),
                ),
                (
                    lambda n=nonlinearity: init.calculate_gain(nonlinearity=n),
                    lambda n=nonlinearity: reference_init.calculate_gain(
                        nonlinearity=n
                    ),
                ),
            )
            for form, (actual_call, expected_call) in enumerate(calls):
                self.assert_calls_match(
                    actual_call,
                    expected_call,
                    case=(nonlinearity, form),
                )
                result = actual_call()
                self.assertIs(type(result), type(value))
                if type(value) is float:
                    self.assertEqual(
                        struct.pack("=d", result), struct.pack("=d", value)
                    )
                else:
                    self.assertEqual(result, value)

            if nonlinearity != "leaky_relu":
                self.assert_calls_match(
                    lambda n=nonlinearity: init.calculate_gain(
                        n, ignored_parameter
                    ),
                    lambda n=nonlinearity: reference_init.calculate_gain(
                        n, ignored_parameter
                    ),
                    case=(nonlinearity, "ignored parameter"),
                )

    def test_leaky_relu_parameter_domain_and_numerical_edges_match(self):
        parameters = (
            None,
            0,
            -0.0,
            1,
            -2,
            0.2,
            _IntegerSlope(3),
            _FloatSlope(0.5),
            _EnumeratedSlope.TWO,
            float("inf"),
            float("-inf"),
            float("nan"),
            float.fromhex("0x0.0000000000001p-1022"),
            10**1000,
            True,
            False,
            "0.2",
            1 + 2j,
            Decimal("0.2"),
            Fraction(1, 5),
            np.bool_(True),
            np.int64(2),
            np.float32(0.2),
            np.float64(0.2),
            [0.2],
            object(),
        )
        for parameter in parameters:
            self.assert_calls_match(
                lambda p=parameter: init.calculate_gain("leaky_relu", p),
                lambda p=parameter: reference_init.calculate_gain(
                    "leaky_relu", p
                ),
                case=(type(parameter).__name__, str(parameter)),
            )

        self.assert_calls_match(
            lambda: init.calculate_gain(nonlinearity="leaky_relu", param=0.2),
            lambda: reference_init.calculate_gain(
                nonlinearity="leaky_relu", param=0.2
            ),
            case="keywords",
        )

    def test_unsupported_and_runtime_duck_typed_nonlinearities_match(self):
        nonlinearities = (
            "ReLU",
            "gelu",
            "",
            None,
            True,
            1,
            ["relu"],
            {"relu": 1},
            np.str_("relu"),
            _NonlinearityAlias("relu"),
            _NonlinearityAlias("linear"),
            np.asarray(["relu", "tanh"]),
        )
        for nonlinearity in nonlinearities:
            self.assert_calls_match(
                lambda n=nonlinearity: init.calculate_gain(n),
                lambda n=nonlinearity: reference_init.calculate_gain(n),
                case=(type(nonlinearity).__name__, str(nonlinearity)),
            )

    def test_python_call_validation_errors_match(self):
        calls = (
            (lambda: init.calculate_gain(), lambda: reference_init.calculate_gain()),
            (
                lambda: init.calculate_gain("relu", None, 1),
                lambda: reference_init.calculate_gain("relu", None, 1),
            ),
            (
                lambda: init.calculate_gain(kind="relu"),
                lambda: reference_init.calculate_gain(kind="relu"),
            ),
            (
                lambda: init.calculate_gain("relu", nonlinearity="relu"),
                lambda: reference_init.calculate_gain(
                    "relu", nonlinearity="relu"
                ),
            ),
            (
                lambda: init.calculate_gain("relu", unsupported=True),
                lambda: reference_init.calculate_gain("relu", unsupported=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(calls):
            self.assert_calls_match(actual_call, expected_call, case=case)


if __name__ == "__main__":
    unittest.main()
