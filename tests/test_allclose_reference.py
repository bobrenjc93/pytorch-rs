import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AllCloseReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def tensor_value_bits(self, tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32).tolist()

    def assert_allclose_calls_match(
        self,
        actual_left,
        actual_right,
        expected_left,
        expected_right,
        **kwargs,
    ):
        actual_results = (
            torch.allclose(actual_left, actual_right, **kwargs),
            actual_left.allclose(actual_right, **kwargs),
        )
        expected_results = (
            reference_torch.allclose(expected_left, expected_right, **kwargs),
            expected_left.allclose(expected_right, **kwargs),
        )
        self.assertEqual(actual_results, expected_results)
        self.assertTrue(all(type(result) is bool for result in actual_results))

    def test_scalar_empty_broadcast_and_strided_inputs_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        expected_dense = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            dtype=reference_torch.float32,
        )
        cases = (
            (
                torch.tensor(1.0),
                torch.tensor([1.0]),
                reference_torch.tensor(1.0),
                reference_torch.tensor([1.0]),
                {},
            ),
            (
                torch.zeros((2, 0, 3)),
                torch.ones((1, 0, 1)),
                reference_torch.zeros((2, 0, 3)),
                reference_torch.ones((1, 0, 1)),
                {},
            ),
            (
                torch.zeros((0,)),
                torch.zeros((1, 0)),
                reference_torch.zeros((0,)),
                reference_torch.zeros((1, 0)),
                {},
            ),
            (
                torch.tensor([[1.0], [1.0]]),
                torch.tensor([1.0, 1.0]),
                reference_torch.tensor([[1.0], [1.0]]),
                reference_torch.tensor([1.0, 1.0]),
                {},
            ),
            (
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([1.0, 2.0]),
                reference_torch.tensor([[1.0], [2.0]]),
                reference_torch.tensor([1.0, 2.0]),
                {},
            ),
            (
                actual_dense[1],
                torch.tensor(
                    np.arange(12, 24, dtype=np.float32).reshape(3, 4).tolist()
                ),
                expected_dense[1],
                reference_torch.tensor(
                    np.arange(12, 24, dtype=np.float32).reshape(3, 4),
                    dtype=reference_torch.float32,
                ),
                {},
            ),
            (
                actual_dense.transpose(0, 2)[1],
                torch.tensor([[1.0, 13.0], [5.0, 17.0], [9.0, 21.0]]),
                expected_dense.transpose(0, 2)[1],
                reference_torch.tensor(
                    [[1.0, 13.0], [5.0, 17.0], [9.0, 21.0]],
                    dtype=reference_torch.float32,
                ),
                {},
            ),
        )
        for case, values in enumerate(cases):
            with self.subTest(case=case):
                self.assert_allclose_calls_match(*values[:4], **values[4])

    def test_special_values_tolerances_and_no_mutation_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        cases = (
            (
                [0.0, -0.0],
                [-0.0, 0.0],
                {},
            ),
            (
                [float("nan")],
                [float("nan")],
                {},
            ),
            (
                [float("nan")],
                [float("nan")],
                {"equal_nan": True},
            ),
            (
                [float("nan")],
                [1.0],
                {"equal_nan": True},
            ),
            (
                [float("inf"), -float("inf")],
                [float("inf"), -float("inf")],
                {},
            ),
            (
                [float("inf")],
                [-float("inf")],
                {},
            ),
            (
                [float("inf")],
                [3.4e38],
                {"rtol": float("inf")},
            ),
            (
                [1.0],
                [1.0 + 1.0e-6],
                {},
            ),
            (
                [1.0],
                [1.0 + 1.0e-4],
                {},
            ),
            (
                [1.0],
                [1.11],
                {"rtol": 0.2, "atol": 0},
            ),
            (
                [float(np.float32(0.001))],
                [0.0],
                {"rtol": 0.0, "atol": 0.001},
            ),
            (
                [float(np.float32(1.0e-45))],
                [0.0],
                {"rtol": 0.0, "atol": 1.0e-45},
            ),
        )
        for case, (left_values, right_values, kwargs) in enumerate(cases):
            with self.subTest(case=case):
                actual_left = torch.tensor(left_values)
                actual_right = torch.tensor(right_values)
                expected_left = reference_torch.tensor(
                    left_values, dtype=reference_torch.float32
                )
                expected_right = reference_torch.tensor(
                    right_values, dtype=reference_torch.float32
                )
                before = (
                    self.tensor_value_bits(actual_left),
                    self.tensor_value_bits(actual_right),
                )
                self.assert_allclose_calls_match(
                    actual_left, actual_right, expected_left, expected_right, **kwargs
                )
                self.assertEqual(
                    (
                        self.tensor_value_bits(actual_left),
                        self.tensor_value_bits(actual_right),
                    ),
                    before,
                )

    def test_keyword_aliases_and_positional_tolerances_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (
                lambda: torch.allclose(input=actual, other=actual),
                lambda: reference_torch.allclose(input=expected, other=expected),
            ),
            (
                lambda: torch.allclose(x=actual, x2=actual),
                lambda: reference_torch.allclose(x=expected, x2=expected),
            ),
            (
                lambda: torch.allclose(a=actual, other=actual),
                lambda: reference_torch.allclose(a=expected, other=expected),
            ),
            (
                lambda: torch.allclose(input=actual, x2=actual),
                lambda: reference_torch.allclose(input=expected, x2=expected),
            ),
            (
                lambda: torch.allclose(actual, actual, 0, 0, False),
                lambda: reference_torch.allclose(expected, expected, 0, 0, False),
            ),
            (
                lambda: actual.allclose(other=actual),
                lambda: expected.allclose(other=expected),
            ),
            (
                lambda: actual.allclose(x2=actual),
                lambda: expected.allclose(x2=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(actual_call(), expected_call())

    def callable_contract(self, module):
        function = module.allclose
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.allclose is function,
            "doc_first_line": next(line for line in function.__doc__.splitlines() if line),
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("allclose"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["allclose"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_ownership_documentation_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def test_binding_type_and_shape_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.allclose(), lambda: reference_torch.allclose()),
            (lambda: torch.allclose(actual), lambda: reference_torch.allclose(expected)),
            (lambda: torch.allclose(None), lambda: reference_torch.allclose(None)),
            (
                lambda: torch.allclose(input=1),
                lambda: reference_torch.allclose(input=1),
            ),
            (
                lambda: torch.allclose(actual, actual, actual),
                lambda: reference_torch.allclose(expected, expected, expected),
            ),
            (
                lambda: torch.allclose(actual, 1),
                lambda: reference_torch.allclose(expected, 1),
            ),
            (
                lambda: torch.allclose(input=actual, other=[]),
                lambda: reference_torch.allclose(input=expected, other=[]),
            ),
            (
                lambda: torch.allclose(foo=actual, other=actual),
                lambda: reference_torch.allclose(foo=expected, other=expected),
            ),
            (
                lambda: torch.allclose(input=actual, b=actual),
                lambda: reference_torch.allclose(input=expected, b=expected),
            ),
            (
                lambda: torch.allclose(actual, actual, extra=True),
                lambda: reference_torch.allclose(expected, expected, extra=True),
            ),
            (
                lambda: torch.allclose(actual, actual, other=actual),
                lambda: reference_torch.allclose(expected, expected, other=expected),
            ),
            (
                lambda: torch.allclose(actual, actual, out=None),
                lambda: reference_torch.allclose(expected, expected, out=None),
            ),
            (
                lambda: torch.allclose(actual, actual, dtype=torch.float32),
                lambda: reference_torch.allclose(
                    expected, expected, dtype=reference_torch.float32
                ),
            ),
            (
                lambda: torch.allclose(actual, actual, device="cpu"),
                lambda: reference_torch.allclose(expected, expected, device="cpu"),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol="x"),
                lambda: reference_torch.allclose(expected, expected, rtol="x"),
            ),
            (
                lambda: torch.allclose(actual, actual, equal_nan=1),
                lambda: reference_torch.allclose(expected, expected, equal_nan=1),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=-1),
                lambda: reference_torch.allclose(expected, expected, rtol=-1),
            ),
            (
                lambda: torch.allclose(actual, actual, atol=float("nan")),
                lambda: reference_torch.allclose(expected, expected, atol=float("nan")),
            ),
            (
                lambda: torch.allclose(torch.zeros((2, 0, 3)), torch.zeros((3, 0, 3))),
                lambda: reference_torch.allclose(
                    reference_torch.zeros((2, 0, 3)),
                    reference_torch.zeros((3, 0, 3)),
                ),
            ),
            (lambda: actual.allclose(), lambda: expected.allclose()),
            (
                lambda: actual.allclose(actual, actual),
                lambda: expected.allclose(expected, expected),
            ),
            (
                lambda: actual.allclose(other=actual, out=None),
                lambda: expected.allclose(other=expected, out=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
