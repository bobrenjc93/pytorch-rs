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
class TopLevelPermuteReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, actual_source, expected_source, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.data_ptr(), actual_source.data_ptr())
            self.assertEqual(expected.data_ptr(), expected_source.data_ptr())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_seeded_forms_values_layout_and_aliasing_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0xA11CE_213)
        shapes = [(), (0,), (2, 0, 3), (1, 3, 2), (2, 3, 4)]
        for _ in range(32):
            rank = int(rng.integers(0, 7))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            if elements:
                actual_source = torch.tensor(
                    values.item() if shape == () else values.tolist()
                )
                expected_source = reference_torch.tensor(
                    values, dtype=reference_torch.float32
                )
            else:
                actual_source = torch.zeros(shape)
                expected_source = reference_torch.zeros(
                    shape, dtype=reference_torch.float32
                )

            if len(shape) >= 2 and shape[0] > 0 and shape[-1] > 0 and case % 3 == 1:
                actual_source = actual_source.transpose(0, -1)[-1]
                expected_source = expected_source.transpose(0, -1)[-1]

            rank = len(actual_source.shape)
            permutation = list(rng.permutation(rank))
            dimensions = tuple(
                axis - rank if (case + index) % 2 else axis
                for index, axis in enumerate(permutation)
            )
            style = case % 7
            if style == 0:
                actual = torch.permute(actual_source, dimensions)
                expected = reference_torch.permute(expected_source, dimensions)
            elif style == 1:
                actual = torch.permute(actual_source, list(dimensions))
                expected = reference_torch.permute(expected_source, list(dimensions))
            elif style == 2:
                actual = torch.permute(input=actual_source, dims=dimensions)
                expected = reference_torch.permute(
                    input=expected_source, dims=dimensions
                )
            elif style == 3:
                actual = torch.permute(dims=list(dimensions), input=actual_source)
                expected = reference_torch.permute(
                    dims=list(dimensions), input=expected_source
                )
            elif style == 4:
                actual = torch.permute(x=actual_source, dims=dimensions)
                expected = reference_torch.permute(x=expected_source, dims=dimensions)
            elif style == 5:
                actual = torch.permute(a=actual_source, dims=list(dimensions))
                expected = reference_torch.permute(
                    a=expected_source, dims=list(dimensions)
                )
            else:
                actual = torch.permute(x1=actual_source, dims=dimensions)
                expected = reference_torch.permute(
                    x1=expected_source, dims=dimensions
                )

            self.assert_matches(
                actual, expected, actual_source, expected_source, case
            )

    def test_autograd_empty_backward_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 2, 3)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = torch.permute(input=actual_leaf, dims=(-1, 0, 1))
        expected = reference_torch.permute(
            input=expected_leaf, dims=(-1, 0, 1)
        )
        self.assert_matches(
            actual, expected, actual_leaf, expected_leaf, "tracked"
        )
        self.assertEqual(
            (actual.requires_grad, actual.is_leaf),
            (expected.requires_grad, expected.is_leaf),
        )
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.permute(actual_empty, [-1, 0, 1]).sum().backward()
        reference_torch.permute(expected_empty, [-1, 0, 1]).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.detach().cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = torch.permute(actual_leaf, (1, 2, 0))
        with reference_torch.no_grad():
            expected_untracked = reference_torch.permute(
                expected_leaf, (1, 2, 0)
            )
        self.assertEqual(
            (actual_untracked.requires_grad, actual_untracked.is_leaf),
            (expected_untracked.requires_grad, expected_untracked.is_leaf),
        )

    def test_scalar_empty_rank_duplicate_range_and_binding_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        actual_scalar = torch.tensor(1.0)
        expected_scalar = reference_torch.tensor(1.0)

        for actual_result, expected_result in (
            (torch.permute(actual_scalar, ()), reference_torch.permute(expected_scalar, ())),
            (torch.permute(actual_scalar, []), reference_torch.permute(expected_scalar, [])),
            (
                torch.permute(actual, [0, True, 2]),
                reference_torch.permute(expected, [0, True, 2]),
            ),
        ):
            self.assertEqual(actual_result.shape, expected_result.shape)
            self.assertEqual(actual_result.stride(), expected_result.stride())

        cases = (
            (lambda: torch.permute(), lambda: reference_torch.permute()),
            (lambda: torch.permute(actual), lambda: reference_torch.permute(expected)),
            (
                lambda: torch.permute(dims=(2, 0, 1)),
                lambda: reference_torch.permute(dims=(2, 0, 1)),
            ),
            (
                lambda: torch.permute(actual, 2, 0, 1),
                lambda: reference_torch.permute(expected, 2, 0, 1),
            ),
            (lambda: torch.permute(1, (0,)), lambda: reference_torch.permute(1, (0,))),
            (
                lambda: torch.permute(input=[], dims=(0,)),
                lambda: reference_torch.permute(input=[], dims=(0,)),
            ),
            (lambda: torch.permute(actual, 1), lambda: reference_torch.permute(expected, 1)),
            (
                lambda: torch.permute(input=actual, dims=1),
                lambda: reference_torch.permute(input=expected, dims=1),
            ),
            (
                lambda: torch.permute(actual, (0, 1)),
                lambda: reference_torch.permute(expected, (0, 1)),
            ),
            (
                lambda: torch.permute(actual, (0, 1, 1)),
                lambda: reference_torch.permute(expected, (0, 1, 1)),
            ),
            (
                lambda: torch.permute(actual, (0, 1, -2)),
                lambda: reference_torch.permute(expected, (0, 1, -2)),
            ),
            (
                lambda: torch.permute(actual, (0, 1, 3)),
                lambda: reference_torch.permute(expected, (0, 1, 3)),
            ),
            (
                lambda: torch.permute(actual, (0, 1, -4)),
                lambda: reference_torch.permute(expected, (0, 1, -4)),
            ),
            (
                lambda: torch.permute(actual, (0, 0, 3)),
                lambda: reference_torch.permute(expected, (0, 0, 3)),
            ),
            (
                lambda: torch.permute(actual, (0, 3, 0)),
                lambda: reference_torch.permute(expected, (0, 3, 0)),
            ),
            (
                lambda: torch.permute(actual_scalar, (-1,)),
                lambda: reference_torch.permute(expected_scalar, (-1,)),
            ),
            (
                lambda: torch.permute(actual, [1.5, 0, 2]),
                lambda: reference_torch.permute(expected, [1.5, 0, 2]),
            ),
            (
                lambda: torch.permute(input=actual, dims=[1.5, 0, 2]),
                lambda: reference_torch.permute(
                    input=expected, dims=[1.5, 0, 2]
                ),
            ),
            (
                lambda: torch.permute(actual, [0, np.bool_(True), 2]),
                lambda: reference_torch.permute(
                    expected, [0, np.bool_(True), 2]
                ),
            ),
            (
                lambda: torch.permute(actual, (2, 0, 1), input=actual),
                lambda: reference_torch.permute(
                    expected, (2, 0, 1), input=expected
                ),
            ),
            (
                lambda: torch.permute(actual, (2, 0, 1), dims=(2, 0, 1)),
                lambda: reference_torch.permute(
                    expected, (2, 0, 1), dims=(2, 0, 1)
                ),
            ),
            (
                lambda: torch.permute(actual, (0, 1), extra=True),
                lambda: reference_torch.permute(expected, (0, 1), extra=True),
            ),
            (
                lambda: torch.permute(actual, [0, 1.5, 2], extra=True),
                lambda: reference_torch.permute(
                    expected, [0, 1.5, 2], extra=True
                ),
            ),
            (
                lambda: torch.permute(actual, [1.5, 0, 2], extra=True),
                lambda: reference_torch.permute(
                    expected, [1.5, 0, 2], extra=True
                ),
            ),
            (
                lambda: torch.permute(x=actual, dims=(2, 0, 1), extra=True),
                lambda: reference_torch.permute(
                    x=expected, dims=(2, 0, 1), extra=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_index_conversion_order_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        class StatefulIndex:
            def __init__(self, name, calls, value):
                self.name = name
                self.calls = calls
                self.value = value

            def __index__(self):
                self.calls.append(self.name)
                return self.value

        for style in ("positional", "keyword", "unexpected"):
            actual_calls = []
            expected_calls = []
            actual_dimensions = [
                StatefulIndex("first", actual_calls, 2),
                StatefulIndex("second", actual_calls, 0),
                StatefulIndex("third", actual_calls, 1),
            ]
            expected_dimensions = [
                StatefulIndex("first", expected_calls, 2),
                StatefulIndex("second", expected_calls, 0),
                StatefulIndex("third", expected_calls, 1),
            ]
            actual = torch.zeros((2, 3, 4))
            expected = reference_torch.zeros((2, 3, 4))
            if style == "positional":
                torch.permute(actual, actual_dimensions)
                reference_torch.permute(expected, expected_dimensions)
            elif style == "keyword":
                torch.permute(input=actual, dims=actual_dimensions)
                reference_torch.permute(
                    input=expected, dims=expected_dimensions
                )
            else:
                self.assert_error_matches(
                    lambda: torch.permute(
                        actual, actual_dimensions, unexpected=True
                    ),
                    lambda: reference_torch.permute(
                        expected, expected_dimensions, unexpected=True
                    ),
                )
            self.assertEqual(actual_calls, expected_calls)

    def callable_contract(self, module):
        function = module.permute
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
            "owner_callable_identity": owner.permute is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("permute"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["permute"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
