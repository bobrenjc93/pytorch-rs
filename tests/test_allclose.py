import copy
import importlib
import inspect
import pickle
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class AllcloseTests(unittest.TestCase):
    def test_scalar_same_shape_broadcast_empty_and_strided_values(self):
        dense = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).transpose(0, 1)
        offset = dense[1]
        noncontiguous_offset = strided[1]

        cases = (
            ("scalar equal", torch.tensor(1.0), torch.tensor(1.0), {}, True),
            ("scalar mismatch", torch.tensor(1.0), torch.tensor(1.1), {}, False),
            (
                "same shape",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                {},
                True,
            ),
            (
                "same shape false",
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
                {},
                False,
            ),
            (
                "broadcast",
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([[1.0, 1.0, 1.0]]),
                {},
                False,
            ),
            (
                "broadcast tolerance",
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([[1.0, 1.0, 1.0]]),
                {"rtol": 1.0, "atol": 0.0},
                True,
            ),
            ("empty", torch.zeros((2, 0, 3)), torch.ones((1,)), {}, True),
            (
                "empty broadcast",
                torch.zeros((2, 0)),
                torch.ones((2, 1)),
                {},
                True,
            ),
            (
                "offset",
                offset,
                torch.tensor(np.arange(12, 24, dtype=np.float32).reshape(3, 4).tolist()),
                {},
                True,
            ),
            (
                "noncontiguous",
                strided,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                {},
                True,
            ),
            (
                "noncontiguous offset",
                noncontiguous_offset,
                torch.tensor([4.0, 5.0, 6.0]),
                {},
                True,
            ),
        )
        for case, left, right, kwargs, expected in cases:
            with self.subTest(case=case):
                result = torch.allclose(left, right, **kwargs)
                self.assertIs(type(result), bool)
                self.assertIs(result, expected)

    def test_nan_infinity_signed_zero_and_tolerances(self):
        finite = torch.tensor([1.0, 1000.0])
        close = torch.tensor([1.0 + 1e-6, 1000.009])
        not_close = torch.tensor([1.0 + 1e-3, 1000.2])
        subnormal_reference = np.float32(1e-38)
        subnormal_neighbor = np.nextafter(
            subnormal_reference, np.float32(np.inf), dtype=np.float32
        )

        cases = (
            (torch.tensor([0.0, -0.0]), torch.tensor([-0.0, 0.0]), {}, True),
            (torch.tensor([float("nan")]), torch.tensor([float("nan")]), {}, False),
            (
                torch.tensor([1.0, float("nan")]),
                torch.tensor([1.0, float("nan")]),
                {"equal_nan": True},
                True,
            ),
            (
                torch.tensor([float("nan")]),
                torch.tensor([1.0]),
                {"equal_nan": True},
                False,
            ),
            (
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                {},
                True,
            ),
            (torch.tensor([float("inf")]), torch.tensor([-float("inf")]), {}, False),
            (
                torch.tensor([float("inf")]),
                torch.tensor([1.0]),
                {"rtol": float("inf"), "atol": float("inf")},
                False,
            ),
            (
                torch.tensor([float(subnormal_neighbor)]),
                torch.tensor([float(subnormal_reference)]),
                {"rtol": 1.0e-7, "atol": 0.0},
                True,
            ),
            (
                torch.tensor([3.0e38]),
                torch.tensor([-3.0e38]),
                {"rtol": 0.0, "atol": float("inf")},
                False,
            ),
            (finite, close, {}, True),
            (finite, not_close, {}, False),
            (finite, not_close, {"rtol": 1.0e-3, "atol": 1.0e-3}, True),
        )
        for left, right, kwargs, expected in cases:
            with self.subTest(left=left.tolist(), right=right.tolist(), kwargs=kwargs):
                self.assertIs(torch.allclose(left, right, **kwargs), expected)

    def test_supported_argument_forms_and_numpy_tolerances(self):
        left = torch.tensor([1.0, 2.0])
        right = torch.tensor([1.0, 2.0 + 1e-6])
        forms = (
            lambda: torch.allclose(left, right),
            lambda: torch.allclose(left, right, 1e-5),
            lambda: torch.allclose(left, right, 1e-5, 1e-8),
            lambda: torch.allclose(left, right, 1e-5, 1e-8, False),
            lambda: torch.allclose(input=left, other=right),
            lambda: torch.allclose(x=left, other=right),
            lambda: torch.allclose(a=left, x2=right),
            lambda: torch.allclose(x1=left, x2=right, rtol=np.float32(1e-5)),
            lambda: torch.allclose(left, x2=right, atol=np.float64(1e-8)),
            lambda: torch.allclose(left, right, rtol=np.bool_(True)),
            lambda: torch.allclose(left, right, rtol=np.complex64(1 + 2j)),
            lambda: torch.allclose(left, right, rtol=True, atol=False),
        )
        for call in forms:
            with self.subTest(call=call):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = call()
                self.assertIs(result, True)

    def test_errors_and_unsupported_surface(self):
        left = torch.tensor([1.0])
        right = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.allclose(),
                TypeError,
                'allclose() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.allclose(left),
                TypeError,
                'allclose() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.allclose(left, right, 1e-5, 1e-8, False, None),
                TypeError,
                "allclose() takes from 2 to 5 positional arguments but 6 were given",
            ),
            (
                lambda: torch.allclose(np.array([1.0], dtype=np.float32), right),
                TypeError,
                "allclose(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: torch.allclose([1.0], right),
                TypeError,
                "allclose(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.allclose(left, 1.0),
                TypeError,
                "allclose(): argument 'other' (position 2) must be Tensor, not float",
            ),
            (
                lambda: torch.allclose(left, right, rtol=None),
                TypeError,
                "allclose(): argument 'rtol' must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(left, right, rtol=None, out=None),
                TypeError,
                "allclose(): argument 'rtol' must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(left, right, None),
                TypeError,
                "allclose(): argument 'rtol' (position 3) must be float, not NoneType",
            ),
            (
                lambda: torch.allclose(left, right, atol=torch.tensor(1e-8)),
                TypeError,
                "allclose(): argument 'atol' must be float, not Tensor",
            ),
            (
                lambda: torch.allclose(left, right, equal_nan=1),
                TypeError,
                "allclose(): argument 'equal_nan' must be bool, not int",
            ),
            (
                lambda: torch.allclose(left, right, equal_nan=np.bool_(True)),
                TypeError,
                "allclose(): argument 'equal_nan' must be bool, not numpy.bool",
            ),
            (
                lambda: torch.allclose(left, right, 1e-5, 1e-8, 1),
                TypeError,
                "allclose(): argument 'equal_nan' (position 5) must be bool, not int",
            ),
            (
                lambda: torch.allclose(left, right, rtol=-1e-5),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got -1e-05",
            ),
            (
                lambda: torch.allclose(left, right, atol=-1e-8),
                RuntimeError,
                "atol must be greater than or equal to zero, but got -1e-08",
            ),
            (
                lambda: torch.allclose(left, right, rtol=float("nan")),
                RuntimeError,
                "rtol must be greater than or equal to zero, but got nan",
            ),
            (
                lambda: torch.allclose(left, right, out=None),
                TypeError,
                "allclose() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.allclose(left, right, dtype=torch.float32),
                TypeError,
                "allclose() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.allclose(left, right, device=torch.device("cpu")),
                TypeError,
                "allclose() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.allclose(left, right, input=left),
                TypeError,
                "allclose() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.allclose(left, right, x=left),
                TypeError,
                "allclose() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.allclose(left, right, x2=right),
                TypeError,
                "allclose() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.allclose(torch.ones((2, 3)), torch.ones((2, 2))),
                RuntimeError,
                "The size of tensor a (3) must match the size of tensor b (2) "
                "at non-singleton dimension 1",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch, "isclose"))
        self.assertFalse(hasattr(torch, "bool"))

    def test_no_mutation_or_autograd_side_effects(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        view = leaf.transpose(0, 1)[1]
        other = torch.tensor([2.0, 5.0])
        before = (
            view.tolist(),
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.data_ptr(),
            view.requires_grad,
            view.is_leaf,
            leaf.grad,
            other.tolist(),
            other.shape,
            other.stride(),
            other.storage_offset(),
            other.data_ptr(),
        )

        self.assertIs(torch.allclose(view, other), True)
        after = (
            view.tolist(),
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.data_ptr(),
            view.requires_grad,
            view.is_leaf,
            leaf.grad,
            other.tolist(),
            other.shape,
            other.stride(),
            other.storage_offset(),
            other.data_ptr(),
        )
        self.assertEqual(after, before)

    def test_torch_function_modes_and_overrides_are_observed(self):
        left = torch.tensor([1.0])
        right = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.allclose(left, right, rtol=0.0), marker)
        self.assertEqual(mode.calls, [(torch.allclose, (), (left, right), {"rtol": 0.0})])

        bad_mode = RecordingMode()
        with bad_mode:
            with self.assertRaisesRegex(
                TypeError,
                r"^allclose\(\): argument 'rtol' must be float, not NoneType$",
            ):
                torch.allclose(left, right, rtol=None)
        self.assertEqual(bad_mode.calls, [])

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((self.label, func, types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                result = torch.allclose(left, right, 0.0, 0.0)
        self.assertIs(result, True)
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.allclose for call in calls))
        self.assertTrue(all(call[2] == () for call in calls))
        self.assertTrue(all(call[3] == (left, right, 0.0, 0.0) for call in calls))

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.allclose(Override(), right), marker)
        self.assertEqual(len(override_calls), 1)
        self.assertIs(override_calls[0][0], torch.allclose)
        self.assertEqual(override_calls[0][1], (Override,))
        self.assertIs(torch.allclose(Override(), right, rtol=-1e-5), marker)

        class OptionalOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        optional = OptionalOverride()
        optional_mode = RecordingMode()
        with optional_mode:
            self.assertIs(torch.allclose(left, right, rtol=optional), marker)
        self.assertEqual(
            optional_mode.calls,
            [(torch.allclose, (OptionalOverride,), (left, right), {"rtol": optional})],
        )
        self.assertEqual(OptionalOverride.calls, [])

        for name in ("rtol", "atol", "equal_nan"):
            with self.subTest(keyword=name):
                OptionalOverride.calls.clear()
                optional = OptionalOverride()
                self.assertIs(torch.allclose(left, right, **{name: optional}), marker)
                self.assertEqual(len(OptionalOverride.calls), 1)
                func, types, args, kwargs = OptionalOverride.calls[0]
                self.assertIs(func, torch.allclose)
                self.assertEqual(types, (OptionalOverride,))
                self.assertEqual(args, (left, right))
                self.assertEqual(kwargs, {name: optional})

        positional_cases = (
            ((), lambda optional: torch.allclose(left, right, optional)),
            ((0.0,), lambda optional: torch.allclose(left, right, 0.0, optional)),
            (
                (0.0, 0.0),
                lambda optional: torch.allclose(left, right, 0.0, 0.0, optional),
            ),
        )
        for preceding, call in positional_cases:
            with self.subTest(position=len(preceding) + 3):
                OptionalOverride.calls.clear()
                optional = OptionalOverride()
                self.assertIs(call(optional), marker)
                self.assertEqual(len(OptionalOverride.calls), 1)
                func, types, args, kwargs = OptionalOverride.calls[0]
                self.assertIs(func, torch.allclose)
                self.assertEqual(types, (OptionalOverride,))
                self.assertEqual(args, (left, right, *preceding, optional))
                self.assertIsNone(kwargs)

        OptionalOverride.calls.clear()
        with self.assertRaisesRegex(
            TypeError, r"^allclose\(\) got an unexpected keyword argument 'out'$"
        ):
            torch.allclose(left, right, rtol=OptionalOverride(), out=None)
        self.assertEqual(OptionalOverride.calls, [])

    def test_callable_metadata_pickling_wildcard_and_reload(self):
        function = torch.allclose
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "allclose")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.allclose")
        self.assertEqual(function.__module__, "torch")
        self.assertIn(
            "allclose(input: Tensor, other: Tensor, rtol: float = 1e-05, "
            "atol: float = 1e-08, equal_nan: bool = False) -> bool",
            function.__doc__,
        )
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method allclose of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.allclose, function)
        for action in (
            lambda: setattr(owner, "allclose", None),
            lambda: delattr(owner, "allclose"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.allclose, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("allclose"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["allclose"], function)
        self.assertIs(importlib.reload(torch).allclose, function)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AllcloseReferenceTests(unittest.TestCase):
    def assert_allclose_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        actual_error = actual_raised.exception

        try:
            expected = expected_call()
        except Exception as expected_error:
            self.assertIs(type(actual_error), type(expected_error))
            self.assertEqual(str(actual_error), str(expected_error))
        else:
            raise AssertionError(
                f"actual call raised {actual_error!r}, but reference returned {expected!r}"
            )

    def assert_result_matches(self, actual_call, expected_call):
        actual = actual_call()
        expected = expected_call()
        self.assertIs(type(actual), bool)
        self.assertEqual(actual, expected)

    def test_values_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_matrix = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=reference_torch.float32,
        )
        actual_strided = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).transpose(0, 1)
        expected_strided = reference_torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            dtype=reference_torch.float32,
        ).transpose(0, 1)
        actual_offset = torch.tensor(
            [[9.0, 8.0, 7.0], [1.0, 2.0, 3.0]]
        )[1]
        expected_offset = reference_torch.tensor(
            [[9.0, 8.0, 7.0], [1.0, 2.0, 3.0]],
            dtype=reference_torch.float32,
        )[1]

        cases = (
            (
                lambda: torch.allclose(torch.tensor(1.0), torch.tensor(1.0)),
                lambda: reference_torch.allclose(
                    reference_torch.tensor(1.0, dtype=reference_torch.float32),
                    reference_torch.tensor(1.0, dtype=reference_torch.float32),
                ),
            ),
            (
                lambda: torch.allclose(actual_matrix, actual_matrix.clone()),
                lambda: reference_torch.allclose(expected_matrix, expected_matrix.clone()),
            ),
            (
                lambda: torch.allclose(
                    actual_matrix,
                    torch.tensor([[1.001, 2.001, 3.001], [4.001, 5.001, 6.001]]),
                ),
                lambda: reference_torch.allclose(
                    expected_matrix,
                    reference_torch.tensor(
                        [[1.001, 2.001, 3.001], [4.001, 5.001, 6.001]],
                        dtype=reference_torch.float32,
                    ),
                ),
            ),
            (
                lambda: torch.allclose(
                    torch.tensor([[1.0], [2.0]]),
                    torch.tensor([[1.0, 1.0, 1.0]]),
                    rtol=1.0,
                    atol=0.0,
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([[1.0], [2.0]], dtype=reference_torch.float32),
                    reference_torch.tensor([[1.0, 1.0, 1.0]], dtype=reference_torch.float32),
                    rtol=1.0,
                    atol=0.0,
                ),
            ),
            (
                lambda: torch.allclose(torch.zeros((2, 0)), torch.ones((1,))),
                lambda: reference_torch.allclose(
                    reference_torch.zeros((2, 0), dtype=reference_torch.float32),
                    reference_torch.ones((1,), dtype=reference_torch.float32),
                ),
            ),
            (
                lambda: torch.allclose(actual_strided, actual_matrix),
                lambda: reference_torch.allclose(expected_strided, expected_matrix),
            ),
            (
                lambda: torch.allclose(actual_offset, torch.tensor([1.0, 2.0, 3.0])),
                lambda: reference_torch.allclose(
                    expected_offset,
                    reference_torch.tensor([1.0, 2.0, 3.0], dtype=reference_torch.float32),
                ),
            ),
            (
                lambda: torch.allclose(
                    torch.tensor([0.0, -0.0]),
                    torch.tensor([-0.0, 0.0]),
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([0.0, -0.0], dtype=reference_torch.float32),
                    reference_torch.tensor([-0.0, 0.0], dtype=reference_torch.float32),
                ),
            ),
            (
                lambda: torch.allclose(
                    torch.tensor([float("nan")]),
                    torch.tensor([float("nan")]),
                    equal_nan=True,
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([float("nan")], dtype=reference_torch.float32),
                    reference_torch.tensor([float("nan")], dtype=reference_torch.float32),
                    equal_nan=True,
                ),
            ),
            (
                lambda: torch.allclose(
                    torch.tensor([float("inf")]),
                    torch.tensor([1.0]),
                    rtol=float("inf"),
                    atol=float("inf"),
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([float("inf")], dtype=reference_torch.float32),
                    reference_torch.tensor([1.0], dtype=reference_torch.float32),
                    rtol=float("inf"),
                    atol=float("inf"),
                ),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_result_matches(actual_call, expected_call)

    def test_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.allclose(), lambda: reference_torch.allclose()),
            (lambda: torch.allclose(actual), lambda: reference_torch.allclose(expected)),
            (
                lambda: torch.allclose(actual, np.array([1.0], dtype=np.float32)),
                lambda: reference_torch.allclose(
                    expected, np.array([1.0], dtype=np.float32)
                ),
            ),
            (
                lambda: torch.allclose(actual, actual, out=None),
                lambda: reference_torch.allclose(expected, expected, out=None),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=None),
                lambda: reference_torch.allclose(expected, expected, rtol=None),
            ),
            (
                lambda: torch.allclose(actual, actual, equal_nan=1),
                lambda: reference_torch.allclose(expected, expected, equal_nan=1),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=-1e-5),
                lambda: reference_torch.allclose(expected, expected, rtol=-1e-5),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=index):
                self.assert_allclose_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
