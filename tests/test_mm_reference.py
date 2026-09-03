import copy
import importlib
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
class TopLevelMmReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.mm differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

        actual_values = np.asarray(actual)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, classifications=True):
            np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(expected_values))
            non_nan = ~np.isnan(expected_values)
            np.testing.assert_array_equal(
                np.signbit(actual_values[non_nan]), np.signbit(expected_values[non_nan])
            )
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def layout_cases(self, module):
        offset_left = module.tensor(
            np.arange(24, dtype=np.float32).reshape(3, 2, 4).tolist(),
            dtype=module.float32,
        )[1]
        offset_right = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            dtype=module.float32,
        )[1]
        strided_left = module.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
        ).transpose(0, 1)
        strided_right = module.tensor(
            [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]], dtype=module.float32
        ).transpose(0, 1)
        return (
            (
                "square",
                module.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
                ),
                module.tensor(
                    [[5.0, 6.0], [7.0, 8.0]], dtype=module.float32
                ),
            ),
            (
                "rectangular",
                module.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
                ),
                module.tensor(
                    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
                    dtype=module.float32,
                ),
            ),
            (
                "empty rows",
                module.zeros((0, 3), dtype=module.float32),
                module.ones((3, 2), dtype=module.float32),
            ),
            (
                "empty inner",
                module.ones((2, 0), dtype=module.float32),
                module.zeros((0, 3), dtype=module.float32),
            ),
            ("offset", offset_left, offset_right),
            ("noncontiguous", strided_left, strided_right),
            (
                "signed zero",
                module.tensor(
                    [[-0.0, 0.0], [0.0, -0.0]], dtype=module.float32
                ),
                module.tensor(
                    [[1.0, -1.0], [-1.0, 1.0]], dtype=module.float32
                ),
            ),
            (
                "nan and infinity",
                module.tensor(
                    [
                        [float("inf"), 1.0],
                        [float("-inf"), -1.0],
                        [float("nan"), 2.0],
                    ],
                    dtype=module.float32,
                ),
                module.tensor(
                    [[1.0, -1.0], [0.5, 1.0]], dtype=module.float32
                ),
            ),
        )

    def test_rank_two_results_layouts_and_edge_values_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            calls = (
                (
                    "positional",
                    lambda: torch.mm(actual_left, actual_right),
                    lambda: reference_torch.mm(expected_left, expected_right),
                ),
                (
                    "canonical keywords",
                    lambda: torch.mm(input=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(input=expected_left, mat2=expected_right),
                ),
                (
                    "x alias",
                    lambda: torch.mm(x=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(x=expected_left, mat2=expected_right),
                ),
                (
                    "a alias",
                    lambda: torch.mm(a=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(a=expected_left, mat2=expected_right),
                ),
                (
                    "x1 alias",
                    lambda: torch.mm(x1=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(x1=expected_left, mat2=expected_right),
                ),
                (
                    "positional mat2 keyword",
                    lambda: torch.mm(actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(expected_left, mat2=expected_right),
                ),
                (
                    "out none",
                    lambda: torch.mm(actual_left, actual_right, out=None),
                    lambda: reference_torch.mm(expected_left, expected_right, out=None),
                ),
            )
            for style, actual_call, expected_call in calls:
                self.assert_matches(actual_call(), expected_call(), case=(case, style))

    def dispatch_observation(self, module):
        left = module.tensor([[1.0]], dtype=module.float32)
        right = module.tensor([[2.0]], dtype=module.float32)
        function = module.mm
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            (lambda: function(left, right), None),
            (lambda: function(input=left, mat2=right), ("input", "mat2")),
            (lambda: function(x=left, mat2=right), ("x", "mat2")),
            (lambda: function(left, right, out=None), ("out",)),
        )
        for call, keywords in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    len(args) == 2 and args[0] is left and args[1] is right,
                    kwargs is None,
                    kwargs is not None
                    and tuple(kwargs) == keywords
                    and (
                        keywords == ("out",)
                        or kwargs[keywords[0]] is left and kwargs[keywords[1]] is right
                    ),
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value, right), None),
            (lambda value: function(left, value), None),
            (lambda value: function(input=left, mat2=value), "mat2"),
            (lambda value: function(left, right, out=value), "out"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (Override,),
                    len(args),
                    kwargs is None,
                    kwargs is not None and keyword in kwargs and kwargs[keyword] is value,
                )
            )

        return mode_observations, override_observations

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_supported_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([[1.0]], dtype=torch.float32)
        expected = reference_torch.tensor([[1.0]], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.mm(actual), lambda: reference_torch.mm(expected)),
            (
                lambda: torch.mm([], actual),
                lambda: reference_torch.mm([], expected),
            ),
            (
                lambda: torch.mm(actual, []),
                lambda: reference_torch.mm(expected, []),
            ),
            (
                lambda: torch.mm(actual, mat2=[]),
                lambda: reference_torch.mm(expected, mat2=[]),
            ),
            (
                lambda: torch.mm(actual, actual, extra=True),
                lambda: reference_torch.mm(expected, expected, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module):
        function = module.mm
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
            "owner_callable_identity": owner.mm is function,
            "import_identity": importlib.import_module(module.__name__).mm is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("mm"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["mm"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_import_wildcard_copy_and_pickle_match_pytorch_2_13(
        self,
    ):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

    def test_reload_preserves_actual_mm_callable(self):
        function = torch.mm
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.mm, function)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.mm, function)

    def test_deliberately_unsupported_boundaries_remain_explicit(self):
        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        result = torch.mm(left, right)
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)

        destination = torch.full((1, 1), 17.0)
        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(left, right, out=destination)
        self.assertEqual(destination.tolist(), [[17.0]])

        with self.assertRaisesRegex(TypeError, "received an invalid combination"):
            torch.mm(left, right, torch.float32)
        for name in ("bmm", "addmm", "mv"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))

        for input, mat2 in (
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
        ):
            with self.subTest(input=input.shape, mat2=mat2.shape):
                with self.assertRaises(RuntimeError) as raised:
                    torch.mm(input, mat2)
                self.assertIn("requires two rank-2 tensors", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
