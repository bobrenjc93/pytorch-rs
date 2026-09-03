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
            np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist(),
            dtype=module.float32,
        )[1]
        offset_right = module.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist(),
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
                module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32),
                module.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=module.float32),
            ),
            (
                "rectangular",
                module.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=module.float32,
                ),
                module.tensor(
                    [
                        [7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0],
                    ],
                    dtype=module.float32,
                ),
            ),
            (
                "empty rows",
                module.zeros((0, 3), dtype=module.float32),
                module.ones((3, 2), dtype=module.float32),
            ),
            (
                "empty columns",
                module.ones((2, 3), dtype=module.float32),
                module.zeros((3, 0), dtype=module.float32),
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
                module.tensor([[-0.0, 0.0], [0.0, -0.0]], dtype=module.float32),
                module.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=module.float32),
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
                module.tensor([[1.0, -1.0], [0.5, 1.0]], dtype=module.float32),
            ),
        )

    def test_rank_two_values_layouts_edges_and_out_none_match_pytorch_2_13(self):
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
                    "explicit out none",
                    lambda: torch.mm(actual_left, actual_right, out=None),
                    lambda: reference_torch.mm(expected_left, expected_right, out=None),
                ),
                (
                    "input alias x",
                    lambda: torch.mm(x=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(x=expected_left, mat2=expected_right),
                ),
                (
                    "input alias a",
                    lambda: torch.mm(a=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(a=expected_left, mat2=expected_right),
                ),
                (
                    "input alias x1",
                    lambda: torch.mm(x1=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(x1=expected_left, mat2=expected_right),
                ),
            )
            for style, actual_call, expected_call in calls:
                self.assert_matches(actual_call(), expected_call(), case=(case, style))

    def test_rank_and_shape_errors_match_pytorch_2_13(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
            ((), (1, 1)),
            ((2,), (2, 2)),
            ((2, 2), (2,)),
            ((1, 2, 2), (2, 2)),
            ((2, 2), (1, 2, 2)),
        ):
            actual_left = torch.zeros(left_shape)
            actual_right = torch.zeros(right_shape)
            expected_left = reference_torch.zeros(left_shape)
            expected_right = reference_torch.zeros(right_shape)
            with self.subTest(left=left_shape, right=right_shape):
                self.assert_error_matches(
                    lambda: torch.mm(actual_left, actual_right),
                    lambda: reference_torch.mm(expected_left, expected_right),
                )

    def dispatch_observation(self, module):
        left = module.tensor([[1.0]])
        right = module.tensor([[2.0]])
        destination = module.tensor([[0.0]])
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

        for call, keyword_names in (
            (lambda: function(left, right), None),
            (lambda: function(input=left, mat2=right), ("input", "mat2")),
            (lambda: function(left, right, out=None), ("out",)),
            (lambda: function(left, right, out=destination), ("out",)),
        ):
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
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
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
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        order = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("left", tuple(item.__name__ for item in types)))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("right", tuple(item.__name__ for item in types)))
                return marker

        both_result = function(LeftOverride(), RightOverride())

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(input=left, mat2=FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, unexpected=True),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )
            else:
                invalid_observations.append(None)

        return (
            mode_observations,
            override_observations,
            both_result is marker,
            order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        function = module.mm
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        object_import = {}
        exec(f"from {module.__name__} import mm", object_import)
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("mm"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "object_import_identity": object_import["mm"] is function,
            "wildcard_identity": wildcard_namespace["mm"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def reload_contract(self, module):
        native = module._C
        function = module.mm
        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module.mm is function,
            module.mm(module.tensor([[1.0]]), module.tensor([[2.0]])).tolist(),
        )

    def test_callable_metadata_import_reload_copy_and_pickle_match_pytorch_2_13(
        self,
    ):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
