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


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
        dtype=module.float32,
    )[1]
    offset_mat2 = module.tensor(
        np.arange(40, dtype=np.float32).reshape(2, 4, 5).tolist(),
        dtype=module.float32,
    )[1]

    noncontiguous_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
    ).transpose(0, 1)
    noncontiguous_mat2 = module.tensor(
        [[7.0, 10.0, 13.0], [8.0, 11.0, 14.0], [9.0, 12.0, 15.0]],
        dtype=module.float32,
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
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
            ),
            module.tensor(
                [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
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
        ("offset", offset_left, offset_mat2),
        ("noncontiguous", noncontiguous_left, noncontiguous_mat2),
        (
            "signed zero",
            module.tensor([[-0.0, 0.0], [0.0, -0.0]], dtype=module.float32),
            module.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=module.float32),
        ),
        (
            "nan and inf",
            module.tensor(
                [[float("inf"), 1.0], [float("-inf"), -1.0], [float("nan"), 2.0]],
                dtype=module.float32,
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]], dtype=module.float32),
        ),
    )


def invalid_mm_overload(summary):
    return (
        "mm() received an invalid combination of arguments - got "
        f"({summary}), but expected one of:\n"
        " * (Tensor input, Tensor mat2, *, Tensor out = None)\n"
        " * (Tensor input, Tensor mat2, torch.dtype out_dtype, *, Tensor out = None)\n"
    )


def first_operand_alias_pairs():
    aliases = ("input", "x", "a", "x1")
    for index, left_alias in enumerate(aliases):
        for right_alias in aliases[index + 1 :]:
            yield left_alias, right_alias


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

    def test_rank_two_results_layouts_and_edges_match_pytorch_2_13(self):
        actual_cases = mm_layout_cases(torch)
        expected_cases = mm_layout_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_mat2 = actual_case
            expected_name, expected_left, expected_mat2 = expected_case
            self.assertEqual(case, expected_name)
            calls = (
                (
                    "positional",
                    lambda: torch.mm(actual_left, actual_mat2),
                    lambda: reference_torch.mm(expected_left, expected_mat2),
                ),
                (
                    "canonical keywords",
                    lambda: torch.mm(input=actual_left, mat2=actual_mat2),
                    lambda: reference_torch.mm(input=expected_left, mat2=expected_mat2),
                ),
                (
                    "x alias",
                    lambda: torch.mm(x=actual_left, mat2=actual_mat2),
                    lambda: reference_torch.mm(x=expected_left, mat2=expected_mat2),
                ),
                (
                    "a alias",
                    lambda: torch.mm(a=actual_left, mat2=actual_mat2),
                    lambda: reference_torch.mm(a=expected_left, mat2=expected_mat2),
                ),
                (
                    "x1 alias",
                    lambda: torch.mm(x1=actual_left, mat2=actual_mat2),
                    lambda: reference_torch.mm(x1=expected_left, mat2=expected_mat2),
                ),
                (
                    "out none",
                    lambda: torch.mm(actual_left, actual_mat2, out=None),
                    lambda: reference_torch.mm(expected_left, expected_mat2, out=None),
                ),
            )
            for style, actual_call, expected_call in calls:
                self.assert_matches(actual_call(), expected_call(), case=(case, style))

    def test_rank_and_shape_errors_match_pytorch_2_13(self):
        for actual_call, expected_call in (
            (
                lambda: torch.mm(torch.zeros((2, 3)), torch.zeros((4, 2))),
                lambda: reference_torch.mm(
                    reference_torch.zeros((2, 3)), reference_torch.zeros((4, 2))
                ),
            ),
            (
                lambda: torch.mm(torch.tensor(1.0), torch.ones((1, 1))),
                lambda: reference_torch.mm(
                    reference_torch.tensor(1.0), reference_torch.ones((1, 1))
                ),
            ),
            (
                lambda: torch.mm(torch.ones((2,)), torch.ones((2, 2))),
                lambda: reference_torch.mm(
                    reference_torch.ones((2,)), reference_torch.ones((2, 2))
                ),
            ),
            (
                lambda: torch.mm(torch.ones((1, 2, 2)), torch.ones((2, 2))),
                lambda: reference_torch.mm(
                    reference_torch.ones((1, 2, 2)), reference_torch.ones((2, 2))
                ),
            ),
            (
                lambda: torch.mm(torch.ones((1, 1)), torch.ones((1,))),
                lambda: reference_torch.mm(
                    reference_torch.ones((1, 1)), reference_torch.ones((1,))
                ),
            ),
            (
                lambda: torch.mm(torch.ones((1, 1)), torch.ones((1, 1, 1))),
                lambda: reference_torch.mm(
                    reference_torch.ones((1, 1)), reference_torch.ones((1, 1, 1))
                ),
            ),
            (
                lambda: torch.mm([], torch.ones((1, 1))),
                lambda: reference_torch.mm([], reference_torch.ones((1, 1))),
            ),
            (
                lambda: torch.mm(torch.ones((1, 1)), []),
                lambda: reference_torch.mm(reference_torch.ones((1, 1)), []),
            ),
            (
                lambda: torch.mm(input=None, mat2=torch.ones((1, 1))),
                lambda: reference_torch.mm(
                    input=None, mat2=reference_torch.ones((1, 1))
                ),
            ),
            (
                lambda: torch.mm(input=torch.ones((1, 1)), other=torch.ones((1, 1))),
                lambda: reference_torch.mm(
                    input=reference_torch.ones((1, 1)),
                    other=reference_torch.ones((1, 1)),
                ),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def dispatch_observation(self, module):
        left = module.tensor([[1.0]], dtype=module.float32)
        mat2 = module.tensor([[2.0]], dtype=module.float32)
        out = module.tensor([[0.0]], dtype=module.float32)
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
            (lambda: function(left, mat2), None, 2),
            (lambda: function(input=left, mat2=mat2), ("input", "mat2"), 0),
            (lambda: function(left, mat2, out=None), ("out",), 2),
            (lambda: function(input=left, mat2=mat2, out=out), ("input", "mat2", "out"), 0),
        )
        for call, keywords, positional_count in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args) == positional_count,
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == keywords,
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
            (lambda value: function(value, mat2), None),
            (lambda value: function(left, value), None),
            (lambda value: function(input=left, mat2=value), "mat2"),
            (lambda value: function(left, mat2, out=value), "out"),
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

        out_order = []

        class Mat2Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_order.append(("mat2", tuple(item.__name__ for item in types)))
                return NotImplemented

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_order.append(("out", tuple(item.__name__ for item in types)))
                return marker

        out_result = function(left, Mat2Override(), out=OutOverride())

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(("base", tuple(item.__name__ for item in types)))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), DerivedOverride())

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(input=left, mat2=FallbackOverride())

        class InvalidOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        invalid_observations = []
        invalid_calls = [
            lambda: function([], mat2),
            lambda: function(left, []),
            lambda: function(left, mat2, unexpected=True),
        ]
        for left_alias, right_alias in first_operand_alias_pairs():
            invalid_calls.append(
                lambda left_alias=left_alias, right_alias=right_alias: function(
                    **{left_alias: left, right_alias: left, "mat2": mat2}
                )
            )
            invalid_calls.append(
                lambda left_alias=left_alias, right_alias=right_alias: function(
                    **{left_alias: InvalidOverride(), right_alias: left, "mat2": mat2}
                )
            )
            invalid_calls.append(
                lambda left_alias=left_alias, right_alias=right_alias: function(
                    **{left_alias: left, right_alias: InvalidOverride(), "mat2": mat2}
                )
            )
        for call in invalid_calls:
            invalid_mode = RecordingMode()
            InvalidOverride.calls.clear()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (
                        type(error).__name__,
                        str(error),
                        len(invalid_mode.calls),
                        len(InvalidOverride.calls),
                    )
                )

        return (
            mode_observations,
            override_observations,
            both_result is marker,
            order,
            out_result is marker,
            out_order,
            subclass_result is marker,
            subclass_order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(self.dispatch_observation(torch), self.dispatch_observation(reference_torch))

    def callable_contract(self, module):
        function = module.mm
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        direct_namespace = {}
        exec(f"from {module.__name__} import mm", direct_namespace)
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
            "direct_import_identity": direct_namespace["mm"] is function,
            "wildcard_identity": wildcard_namespace["mm"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

        old = torch.mm
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.mm, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.mm, old)

    def test_out_dtype_bmm_addmm_and_concrete_out_remain_out_of_scope(self):
        actual_destination = torch.tensor([[17.0]], dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, r"^mm\(\): the 'out' argument is not supported$"
        ):
            torch.mm(
                torch.ones((1, 1), dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.float32),
                out=actual_destination,
            )
        self.assertEqual(actual_destination.tolist(), [[17.0]])

        with self.assertRaisesRegex(
            TypeError,
            f"^{re.escape(invalid_mm_overload('Tensor, Tensor, out_dtype=torch.dtype'))}$",
        ):
            torch.mm(
                torch.ones((1, 1), dtype=torch.float32),
                torch.ones((1, 1), dtype=torch.float32),
                out_dtype=torch.float32,
            )

        for name in ("bmm", "addmm"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
