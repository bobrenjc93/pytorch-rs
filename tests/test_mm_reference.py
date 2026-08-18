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
        np.arange(24, dtype=np.float32).reshape(4, 2, 3).tolist()
    )[1]
    offset_right = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 3, 2).tolist()
    )[1]
    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]
    ).transpose(0, 1)

    return (
        (
            "contiguous",
            module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            module.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        ),
        ("offset contiguous", offset_left, offset_right),
        ("strided", strided_left, strided_right),
        ("empty rows", module.zeros((0, 3)), module.ones((3, 4))),
        ("empty inner", module.ones((2, 0)), module.zeros((0, 3))),
        ("empty columns", module.ones((2, 3)), module.zeros((3, 0))),
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MmReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.mm differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_rank_two_layouts_values_and_aliases_match_pytorch_2_13(self):
        actual_cases = mm_layout_cases(torch)
        expected_cases = mm_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
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
                    lambda: reference_torch.mm(
                        input=expected_left, mat2=expected_right
                    ),
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
            )
            for style, actual_call, expected_call in calls:
                self.assert_matches(
                    actual_call(), expected_call(), case=(case, style)
                )

    def test_shape_and_non_matrix_errors_match_pytorch_2_13(self):
        shape_cases = (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        )
        for left_shape, right_shape in shape_cases:
            actual_left = torch.zeros(left_shape)
            actual_right = torch.zeros(right_shape)
            expected_left = reference_torch.zeros(left_shape)
            expected_right = reference_torch.zeros(right_shape)
            for actual_call, expected_call in (
                (
                    lambda: torch.mm(actual_left, actual_right),
                    lambda: reference_torch.mm(expected_left, expected_right),
                ),
                (
                    lambda: torch.mm(input=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(
                        input=expected_left, mat2=expected_right
                    ),
                ),
            ):
                with self.subTest(left=left_shape, right=right_shape):
                    self.assert_error_matches(actual_call, expected_call)

        rank_cases = (
            ((), (1, 1)),
            ((2,), (2, 2)),
            ((2, 2), (2,)),
            ((1, 2, 2), (2, 2)),
            ((2, 2), (1, 2, 2)),
        )
        for left_shape, right_shape in rank_cases:
            actual_left = torch.ones(left_shape)
            actual_right = torch.ones(right_shape)
            expected_left = reference_torch.ones(left_shape)
            expected_right = reference_torch.ones(right_shape)
            with self.subTest(left=left_shape, right=right_shape):
                self.assert_error_matches(
                    lambda: torch.mm(actual_left, actual_right),
                    lambda: reference_torch.mm(expected_left, expected_right),
                )

    def dispatch_observation(self, module):
        left = module.tensor([[1.0]])
        right = module.tensor([[2.0]])
        vector = module.tensor([1.0])
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
            (lambda: function(vector, right), None),
            (lambda: function(left, right, out=left), ("out",)),
            (
                lambda: function(left, right, out_dtype=module.float32),
                ("out_dtype",),
            ),
            (
                lambda: function(
                    input=left, mat2=right, out_dtype=module.float32
                ),
                ("input", "mat2", "out_dtype"),
            ),
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
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keywords,
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
            (
                lambda value: function(
                    value, right, out_dtype=module.float32
                ),
                None,
            ),
            (lambda value: function(left, right, out_dtype=value), "out_dtype"),
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

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), DerivedOverride())

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=left, mat2=right)

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
            lambda: function(left),
            lambda: function(left, right, extra=True),
            lambda: function(
                left,
                right,
                module.float32,
                out_dtype=module.float32,
            ),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            tuple(np.asarray(forwarded).reshape(-1)),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_torch_function_modes_and_overrides_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.ones((1, 1))
        expected = reference_torch.ones((1, 1))
        cases = (
            (lambda: torch.mm(), lambda: reference_torch.mm()),
            (lambda: torch.mm(actual), lambda: reference_torch.mm(expected)),
            (
                lambda: torch.mm(actual, actual, actual),
                lambda: reference_torch.mm(expected, expected, expected),
            ),
            (lambda: torch.mm([], actual), lambda: reference_torch.mm([], expected)),
            (lambda: torch.mm(actual, []), lambda: reference_torch.mm(expected, [])),
            (
                lambda: torch.mm(input=None, mat2=actual),
                lambda: reference_torch.mm(input=None, mat2=expected),
            ),
            (
                lambda: torch.mm(input=actual, mat2=None),
                lambda: reference_torch.mm(input=expected, mat2=None),
            ),
            (
                lambda: torch.mm(input=actual, other=actual),
                lambda: reference_torch.mm(input=expected, other=expected),
            ),
            (
                lambda: torch.mm(foo=actual, mat2=actual),
                lambda: reference_torch.mm(foo=expected, mat2=expected),
            ),
            (
                lambda: torch.mm(x1=actual, x2=actual),
                lambda: reference_torch.mm(x1=expected, x2=expected),
            ),
            (
                lambda: torch.mm(actual, actual, input=actual),
                lambda: reference_torch.mm(expected, expected, input=expected),
            ),
            (
                lambda: torch.mm(actual, actual, mat2=actual),
                lambda: reference_torch.mm(expected, expected, mat2=expected),
            ),
            (
                lambda: torch.mm(actual, actual, extra=True),
                lambda: reference_torch.mm(expected, expected, extra=True),
            ),
            (
                lambda: torch.mm([], actual, extra=True),
                lambda: reference_torch.mm([], expected, extra=True),
            ),
            (
                lambda: torch.mm(actual, actual, out=[]),
                lambda: reference_torch.mm(expected, expected, out=[]),
            ),
            (
                lambda: torch.mm(actual, actual, out_dtype=1),
                lambda: reference_torch.mm(expected, expected, out_dtype=1),
            ),
            (
                lambda: torch.mm(
                    actual,
                    actual,
                    torch.float32,
                    out_dtype=torch.float32,
                ),
                lambda: reference_torch.mm(
                    expected,
                    expected,
                    reference_torch.float32,
                    out_dtype=reference_torch.float32,
                ),
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("mm"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["mm"] is function,
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
