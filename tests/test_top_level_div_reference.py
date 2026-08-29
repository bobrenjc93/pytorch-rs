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
class TopLevelDivReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.div differentials require pinned PyTorch 2.13.0")

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
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_values_layouts_ieee_empties_and_canonical_keywords_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])

        calls = (
            (
                "positional tensors",
                lambda: torch.div(actual_left, actual_right),
                lambda: reference_torch.div(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.div(input=actual_left, other=actual_right),
                lambda: reference_torch.div(
                    input=expected_left, other=expected_right
                ),
            ),
            (
                "x/other aliases",
                lambda: torch.div(x=actual_left, other=actual_right),
                lambda: reference_torch.div(x=expected_left, other=expected_right),
            ),
            (
                "a/other aliases",
                lambda: torch.div(a=actual_left, other=actual_right),
                lambda: reference_torch.div(a=expected_left, other=expected_right),
            ),
            (
                "x1/x2 aliases",
                lambda: torch.div(x1=actual_left, x2=actual_right),
                lambda: reference_torch.div(x1=expected_left, x2=expected_right),
            ),
            (
                "input/x2 aliases",
                lambda: torch.div(input=actual_left, x2=actual_right),
                lambda: reference_torch.div(input=expected_left, x2=expected_right),
            ),
            (
                "none options",
                lambda: torch.div(
                    input=actual_left,
                    other=actual_right,
                    rounding_mode=None,
                    out=None,
                ),
                lambda: reference_torch.div(
                    input=expected_left,
                    other=expected_right,
                    rounding_mode=None,
                    out=None,
                ),
            ),
            (
                "tensor/scalar",
                lambda: torch.div(actual_left[1], np.float32(-0.0)),
                lambda: reference_torch.div(expected_left[1], np.float32(-0.0)),
            ),
            (
                "keyword tensor/scalar",
                lambda: torch.div(input=actual_left[1], other=-2.5),
                lambda: reference_torch.div(input=expected_left[1], other=-2.5),
            ),
            (
                "alias tensor/scalar",
                lambda: torch.div(x=actual_left[1], other=-2.5),
                lambda: reference_torch.div(x=expected_left[1], other=-2.5),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.div(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.div(
                expected_empty, reference_torch.ones((1, 1, 2))
            ),
            case="strided broadcast empty",
        )

        numerators = np.asarray(
            (0x3F80_0000, 0xBF80_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        denominators = np.asarray(
            (0x8000_0000, 0x0000_0000, 0x7F80_0000, 0xFF80_0000, 0x3F80_0000),
            dtype=np.uint32,
        )
        actual_values = memoryview(numerators.view(np.float32))
        expected_values = memoryview(numerators.view(np.float32))
        actual_denominators = memoryview(denominators.view(np.float32))
        expected_denominators = memoryview(denominators.view(np.float32))
        self.assert_matches(
            torch.div(torch.tensor(actual_values), torch.tensor(actual_denominators)),
            reference_torch.div(
                reference_torch.tensor(expected_values),
                reference_torch.tensor(expected_denominators),
            ),
            case="IEEE special values",
        )

    def test_detached_and_no_grad_inputs_match_pytorch_2_13(self):
        actual_left = torch.tensor([[2.0, 4.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 4.0]], requires_grad=True)
        actual_right = torch.tensor([[1.0], [2.0]], requires_grad=True)
        expected_right = reference_torch.tensor([[1.0], [2.0]], requires_grad=True)

        self.assert_matches(
            torch.div(actual_left.detach(), actual_right.detach()),
            reference_torch.div(expected_left.detach(), expected_right.detach()),
            case="detached",
        )
        with torch.no_grad():
            actual = torch.div(actual_left, actual_right)
        with reference_torch.no_grad():
            expected = reference_torch.div(expected_left, expected_right)
        self.assert_matches(actual, expected, case="no_grad")

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([6.0])
        right = module.tensor([3.0])
        destination = module.zeros((1,))
        function = module.div
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 3.0), None),
            (lambda: function(x1=left, x2=right), ("x1", "x2")),
            (
                lambda: function(
                    input=left, other=right, rounding_mode=None, out=None
                ),
                ("input", "other", "rounding_mode", "out"),
            ),
            (lambda: function(left, right, out=destination), ("out",)),
            (
                lambda: function(left, right, rounding_mode="trunc"),
                ("rounding_mode",),
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

        for call, expected_args, expected_keywords in (
            (lambda value: function(value, right), 2, None),
            (lambda value: function(left, value), 2, None),
            (lambda value: function(input=left, other=value), 0, ("input", "other")),
            (lambda value: function(left, right, out=value), 2, ("out",)),
            (
                lambda value: function(left, right, rounding_mode=value),
                2,
                ("rounding_mode",),
            ),
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
                    len(args) == expected_args,
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == expected_keywords,
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

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_order.append(("base", tuple(item.__name__ for item in types)))
                return NotImplemented

        class DerivedOutOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_order.append(("derived", tuple(item.__name__ for item in types)))
                return marker

        out_order_result = function(BaseOverride(), right, out=DerivedOutOverride())

        return (
            mode_observations,
            override_observations,
            both_result is marker,
            order,
            out_order_result is marker,
            out_order,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        function = module.div
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
            "owner_callable_identity": owner.div is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("div"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["div"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def test_shared_error_cases_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.div([], actual), lambda: reference_torch.div([], expected)),
            (lambda: torch.div(actual, []), lambda: reference_torch.div(expected, [])),
            (
                lambda: torch.div(input=None, other=actual),
                lambda: reference_torch.div(input=None, other=expected),
            ),
            (
                lambda: torch.div(actual, np.uint64(2**63)),
                lambda: reference_torch.div(expected, np.uint64(2**63)),
            ),
            (
                lambda: torch.div(actual, 2**64),
                lambda: reference_torch.div(expected, 2**64),
            ),
            (
                lambda: torch.div(actual, -(2**63) - 1),
                lambda: reference_torch.div(expected, -(2**63) - 1),
            ),
            (
                lambda: torch.div(torch.zeros((2, 3)), torch.zeros((4, 2))),
                lambda: reference_torch.div(
                    reference_torch.zeros((2, 3)),
                    reference_torch.zeros((4, 2)),
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
