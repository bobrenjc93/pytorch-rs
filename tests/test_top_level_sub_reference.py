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
class TopLevelSubReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.sub differentials require pinned PyTorch 2.13.0")

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

    def test_values_layouts_ieee_empties_and_argument_forms_match_pytorch_2_13(self):
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
                lambda: torch.sub(actual_left, actual_right),
                lambda: reference_torch.sub(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.sub(input=actual_left, other=actual_right),
                lambda: reference_torch.sub(
                    input=expected_left, other=expected_right
                ),
            ),
            (
                "x/x2 aliases",
                lambda: torch.sub(x=actual_left, x2=actual_right),
                lambda: reference_torch.sub(x=expected_left, x2=expected_right),
            ),
            (
                "x1/x2 aliases",
                lambda: torch.sub(x1=actual_left, x2=actual_right),
                lambda: reference_torch.sub(x1=expected_left, x2=expected_right),
            ),
            (
                "alpha int one",
                lambda: torch.sub(actual_left, actual_right, alpha=1),
                lambda: reference_torch.sub(expected_left, expected_right, alpha=1),
            ),
            (
                "alpha numpy float one",
                lambda: torch.sub(actual_left, actual_right, alpha=np.float32(1.0)),
                lambda: reference_torch.sub(
                    expected_left, expected_right, alpha=np.float32(1.0)
                ),
            ),
            (
                "out none",
                lambda: torch.sub(actual_left, actual_right, out=None),
                lambda: reference_torch.sub(expected_left, expected_right, out=None),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.sub(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.sub(
                expected_empty, reference_torch.ones((1, 1, 2))
            ),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.sub(torch.tensor(values), torch.zeros((5,))),
            reference_torch.sub(
                reference_torch.tensor(values), reference_torch.zeros((5,))
            ),
            case="signed zero and non-finites",
        )

    def test_autograd_shared_operands_empties_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )
        actual_output = torch.sub(
            actual_left.transpose(0, 1), actual_right.transpose(0, 1)
        )
        expected_output = reference_torch.sub(
            expected_left.transpose(0, 1), expected_right.transpose(0, 1)
        )
        self.assert_matches(actual_output, expected_output, case="tracked views")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.sub(actual_shared, actual_shared).sum().backward()
        reference_torch.sub(expected_shared, expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.sub(actual_empty, torch.ones((1, 1, 3))).sum().backward()
        reference_torch.sub(
            expected_empty, reference_torch.ones((1, 1, 3))
        ).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.sub(
                actual_no_grad.transpose(0, 1), torch.ones((1, 2))
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sub(
                expected_no_grad.transpose(0, 1), reference_torch.ones((1, 2))
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(torch.sub(actual_no_grad, torch.ones((1, 2))).requires_grad)
        self.assertTrue(
            reference_torch.sub(expected_no_grad, reference_torch.ones((1, 2))).requires_grad
        )

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        destination = module.tensor([17.0])
        function = module.sub
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
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
            (lambda: function(input=left, other=right), ("input", "other")),
            (lambda: function(x=left, x2=right, alpha=1), ("x", "x2", "alpha")),
            (lambda: function(left, right, alpha=2), ("alpha",)),
            (lambda: function(left, right, out=destination), ("out",)),
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

        for call in (
            lambda value: function(value, right),
            lambda value: function(left, value),
            lambda value: function(left, right, alpha=value),
            lambda value: function(left, right, out=value),
            lambda value: function(input=left, other=value),
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
                    kwargs is not None and tuple(kwargs) in (("alpha",), ("out",), ("input", "other")),
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
                return NotImplemented

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("alpha", tuple(item.__name__ for item in types)))
                return NotImplemented

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("out", tuple(item.__name__ for item in types)))
                return marker

        both_result = function(
            LeftOverride(), RightOverride(), alpha=AlphaOverride(), out=OutOverride()
        )

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
            fallback_result = function(input=left, other=FallbackOverride())

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

        return (
            mode_observations,
            override_observations,
            both_result is marker,
            order,
            subclass_result is marker,
            subclass_order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def test_binding_type_and_shape_errors_match_pytorch_2_13_for_supported_schema(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.sub(), lambda: reference_torch.sub()),
            (lambda: torch.sub(actual), lambda: reference_torch.sub(expected)),
            (
                lambda: torch.sub(actual, actual, actual),
                lambda: reference_torch.sub(expected, expected, expected),
            ),
            (lambda: torch.sub([], actual), lambda: reference_torch.sub([], expected)),
            (lambda: torch.sub(actual, []), lambda: reference_torch.sub(expected, [])),
            (
                lambda: torch.sub(input=None, other=actual),
                lambda: reference_torch.sub(input=None, other=expected),
            ),
            (
                lambda: torch.sub(x1=actual, x2=[]),
                lambda: reference_torch.sub(x1=expected, x2=[]),
            ),
            (
                lambda: torch.sub(actual, actual, input=actual),
                lambda: reference_torch.sub(expected, expected, input=expected),
            ),
            (
                lambda: torch.sub(actual, actual, x2=actual),
                lambda: reference_torch.sub(expected, expected, x2=expected),
            ),
            (lambda: torch.sub(foo=actual), lambda: reference_torch.sub(foo=expected)),
            (
                lambda: torch.sub(actual, actual, extra=True),
                lambda: reference_torch.sub(expected, expected, extra=True),
            ),
            (
                lambda: torch.sub(torch.zeros((2, 3)), torch.zeros((4, 2))),
                lambda: reference_torch.sub(
                    reference_torch.zeros((2, 3)),
                    reference_torch.zeros((4, 2)),
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module):
        function = module.sub
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
            "owner_callable_identity": owner.sub is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("sub"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sub"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )


if __name__ == "__main__":
    unittest.main()
