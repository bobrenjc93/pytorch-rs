import copy
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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelAddReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.add differentials require pinned PyTorch 2.13.0")

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
        actual_right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected_right = reference_torch.tensor([[2.0], [-0.0], [float("inf")]])

        def legacy_actual():
            return torch.add(actual_left, 1, actual_right)

        def legacy_expected():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return reference_torch.add(expected_left, 1, expected_right)

        calls = (
            (
                "positional tensors",
                lambda: torch.add(actual_left, actual_right),
                lambda: reference_torch.add(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.add(input=actual_left, other=actual_right),
                lambda: reference_torch.add(input=expected_left, other=expected_right),
            ),
            (
                "x aliases",
                lambda: torch.add(x=actual_left, x2=actual_right),
                lambda: reference_torch.add(x=expected_left, x2=expected_right),
            ),
            (
                "x1 aliases",
                lambda: torch.add(x1=actual_left, x2=actual_right),
                lambda: reference_torch.add(x1=expected_left, x2=expected_right),
            ),
            (
                "default alpha",
                lambda: torch.add(actual_left, actual_right, alpha=np.float32(1.0)),
                lambda: reference_torch.add(
                    expected_left, expected_right, alpha=np.float32(1.0)
                ),
            ),
            (
                "default numpy bool alpha",
                lambda: torch.add(actual_left, actual_right, alpha=np.bool_(True)),
                lambda: reference_torch.add(
                    expected_left, expected_right, alpha=np.bool_(True)
                ),
            ),
            (
                "out none",
                lambda: torch.add(actual_left, actual_right, out=None),
                lambda: reference_torch.add(expected_left, expected_right, out=None),
            ),
            ("legacy positional default alpha", legacy_actual, legacy_expected),
            (
                "tensor/scalar",
                lambda: torch.add(actual_left[1], np.float32(-0.0)),
                lambda: reference_torch.add(expected_left[1], np.float32(-0.0)),
            ),
            (
                "scalar/tensor",
                lambda: torch.add(np.int64(3), actual_left[1]),
                lambda: reference_torch.add(np.int64(3), expected_left[1]),
            ),
            (
                "keyword scalar/tensor",
                lambda: torch.add(input=-2.5, other=actual_left[1]),
                lambda: reference_torch.add(input=-2.5, other=expected_left[1]),
            ),
            (
                "bool scalar",
                lambda: torch.add(True, actual_left[1]),
                lambda: reference_torch.add(True, expected_left[1]),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.add(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.add(
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
            torch.add(torch.tensor(values), torch.zeros((5,))),
            reference_torch.add(reference_torch.tensor(values), reference_torch.zeros((5,))),
            case="signed zero and non-finites",
        )

    def test_autograd_shared_operands_empties_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )
        actual_output = torch.add(
            actual_left.transpose(0, 1), actual_right.transpose(0, 1)
        )
        expected_output = reference_torch.add(
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
        torch.add(actual_shared, actual_shared).sum().backward()
        reference_torch.add(expected_shared, expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(4.0, actual_scalar).sum().backward()
        reference_torch.add(4.0, expected_scalar).sum().backward()
        self.assert_matches(
            actual_scalar.grad, expected_scalar.grad, case="scalar-first gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(actual_empty, torch.ones((1, 1, 3))).sum().backward()
        reference_torch.add(
            expected_empty, reference_torch.ones((1, 1, 3))
        ).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.add(2.0, actual_no_grad.transpose(0, 1))
        with reference_torch.no_grad():
            expected_untracked = reference_torch.add(
                2.0, expected_no_grad.transpose(0, 1)
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(torch.add(actual_no_grad, 2.0).requires_grad)
        self.assertTrue(reference_torch.add(expected_no_grad, 2.0).requires_grad)

    @staticmethod
    def dispatch_observation(module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        destination = module.tensor([0.0])
        function = module.add
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        def legacy_call():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return function(left, 1, right)

        mode_calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
            (legacy_call, None),
            (lambda: function(input=4.0, other=left, alpha=2), ("input", "other", "alpha")),
            (lambda: function(left, right, alpha=True), ("alpha",)),
            (lambda: function(left, right, out=destination), ("out",)),
        )
        for call, keyword_names in mode_calls:
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
            (lambda value: function(input=left, other=value, alpha=2), "other"),
            (lambda value: function(left, right, alpha=value), "alpha"),
            (lambda value: function(left, right, out=value), "out"),
            (lambda value: function(left, value, right), None),
        ):
            value = Override()
            Override.calls.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
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

        legacy_order = []

        class LegacyAlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                legacy_order.append(
                    (
                        "alpha",
                        func is function,
                        tuple(item.__name__ for item in types),
                        len(args),
                        kwargs is None,
                    )
                )
                return marker

        class LegacyOtherOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                legacy_order.append(
                    (
                        "other",
                        func is function,
                        tuple(item.__name__ for item in types),
                        len(args),
                        kwargs is None,
                    )
                )
                return object()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            legacy_result = function(left, LegacyAlphaOverride(), LegacyOtherOverride())

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
            fallback_result = function(input=left, other=FallbackOverride(), alpha=2)

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, alpha=[]),
            lambda: function(left, right, dtype=module.float32),
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
            legacy_result is marker,
            legacy_order,
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
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    @staticmethod
    def callable_contract(module):
        function = module.add
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
            "owner_callable_identity": owner.add is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("add"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["add"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "no_inplace": not hasattr(module, "add_"),
        }

    def test_callable_metadata_documentation_exports_and_reload_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

    def test_binding_type_bool_and_shape_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.add(), lambda: reference_torch.add()),
            (lambda: torch.add(actual), lambda: reference_torch.add(expected)),
            (lambda: torch.add(2), lambda: reference_torch.add(2)),
            (
                lambda: torch.add(actual, actual, actual),
                lambda: reference_torch.add(expected, expected, expected),
            ),
            (
                lambda: torch.add(actual, actual, 1),
                lambda: reference_torch.add(expected, expected, 1),
            ),
            (
                lambda: torch.add([], actual),
                lambda: reference_torch.add([], expected),
            ),
            (
                lambda: torch.add(actual, []),
                lambda: reference_torch.add(expected, []),
            ),
            (
                lambda: torch.add(input=None, other=actual),
                lambda: reference_torch.add(input=None, other=expected),
            ),
            (
                lambda: torch.add(actual, actual, input=actual),
                lambda: reference_torch.add(expected, expected, input=expected),
            ),
            (
                lambda: torch.add(actual, actual, x2=actual),
                lambda: reference_torch.add(expected, expected, x2=expected),
            ),
            (
                lambda: torch.add(actual, actual, extra=True),
                lambda: reference_torch.add(expected, expected, extra=True),
            ),
            (
                lambda: torch.add(actual, actual, alpha=[]),
                lambda: reference_torch.add(expected, expected, alpha=[]),
            ),
            (
                lambda: torch.add(actual, actual, alpha=None),
                lambda: reference_torch.add(expected, expected, alpha=None),
            ),
            (
                lambda: torch.add(actual, actual, alpha=True),
                lambda: reference_torch.add(expected, expected, alpha=True),
            ),
            (
                lambda: torch.add(actual, np.uint64(2**63)),
                lambda: reference_torch.add(expected, np.uint64(2**63)),
            ),
            (
                lambda: torch.add(2**64, actual),
                lambda: reference_torch.add(2**64, expected),
            ),
            (
                lambda: torch.add(actual, -(2**63) - 1),
                lambda: reference_torch.add(expected, -(2**63) - 1),
            ),
            (
                lambda: torch.add(torch.zeros((2, 3)), torch.zeros((4, 2))),
                lambda: reference_torch.add(
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
