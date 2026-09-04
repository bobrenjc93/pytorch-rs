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


MATCHING_DENSE_LEFT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC1_2345,
        0x3F80_0000,
        0x7F81_2345,
        0xFF81_2345,
    ),
    dtype=np.uint32,
)
MATCHING_DENSE_RIGHT_BITS = np.asarray(
    (
        0x8000_0000,
        0x0000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FCA_BCDE,
        0x7FCA_BCDE,
        0xFF85_6789,
        0x7F85_6789,
    ),
    dtype=np.uint32,
)


def matching_dense_view(module, bits, *, offset=False, requires_grad=False):
    storage_bits = bits
    if offset:
        storage_bits = np.concatenate([np.zeros(bits.size, dtype=np.uint32), bits])
    tensor = module.tensor(
        memoryview(storage_bits.view(np.float32)),
        requires_grad=requires_grad,
    )
    if offset:
        return tensor.reshape((2, 2, 4))[1].transpose(0, 1)
    return tensor.reshape((2, 4)).transpose(0, 1)


def tensor_bits(tensor):
    return np.asarray(tensor).reshape(-1).view(np.uint32).copy()


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
                "sub positional tensors",
                lambda: torch.sub(actual_left, actual_right),
                lambda: reference_torch.sub(expected_left, expected_right),
            ),
            (
                "sub canonical keywords",
                lambda: torch.sub(input=actual_left, other=actual_right),
                lambda: reference_torch.sub(
                    input=expected_left, other=expected_right
                ),
            ),
            (
                "sub aliases",
                lambda: torch.sub(x1=actual_left, x2=actual_right),
                lambda: reference_torch.sub(x1=expected_left, x2=expected_right),
            ),
            (
                "sub alpha one",
                lambda: torch.sub(actual_left, actual_right, alpha=np.float32(1.0)),
                lambda: reference_torch.sub(
                    expected_left, expected_right, alpha=np.float32(1.0)
                ),
            ),
            (
                "sub numpy bool alpha true",
                lambda: torch.sub(actual_left, actual_right, alpha=np.bool_(True)),
                lambda: reference_torch.sub(
                    expected_left, expected_right, alpha=np.bool_(True)
                ),
            ),
            (
                "subtract alias",
                lambda: torch.subtract(x=actual_left, other=actual_right),
                lambda: reference_torch.subtract(x=expected_left, other=expected_right),
            ),
            (
                "subtract scalar overload positional default alpha",
                lambda: torch.subtract(actual_left[1], 2, 1),
                lambda: reference_torch.subtract(expected_left[1], 2, 1),
            ),
            (
                "tensor/scalar",
                lambda: torch.sub(actual_left[1], np.float32(-0.0)),
                lambda: reference_torch.sub(expected_left[1], np.float32(-0.0)),
            ),
            (
                "scalar/tensor",
                lambda: torch.sub(np.int64(3), actual_left[1]),
                lambda: reference_torch.sub(np.int64(3), expected_left[1]),
            ),
            (
                "keyword scalar/tensor",
                lambda: torch.sub(input=-2.5, other=actual_left[1]),
                lambda: reference_torch.sub(input=-2.5, other=expected_left[1]),
            ),
            (
                "numpy bool scalar",
                lambda: torch.sub(actual_left[1], np.bool_(True)),
                lambda: reference_torch.sub(expected_left[1], np.bool_(True)),
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
            reference_torch.sub(reference_torch.tensor(values), reference_torch.zeros((5,))),
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

        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.sub(4.0, actual_scalar).sum().backward()
        reference_torch.sub(4.0, expected_scalar).sum().backward()
        self.assert_matches(
            actual_scalar.grad, expected_scalar.grad, case="scalar-first gradient"
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
            actual_untracked = torch.sub(2.0, actual_no_grad.transpose(0, 1))
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sub(
                2.0, expected_no_grad.transpose(0, 1)
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(torch.sub(actual_no_grad, 2.0).requires_grad)
        self.assertTrue(reference_torch.sub(expected_no_grad, 2.0).requires_grad)

    def test_same_shape_matching_dense_views_match_pytorch_2_13(self):
        for function_name in ("sub", "subtract"):
            function = getattr(torch, function_name)
            reference_function = getattr(reference_torch, function_name)
            for offset in (False, True):
                actual_left = matching_dense_view(
                    torch, MATCHING_DENSE_LEFT_BITS, offset=offset
                )
                actual_right = matching_dense_view(
                    torch, MATCHING_DENSE_RIGHT_BITS, offset=offset
                )
                expected_left = matching_dense_view(
                    reference_torch, MATCHING_DENSE_LEFT_BITS, offset=offset
                )
                expected_right = matching_dense_view(
                    reference_torch, MATCHING_DENSE_RIGHT_BITS, offset=offset
                )
                actual_left_before = tensor_bits(actual_left)
                actual_right_before = tensor_bits(actual_right)

                self.assert_matches(
                    function(actual_left, actual_right),
                    reference_function(expected_left, expected_right),
                    case=(function_name, "offset transposed" if offset else "transposed"),
                )
                np.testing.assert_array_equal(tensor_bits(actual_left), actual_left_before)
                np.testing.assert_array_equal(tensor_bits(actual_right), actual_right_before)

            actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
            expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
            self.assert_matches(
                function(actual_empty, actual_empty),
                reference_function(expected_empty, expected_empty),
                case=(function_name, "empty transposed"),
            )

            actual_no_grad_left = matching_dense_view(
                torch, MATCHING_DENSE_LEFT_BITS, offset=True, requires_grad=True
            )
            actual_no_grad_right = matching_dense_view(
                torch, MATCHING_DENSE_RIGHT_BITS, offset=True, requires_grad=True
            )
            expected_no_grad_left = matching_dense_view(
                reference_torch,
                MATCHING_DENSE_LEFT_BITS,
                offset=True,
                requires_grad=True,
            )
            expected_no_grad_right = matching_dense_view(
                reference_torch,
                MATCHING_DENSE_RIGHT_BITS,
                offset=True,
                requires_grad=True,
            )
            with torch.no_grad():
                actual_untracked = function(actual_no_grad_left, actual_no_grad_right)
            with reference_torch.no_grad():
                expected_untracked = reference_function(
                    expected_no_grad_left, expected_no_grad_right
                )
            self.assert_matches(
                actual_untracked,
                expected_untracked,
                case=(function_name, "no_grad offset transposed"),
            )

    def test_same_shape_matching_dense_active_autograd_matches_pytorch_2_13(self):
        values = np.arange(8, dtype=np.float32).reshape(2, 4)
        other_values = (16.0 - values).astype(np.float32)
        for function_name in ("sub", "subtract"):
            actual_left_leaf = torch.tensor(values.tolist(), requires_grad=True)
            actual_right_leaf = torch.tensor(other_values.tolist(), requires_grad=True)
            expected_left_leaf = reference_torch.tensor(
                values.tolist(), requires_grad=True
            )
            expected_right_leaf = reference_torch.tensor(
                other_values.tolist(), requires_grad=True
            )
            actual_output = getattr(torch, function_name)(
                actual_left_leaf.transpose(0, 1),
                actual_right_leaf.transpose(0, 1),
            )
            expected_output = getattr(reference_torch, function_name)(
                expected_left_leaf.transpose(0, 1),
                expected_right_leaf.transpose(0, 1),
            )

            self.assert_matches(
                actual_output, expected_output, case=(function_name, "tracked transposed")
            )
            actual_output.sum().backward()
            expected_output.sum().backward()
            np.testing.assert_array_equal(
                np.asarray(actual_left_leaf.grad), expected_left_leaf.grad.numpy()
            )
            np.testing.assert_array_equal(
                np.asarray(actual_right_leaf.grad), expected_right_leaf.grad.numpy()
            )

    @staticmethod
    def dispatch_observation(module, function_name):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        destination = module.tensor([0.0])
        function = getattr(module, function_name)
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
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
        for function_name in ("sub", "subtract"):
            with self.subTest(function=function_name):
                self.assertEqual(
                    self.dispatch_observation(torch, function_name),
                    self.dispatch_observation(reference_torch, function_name),
                )

    @staticmethod
    def callable_contract(module, function_name):
        function = getattr(module, function_name)
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
            "owner_callable_identity": getattr(owner, function_name) is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count(function_name),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace[function_name] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        for function_name in ("sub", "subtract"):
            with self.subTest(function=function_name):
                self.assertEqual(
                    self.callable_contract(torch, function_name),
                    self.callable_contract(reference_torch, function_name),
                )

    def test_binding_type_bool_and_scalar_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        for actual_function, expected_function in (
            (torch.sub, reference_torch.sub),
            (torch.subtract, reference_torch.subtract),
        ):
            cases = (
                (lambda: actual_function(), lambda: expected_function()),
                (lambda: actual_function(actual), lambda: expected_function(expected)),
                (lambda: actual_function(2), lambda: expected_function(2)),
                (
                    lambda: actual_function(actual, actual, actual),
                    lambda: expected_function(expected, expected, expected),
                ),
                (
                    lambda: actual_function(actual, actual, 1),
                    lambda: expected_function(expected, expected, 1),
                ),
                (
                    lambda: actual_function(3, actual, 1),
                    lambda: expected_function(3, expected, 1),
                ),
                (
                    lambda: actual_function([], actual),
                    lambda: expected_function([], expected),
                ),
                (
                    lambda: actual_function(actual, []),
                    lambda: expected_function(expected, []),
                ),
                (
                    lambda: actual_function(input=None, other=actual),
                    lambda: expected_function(input=None, other=expected),
                ),
                (
                    lambda: actual_function(actual, actual, input=actual),
                    lambda: expected_function(expected, expected, input=expected),
                ),
                (
                    lambda: actual_function(actual, actual, x2=actual),
                    lambda: expected_function(expected, expected, x2=expected),
                ),
                (
                    lambda: actual_function(actual, actual, extra=True),
                    lambda: expected_function(expected, expected, extra=True),
                ),
                (
                    lambda: actual_function(actual, actual, alpha=[]),
                    lambda: expected_function(expected, expected, alpha=[]),
                ),
                (
                    lambda: actual_function(actual, actual, alpha=None),
                    lambda: expected_function(expected, expected, alpha=None),
                ),
                (
                    lambda: actual_function(actual, True),
                    lambda: expected_function(expected, True),
                ),
                (
                    lambda: actual_function(True, actual),
                    lambda: expected_function(True, expected),
                ),
                (
                    lambda: actual_function(actual, actual, alpha=True),
                    lambda: expected_function(expected, expected, alpha=True),
                ),
                (
                    lambda: actual_function(actual, np.uint64(2**63)),
                    lambda: expected_function(expected, np.uint64(2**63)),
                ),
                (
                    lambda: actual_function(2**64, actual),
                    lambda: expected_function(2**64, expected),
                ),
                (
                    lambda: actual_function(actual, -(2**63) - 1),
                    lambda: expected_function(expected, -(2**63) - 1),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(function=actual_function.__name__, case=case):
                    self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
