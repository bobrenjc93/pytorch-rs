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


def tensor_from_bits(module, bits):
    values = np.asarray(bits, dtype=np.uint32)
    return module.tensor(memoryview(values.view(np.float32)))


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelAddReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.add differentials require pinned PyTorch 2.13.0")

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

    def test_supported_values_layouts_ieee_empties_and_argument_forms_match_pytorch_2_13(self):
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
                lambda: torch.add(actual_left, actual_right),
                lambda: reference_torch.add(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.add(input=actual_left, other=actual_right),
                lambda: reference_torch.add(input=expected_left, other=expected_right),
            ),
            (
                "x1/x2 aliases",
                lambda: torch.add(x1=actual_left, x2=actual_right),
                lambda: reference_torch.add(x1=expected_left, x2=expected_right),
            ),
            (
                "input/x2 aliases",
                lambda: torch.add(input=actual_left, x2=actual_right),
                lambda: reference_torch.add(input=expected_left, x2=expected_right),
            ),
            (
                "x1/other aliases",
                lambda: torch.add(x1=actual_left, other=actual_right),
                lambda: reference_torch.add(x1=expected_left, other=expected_right),
            ),
            (
                "explicit alpha one",
                lambda: torch.add(actual_left, actual_right, alpha=1),
                lambda: reference_torch.add(expected_left, expected_right, alpha=1),
            ),
            (
                "tensor alpha one",
                lambda: torch.add(actual_left, actual_right, alpha=torch.tensor(1.0)),
                lambda: reference_torch.add(
                    expected_left,
                    expected_right,
                    alpha=reference_torch.tensor(1.0),
                ),
            ),
            (
                "alpha one out none",
                lambda: torch.add(
                    input=actual_left,
                    other=actual_right,
                    alpha=np.float32(1.0),
                    out=None,
                ),
                lambda: reference_torch.add(
                    input=expected_left,
                    other=expected_right,
                    alpha=np.float32(1.0),
                    out=None,
                ),
            ),
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
            torch.add(-0.0, torch.tensor(values)),
            reference_torch.add(-0.0, reference_torch.tensor(values)),
            case="signed zero and non-finites",
        )

        nan_bits_left = (
            0x7FC1_1111,
            0x7FC1_1111,
            0x3F80_0000,
            0x7F81_1111,
            0xFF81_1111,
        )
        nan_bits_right = (
            0x7FC2_2222,
            0x3F80_0000,
            0x7FC2_2222,
            0x3F80_0000,
            0x7F82_2222,
        )
        actual_nan_left = tensor_from_bits(torch, nan_bits_left)
        actual_nan_right = tensor_from_bits(torch, nan_bits_right)
        expected_nan_left = tensor_from_bits(reference_torch, nan_bits_left)
        expected_nan_right = tensor_from_bits(reference_torch, nan_bits_right)
        scalar_nan = np.asarray([0x7FC3_3333], dtype=np.uint32).view(np.float32)[0]
        for case, actual_call, expected_call in (
            (
                "tensor/tensor NaN payload precedence",
                lambda: torch.add(actual_nan_left, actual_nan_right),
                lambda: reference_torch.add(expected_nan_left, expected_nan_right),
            ),
            (
                "tensor/scalar NaN payload precedence",
                lambda: torch.add(actual_nan_left, scalar_nan),
                lambda: reference_torch.add(expected_nan_left, scalar_nan),
            ),
            (
                "scalar/tensor NaN payload precedence",
                lambda: torch.add(scalar_nan, actual_nan_left),
                lambda: reference_torch.add(scalar_nan, expected_nan_left),
            ),
            (
                "rank-zero tensor NaN payload precedence",
                lambda: torch.add(
                    actual_nan_left,
                    tensor_from_bits(torch, [0x7FC4_4444]).reshape(()),
                ),
                lambda: reference_torch.add(
                    expected_nan_left,
                    tensor_from_bits(reference_torch, [0x7FC4_4444]).reshape(()),
                ),
            ),
        ):
            self.assert_matches(actual_call(), expected_call(), case=case)

    def test_scalar_autograd_and_no_grad_tensor_tensor_match_pytorch_2_13(self):
        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(4.0, actual_scalar).sum().backward()
        reference_torch.add(4.0, expected_scalar).sum().backward()
        self.assert_matches(
            actual_scalar.grad, expected_scalar.grad, case="scalar-first gradient"
        )

        actual_view_leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_view_leaf = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        torch.add(actual_view_leaf.transpose(0, 1), 3.0).sum().backward()
        reference_torch.add(expected_view_leaf.transpose(0, 1), 3.0).sum().backward()
        self.assert_matches(
            actual_view_leaf.grad,
            expected_view_leaf.grad,
            case="strided tensor/scalar gradient",
        )

        actual_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        actual_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        expected_right = reference_torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.add(
                actual_left.transpose(0, 1), actual_right.transpose(0, 1)
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.add(
                expected_left.transpose(0, 1), expected_right.transpose(0, 1)
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad views")

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        out = module.zeros((1,))
        complex_scalar = np.complex64(1j)
        function = module.add
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
            (lambda: function(left, right), 2, None),
            (lambda: function(left, 4.0), 2, None),
            (lambda: function(4.0, left), 2, None),
            (lambda: function(left, 1j), 2, None),
            (lambda: function(1j, left), 2, None),
            (lambda: function(left, complex_scalar), 2, None),
            (lambda: function(complex_scalar, left), 2, None),
            (lambda: function(1j, 2j), 2, None),
            (lambda: function(2.0, 3.0), 2, None),
            (lambda: function(left, right, alpha=2), 2, ("alpha",)),
            (lambda: function(left, right, out=out), 2, ("out",)),
            (
                lambda: function(input=4.0, other=left, alpha=1, out=None),
                0,
                ("input", "other", "alpha", "out"),
            ),
            (lambda: function(x1=left, x2=right), 0, ("x1", "x2")),
        )
        for call, args_length, keywords in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args) == args_length,
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
            lambda value: function(value, 4.0),
            lambda value: function(4.0, value),
            lambda value: function(value, 1j),
            lambda value: function(1j, value),
            lambda value: function(value, complex_scalar),
            lambda value: function(input=left, other=value),
            lambda value: function(left, right, alpha=value),
            lambda value: function(left, right, out=value),
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
                    kwargs is not None
                    and tuple(kwargs) in {("input", "other"), ("alpha",), ("out",)},
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

        wide_mode = RecordingMode()
        with wide_mode:
            wide_result = function(left, np.uint64(2**63))

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, alpha=[]),
            lambda: function(left, right, out=[]),
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
            wide_result is marker,
            len(wide_mode.calls),
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
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
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )


if __name__ == "__main__":
    unittest.main()
