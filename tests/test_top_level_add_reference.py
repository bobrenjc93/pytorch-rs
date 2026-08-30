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

    def test_scalar_values_layouts_empties_and_argument_forms_match_pytorch_2_13(self):
        actual_source = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_source = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_offset = actual_source[1]
        expected_offset = expected_source[1]

        calls = (
            (
                "tensor/scalar",
                lambda: torch.add(actual_offset, 2.5),
                lambda: reference_torch.add(expected_offset, 2.5),
            ),
            (
                "scalar/tensor",
                lambda: torch.add(np.int64(3), actual_offset),
                lambda: reference_torch.add(np.int64(3), expected_offset),
            ),
            (
                "canonical keywords",
                lambda: torch.add(input=actual_offset, other=np.float32(-0.0)),
                lambda: reference_torch.add(
                    input=expected_offset, other=np.float32(-0.0)
                ),
            ),
            (
                "scalar/tensor keywords",
                lambda: torch.add(input=True, other=actual_offset),
                lambda: reference_torch.add(input=True, other=expected_offset),
            ),
            (
                "legacy aliases",
                lambda: torch.add(x=actual_offset, x2=np.bool_(True)),
                lambda: reference_torch.add(x=expected_offset, x2=np.bool_(True)),
            ),
            (
                "alpha int and out none",
                lambda: torch.add(actual_offset, 2.0, alpha=1, out=None),
                lambda: reference_torch.add(expected_offset, 2.0, alpha=1, out=None),
            ),
            (
                "alpha tensor default",
                lambda: torch.add(actual_offset, 2.0, alpha=torch.tensor(1.0)),
                lambda: reference_torch.add(
                    expected_offset, 2.0, alpha=reference_torch.tensor(1.0)
                ),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.add(actual_empty, -2.0),
            reference_torch.add(expected_empty, -2.0),
            case="strided empty",
        )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF85_4321,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.add(-0.0, torch.tensor(values)),
            reference_torch.add(-0.0, reference_torch.tensor(values)),
            case="signed zero and non-finites",
        )

    def test_scalar_autograd_empty_views_and_no_grad_match_pytorch_2_13(self):
        actual_leaf = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_leaf = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_output = torch.add(4.0, actual_leaf.transpose(0, 1))
        expected_output = reference_torch.add(4.0, expected_leaf.transpose(0, 1))
        self.assert_matches(actual_output, expected_output, case="tracked view")
        actual_output.sum().backward()
        expected_output.sum().backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="gradient")

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(actual_empty.transpose(0, 2)[1], 7.0).sum().backward()
        reference_torch.add(expected_empty.transpose(0, 2)[1], 7.0).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.add(input=actual_no_grad, other=2.0, out=None)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.add(
                input=expected_no_grad, other=2.0, out=None
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad")
        self.assertTrue(torch.add(actual_no_grad, 2.0).requires_grad)
        self.assertTrue(reference_torch.add(expected_no_grad, 2.0).requires_grad)

    def dispatch_contract(self, module):
        tensor = module.tensor([1.0])
        destination = module.tensor([0.0])
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
        for call, keywords in (
            (lambda: function(tensor, 2.0), None),
            (lambda: function(2.0, tensor), None),
            (lambda: function(tensor, tensor), None),
            (lambda: function(input=2.0, other=tensor, alpha=2, out=destination), ("input", "other", "alpha", "out")),
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

        for call, keywords in (
            (lambda value: function(value, tensor), None),
            (lambda value: function(tensor, value), None),
            (lambda value: function(tensor, 2.0, alpha=value), ("alpha",)),
            (lambda value: function(tensor, 2.0, out=value), ("out",)),
        ):
            Override.calls.clear()
            result = call(Override())
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (Override,),
                    len(args),
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == keywords,
                )
            )

        invalid_observations = []
        for call in (
            lambda: function([], tensor),
            lambda: function(tensor, []),
            lambda: function(tensor, 2.0, alpha="1"),
            lambda: function(tensor, 2.0, unexpected=True),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )

        return mode_observations, override_observations, invalid_observations

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(self.dispatch_contract(torch), self.dispatch_contract(reference_torch))

    def test_parser_errors_for_shared_supported_schema_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.add(), lambda: reference_torch.add()),
            (lambda: torch.add(actual), lambda: reference_torch.add(expected)),
            (lambda: torch.add(input=actual), lambda: reference_torch.add(input=expected)),
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
                lambda: torch.add(actual, 2.0, input=actual),
                lambda: reference_torch.add(expected, 2.0, input=expected),
            ),
            (lambda: torch.add(foo=actual), lambda: reference_torch.add(foo=expected)),
            (
                lambda: torch.add(actual, 2.0, extra=True),
                lambda: reference_torch.add(expected, 2.0, extra=True),
            ),
            (
                lambda: torch.add(actual, 2.0, alpha="1"),
                lambda: reference_torch.add(expected, 2.0, alpha="1"),
            ),
            (
                lambda: torch.add(actual, 2.0, out=[]),
                lambda: reference_torch.add(expected, 2.0, out=[]),
            ),
            (
                lambda: torch.add(actual, 2**64),
                lambda: reference_torch.add(expected, 2**64),
            ),
            (
                lambda: torch.add(actual, np.uint64(2**63)),
                lambda: reference_torch.add(expected, np.uint64(2**63)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

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
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))


if __name__ == "__main__":
    unittest.main()
