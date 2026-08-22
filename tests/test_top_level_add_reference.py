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
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def test_values_ieee_layout_and_unit_alpha_match_pytorch_2_13(self):
        actual_source = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)[1]
        expected_source = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)[1]
        actual_unit = torch.tensor(1.0)
        expected_unit = reference_torch.tensor(1.0)
        calls = (
            (
                "tensor/scalar",
                lambda: torch.add(actual_source, -2.5),
                lambda: reference_torch.add(expected_source, -2.5),
            ),
            (
                "scalar/tensor",
                lambda: torch.add(np.int64(3), actual_source),
                lambda: reference_torch.add(np.int64(3), expected_source),
            ),
            (
                "keywords",
                lambda: torch.add(
                    input=actual_source, other=np.bool_(True), alpha=np.float32(1)
                ),
                lambda: reference_torch.add(
                    input=expected_source,
                    other=np.bool_(True),
                    alpha=np.float32(1),
                ),
            ),
            (
                "aliases and tensor alpha",
                lambda: torch.add(x1=2.0, x2=actual_source, alpha=actual_unit),
                lambda: reference_torch.add(
                    x1=2.0, x2=expected_source, alpha=expected_unit
                ),
            ),
            (
                "explicit out none",
                lambda: torch.add(actual_source, 2, out=None),
                lambda: reference_torch.add(expected_source, 2, out=None),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.add(-2.0, actual_empty),
            reference_torch.add(-2.0, expected_empty),
            case="strided empty",
        )

        tensor_bits = np.asarray(
            (0x7FC1_2345, 0xFFC5_4321, 0x0000_0000, 0x8000_0000),
            dtype=np.uint32,
        )
        scalar = np.asarray((0x7FC6_789A,), dtype=np.uint32).view(np.float32)[0]
        actual_special = torch.tensor(memoryview(tensor_bits.view(np.float32)))
        expected_special = reference_torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        )
        self.assert_matches(
            torch.add(actual_special, scalar),
            reference_torch.add(expected_special, scalar),
            case="tensor-first NaN payload",
        )
        self.assert_matches(
            torch.add(scalar, actual_special),
            reference_torch.add(scalar, expected_special),
            case="scalar-first NaN payload",
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        actual_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        actual_output = torch.add(5.0, actual_leaf.transpose(0, 1))
        expected_output = reference_torch.add(5.0, expected_leaf.transpose(0, 1))
        self.assert_matches(actual_output, expected_output, case="tracked view")
        actual_output.sum().backward()
        expected_output.sum().backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="gradient")

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(actual_empty, 7).sum().backward()
        reference_torch.add(expected_empty, 7).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_tracked = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_tracked = reference_torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.add(actual_tracked, 2)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.add(expected_tracked, 2)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad")

    def dispatch_observation(self, module):
        tensor = module.tensor([1.0])
        other = module.tensor([2.0])
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

        observations = []
        calls = (
            lambda: function(tensor, 2),
            lambda: function(2, tensor),
            lambda: function(tensor, other),
            lambda: function(2, 3),
            lambda: function(tensor, 2, alpha=2),
            lambda: function(tensor, 2, out=destination),
            lambda: function(input=tensor, other=2, alpha=1, out=None),
        )
        for call in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    None if kwargs is None else tuple(kwargs),
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=2, other=tensor, alpha=1, out=None)

        invalid = RecordingMode()
        try:
            with invalid:
                function(tensor, [])
        except Exception as error:
            invalid_observation = (type(error).__name__, str(error), len(invalid.calls))
        else:
            invalid_observation = None

        return (
            observations,
            order,
            tuple(forwarded.shape),
            forwarded.stride(),
            np.asarray(forwarded.detach() if module is reference_torch else forwarded).tolist(),
            invalid_observation,
            module.overrides._get_current_function_mode_stack(),
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def multi_operand_override_observation(self, module):
        events = []

        class Base:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "base",
                        func is module.add,
                        tuple(item.__name__ for item in types),
                        len(args),
                        None if kwargs is None else tuple(kwargs),
                    )
                )
                return "base result"

        class Derived(Base):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "derived",
                        func is module.add,
                        tuple(item.__name__ for item in types),
                        len(args),
                        None if kwargs is None else tuple(kwargs),
                    )
                )
                return "derived result"

        result = module.add(Base(), Derived, alpha=Derived())
        return result, events

    def test_multi_operand_override_deduplication_matches_pytorch_2_13(self):
        self.assertEqual(
            self.multi_operand_override_observation(torch),
            self.multi_operand_override_observation(reference_torch),
        )

    def test_binding_and_conversion_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.add(), lambda: reference_torch.add()),
            (lambda: torch.add(actual), lambda: reference_torch.add(expected)),
            (lambda: torch.add(2), lambda: reference_torch.add(2)),
            (
                lambda: torch.add(actual, 2, 1, actual),
                lambda: reference_torch.add(expected, 2, 1, expected),
            ),
            (lambda: torch.add([], actual), lambda: reference_torch.add([], expected)),
            (lambda: torch.add(actual, []), lambda: reference_torch.add(expected, [])),
            (
                lambda: torch.add(input=None, other=actual),
                lambda: reference_torch.add(input=None, other=expected),
            ),
            (
                lambda: torch.add(actual, 2, input=actual),
                lambda: reference_torch.add(expected, 2, input=expected),
            ),
            (
                lambda: torch.add(actual, 2, extra=True),
                lambda: reference_torch.add(expected, 2, extra=True),
            ),
            (
                lambda: torch.add(actual, 2, alpha=None),
                lambda: reference_torch.add(expected, 2, alpha=None),
            ),
            (
                lambda: torch.add(actual, 2, alpha=True),
                lambda: reference_torch.add(expected, 2, alpha=True),
            ),
            (
                lambda: torch.add(actual, 2, alpha=torch.tensor([1.0])),
                lambda: reference_torch.add(
                    expected, 2, alpha=reference_torch.tensor([1.0])
                ),
            ),
            (
                lambda: torch.add(actual, 2, out=[]),
                lambda: reference_torch.add(expected, 2, out=[]),
            ),
            (
                lambda: torch.add(actual, np.uint64(2**63)),
                lambda: reference_torch.add(expected, np.uint64(2**63)),
            ),
            (
                lambda: torch.add(actual, 2**64),
                lambda: reference_torch.add(expected, 2**64),
            ),
            (
                lambda: torch.add(-(2**63) - 1, actual),
                lambda: reference_torch.add(-(2**63) - 1, expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module):
        function = module.add
        owner = function.__reduce__()[1][0]
        namespace = {}
        exec(f"from {module.__name__} import *", namespace)
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
            "owner_identity": owner is module._C._VariableFunctionsClass,
            "callable_identity": owner.add is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("add"),
            "wildcard_identity": namespace["add"] is function,
            "pickle": tuple(
                pickle.loads(pickle.dumps(function, protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )


if __name__ == "__main__":
    unittest.main()
