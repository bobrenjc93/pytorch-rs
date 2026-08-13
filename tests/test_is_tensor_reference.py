import inspect
import types
import typing
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class ConversionTrap:
    def __init__(self):
        self.calls = []

    def called(self, name):
        self.calls.append(name)
        raise AssertionError(f"is_tensor invoked {name}")

    def __torch_function__(self, *args, **kwargs):
        self.called("__torch_function__")

    def __torch_dispatch__(self, *args, **kwargs):
        self.called("__torch_dispatch__")

    def __array__(self, *args, **kwargs):
        self.called("__array__")

    @property
    def __array_interface__(self):
        self.called("__array_interface__")

    @property
    def __array_struct__(self):
        self.called("__array_struct__")

    @property
    def __cuda_array_interface__(self):
        self.called("__cuda_array_interface__")

    def __dlpack__(self, *args, **kwargs):
        self.called("__dlpack__")

    def __dlpack_device__(self):
        self.called("__dlpack_device__")

    def __iter__(self):
        self.called("__iter__")

    def __len__(self):
        self.called("__len__")

    def __index__(self):
        self.called("__index__")

    def __float__(self):
        self.called("__float__")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_tensor differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        return (
            module.tensor(3.5),
            module.tensor([1.0, 2.0]),
            module.zeros((2, 0, 3)),
            module.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
        )

    def non_tensor_cases(self, module, conversion_trap):
        return (
            None,
            True,
            1,
            1.5,
            2.0j,
            "tensor",
            b"tensor",
            [],
            [1.0, 2.0],
            (1.0, 2.0),
            range(3),
            np.float32(1.0),
            np.array(1.0, dtype=np.float32),
            np.arange(4, dtype=np.float32),
            np.dtype(np.float32),
            module.float32,
            module.device("cpu"),
            module.contiguous_format,
            module.preserve_format,
            inspect.getattr_static(module.Tensor, "dtype"),
            inspect.getattr_static(module.Tensor, "device"),
            module.Tensor,
            conversion_trap,
        )

    def test_results_and_conversion_hook_behavior_match_pytorch_2_13(self):
        actual_tensors = self.tensor_cases(torch)
        expected_tensors = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_tensors, expected_tensors, strict=True)
        ):
            with self.subTest(kind="tensor", case=case):
                actual_result = torch.is_tensor(actual)
                expected_result = reference_torch.is_tensor(expected)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertIs(actual_result, expected_result)

        actual_trap = ConversionTrap()
        expected_trap = ConversionTrap()
        actual_values = self.non_tensor_cases(torch, actual_trap)
        expected_values = self.non_tensor_cases(reference_torch, expected_trap)
        for case, (actual, expected) in enumerate(
            zip(actual_values, expected_values, strict=True)
        ):
            with self.subTest(kind="non-tensor", case=case):
                actual_result = torch.is_tensor(actual)
                expected_result = reference_torch.is_tensor(expected)
                self.assertIs(type(actual_result), type(expected_result))
                self.assertIs(actual_result, expected_result)
        self.assertEqual(actual_trap.calls, expected_trap.calls)
        self.assertEqual(actual_trap.calls, [])

    def rebound_tensor_outcomes(self, module):
        native_tensor_type = module.Tensor
        tensor = module.tensor([1.0])
        try:
            module.Tensor = int
            integer_binding = (
                module.is_tensor(1),
                module.is_tensor(tensor),
                module.is_tensor("tensor"),
            )

            module.Tensor = (int, str)
            tuple_binding = module.is_tensor("tensor")

            module.Tensor = 42
            try:
                module.is_tensor(1)
            except Exception as error:
                invalid_binding = (type(error).__name__, str(error))
            else:
                self.fail(f"{module.__name__}.is_tensor accepted a non-type binding")
        finally:
            module.Tensor = native_tensor_type

        return integer_binding, tuple_binding, invalid_binding, module.is_tensor(tensor)

    def test_live_public_tensor_rebinding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.rebound_tensor_outcomes(torch),
            self.rebound_tensor_outcomes(reference_torch),
        )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.is_tensor
        expected = reference_torch.is_tensor
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        actual_hints = typing.get_type_hints(actual)
        expected_hints = typing.get_type_hints(expected)
        self.assertEqual(actual_hints.keys(), expected_hints.keys())
        self.assertIs(actual_hints["obj"], expected_hints["obj"])
        self.assertIs(
            typing.get_origin(actual_hints["return"]),
            typing.get_origin(expected_hints["return"]),
        )
        self.assertEqual(typing.get_args(actual_hints["return"]), (torch.Tensor,))
        self.assertEqual(
            typing.get_args(expected_hints["return"]), (reference_torch.Tensor,)
        )
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.is_tensor(), lambda: reference_torch.is_tensor()),
            (
                lambda: torch.is_tensor(actual, actual),
                lambda: reference_torch.is_tensor(expected, expected),
            ),
            (
                lambda: torch.is_tensor(obj=actual),
                lambda: reference_torch.is_tensor(obj=expected),
            ),
            (
                lambda: torch.is_tensor(actual, obj=actual),
                lambda: reference_torch.is_tensor(expected, obj=expected),
            ),
            (
                lambda: torch.is_tensor(input=actual),
                lambda: reference_torch.is_tensor(input=expected),
            ),
            (
                lambda: torch.is_tensor(extra=actual),
                lambda: reference_torch.is_tensor(extra=expected),
            ),
            (
                lambda: torch.is_tensor(actual, extra=actual),
                lambda: reference_torch.is_tensor(expected, extra=expected),
            ),
            (
                lambda: torch.is_tensor(obj=actual, extra=actual),
                lambda: reference_torch.is_tensor(obj=expected, extra=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
