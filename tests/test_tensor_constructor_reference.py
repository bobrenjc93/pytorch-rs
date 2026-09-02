import array
import copy
import importlib
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def float32_bits(tensor):
    value = tensor.detach() if tensor.requires_grad else tensor
    return np.asarray(value).reshape(-1).view(np.uint32).tolist()


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorConstructorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("tensor differentials require pinned PyTorch 2.13.0")

    def tensor_contract(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "is_pinned": tensor.is_pinned(),
            "bits": float32_bits(tensor),
        }

    def assert_tensor_matches_reference(self, data, *, actual_kwargs=None, expected_kwargs=None):
        actual_kwargs = dict(actual_kwargs or {})
        expected_kwargs = dict(expected_kwargs or {})
        actual = torch.tensor(data, **actual_kwargs)
        expected = reference_torch.tensor(data, **expected_kwargs)
        self.assertEqual(self.tensor_contract(actual), self.tensor_contract(expected))
        if actual.numel() and expected.numel():
            self.assertNotEqual(actual.data_ptr(), 0)
            self.assertNotEqual(expected.data_ptr(), 0)

    def assert_error_matches(self, actual_call, expected_call, *, message_contains=None):
        with self.assertRaises(BaseException) as actual_raised:
            actual_call()
        with self.assertRaises(BaseException) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        if message_contains is not None:
            self.assertIn(message_contains, str(actual_raised.exception))
            self.assertIn(message_contains, str(expected_raised.exception))
            return
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_pin_memory_false_scalar_sequences_and_buffers_match_pytorch_2_13(self):
        cases = (
            ("scalar", -0.0, {}, {}),
            ("empty sequence", [], {}, {}),
            ("nested empty sequence", [[], []], {}, {}),
            (
                "nested float sequence",
                [[-0.0, 1.5], [float("inf"), float("nan")]],
                {},
                {},
            ),
            (
                "bytearray buffer",
                bytearray((0, 1, 127, 128, 255)),
                {"dtype": torch.float32},
                {"dtype": reference_torch.float32},
            ),
            (
                "float array buffer",
                array.array("f", [-0.0, 1.5, float("inf")]),
                {"dtype": torch.float32},
                {"dtype": reference_torch.float32},
            ),
            (
                "double memoryview buffer",
                memoryview(array.array("d", [-2.5, -0.0, 3.25])),
                {"dtype": torch.float32},
                {"dtype": reference_torch.float32},
            ),
            (
                "character buffer with explicit dtype",
                memoryview(b"ab").cast("c"),
                {"dtype": torch.float32},
                {"dtype": reference_torch.float32},
            ),
        )
        for case, data, actual_options, expected_options in cases:
            with self.subTest(case=case):
                self.assert_tensor_matches_reference(
                    data,
                    actual_kwargs={"pin_memory": False, **actual_options},
                    expected_kwargs={"pin_memory": False, **expected_options},
                )

    def test_requires_grad_and_no_grad_match_pytorch_2_13(self):
        actual_leaf = torch.tensor([2.0, 3.0], pin_memory=False, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [2.0, 3.0], pin_memory=False, requires_grad=True
        )
        self.assertEqual(
            self.tensor_contract(actual_leaf),
            self.tensor_contract(expected_leaf),
        )

        with torch.no_grad():
            actual_default = torch.tensor([1.0], pin_memory=False)
            actual_tracked = torch.tensor([1.0], pin_memory=False, requires_grad=True)
        with reference_torch.no_grad():
            expected_default = reference_torch.tensor([1.0], pin_memory=False)
            expected_tracked = reference_torch.tensor(
                [1.0], pin_memory=False, requires_grad=True
            )
        self.assertEqual(
            self.tensor_contract(actual_default),
            self.tensor_contract(expected_default),
        )
        self.assertEqual(
            self.tensor_contract(actual_tracked),
            self.tensor_contract(expected_tracked),
        )

        (actual_leaf * 4.0).sum().backward()
        (expected_leaf * 4.0).sum().backward()
        self.assertEqual(
            self.tensor_contract(actual_leaf.grad),
            self.tensor_contract(expected_leaf.grad),
        )

    def test_dtype_device_pin_memory_and_requires_grad_precedence_match_pytorch_2_13(self):
        exact_call_pairs = (
            (
                lambda: torch.tensor(dtype=object(), pin_memory=0),
                lambda: reference_torch.tensor(dtype=object(), pin_memory=0),
            ),
            (
                lambda: torch.tensor([1.0], dtype=object(), pin_memory=0),
                lambda: reference_torch.tensor([1.0], dtype=object(), pin_memory=0),
            ),
            (
                lambda: torch.tensor([1.0], device="cuda", pin_memory=0),
                lambda: reference_torch.tensor([1.0], device="cuda", pin_memory=0),
            ),
            (
                lambda: torch.tensor([1.0], pin_memory=0, requires_grad=0),
                lambda: reference_torch.tensor([1.0], pin_memory=0, requires_grad=0),
            ),
            (
                lambda: torch.tensor([1.0], pin_memory=True, requires_grad=0),
                lambda: reference_torch.tensor([1.0], pin_memory=True, requires_grad=0),
            ),
            (
                lambda: torch.tensor([1.0], unexpected=True, pin_memory=0),
                lambda: reference_torch.tensor([1.0], unexpected=True, pin_memory=0),
            ),
            (
                lambda: torch.tensor([1.0], unexpected=True, pin_memory=False),
                lambda: reference_torch.tensor([1.0], unexpected=True, pin_memory=False),
            ),
        )
        for actual_call, expected_call in exact_call_pairs:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(actual_call, expected_call)
        self.assert_error_matches(
            lambda: torch.tensor([1.0], device=object(), pin_memory=0),
            lambda: reference_torch.tensor([1.0], device=object(), pin_memory=0),
            message_contains="argument 'device'",
        )

    def callable_contract(self, module):
        function = module.tensor
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "all_count": module.__all__.count("tensor"),
            "wildcard_identity": wildcard_namespace["tensor"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_import_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

        old = torch.tensor
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.tensor, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.tensor, old)


if __name__ == "__main__":
    unittest.main()
