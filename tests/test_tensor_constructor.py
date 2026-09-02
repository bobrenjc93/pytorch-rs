import array
import copy
import importlib
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


def float32_bits(tensor):
    value = tensor.detach() if tensor.requires_grad else tensor
    return np.asarray(value).reshape(-1).view(np.uint32).tolist()


class TensorConstructorTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected):
        self.assertIs(type(actual), torch.Tensor)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertIs(actual.layout, torch.strided)
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertFalse(actual.is_pinned())
        self.assertEqual(float32_bits(actual), float32_bits(expected))

    def test_unpinned_pin_memory_options_preserve_scalar_sequence_and_buffer_conversion(self):
        supported_buffers = (
            bytearray((0, 1, 127, 128, 255)),
            array.array("f", [-0.0, 1.5, float("inf")]),
            memoryview(array.array("d", [-2.5, -0.0, 3.25])),
        )
        cases = (
            ("scalar", -0.0, {}),
            ("empty sequence", [], {}),
            ("nested empty sequence", [[], []], {}),
            (
                "nested float sequence",
                [[-0.0, 1.5], [float("inf"), float("nan")]],
                {},
            ),
            *(("buffer", source, {"dtype": torch.float32}) for source in supported_buffers),
            (
                "character buffer with explicit dtype",
                memoryview(b"ab").cast("c"),
                {"dtype": torch.float32},
            ),
        )
        for case, data, options in cases:
            expected = torch.tensor(data, **options)
            for pin_memory in (None, False):
                with self.subTest(case=case, pin_memory=pin_memory):
                    actual = torch.tensor(data, pin_memory=pin_memory, **options)
                    self.assert_tensor_matches(actual, expected)
                    if actual.numel() and expected.numel():
                        self.assertNotEqual(actual.data_ptr(), expected.data_ptr())

    def test_requires_grad_and_no_grad_stay_factory_controlled(self):
        leaf = torch.tensor([2.0, 3.0], pin_memory=False, requires_grad=True)
        self.assertTrue(leaf.requires_grad)
        self.assertTrue(leaf.is_leaf)
        self.assertFalse(leaf.is_pinned())
        (leaf * 4.0).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [4.0, 4.0])

        with torch.no_grad():
            default = torch.tensor([1.0], pin_memory=None)
            tracked = torch.tensor([1.0], pin_memory=False, requires_grad=True)
        self.assertFalse(default.requires_grad)
        self.assertTrue(default.is_leaf)
        self.assertTrue(tracked.requires_grad)
        self.assertTrue(tracked.is_leaf)
        self.assertFalse(default.is_pinned())
        self.assertFalse(tracked.is_pinned())

    def test_pin_memory_rejects_invalid_values_true_and_non_cpu_devices(self):
        for pin_memory in (0, 1, "false", object(), np.bool_(False)):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^tensor\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.tensor([1.0], pin_memory=pin_memory)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.tensor([1.0], pin_memory=True)

        for device in ("cuda", "cuda:0", "meta", "mps"):
            for pin_memory in (False, True):
                with self.subTest(device=device, pin_memory=pin_memory):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^tensor\(\): device '{re.escape(device)}' is not supported; only 'cpu' is implemented$",
                    ):
                        torch.tensor([1.0], device=device, pin_memory=pin_memory)

    def test_dtype_device_pin_memory_and_requires_grad_error_precedence(self):
        cases = (
            (
                lambda: torch.tensor(dtype=object(), pin_memory=0),
                TypeError,
                'tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.tensor([1.0], dtype=object(), pin_memory=0),
                TypeError,
                "tensor(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.tensor([1.0], device=object(), pin_memory=0),
                TypeError,
                "tensor(): argument 'device' must be torch.device or str, not object",
            ),
            (
                lambda: torch.tensor([1.0], device="cuda", pin_memory=0),
                TypeError,
                "tensor(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.tensor([1.0], pin_memory=0, requires_grad=0),
                TypeError,
                "tensor(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.tensor([1.0], pin_memory=True, requires_grad=0),
                TypeError,
                "tensor(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.tensor([1.0], unexpected=True, pin_memory=0),
                TypeError,
                "tensor(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.tensor([1.0], unexpected=True, pin_memory=False),
                TypeError,
                "tensor() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_unsupported_dtype_and_existing_data_errors_remain_unchanged(self):
        for dtype in ("float32", np.dtype("float32"), np.float32, float, object()):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    torch.tensor([1.0], dtype=dtype, pin_memory=False)

        with self.assertRaisesRegex(TypeError, "^new\\(\\): invalid data type 'bytes'$"):
            torch.tensor(b"ab", pin_memory=False)
        with self.assertRaisesRegex(RuntimeError, "^Could not infer dtype of object$"):
            torch.tensor(object(), pin_memory=None)
        with self.assertRaisesRegex(
            TypeError,
            "^must be real number, not object$",
        ):
            torch.tensor(object(), dtype=torch.float32, pin_memory=False)
        with self.assertRaisesRegex(
            ValueError,
            "^expected a rectangular sequence, but nested shapes differ$",
        ):
            torch.tensor([[1.0], [2.0, 3.0]], pin_memory=False)

    def test_callable_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.tensor
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "tensor")
        self.assertEqual(function.__module__, native.__name__)
        self.assertIs(native.tensor, function)
        self.assertEqual(package.__all__.count("tensor"), 1)
        self.assertIs(wildcard_namespace["tensor"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.tensor, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.tensor, function)
        self.assertEqual(package.__all__.count("tensor"), 1)


if __name__ == "__main__":
    unittest.main()
