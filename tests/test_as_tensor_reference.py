import array
import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AsTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("as_tensor differentials require pinned PyTorch 2.13.0")

    def tensor_observation(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
        }

    def error_observation(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.as_tensor
        expected = reference_torch.as_tensor
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            torch.__all__.count("as_tensor"),
            reference_torch.__all__.count("as_tensor"),
        )

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["as_tensor"], actual)

        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)

    def test_exact_tensor_identity_and_indexed_cpu_copy_match_pytorch_2_13(self):
        actual_base = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            requires_grad=True,
        )
        expected_base = reference_torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_source = actual_base.transpose(0, 1)
        expected_source = expected_base.transpose(0, 1)

        metadata_factories = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"device": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"dtype": module.float32, "device": "cpu"},
        )
        for metadata_factory in metadata_factories:
            actual_kwargs = metadata_factory(torch)
            expected_kwargs = metadata_factory(reference_torch)
            with self.subTest(kwargs=actual_kwargs):
                actual = torch.as_tensor(actual_source, **actual_kwargs)
                expected = reference_torch.as_tensor(expected_source, **expected_kwargs)
                self.assertEqual(actual is actual_source, expected is expected_source)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

    def test_supported_copy_inputs_match_pytorch_2_13_values(self):
        cases = (
            lambda module: module.as_tensor(1.5),
            lambda module: module.as_tensor([[1, 2], [3, 4]], dtype=module.float32),
            lambda module: module.as_tensor(
                memoryview(array.array("f", [1.0, 2.0, 3.0]))
            ),
        )
        for create in cases:
            with self.subTest(create=create):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

        actual_array = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
        expected_array = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
        actual = torch.as_tensor(actual_array)
        expected = reference_torch.as_tensor(expected_array)
        self.assertEqual(
            self.tensor_observation(torch, actual),
            self.tensor_observation(reference_torch, expected),
        )

    def test_common_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda module: module.as_tensor(),
                lambda module: module.as_tensor(),
            ),
            (
                lambda module: module.as_tensor([1.0], module.float32),
                lambda module: module.as_tensor([1.0], module.float32),
            ),
            (
                lambda module: module.as_tensor([1.0], requires_grad=True),
                lambda module: module.as_tensor([1.0], requires_grad=True),
            ),
            (
                lambda module: module.as_tensor([1.0], data=[2.0]),
                lambda module: module.as_tensor([1.0], data=[2.0]),
            ),
            (
                lambda module: module.as_tensor([1.0], dtype="float32"),
                lambda module: module.as_tensor([1.0], dtype="float32"),
            ),
            (
                lambda module: module.as_tensor([1.0], dtype=object()),
                lambda module: module.as_tensor([1.0], dtype=object()),
            ),
            (
                lambda module: module.as_tensor([1.0], dtype=module.device("cpu")),
                lambda module: module.as_tensor([1.0], dtype=module.device("cpu")),
            ),
            (
                lambda module: module.as_tensor([1.0], device=object()),
                lambda module: module.as_tensor([1.0], device=object()),
            ),
            (
                lambda module: module.as_tensor([1.0], device=module.float32),
                lambda module: module.as_tensor([1.0], device=module.float32),
            ),
            (
                lambda module: module.as_tensor([1.0], device="cpu:01"),
                lambda module: module.as_tensor([1.0], device="cpu:01"),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(actual_call=actual_call):
                actual_type, actual_message = self.error_observation(
                    lambda actual_call=actual_call: actual_call(torch)
                )
                expected_type, expected_message = self.error_observation(
                    lambda expected_call=expected_call: expected_call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)


if __name__ == "__main__":
    unittest.main()
