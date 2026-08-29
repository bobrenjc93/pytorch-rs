import array
import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


class AsTensorTests(unittest.TestCase):
    def assert_tensor_observation(self, tensor, values, shape, stride):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.tolist(), values)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))

    def test_exact_native_tensor_identity_preserves_metadata_and_autograd(self):
        base = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            requires_grad=True,
        )
        source = base.transpose(0, 1)
        metadata = (
            {},
            {"dtype": None},
            {"device": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"dtype": torch.float32, "device": "cpu"},
            {"dtype": torch.float, "device": torch.device("cpu")},
        )

        for kwargs in metadata:
            with self.subTest(kwargs=kwargs):
                result = torch.as_tensor(source, **kwargs)
                self.assertIs(result, source)
                self.assert_tensor_observation(
                    result,
                    [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                    (3, 2),
                    (1, 3),
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)

    def test_indexed_cpu_metadata_copies_exact_native_tensor(self):
        source = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            requires_grad=True,
        ).transpose(0, 1)

        for device in ("cpu:0", torch.device("cpu", 2)):
            with self.subTest(device=device):
                result = torch.as_tensor(source, device=device)
                self.assertIsNot(result, source)
                self.assertNotEqual(result.data_ptr(), source.data_ptr())
                self.assert_tensor_observation(
                    result,
                    [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]],
                    (3, 2),
                    (1, 3),
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)

    def test_supported_non_tensor_inputs_use_the_tensor_copy_path(self):
        self.assert_tensor_observation(torch.as_tensor(1.5), 1.5, (), ())
        self.assert_tensor_observation(
            torch.as_tensor([[1, 2], [3, 4]], dtype=torch.float),
            [[1.0, 2.0], [3.0, 4.0]],
            (2, 2),
            (2, 1),
        )

        buffer_source = array.array("f", [1.0, 2.0, 3.0])
        buffer_tensor = torch.as_tensor(memoryview(buffer_source), dtype=torch.float32)
        buffer_source[0] = 99.0
        self.assert_tensor_observation(buffer_tensor, [1.0, 2.0, 3.0], (3,), (1,))

        numpy_source = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
        numpy_tensor = torch.as_tensor(numpy_source)
        numpy_source[0] = 99.0
        self.assert_tensor_observation(numpy_tensor, [4.0, 5.0, 6.0], (3,), (1,))

    def test_binding_and_unsupported_metadata_errors(self):
        with self.assertRaisesRegex(
            TypeError, 'as_tensor\\(\\) missing 1 required positional arguments: "data"'
        ):
            torch.as_tensor()
        with self.assertRaisesRegex(
            TypeError, "as_tensor\\(\\) takes 1 positional argument but 2 were given"
        ):
            torch.as_tensor([1.0], torch.float32)
        with self.assertRaisesRegex(
            TypeError, "as_tensor\\(\\) got an unexpected keyword argument 'requires_grad'"
        ):
            torch.as_tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(
            TypeError, "as_tensor\\(\\) got multiple values for argument 'data'"
        ):
            torch.as_tensor([1.0], data=[2.0])

        for dtype in (
            "float32",
            np.dtype("float32"),
            np.float32,
            float,
            object(),
            torch.device("cpu"),
        ):
            with self.subTest(argument="dtype", value=dtype):
                with self.assertRaises(TypeError):
                    torch.as_tensor([1.0], dtype=dtype)
                with self.assertRaises(TypeError):
                    torch.as_tensor(torch.tensor([1.0]), dtype=dtype)

        for device in (object(), 0, b"cpu", torch.float32):
            with self.subTest(argument="device", value=device):
                with self.assertRaises(TypeError):
                    torch.as_tensor([1.0], device=device)
                with self.assertRaises(TypeError):
                    torch.as_tensor(torch.tensor([1.0]), device=device)

        for device in ("cuda", "meta", "mps", "cpu:01"):
            with self.subTest(argument="device", value=device):
                with self.assertRaises(RuntimeError):
                    torch.as_tensor([1.0], device=device)
                with self.assertRaises(RuntimeError):
                    torch.as_tensor(torch.tensor([1.0]), device=device)

    def test_public_metadata_wildcard_and_mode_dispatch(self):
        function = torch.as_tensor
        owner = torch._C._VariableFunctionsClass
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "as_tensor")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.as_tensor")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertIs(function, owner.as_tensor)
        self.assertEqual(torch.__all__.count("as_tensor"), 1)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["as_tensor"], function)

        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = torch.as_tensor([1.0])
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        func, types_, args, kwargs = mode.calls[0]
        self.assertIs(func, function)
        self.assertEqual(types_, ())
        self.assertEqual(args, ([1.0],))
        self.assertIsNone(kwargs)

    def test_invalid_calls_do_not_dispatch_to_torch_function_mode(self):
        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        cases = (
            ("missing data", lambda: torch.as_tensor()),
            ("extra positional", lambda: torch.as_tensor([1.0], torch.float32)),
            ("unexpected keyword", lambda: torch.as_tensor([1.0], requires_grad=True)),
            ("duplicate data", lambda: torch.as_tensor([1.0], data=[2.0])),
            ("invalid dtype", lambda: torch.as_tensor([1.0], dtype="float32")),
            ("invalid device object", lambda: torch.as_tensor([1.0], device=object())),
            (
                "invalid device dtype",
                lambda: torch.as_tensor([1.0], device=torch.float32),
            ),
        )
        for name, call in cases:
            mode = RecordingMode()
            with self.subTest(name=name):
                with self.assertRaises(TypeError):
                    with mode:
                        call()
                self.assertEqual(mode.calls, [])


if __name__ == "__main__":
    unittest.main()
