import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class DeviceMetadata:
    def __init__(self, kind, calls, error=None):
        self.kind = kind
        self.calls = calls
        self.error = error

    @property
    def type(self):
        self.calls.append("device.type")
        if self.error is not None:
            raise self.error
        return self.kind


class TensorMetadata:
    def __init__(
        self,
        kind,
        dtype,
        calls,
        *,
        device_error=None,
        device_type_error=None,
        dtype_error=None,
    ):
        self.kind = kind
        self.value_dtype = dtype
        self.calls = calls
        self.device_error = device_error
        self.device_type_error = device_type_error
        self.dtype_error = dtype_error

    @property
    def device(self):
        self.calls.append("tensor.device")
        if self.device_error is not None:
            raise self.device_error
        return DeviceMetadata(self.kind, self.calls, self.device_type_error)

    @property
    def dtype(self):
        self.calls.append("tensor.dtype")
        if self.dtype_error is not None:
            raise self.dtype_error
        return self.value_dtype


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnIsAcceptableReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.is_acceptable differentials require pinned "
                "PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def fresh_cudnn_module(self, root):
        module_name = f"{root.__name__}.backends.cudnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "cudnn"):
            del root.backends.cudnn
        module = importlib.import_module(module_name)
        root.backends.cudnn = module
        return module

    def tensor_cases(self, module):
        leaf = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
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

    def test_cpu_tensor_results_and_state_match_pytorch_2_13(self):
        actual_tensors = self.tensor_cases(torch)
        expected_tensors = self.tensor_cases(reference_torch)
        for case, (actual_tensor, expected_tensor) in enumerate(
            zip(actual_tensors, expected_tensors, strict=True)
        ):
            with self.subTest(case=case):
                actual_before = (
                    actual_tensor.tolist(),
                    actual_tensor.shape,
                    actual_tensor.stride(),
                    actual_tensor.storage_offset(),
                    actual_tensor.data_ptr(),
                )
                expected_before = (
                    expected_tensor.tolist(),
                    expected_tensor.shape,
                    expected_tensor.stride(),
                    expected_tensor.storage_offset(),
                    expected_tensor.data_ptr(),
                )
                actual = torch.backends.cudnn.is_acceptable(actual_tensor)
                expected = reference_torch.backends.cudnn.is_acceptable(
                    expected_tensor
                )
                self.assertIs(type(actual), type(expected))
                self.assertIs(actual, expected)
                self.assertIs(actual, False)
                self.assertEqual(
                    (
                        actual_tensor.tolist(),
                        actual_tensor.shape,
                        actual_tensor.stride(),
                        actual_tensor.storage_offset(),
                        actual_tensor.data_ptr(),
                    ),
                    actual_before,
                )
                self.assertEqual(
                    (
                        expected_tensor.tolist(),
                        expected_tensor.shape,
                        expected_tensor.stride(),
                        expected_tensor.storage_offset(),
                        expected_tensor.data_ptr(),
                    ),
                    expected_before,
                )

    def test_signature_metadata_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cudnn")
        expected_module = importlib.import_module("torch.backends.cudnn")
        actual = actual_module.is_acceptable
        expected = expected_module.is_acceptable

        self.assertIs(torch.backends.cudnn, actual_module)
        self.assertIs(reference_torch.backends.cudnn, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(type(actual_module).__name__, type(expected_module).__name__)
        self.assertEqual(
            type(actual_module).__module__.replace("torch_rs", "torch"),
            type(expected_module).__module__,
        )
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            hasattr(actual_module, "__all__"),
            hasattr(expected_module, "__all__"),
        )
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {name for name in vars(expected_module) if not name.startswith("_")},
        )
        self.assertIs(actual, actual_module.m.is_acceptable)
        self.assertIs(expected, expected_module.m.is_acceptable)

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_module)
        self.assertIs(inspect.getmodule(expected), expected_module)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.cudnn")
        expected_module = importlib.import_module("torch.backends.cudnn")
        actual = actual_module.is_acceptable
        expected = expected_module.is_acceptable

        for package_name, module, function in (
            ("torch_rs", actual_module, actual),
            ("torch", expected_module, expected),
        ):
            backend_import = {}
            function_import = {}
            exec(f"from {package_name}.backends import cudnn", backend_import)
            exec(
                f"from {package_name}.backends.cudnn import is_acceptable",
                function_import,
            )
            self.assertIs(backend_import["cudnn"], module)
            self.assertIs(function_import["is_acceptable"], function)

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.cudnn import *", actual_child_wildcard)
        exec("from torch.backends.cudnn import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {name for name in expected_child_wildcard if not name.startswith("__")},
        )

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.cudnn
        old_function = module.is_acceptable
        namespace = module.__dict__
        tensor = root.tensor([1.0])
        reloaded = importlib.reload(module)
        new_function = module.is_acceptable

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale cuDNN tensor eligibility function remained pickleable")

        contract = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            old_function is not new_function,
            reloaded.is_acceptable is new_function,
            new_function(tensor) is False,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )
        self.fresh_cudnn_module(root)
        return contract

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.backends.cudnn.is_acceptable
        expected = reference_torch.backends.cudnn.is_acceptable
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0])
        self.assertIs(actual(tensor=actual_tensor), False)
        self.assertIs(expected(tensor=expected_tensor), False)
        cases = (
            (lambda: actual(), lambda: expected()),
            (
                lambda: actual(actual_tensor, actual_tensor),
                lambda: expected(expected_tensor, expected_tensor),
            ),
            (
                lambda: actual(input=actual_tensor),
                lambda: expected(input=expected_tensor),
            ),
            (
                lambda: actual(actual_tensor, tensor=actual_tensor),
                lambda: expected(expected_tensor, tensor=expected_tensor),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def metadata_observations(self, module):
        function = module.backends.cudnn.is_acceptable
        observations = []

        def observe(value, calls):
            try:
                result = function(value)
            except Exception as error:
                outcome = ("error", type(error).__name__, str(error), error.args)
            else:
                outcome = ("result", type(result).__name__, result)
            observations.append((outcome, tuple(calls)))

        observe(None, [])

        calls = []
        observe(
            TensorMetadata(
                "cpu",
                None,
                calls,
                dtype_error=AssertionError("dtype should not be read"),
            ),
            calls,
        )

        calls = []
        observe(TensorMetadata("cuda", object(), calls), calls)

        for location in ("device", "device.type", "dtype"):
            calls = []
            error = RuntimeError(f"{location} failed")
            observe(
                TensorMetadata(
                    "cuda",
                    module.float32,
                    calls,
                    device_error=error if location == "device" else None,
                    device_type_error=error if location == "device.type" else None,
                    dtype_error=error if location == "dtype" else None,
                ),
                calls,
            )

        calls = []
        observe(TensorMetadata("cuda", [], calls), calls)
        return observations

    def test_non_tensor_error_order_matches_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_observations(torch),
            self.metadata_observations(reference_torch),
        )

    def test_one_h100_exposes_the_device_and_execution_boundary(self):
        if not reference_torch.backends.cudnn.is_available():
            self.skipTest("requires a cuDNN-built reference PyTorch")
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        actual_cpu = torch.tensor([1.0])
        expected_cpu = reference_torch.tensor([1.0])
        expected_cuda = reference_torch.tensor([1.0], device="cuda:0")
        self.assertIs(torch.backends.cudnn.is_acceptable(actual_cpu), False)
        self.assertIs(
            reference_torch.backends.cudnn.is_acceptable(expected_cpu),
            False,
        )
        self.assertIs(
            reference_torch.backends.cudnn.is_acceptable(expected_cuda),
            True,
        )
        self.assertEqual(expected_cuda.device.type, "cuda")
        self.assertEqual(expected_cuda.device.index, 0)
        self.assertEqual(reference_torch.version.cuda, "13.0")
        self.assertGreater(reference_torch.backends.cudnn.version(), 0)

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="cuda:0")

    def test_configuration_and_execution_surface_remains_unsupported(self):
        actual = torch.backends.cudnn
        expected = reference_torch.backends.cudnn
        self.assertTrue(hasattr(actual, "is_acceptable"))
        self.assertTrue(hasattr(expected, "is_acceptable"))
        for name in (
            "CUDNN_TENSOR_DTYPES",
            "allow_tf32",
            "benchmark",
            "benchmark_limit",
            "conv",
            "depthwise_kernel",
            "deterministic",
            "enabled",
            "flags",
            "fp32_precision",
            "rnn",
            "set_flags",
            "version",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))
                self.assertTrue(hasattr(expected, name))

        for name in (
            "_cudnn",
            "_get_cudnn_enabled",
            "_set_cudnn_enabled",
            "_get_cudnn_benchmark",
            "_set_cudnn_benchmark",
        ):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch._C, name))
                self.assertTrue(hasattr(reference_torch._C, name))


if __name__ == "__main__":
    unittest.main()
