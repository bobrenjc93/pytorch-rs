import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch_rs as torch


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


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


class RejectContains:
    def __contains__(self, value):
        raise AssertionError(f"dtype eligibility was probed for {value!r}")


class RejectCudnnBuildProbe:
    @property
    def _C(self):
        raise AssertionError("the cuDNN build flag was probed")


class CudnnIsAcceptableTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        return (
            torch.tensor(3.5),
            torch.tensor([1.0, 2.0]),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
        )

    def tensor_snapshot(self, tensor):
        return (
            tensor.tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def test_native_cpu_float32_tensors_return_exact_false_without_side_effects(self):
        cudnn = torch.backends.cudnn
        function = cudnn.is_acceptable
        self.assertEqual(
            function.__code__.co_names,
            (
                "device",
                "type",
                "dtype",
                "_CUDNN_TENSOR_DTYPES",
                "torch",
                "_C",
                "_has_cudnn",
            ),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_CUDNN": "1"},
            {"CUDA_VISIBLE_DEVICES": ""},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
                "USE_CUDNN": "1",
            },
        )
        tensors = self.tensor_cases()
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for case, tensor in enumerate(tensors):
                        with self.subTest(case=case, shape=tensor.shape):
                            before = self.tensor_snapshot(tensor)
                            with mock.patch.object(
                                cudnn.m,
                                "_CUDNN_TENSOR_DTYPES",
                                RejectContains(),
                            ), mock.patch.object(
                                cudnn.m,
                                "torch",
                                RejectCudnnBuildProbe(),
                            ):
                                result = function(tensor)
                            self.assertIs(type(result), bool)
                            self.assertIs(result, False)
                            self.assertEqual(self.tensor_snapshot(tensor), before)

    def test_signature_metadata_and_module_proxy_identity_match_pytorch_2_13(self):
        cudnn = importlib.import_module("torch_rs.backends.cudnn")
        function = cudnn.is_acceptable

        self.assertIs(torch.backends.cudnn, cudnn)
        self.assertIs(sys.modules["torch_rs.backends.cudnn"], cudnn)
        self.assertIsInstance(cudnn, types.ModuleType)
        self.assertEqual(type(cudnn).__name__, "CudnnModule")
        self.assertEqual(type(cudnn).__module__, "torch_rs.backends.cudnn")
        self.assertIsNone(cudnn.__doc__)
        self.assertFalse(hasattr(cudnn, "__all__"))
        self.assertEqual(
            {name for name in vars(cudnn) if not name.startswith("_")},
            {"m"},
        )
        self.assertIs(function, cudnn.m.is_acceptable)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(tensor)")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_acceptable")
        self.assertEqual(function.__qualname__, "is_acceptable")
        self.assertEqual(function.__module__, "torch_rs.backends.cudnn")
        self.assertIs(inspect.getmodule(function), cudnn)
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        cudnn = importlib.import_module("torch_rs.backends.cudnn")
        function = cudnn.is_acceptable

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import cudnn", backend_import)
        exec(
            "from torch_rs.backends.cudnn import is_acceptable",
            function_import,
        )
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.cudnn import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["cudnn"], cudnn)
        self.assertIs(function_import["is_acceptable"], function)
        self.assertIs(parent_wildcard["cudnn"], cudnn)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"m"},
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cudnn", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_the_function_through_the_module_proxy(self):
        backends = torch.backends
        cudnn = backends.cudnn
        old_function = cudnn.is_acceptable
        namespace = cudnn.__dict__
        tensor = torch.tensor([1.0])

        try:
            reloaded = importlib.reload(cudnn)

            self.assertIsNot(reloaded, cudnn)
            self.assertIs(cudnn.__dict__, namespace)
            self.assertIs(backends.cudnn, cudnn)
            self.assertIs(sys.modules[cudnn.__name__], reloaded)
            self.assertIs(reloaded.m, cudnn)
            self.assertIsNot(cudnn.is_acceptable, old_function)
            self.assertIs(reloaded.is_acceptable, cudnn.is_acceptable)
            self.assertIs(cudnn.is_acceptable(tensor), False)
            self.assertIs(copy.copy(cudnn.is_acceptable), cudnn.is_acceptable)
            self.assertIs(copy.deepcopy(cudnn.is_acceptable), cudnn.is_acceptable)
            self.assertIs(
                pickle.loads(pickle.dumps(cudnn.is_acceptable)),
                cudnn.is_acceptable,
            )
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(old_function)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                "Can't pickle <function is_acceptable at 0x...>: "
                "it's not the same object as torch_rs.backends.cudnn.is_acceptable",
            )
        finally:
            fresh_cudnn_module()

    def test_argument_binding_errors_match_pytorch_2_13(self):
        function = torch.backends.cudnn.is_acceptable
        tensor = torch.tensor([1.0])
        self.assertIs(function(tensor=tensor), False)
        cases = (
            (
                lambda: function(),
                "is_acceptable() missing 1 required positional argument: 'tensor'",
            ),
            (
                lambda: function(tensor, tensor),
                "is_acceptable() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(input=tensor),
                "is_acceptable() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: function(tensor, tensor=tensor),
                "is_acceptable() got multiple values for argument 'tensor'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_non_tensor_metadata_errors_follow_device_then_dtype_order(self):
        function = torch.backends.cudnn.is_acceptable

        with self.assertRaises(AttributeError) as raised:
            function(None)
        self.assertEqual(
            str(raised.exception),
            "'NoneType' object has no attribute 'device'",
        )

        calls = []
        cpu_like = TensorMetadata(
            "cpu",
            None,
            calls,
            dtype_error=AssertionError("dtype should not be read"),
        )
        self.assertIs(function(cpu_like), False)
        self.assertEqual(calls, ["tensor.device", "device.type"])

        for location, expected_calls in (
            ("device", ["tensor.device"]),
            ("device.type", ["tensor.device", "device.type"]),
            (
                "dtype",
                ["tensor.device", "device.type", "tensor.dtype"],
            ),
        ):
            calls = []
            error = RuntimeError(f"{location} failed")
            value = TensorMetadata(
                "cuda",
                torch.float32,
                calls,
                device_error=error if location == "device" else None,
                device_type_error=error if location == "device.type" else None,
                dtype_error=error if location == "dtype" else None,
            )
            with self.subTest(location=location):
                with self.assertRaises(RuntimeError) as raised:
                    function(value)
                self.assertIs(raised.exception, error)
                self.assertEqual(calls, expected_calls)

        calls = []
        with self.assertRaisesRegex(TypeError, "^unhashable type: 'list'$"):
            function(TensorMetadata("cuda", [], calls))
        self.assertEqual(calls, ["tensor.device", "device.type", "tensor.dtype"])

    def test_configuration_execution_and_cuda_tensors_remain_unsupported(self):
        cudnn = torch.backends.cudnn
        self.assertIs(cudnn.is_acceptable(torch.tensor([1.0])), False)
        for name in (
            "CUDNN_TENSOR_DTYPES",
            "allow_tf32",
            "benchmark",
            "benchmark_limit",
            "conv",
            "deterministic",
            "enabled",
            "flags",
            "fp32_precision",
            "set_flags",
            "version",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cudnn, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_importing_and_calling_does_not_probe_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cudnn", "cupy", "nvidia", "numpy", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
    USE_CUDNN="1",
)
import torch_rs as torch
from torch_rs.backends import cudnn
from torch_rs.backends.cudnn import is_acceptable

tensor = torch.tensor([1.0])
before = (tensor.tolist(), tensor.data_ptr(), tensor.storage_offset())
assert torch.backends.cudnn is cudnn
assert cudnn.is_acceptable is is_acceptable
assert is_acceptable(tensor) is False
assert (tensor.tolist(), tensor.data_ptr(), tensor.storage_offset()) == before
assert not hasattr(torch, "cuda")
assert not hasattr(cudnn, "version")
assert not hasattr(cudnn, "flags")
assert not any(
    name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    for name in sys.modules
)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
