import importlib.util
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import torch_rs as torch
from torch_rs import (
    CudaBenchmarkTensor,
    _cuda_buffer,
    _cuda_benchmark_tensor,
    _compiler_state,
    _cuda_driver_probe,
    _cuda_pointwise_kernel,
    _cuda_pointwise_reduce_workload,
    _cuda_runtime_ownership,
    _cuda_runtime_roundtrip,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_compile_cuda.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_compile_cuda_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_compile_cuda = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_compile_cuda
spec.loader.exec_module(benchmark_compile_cuda)


def _reference_cuda_probe(cuda_visible_devices="0"):
    script = r"""
import json
try:
    import torch
except ImportError:
    print(json.dumps({"imported": False}))
else:
    available = bool(torch.cuda.is_available())
    print(json.dumps({
        "imported": True,
        "version": torch.__version__,
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices},
        text=True,
    )
    if completed.returncode != 0:
        return {"imported": False, "error": completed.stdout + completed.stderr}
    return json.loads(completed.stdout)


def _synthetic_private_buffer(
    *,
    name="synthetic",
    runtime=None,
    shape=(2,),
    device_type="cuda",
    device_index=0,
    device="cuda:0",
    dtype="torch.float32",
):
    buffer = object.__new__(_cuda_buffer.PrivateCudaFloat32Buffer)
    buffer.runtime = runtime
    buffer.name = name
    buffer.shape = tuple(shape)
    buffer.stride = _cuda_buffer.contiguous_stride(buffer.shape)
    buffer.element_count = _cuda_buffer.element_count(buffer.shape)
    buffer.byte_count = buffer.element_count * _cuda_buffer.FLOAT32_ITEMSIZE
    buffer.device_index = device_index
    buffer._pointer = SimpleNamespace(value=1)
    buffer._closed = False
    buffer.malloc_call = {"result": 0}

    def metadata():
        return {
            "shape": list(buffer.shape),
            "stride": list(buffer.stride),
            "storage_offset": 0,
            "dtype": dtype,
            "device": device,
            "device_type": device_type,
            "device_index": device_index,
            "requires_grad": False,
            "is_contiguous": True,
        }

    buffer.metadata = metadata
    return buffer


def _successful_synthetic_readback(buffer, checksum="checksum"):
    return _cuda_buffer.PrivateCudaHostReadback(
        payload=b"\0" * buffer.byte_count,
        copy_call={
            "result": 0,
            "kind": "cudaMemcpyDeviceToHost",
            "byte_count": buffer.byte_count,
            "buffer_name": buffer.name,
            "device_index": buffer.device_index,
        },
        sync_call={
            "result": 0,
            "buffer_name": buffer.name,
            "device_index": buffer.device_index,
        },
        checksum=checksum,
    )


class _FakeCudaRuntime:
    def __init__(self):
        self.current_device = 0
        self.next_pointer = 0x1000
        self.set_devices = []
        self.get_devices = []
        self.mallocs = []
        self.frees = []
        self.syncs = []
        self.copies = []

    def cudaGetDeviceCount(self, value):
        value._obj.value = 1
        return 0

    def cudaSetDevice(self, device):
        self.current_device = int(device)
        self.set_devices.append(self.current_device)
        return 0

    def cudaGetDevice(self, value):
        value._obj.value = self.current_device
        self.get_devices.append(self.current_device)
        return 0

    def cudaMalloc(self, pointer, byte_count):
        allocation = self.next_pointer
        self.next_pointer += max(int(byte_count), 1) + 0x1000
        pointer._obj.value = allocation
        self.mallocs.append(
            {
                "pointer": allocation,
                "byte_count": int(byte_count),
            }
        )
        return 0

    def cudaMemcpy(self, destination, source, byte_count, kind):
        del source
        self.copies.append(
            {
                "byte_count": int(byte_count),
                "kind": int(kind),
            }
        )
        if int(kind) == _cuda_buffer.CUDA_MEMCPY_DEVICE_TO_HOST:
            destination_address = getattr(destination, "value", None)
            if destination_address is None:
                destination_address = int(destination)
            ctypes.memset(destination_address, 0, int(byte_count))
        return 0

    def cudaDeviceSynchronize(self):
        self.syncs.append(len(self.syncs) + 1)
        return 0

    def cudaFree(self, pointer):
        self.frees.append(int(pointer.value))
        return 0


class _FakePointwiseReduceLibrary:
    def __init__(self, runtime=None):
        self.runtime = runtime

    def torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
        self,
        x,
        bias,
        output,
        rows,
        columns,
        blocks_out,
        threads_out,
        launch_error_out,
    ):
        del x, bias, output, columns
        blocks_out._obj.value = int(rows)
        threads_out._obj.value = 32
        launch_error_out._obj.value = 0
        return 0

    def torch_rs_private_h100_pointwise_reduce_float32_device0_guarded_launch_async_v1(
        self,
        x,
        bias,
        output,
        rows,
        columns,
        report,
    ):
        report = report._obj
        report.set_device_error = 0
        report.get_device_error = 0
        report.observed_device = 0
        report.launch_error = 0
        runtime = getattr(self, "runtime", None)
        if runtime is None:
            runtime = getattr(self, "observed_runtime", None)
        if runtime is not None:
            report.set_device_error = int(runtime.cudaSetDevice(0))
            current_device = ctypes.c_int(-1)
            report.get_device_error = int(
                runtime.cudaGetDevice(ctypes.byref(current_device))
            )
            report.observed_device = int(current_device.value)
        if report.set_device_error != 0:
            return report.set_device_error
        if report.get_device_error != 0:
            return report.get_device_error
        if report.observed_device != 0:
            return -3

        blocks = ctypes.c_int(0)
        threads = ctypes.c_int(0)
        launch_error = ctypes.c_int(0)
        result = self.torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
            x,
            bias,
            output,
            rows,
            columns,
            ctypes.byref(blocks),
            ctypes.byref(threads),
            ctypes.byref(launch_error),
        )
        report.launch_error = int(launch_error.value)
        return int(result)


def _fake_prepared_executor(
    runtime=None,
    required_cuda_visible_devices=None,
    kernel_library=None,
):
    runtime = runtime or _FakeCudaRuntime()
    kernel_library = kernel_library or _FakePointwiseReduceLibrary(runtime)
    preparation = {
        "schema_version": (
            _cuda_pointwise_reduce_workload
            .POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION
        ),
        "implementation": "torch_rs",
        "status": "ok",
        "reason": "compiled pointwise-reduce executor prepared",
        "workload_version": (
            _cuda_pointwise_reduce_workload
            .POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION
        ),
        "compile_backend": "inductor",
        "compile_fullgraph": True,
        "compile_dynamic": False,
        "native_cuda_compile": True,
        "eager_fallback": False,
        "forwarded_to_pytorch": False,
        "public_torch_cuda_api": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_cuda_visible_devices": required_cuda_visible_devices,
        "cuda_visible_devices_match": (
            required_cuda_visible_devices is None
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            == required_cuda_visible_devices
        ),
        "single_visible_cuda_device": True,
        "cpu_fallback": False,
        "device_type": "cuda",
        "device_index": 0,
        "dtype": "float32",
        "workload_shape": list(_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE),
        "output_shape": list(_cuda_pointwise_reduce_workload.OUTPUT_SHAPE),
        "driver": {},
        "runtime": {},
        "device_0": {},
        "gpu": {},
        "nvcc": {"available": True},
        "build": {"build_key": "fake-build", "library_path": "fake.so"},
        "kernel_library": {
            "loaded": True,
            "function": "torch_rs_private_h100_pointwise_reduce_float32_v1",
            "async_function": (
                "torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1"
            ),
        },
        "prepared": True,
        "setup_hoisted_to_compile_wrapper": True,
        "invocation_count": 0,
        "calls": {},
    }
    executor = _cuda_pointwise_reduce_workload.H100Float32PointwiseReduceCompiledExecutor(
        runtime=runtime,
        runtime_library="fake-cudart",
        runtime_load_error=None,
        driver_probe={"driver": {}, "device_0": {}},
        nvcc={"available": True},
        build={"build_key": "fake-build", "library_path": "fake.so"},
        kernel_library=kernel_library,
        kernel_library_evidence=preparation["kernel_library"],
        required_cuda_visible_devices=required_cuda_visible_devices,
        preparation=preparation,
    )
    return executor, runtime


def _fake_input_tensor(runtime, name, shape):
    buffer = _cuda_buffer.PrivateCudaFloat32Buffer(
        runtime,
        shape,
        name=name,
        device_index=0,
    )
    return CudaBenchmarkTensor(
        buffer,
        readback=_successful_synthetic_readback(buffer, checksum=f"{name}-checksum"),
        checksum_name=f"{name}_checksum_v1",
    )


class CompileCudaBenchmarkTests(unittest.TestCase):
    def _require_h100_reference_torch(self):
        probe = _reference_cuda_probe()
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")
        if benchmark_compile_cuda._version_without_local(
            probe["version"],
        ) != benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION:
            self.skipTest(
                "requires PyTorch "
                f"{benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION}, "
                f"got {probe['version']}"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            self.skipTest("requires CUDA_VISIBLE_DEVICES=0")

        import torch as reference_torch

        return reference_torch

    def test_checked_in_cuda_prepared_executor_artifact_records_reuse(self):
        artifact = (
            REPOSITORY_ROOT
            / "docs"
            / "benchmark-data"
            / "torch-compile-cuda-h100-runtime-ownership-v10.json"
        )
        report = json.loads(artifact.read_text(encoding="utf-8"))
        candidate = report["candidate"]
        reference = report["reference_workload"]
        reuse = candidate["prepared_executor_reuse"]
        unprepared = candidate["unprepared_compatibility_comparison"]

        self.assertEqual(
            report["environment"]["benchmark_version"],
            benchmark_compile_cuda.BENCHMARK_VERSION,
        )
        self.assertEqual(candidate["status"], "ok")
        self.assertIs(
            candidate["eligibility"]["eligible_cuda_compile_evidence"],
            True,
        )
        self.assertEqual(candidate["eligibility"]["rejection_reasons"], [])
        self.assertEqual(reuse["before_first_call_invocation_count"], 0)
        self.assertEqual(
            reuse["observed_invocation_count"],
            reuse["expected_invocation_count"],
        )
        self.assertEqual(
            reuse["last_invocation_index"],
            reuse["expected_invocation_count"],
        )
        self.assertIs(reuse["steady_state_reused_prepared_executor"], True)
        self.assertIs(reuse["setup_hoisted_to_compile_wrapper"], True)
        self.assertEqual(
            candidate["prepared_executor"]["preparation_id"],
            candidate["prepared_executor_after_timing"]["preparation_id"],
        )
        self.assertLess(
            candidate["steady"]["median_us"],
            unprepared["steady"]["median_us"],
        )
        self.assertGreater(
            report["aggregates"]["torch_rs_cuda_compile_score_percent"],
            21.3,
        )
        self.assertGreater(
            reuse["prepared_vs_unprepared_steady_speedup"],
            2.0,
        )
        self.assertIs(reuse["output_pool_enabled"], True)
        self.assertGreaterEqual(reuse["output_pool_initial_capacity"], 2)
        self.assertEqual(reuse["output_pool_live_buffers_after_timing"], 0)
        self.assertIs(reuse["steady_state_reused_output_buffer"], True)
        self.assertIs(reuse["steady_state_output_allocated_in_execute"], False)
        self.assertIs(candidate["timing_boundary"]["compiled_calls_only"], True)
        self.assertIs(
            candidate["timing_boundary"]["materialization_outside_timed_region"],
            True,
        )
        self.assertIs(reference["timing_boundary"]["compiled_calls_only"], True)
        self.assertIs(
            reference["timing_boundary"]["materialization_outside_timed_region"],
            True,
        )
        self.assertEqual(
            reference["timing_boundary"][
                "explicit_cuda_synchronize_before_timed_region"
            ],
            candidate["timing_boundary"][
                "explicit_cuda_synchronize_before_timed_region"
            ],
        )
        self.assertEqual(
            reference["timing_boundary"][
                "explicit_cuda_synchronize_after_timed_region"
            ],
            candidate["timing_boundary"][
                "explicit_cuda_synchronize_after_timed_region"
            ],
        )
        self.assertIs(candidate["compile_execution"]["readback_deferred"], True)
        self.assertIs(candidate["compile_execution"]["output_materialized"], True)
        self.assertIs(
            candidate["compile_execution"]["kernel_synchronized_in_call"],
            False,
        )
        self.assertIsNone(
            candidate["compile_execution"]["launch"]["sync_error"]["result"]
        )
        self.assertEqual(
            candidate["compile_execution"]["calls"][
                "cudaSetDevice_before_launch"
            ]["result"],
            0,
        )
        self.assertEqual(
            candidate["compile_execution"]["calls"][
                "cudaGetDevice_before_launch"
            ]["value"],
            0,
        )
        self.assertIs(
            candidate["last_compile_execution"]["output_buffer_pool"][
                "reused_released_allocation"
            ],
            True,
        )
        self.assertEqual(
            candidate["steady_checksums"],
            [report["reference_workload"]["cold_checksum"]],
        )
        self.assertEqual(unprepared["steady_checksums"], candidate["steady_checksums"])

    def test_torch_rs_cuda_zero_credit_row_is_explicit(self):
        row = benchmark_compile_cuda.torch_rs_zero_credit_unsupported_row(torch)

        self.assertEqual(row["implementation"], "torch_rs")
        self.assertEqual(row["status"], "zero_credit_unsupported")
        self.assertEqual(row["score_credit"], 0.0)
        self.assertFalse(row["eligibility"]["eligible_cuda_compile_evidence"])
        self.assertEqual(row["eligibility"]["score_credit"], 0.0)
        self.assertIn("CPU tensor execution", row["rejected_fallbacks"])
        self.assertIn("backend='eager' compile execution", row["rejected_fallbacks"])
        self.assertIn("does not receive compile credit", row["reason"])

        probes = row["cuda_probes"]
        self.assertIs(probes["cuda_is_available"], False)
        self.assertEqual(probes["cuda_device_count"], 0)
        self.assertIs(probes["cuda_is_initialized"], False)
        self.assertIs(probes["accelerator_is_available"], False)
        self.assertEqual(probes["accelerator_device_count"], 0)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)

    def test_public_cuda_benchmark_tensor_wrapper_is_narrow(self):
        self.assertIn("CudaBenchmarkTensor", torch.__all__)
        self.assertIs(torch.CudaBenchmarkTensor, CudaBenchmarkTensor)
        self.assertFalse(hasattr(torch.cuda, "CudaBenchmarkTensor"))
        self.assertNotIn("CudaBenchmarkTensor", torch.cuda.__all__)
        self.assertEqual(
            torch.cuda.__all__,
            ["device_count", "is_available", "is_initialized"],
        )
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertIs(torch.is_tensor(CudaBenchmarkTensor), False)

    def test_public_cuda_benchmark_tensor_exposes_metadata_snapshot(self):
        buffer = _synthetic_private_buffer()
        wrapper = CudaBenchmarkTensor(
            buffer,
            readback=_successful_synthetic_readback(buffer),
            checksum_name="synthetic_checksum_v1",
        )

        self.assertEqual(wrapper.shape, (2,))
        self.assertEqual(wrapper.stride, (1,))
        self.assertEqual(str(wrapper.dtype), "torch.float32")
        self.assertEqual(wrapper.device, "cuda:0")
        self.assertEqual(wrapper.device_type, "cuda")
        self.assertEqual(wrapper.device_index, 0)
        self.assertIs(wrapper.is_cuda, True)
        self.assertIs(wrapper.is_contiguous, True)
        self.assertIs(wrapper.requires_grad, False)
        self.assertEqual(wrapper.checksum, "checksum")
        self.assertIn("CudaBenchmarkTensor", repr(wrapper))
        self.assertIs(torch.is_tensor(wrapper), False)
        self.assertFalse(hasattr(wrapper, "cpu"))
        self.assertFalse(hasattr(wrapper, "tolist"))
        with self.assertRaises(TypeError):
            wrapper + wrapper

        metadata = wrapper.metadata()
        self.assertEqual(
            metadata["schema_version"],
            _cuda_benchmark_tensor.CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION,
        )
        self.assertEqual(metadata["status"], "ok")
        self.assertEqual(metadata["shape"], [2])
        self.assertEqual(metadata["stride"], [1])
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["device"], "cuda:0")
        self.assertEqual(metadata["device_type"], "cuda")
        self.assertEqual(metadata["device_index"], 0)
        self.assertIs(metadata["is_cuda"], True)
        self.assertIs(metadata["cpu_fallback"], False)
        self.assertEqual(metadata["checksum"], "checksum")
        self.assertEqual(metadata["checksum_name"], "synthetic_checksum_v1")
        self.assertEqual(metadata["operations_supported"], [])
        self.assertIs(metadata["readback"]["synchronized"], True)
        self.assertIs(metadata["readback"]["payload_exposed"], False)
        self.assertNotIn("payload", metadata["readback"])

        metadata["shape"].append(3)
        self.assertEqual(wrapper.metadata()["shape"], [2])

    def test_public_cuda_benchmark_tensor_rejects_non_private_buffers(self):
        with self.assertRaisesRegex(TypeError, "private CUDA float32"):
            CudaBenchmarkTensor(object(), checksum=lambda payload: "checksum")
        with self.assertRaisesRegex(TypeError, "checksum must be callable"):
            CudaBenchmarkTensor(_synthetic_private_buffer())

    def test_public_cuda_benchmark_tensor_rejects_metadata_update_collisions(self):
        buffer = _synthetic_private_buffer()
        for key in ("status", "shape", "device", "is_cuda", "readback"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "unsupported metadata keys"):
                    CudaBenchmarkTensor(
                        buffer,
                        readback=_successful_synthetic_readback(buffer),
                        metadata_updates={key: "spoofed"},
                    )

    def test_public_cuda_benchmark_tensor_rejects_cpu_wrong_device_and_wrong_dtype(self):
        cases = (
            (
                _synthetic_private_buffer(device_type="cpu", device="cpu"),
                ValueError,
                "CUDA buffer metadata",
            ),
            (
                _synthetic_private_buffer(device_index=1, device="cuda:1"),
                ValueError,
                "CUDA device 0",
            ),
            (
                _synthetic_private_buffer(dtype="torch.float64"),
                TypeError,
                "torch.float32",
            ),
        )
        for buffer, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, message):
                    CudaBenchmarkTensor(
                        buffer,
                        readback=_successful_synthetic_readback(buffer),
                    )

    def test_public_cuda_benchmark_tensor_validator_rejects_fallback_device_and_dtype(self):
        base = {
            "schema_version": (
                _cuda_benchmark_tensor.CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION
            ),
            "status": "ok",
            "cpu_fallback": False,
            "device": "cuda:0",
            "device_type": "cuda",
            "device_index": 0,
            "dtype": "torch.float32",
            "is_cuda": True,
            "shape": [benchmark_compile_cuda.WORKLOAD_SHAPE[0]],
            "stride": [1],
            "checksum": "checksum",
            "readback": {"synchronized": True},
        }
        bad_cpu = dict(base, cpu_fallback=True)
        with self.assertRaisesRegex(AssertionError, "CPU fallback"):
            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                bad_cpu
            )
        bad_device = dict(base, device="cuda:1", device_index=1)
        with self.assertRaisesRegex(AssertionError, "CUDA device 0"):
            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                bad_device
            )
        bad_dtype = dict(base, dtype="torch.float64")
        with self.assertRaisesRegex(AssertionError, "torch.float32"):
            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                bad_dtype
            )

    def test_private_cuda_driver_probe_is_not_public_cuda_support(self):
        self.assertNotIn("_cuda_driver_probe", torch.__all__)
        self.assertFalse(hasattr(torch.cuda, "_cuda_driver_probe"))
        self.assertFalse(hasattr(torch.cuda, "driver_probe"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

        probe = _cuda_driver_probe.probe_cuda_driver_device0()
        self.assertEqual(
            probe["schema_version"],
            _cuda_driver_probe.PROBE_SCHEMA_VERSION,
        )
        self.assertEqual(probe["probe"], "torch_rs_private_cuda_driver_device0")
        self.assertIs(probe["public_torch_cuda_api"], False)
        self.assertIn(probe["status"], {"ok", "unavailable", "error"})
        self.assertIn("driver", probe)
        self.assertIn("runtime", probe)
        self.assertIn("cuda_visible_devices", probe)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_float32_buffer_is_not_public_cuda_support(self):
        self.assertNotIn("_cuda_buffer", torch.__all__)
        self.assertFalse(hasattr(torch.cuda, "_cuda_buffer"))
        self.assertFalse(hasattr(torch.cuda, "Float32Buffer"))
        self.assertEqual(
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
            "torch_rs_private_cuda_float32_buffer_v1",
        )

        metadata = _cuda_buffer.float32_metadata((2, 3), device_index=0)
        self.assertEqual(metadata["shape"], [2, 3])
        self.assertEqual(metadata["stride"], [3, 1])
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["device"], "cuda:0")
        self.assertEqual(metadata["device_type"], "cuda")
        self.assertEqual(metadata["device_index"], 0)
        self.assertIs(metadata["requires_grad"], False)
        self.assertIs(metadata["is_contiguous"], True)
        self.assertEqual(_cuda_buffer.element_count((2, 3)), 6)
        self.assertEqual(_cuda_buffer.contiguous_stride((2, 3, 4)), (12, 4, 1))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_runtime_roundtrip_is_not_public_cuda_support(self):
        self.assertNotIn("_cuda_runtime_roundtrip", torch.__all__)
        self.assertFalse(hasattr(torch.cuda, "_cuda_runtime_roundtrip"))
        self.assertFalse(hasattr(torch.cuda, "runtime_roundtrip"))
        self.assertEqual(
            _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM,
            "89c5ee9507c6f91487b4bad190da4a7f",
        )
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_pointwise_kernel_is_not_public_cuda_support(self):
        self.assertNotIn("_cuda_pointwise_kernel", torch.__all__)
        self.assertFalse(hasattr(torch.cuda, "_cuda_pointwise_kernel"))
        self.assertFalse(hasattr(torch.cuda, "pointwise_kernel"))
        self.assertEqual(
            _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM,
            "859e8e6c64e796d56eee827f79e23386",
        )
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_pointwise_reduce_workload_is_not_public_cuda_support(self):
        self.assertNotIn("_cuda_pointwise_reduce_workload", torch.__all__)
        self.assertFalse(hasattr(torch.cuda, "_cuda_pointwise_reduce_workload"))
        self.assertFalse(hasattr(torch.cuda, "pointwise_reduce_workload"))
        self.assertEqual(
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
            benchmark_compile_cuda.WORKLOAD_SHAPE,
        )
        self.assertEqual(_cuda_pointwise_reduce_workload.OUTPUT_SHAPE, (1024,))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_runtime_roundtrip_reports_no_visible_cuda_cleanly(self):
        script = r"""
import json
import torch_rs as torch
from torch_rs import _cuda_runtime_roundtrip

roundtrip = _cuda_runtime_roundtrip.roundtrip_float32_device0()
print(json.dumps({
    "status": roundtrip["status"],
    "reason": roundtrip["reason"],
    "cuda_visible_devices": roundtrip["cuda_visible_devices"],
    "cpu_fallback": roundtrip["cpu_fallback"],
    "checksum_match": roundtrip["checksum_match"],
    "public_cuda_is_available": torch.cuda.is_available(),
    "public_cuda_device_count": torch.cuda.device_count(),
    "public_cuda_is_initialized": torch.cuda.is_initialized(),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        probe = json.loads(completed.stdout)
        self.assertEqual(probe["cuda_visible_devices"], "")
        self.assertEqual(probe["status"], "unavailable")
        self.assertIs(probe["cpu_fallback"], False)
        self.assertIs(probe["checksum_match"], False)
        self.assertIs(probe["public_cuda_is_available"], False)
        self.assertEqual(probe["public_cuda_device_count"], 0)
        self.assertIs(probe["public_cuda_is_initialized"], False)

    def test_private_cuda_runtime_roundtrip_rejects_wrong_visible_mask_early(self):
        script = r"""
import json
import torch_rs as torch
from torch_rs import _cuda_runtime_roundtrip

roundtrip = _cuda_runtime_roundtrip.roundtrip_float32_device0()
print(json.dumps({
    "status": roundtrip["status"],
    "reason": roundtrip["reason"],
    "cuda_visible_devices": roundtrip["cuda_visible_devices"],
    "required_cuda_visible_devices": roundtrip["required_cuda_visible_devices"],
    "cuda_visible_devices_match": roundtrip["cuda_visible_devices_match"],
    "cpu_fallback": roundtrip["cpu_fallback"],
    "device_type": roundtrip["device_type"],
    "device_index": roundtrip["device_index"],
    "buffer_metadata": roundtrip["buffer_metadata"],
    "device_pointer_nonzero": roundtrip["device_pointer_nonzero"],
    "checksum_match": roundtrip["checksum_match"],
    "calls": roundtrip["calls"],
    "public_cuda_is_available": torch.cuda.is_available(),
    "public_cuda_device_count": torch.cuda.device_count(),
    "public_cuda_is_initialized": torch.cuda.is_initialized(),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        probe = json.loads(completed.stdout)
        self.assertEqual(probe["cuda_visible_devices"], "1")
        self.assertEqual(probe["required_cuda_visible_devices"], "0")
        self.assertIs(probe["cuda_visible_devices_match"], False)
        self.assertEqual(probe["status"], "unavailable")
        self.assertEqual(probe["reason"], "CUDA_VISIBLE_DEVICES=0 is required")
        self.assertIs(probe["cpu_fallback"], False)
        self.assertIsNone(probe["device_type"])
        self.assertIsNone(probe["device_index"])
        self.assertIsNone(probe["buffer_metadata"])
        self.assertIs(probe["device_pointer_nonzero"], False)
        self.assertIs(probe["checksum_match"], False)
        self.assertEqual(probe["calls"], {})
        self.assertIs(probe["public_cuda_is_available"], False)
        self.assertEqual(probe["public_cuda_device_count"], 0)
        self.assertIs(probe["public_cuda_is_initialized"], False)

    def test_private_cuda_pointwise_kernel_reports_no_visible_cuda_cleanly(self):
        script = r"""
import json
import torch_rs as torch
from torch_rs import _cuda_pointwise_kernel

pointwise = _cuda_pointwise_kernel.launch_float32_pointwise_device0()
print(json.dumps({
    "status": pointwise["status"],
    "reason": pointwise["reason"],
    "cuda_visible_devices": pointwise["cuda_visible_devices"],
    "cpu_fallback": pointwise["cpu_fallback"],
    "checksum_match": pointwise["checksum_match"],
    "public_cuda_is_available": torch.cuda.is_available(),
    "public_cuda_device_count": torch.cuda.device_count(),
    "public_cuda_is_initialized": torch.cuda.is_initialized(),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        probe = json.loads(completed.stdout)
        self.assertEqual(probe["cuda_visible_devices"], "")
        self.assertEqual(probe["status"], "unavailable")
        self.assertEqual(probe["reason"], "CUDA_VISIBLE_DEVICES=0 is required")
        self.assertIs(probe["cpu_fallback"], False)
        self.assertIs(probe["checksum_match"], False)
        self.assertIs(probe["public_cuda_is_available"], False)
        self.assertEqual(probe["public_cuda_device_count"], 0)
        self.assertIs(probe["public_cuda_is_initialized"], False)

    def test_private_cuda_pointwise_reduce_reports_no_visible_cuda_cleanly(self):
        script = r"""
import json
import torch_rs as torch
from torch_rs import _cuda_pointwise_reduce_workload

rows, columns = _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE
pointwise_reduce = (
    _cuda_pointwise_reduce_workload.launch_h100_float32_pointwise_reduce_device0(
        b"\0" * (rows * columns * 4),
        b"\0" * (columns * 4),
    )
)
print(json.dumps({
    "status": pointwise_reduce["status"],
    "reason": pointwise_reduce["reason"],
    "cuda_visible_devices": pointwise_reduce["cuda_visible_devices"],
    "cpu_fallback": pointwise_reduce["cpu_fallback"],
    "checksum_match": pointwise_reduce["checksum_match"],
    "workload_shape": pointwise_reduce["workload_shape"],
    "public_cuda_tensor_wrapper": pointwise_reduce["public_cuda_tensor_wrapper"],
    "public_cuda_is_available": torch.cuda.is_available(),
    "public_cuda_device_count": torch.cuda.device_count(),
    "public_cuda_is_initialized": torch.cuda.is_initialized(),
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        probe = json.loads(completed.stdout)
        self.assertEqual(probe["cuda_visible_devices"], "")
        self.assertEqual(probe["status"], "unavailable")
        self.assertEqual(probe["reason"], "CUDA_VISIBLE_DEVICES=0 is required")
        self.assertIs(probe["cpu_fallback"], False)
        self.assertIs(probe["checksum_match"], False)
        self.assertEqual(probe["workload_shape"], [1024, 1024])
        self.assertIsNone(probe["public_cuda_tensor_wrapper"])
        self.assertIs(probe["public_cuda_is_available"], False)
        self.assertEqual(probe["public_cuda_device_count"], 0)
        self.assertIs(probe["public_cuda_is_initialized"], False)

    def test_private_cuda_runtime_roundtrip_validator_rejects_cpu_fallback(self):
        with self.assertRaisesRegex(AssertionError, "CPU fallback"):
            benchmark_compile_cuda._require_private_cuda_runtime_roundtrip(
                {
                    "status": "ok",
                    "cpu_fallback": True,
                    "device_type": "cpu",
                    "device_index": None,
                    "checksum_match": True,
                    "device_roundtrip_checksum": (
                        _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM
                    ),
                    "expected_checksum": (
                        _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM
                    ),
                }
            )

    def test_private_cuda_pointwise_kernel_validator_rejects_cpu_fallback(self):
        with self.assertRaisesRegex(AssertionError, "CPU fallback"):
            benchmark_compile_cuda._require_private_cuda_pointwise_kernel(
                {
                    "status": "ok",
                    "cpu_fallback": True,
                    "device_type": "cpu",
                    "device_index": None,
                    "checksum_match": True,
                    "device_output_checksum": (
                        _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM
                    ),
                    "expected_checksum": (
                        _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM
                    ),
                    "launch": {"sync_error": {"result": 0}},
                }
            )

    def test_private_cuda_pointwise_reduce_validator_rejects_cpu_fallback(self):
        with self.assertRaisesRegex(AssertionError, "CPU fallback"):
            benchmark_compile_cuda._require_private_cuda_pointwise_reduce_workload(
                {
                    "status": "ok",
                    "cpu_fallback": True,
                    "device_type": "cpu",
                    "device_index": None,
                    "workload_shape": list(benchmark_compile_cuda.WORKLOAD_SHAPE),
                    "output_shape": [benchmark_compile_cuda.WORKLOAD_SHAPE[0]],
                    "output_metadata_match": True,
                    "checksum_match": True,
                    "device_output_checksum": "checksum",
                    "pytorch_reference_output_checksum": "checksum",
                    "launch": {"sync_error": {"result": 0}},
                }
            )

    def test_private_cuda_runtime_roundtrip_allocates_copies_and_syncs_on_h100(self):
        probe = _reference_cuda_probe()
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            self.skipTest("requires CUDA_VISIBLE_DEVICES=0")

        roundtrip = _cuda_runtime_roundtrip.roundtrip_float32_device0()
        self.assertEqual(
            roundtrip["status"],
            "ok",
            msg=json.dumps(roundtrip, indent=2, sort_keys=True),
        )
        self.assertIs(roundtrip["public_torch_cuda_api"], False)
        self.assertIs(roundtrip["cpu_fallback"], False)
        self.assertEqual(roundtrip["device_type"], "cuda")
        self.assertEqual(roundtrip["device_index"], 0)
        self.assertEqual(roundtrip["cuda_visible_devices"], "0")
        self.assertTrue(roundtrip["cuda_visible_devices_match"])
        self.assertEqual(roundtrip["dtype"], "float32")
        self.assertEqual(
            roundtrip["buffer_schema_version"],
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
        )
        self.assertEqual(
            roundtrip["buffer_metadata"],
            _cuda_buffer.float32_metadata(
                (_cuda_runtime_roundtrip.DEFAULT_ELEMENT_COUNT,),
                device_index=0,
            ),
        )
        self.assertEqual(
            roundtrip["element_count"],
            _cuda_runtime_roundtrip.DEFAULT_ELEMENT_COUNT,
        )
        self.assertEqual(
            roundtrip["host_input_checksum"],
            _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM,
        )
        self.assertEqual(
            roundtrip["device_roundtrip_checksum"],
            _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM,
        )
        self.assertIs(roundtrip["checksum_match"], True)
        self.assertIs(roundtrip["device_pointer_nonzero"], True)
        self.assertEqual(roundtrip["calls"]["cudaGetDeviceCount"]["result"], 0)
        self.assertGreaterEqual(roundtrip["calls"]["cudaGetDeviceCount"]["value"], 1)
        self.assertEqual(roundtrip["calls"]["cudaSetDevice"]["result"], 0)
        self.assertEqual(roundtrip["calls"]["cudaGetDevice"]["result"], 0)
        self.assertEqual(roundtrip["calls"]["cudaMalloc"]["result"], 0)
        self.assertEqual(
            roundtrip["calls"]["cudaMemcpyHostToDevice"]["result"],
            0,
        )
        self.assertEqual(
            roundtrip["calls"]["cudaMemcpyDeviceToHost"]["result"],
            0,
        )
        self.assertEqual(
            roundtrip["calls"]["cudaDeviceSynchronize_after_device_to_host"][
                "result"
            ],
            0,
        )
        self.assertEqual(roundtrip["calls"]["cudaFree"]["result"], 0)
        self.assertIn("H100", roundtrip["device_0"]["name"])
        self.assertEqual(roundtrip["device_0"]["compute_capability"], [9, 0])
        self.assertIsInstance(
            roundtrip["driver"]["version"]["version_text"],
            str,
        )
        self.assertIsInstance(roundtrip["runtime"]["runtime_version_text"], str)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_pointwise_kernel_launches_and_syncs_on_h100(self):
        probe = _reference_cuda_probe()
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            self.skipTest("requires CUDA_VISIBLE_DEVICES=0")

        pointwise = _cuda_pointwise_kernel.launch_float32_pointwise_device0()
        self.assertEqual(
            pointwise["status"],
            "ok",
            msg=json.dumps(pointwise, indent=2, sort_keys=True),
        )
        self.assertIs(pointwise["public_torch_cuda_api"], False)
        self.assertIs(pointwise["cpu_fallback"], False)
        self.assertEqual(pointwise["device_type"], "cuda")
        self.assertEqual(pointwise["device_index"], 0)
        self.assertEqual(pointwise["cuda_visible_devices"], "0")
        self.assertEqual(pointwise["required_cuda_visible_devices"], "0")
        self.assertTrue(pointwise["cuda_visible_devices_match"])
        self.assertTrue(pointwise["single_visible_cuda_device"])
        self.assertEqual(pointwise["dtype"], "float32")
        self.assertEqual(
            pointwise["buffer_schema_version"],
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
        )
        self.assertEqual(
            pointwise["input_metadata"],
            _cuda_buffer.float32_metadata(
                (_cuda_pointwise_kernel.DEFAULT_ELEMENT_COUNT,),
                device_index=0,
            ),
        )
        self.assertEqual(pointwise["output_metadata"], pointwise["input_metadata"])
        self.assertEqual(
            pointwise["element_count"],
            _cuda_pointwise_kernel.DEFAULT_ELEMENT_COUNT,
        )
        self.assertEqual(
            pointwise["host_input_checksum"],
            _cuda_pointwise_kernel.DEFAULT_INPUT_CHECKSUM,
        )
        self.assertEqual(
            pointwise["device_output_checksum"],
            _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM,
        )
        self.assertEqual(
            pointwise["expected_checksum"],
            _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM,
        )
        self.assertIs(pointwise["checksum_match"], True)
        self.assertIs(pointwise["device_input_pointer_nonzero"], True)
        self.assertIs(pointwise["device_output_pointer_nonzero"], True)
        self.assertEqual(pointwise["calls"]["cudaGetDeviceCount"]["result"], 0)
        self.assertGreaterEqual(pointwise["calls"]["cudaGetDeviceCount"]["value"], 1)
        self.assertEqual(pointwise["calls"]["cudaSetDevice"]["result"], 0)
        self.assertEqual(pointwise["calls"]["cudaGetDevice"]["result"], 0)
        self.assertEqual(pointwise["calls"]["cudaMalloc_input"]["result"], 0)
        self.assertEqual(pointwise["calls"]["cudaMalloc_output"]["result"], 0)
        self.assertEqual(
            pointwise["calls"]["cudaMemcpyHostToDevice"]["result"],
            0,
        )
        self.assertEqual(pointwise["launch"]["result"], 0)
        self.assertEqual(pointwise["launch"]["blocks"], 16)
        self.assertEqual(pointwise["launch"]["threads_per_block"], 256)
        self.assertEqual(pointwise["launch"]["launch_error"]["result"], 0)
        self.assertEqual(pointwise["launch"]["sync_error"]["result"], 0)
        self.assertEqual(
            pointwise["calls"]["cudaMemcpyDeviceToHost"]["result"],
            0,
        )
        self.assertEqual(
            pointwise["calls"]["cudaDeviceSynchronize_after_device_to_host"][
                "result"
            ],
            0,
        )
        self.assertEqual(pointwise["calls"]["cudaFree_input"]["result"], 0)
        self.assertEqual(pointwise["calls"]["cudaFree_output"]["result"], 0)
        self.assertIn("H100", pointwise["device_0"]["name"])
        self.assertEqual(pointwise["device_0"]["compute_capability"], [9, 0])
        self.assertIn("H100", pointwise["gpu"]["name"])
        self.assertEqual(pointwise["gpu"]["compute_capability"], [9, 0])
        self.assertIsInstance(
            pointwise["driver"]["version"]["version_text"],
            str,
        )
        self.assertIsInstance(pointwise["runtime"]["runtime_version_text"], str)
        self.assertIs(pointwise["nvcc"]["available"], True)
        self.assertEqual(pointwise["nvcc"]["version"]["returncode"], 0)
        self.assertIn("Cuda compilation tools", pointwise["nvcc"]["version"]["stdout"])
        self.assertEqual(pointwise["build"]["architecture"], "sm_90")
        self.assertIs(pointwise["kernel_library"]["loaded"], True)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_pointwise_reduce_matches_pytorch_compile_on_h100(self):
        probe = _reference_cuda_probe()
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")
        if benchmark_compile_cuda._version_without_local(
            probe["version"],
        ) != benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION:
            self.skipTest(
                "requires PyTorch "
                f"{benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION}, "
                f"got {probe['version']}"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            self.skipTest("requires CUDA_VISIBLE_DEVICES=0")

        import torch as reference_torch

        args = SimpleNamespace(warmups=0, samples=1, repeats=1)
        pytorch_reference, buffers = benchmark_compile_cuda._run_pytorch_reference(
            reference_torch,
            args,
        )

        launch_pointwise_reduce = (
            _cuda_pointwise_reduce_workload.launch_h100_float32_pointwise_reduce_device0
        )
        pointwise_reduce = launch_pointwise_reduce(
            buffers["x_host_bytes"],
            buffers["bias_host_bytes"],
            expected_output_bytes=buffers["expected_output_bytes"],
            expected_output_checksum=buffers["expected_output_checksum"],
            expected_output_metadata=buffers["expected_output_metadata"],
        )
        self.assertEqual(
            pointwise_reduce["status"],
            "ok",
            msg=json.dumps(pointwise_reduce, indent=2, sort_keys=True),
        )
        self.assertIs(pointwise_reduce["public_torch_cuda_api"], False)
        self.assertIs(pointwise_reduce["cpu_fallback"], False)
        self.assertEqual(pointwise_reduce["device_type"], "cuda")
        self.assertEqual(pointwise_reduce["device_index"], 0)
        self.assertEqual(pointwise_reduce["cuda_visible_devices"], "0")
        self.assertEqual(pointwise_reduce["required_cuda_visible_devices"], "0")
        self.assertTrue(pointwise_reduce["cuda_visible_devices_match"])
        self.assertTrue(pointwise_reduce["single_visible_cuda_device"])
        self.assertEqual(pointwise_reduce["dtype"], "float32")
        self.assertEqual(
            pointwise_reduce["buffer_schema_version"],
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
        )
        self.assertEqual(pointwise_reduce["workload_shape"], [1024, 1024])
        self.assertEqual(pointwise_reduce["output_shape"], [1024])
        self.assertEqual(
            pointwise_reduce["input_metadata"],
            [
                _cuda_buffer.float32_metadata((1024, 1024), device_index=0),
                _cuda_buffer.float32_metadata((1024,), device_index=0),
            ],
        )
        self.assertEqual(
            pointwise_reduce["pytorch_reference_output_checksum"],
            pytorch_reference["cold_checksum"],
        )
        self.assertEqual(
            pointwise_reduce["device_output_checksum"],
            pytorch_reference["cold_checksum"],
        )
        self.assertIs(pointwise_reduce["checksum_match"], True)
        self.assertEqual(
            pointwise_reduce["output_metadata"],
            pytorch_reference["output_metadata"],
        )
        wrapper = pointwise_reduce["public_cuda_tensor_wrapper"]
        self.assertEqual(
            wrapper["schema_version"],
            _cuda_benchmark_tensor.CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION,
        )
        self.assertEqual(wrapper["status"], "ok")
        self.assertEqual(wrapper["shape"], [1024])
        self.assertEqual(wrapper["stride"], [1])
        self.assertEqual(wrapper["dtype"], "torch.float32")
        self.assertEqual(wrapper["device"], "cuda:0")
        self.assertEqual(wrapper["device_type"], "cuda")
        self.assertEqual(wrapper["device_index"], 0)
        self.assertIs(wrapper["is_cuda"], True)
        self.assertIs(wrapper["cpu_fallback"], False)
        self.assertEqual(
            wrapper["checksum"],
            pointwise_reduce["device_output_bytes_checksum"],
        )
        self.assertIs(wrapper["readback"]["synchronized"], True)
        self.assertIs(wrapper["readback"]["payload_exposed"], False)
        self.assertNotIn("payload", wrapper["readback"])
        self.assertIs(pointwise_reduce["output_metadata_match"], True)
        self.assertIs(
            pointwise_reduce["output_comparison"]["exact_bytes_match"],
            True,
        )
        self.assertEqual(
            pointwise_reduce["output_comparison"]["mismatched_element_count"],
            0,
        )
        self.assertIs(pointwise_reduce["device_x_pointer_nonzero"], True)
        self.assertIs(pointwise_reduce["device_bias_pointer_nonzero"], True)
        self.assertIs(pointwise_reduce["device_output_pointer_nonzero"], True)
        self.assertEqual(pointwise_reduce["calls"]["cudaGetDeviceCount"]["result"], 0)
        self.assertGreaterEqual(
            pointwise_reduce["calls"]["cudaGetDeviceCount"]["value"],
            1,
        )
        self.assertEqual(pointwise_reduce["calls"]["cudaSetDevice"]["result"], 0)
        self.assertEqual(pointwise_reduce["calls"]["cudaGetDevice"]["result"], 0)
        self.assertEqual(pointwise_reduce["calls"]["cudaMalloc_x"]["result"], 0)
        self.assertEqual(pointwise_reduce["calls"]["cudaMalloc_bias"]["result"], 0)
        self.assertEqual(pointwise_reduce["calls"]["cudaMalloc_output"]["result"], 0)
        self.assertEqual(
            pointwise_reduce["calls"]["cudaMemcpyHostToDevice_x"]["result"],
            0,
        )
        self.assertEqual(
            pointwise_reduce["calls"]["cudaMemcpyHostToDevice_bias"]["result"],
            0,
        )
        self.assertEqual(pointwise_reduce["launch"]["result"], 0)
        self.assertEqual(pointwise_reduce["launch"]["blocks"], 1024)
        self.assertEqual(pointwise_reduce["launch"]["threads_per_block"], 32)
        self.assertEqual(pointwise_reduce["launch"]["launch_error"]["result"], 0)
        self.assertEqual(pointwise_reduce["launch"]["sync_error"]["result"], 0)
        self.assertEqual(
            pointwise_reduce["calls"]["cudaMemcpyDeviceToHost_output"]["result"],
            0,
        )
        self.assertEqual(
            pointwise_reduce["calls"]["cudaDeviceSynchronize_after_device_to_host"][
                "result"
            ],
            0,
        )
        self.assertEqual(pointwise_reduce["calls"]["cudaFree_x"]["result"], 0)
        self.assertEqual(pointwise_reduce["calls"]["cudaFree_bias"]["result"], 0)
        self.assertEqual(pointwise_reduce["calls"]["cudaFree_output"]["result"], 0)
        self.assertIn("H100", pointwise_reduce["device_0"]["name"])
        self.assertEqual(pointwise_reduce["device_0"]["compute_capability"], [9, 0])
        self.assertIn("H100", pointwise_reduce["gpu"]["name"])
        self.assertEqual(pointwise_reduce["gpu"]["compute_capability"], [9, 0])
        self.assertIsInstance(
            pointwise_reduce["driver"]["version"]["version_text"],
            str,
        )
        self.assertIsInstance(
            pointwise_reduce["runtime"]["runtime_version_text"],
            str,
        )
        self.assertIs(pointwise_reduce["nvcc"]["available"], True)
        self.assertEqual(pointwise_reduce["nvcc"]["version"]["returncode"], 0)
        self.assertIn(
            "Cuda compilation tools",
            pointwise_reduce["nvcc"]["version"]["stdout"],
        )
        self.assertEqual(pointwise_reduce["build"]["architecture"], "sm_90")
        self.assertIs(pointwise_reduce["kernel_library"]["loaded"], True)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_torch_compile_inductor_pointwise_reduce_runs_native_cuda_on_h100(self):
        reference_torch = self._require_h100_reference_torch()

        args = SimpleNamespace(warmups=0, samples=1, repeats=1)
        _pytorch_reference, buffers = benchmark_compile_cuda._run_pytorch_reference(
            reference_torch,
            args,
        )
        input_bundle, input_evidence = (
            _cuda_pointwise_reduce_workload.make_h100_float32_pointwise_reduce_inputs_device0(
                buffers["x_host_bytes"],
                buffers["bias_host_bytes"],
            )
        )
        self.assertEqual(
            input_evidence["status"],
            "ok",
            msg=json.dumps(input_evidence, indent=2, sort_keys=True),
        )
        self.assertIsNotNone(input_bundle)
        self.addCleanup(input_bundle.close)

        compiled = torch.compile(
            benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        output = compiled(*input_bundle.inputs)
        self.addCleanup(output._torch_rs_close_private_cuda_buffer)

        self.assertIs(type(output), CudaBenchmarkTensor)
        metadata, execution = benchmark_compile_cuda._require_torch_rs_cuda_compile_output(
            output,
            expected_checksum=buffers["expected_output_checksum"],
            expected_output_bytes=buffers["expected_output_bytes"],
            expected_output_metadata=buffers["expected_output_metadata"],
        )
        self.assertEqual(metadata["shape"], [1024])
        self.assertEqual(metadata["dtype"], "torch.float32")
        self.assertEqual(metadata["device_type"], "cuda")
        self.assertEqual(metadata["device_index"], 0)
        self.assertIs(metadata["native_cuda_compile"], True)
        self.assertEqual(metadata["compile_backend"], "inductor")
        self.assertIs(metadata["compile_fullgraph"], True)
        self.assertIs(metadata["compile_dynamic"], False)
        self.assertIs(metadata["eager_fallback"], False)
        self.assertIs(metadata["forwarded_to_pytorch"], False)
        self.assertEqual(execution["status"], "ok")
        self.assertEqual(execution["workload_shape"], [1024, 1024])
        self.assertEqual(execution["output_shape"], [1024])
        self.assertEqual(
            execution["device_output_checksum"],
            buffers["expected_output_checksum"],
        )
        self.assertIs(execution["output_comparison"]["exact_bytes_match"], True)
        self.assertEqual(
            execution["output_comparison"]["mismatched_element_count"],
            0,
        )
        self.assertEqual(execution["launch"]["result"], 0)
        self.assertEqual(execution["launch"]["blocks"], 1024)
        self.assertEqual(execution["launch"]["threads_per_block"], 32)
        self.assertIsNone(execution["launch"]["sync_error"]["result"])
        self.assertIs(
            execution["launch"]["sync_error"][
                "deferred_to_explicit_timing_boundary"
            ],
            True,
        )
        self.assertIs(execution["kernel_synchronized_in_call"], False)
        self.assertIs(execution["readback_deferred"], True)
        self.assertIs(execution["output_materialized"], True)
        self.assertIs(execution["readback_synchronized"], True)
        self.assertIs(execution["output_buffer_pool"]["enabled"], True)
        self.assertGreaterEqual(
            execution["output_buffer_pool"]["pool_initial_capacity"],
            2,
        )
        self.assertIs(
            execution["output_buffer_pool"]["allocated_in_execute"],
            False,
        )
        self.assertIs(execution["kernel_library"]["loaded"], True)

        classification = benchmark_compile_cuda.classify_torch_rs_cuda_compile_evidence(
            {
                "implementation": "torch_rs",
                "status": "ok",
                "workload_version": benchmark_compile_cuda.WORKLOAD_VERSION,
                "compile_backend": "inductor",
                "compile_fullgraph": True,
                "compile_dynamic": False,
                "input_device_type": "cuda",
                "output_device_type": "cuda",
                "native_cuda_compile": True,
                "eager_fallback": False,
                "forwarded_to_pytorch": False,
            }
        )
        self.assertIs(classification["eligible_cuda_compile_evidence"], True)
        self.assertEqual(classification["score_credit"], 1.0)
        self.assertEqual(classification["rejection_reasons"], [])
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_torch_compile_inductor_pointwise_reduce_prepares_executor_once(self):
        class FakePreparedExecutor:
            def __init__(self):
                self.calls = 0

            def metadata(self):
                return {
                    "schema_version": (
                        _cuda_pointwise_reduce_workload
                        .POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION
                    ),
                    "status": "ok",
                    "prepared": True,
                    "setup_hoisted_to_compile_wrapper": True,
                    "preparation_id": "fake-prepared-executor",
                    "invocation_count": self.calls,
                }

            def execute(self, x, bias):
                self.calls += 1
                return {
                    "call_count": self.calls,
                    "x": x,
                    "bias": bias,
                }

        prepared_executor = FakePreparedExecutor()
        prepare_calls = []

        def prepare_executor(*, required_cuda_visible_devices="0"):
            prepare_calls.append(required_cuda_visible_devices)
            return prepared_executor

        with unittest.mock.patch.object(
            _cuda_pointwise_reduce_workload,
            "prepare_h100_float32_pointwise_reduce_compiled_executor_device0",
            side_effect=prepare_executor,
        ):
            compiled = torch.compile(
                benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )

        self.assertEqual(prepare_calls, ["0"])
        self.assertIs(
            compiled._torch_rs_cuda_compile_executor,
            prepared_executor,
        )
        self.assertEqual(
            compiled._torch_rs_cuda_compile_preparation["invocation_count"],
            0,
        )

        first = compiled("x", "bias")
        second = compiled("x", "bias")

        self.assertEqual(first["call_count"], 1)
        self.assertEqual(second["call_count"], 2)
        self.assertEqual(prepare_calls, ["0"])
        self.assertEqual(
            compiled._torch_rs_cuda_compile_executor.metadata()[
                "invocation_count"
            ],
            2,
        )

    def test_torch_compile_inductor_pointwise_reduce_reprepares_after_reset(
        self,
    ):
        class FakePreparedExecutor:
            def __init__(self, name):
                self.name = name
                self.calls = 0
                self.closed = False

            def metadata(self):
                return {
                    "schema_version": (
                        _cuda_pointwise_reduce_workload
                        .POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION
                    ),
                    "status": "ok",
                    "prepared": True,
                    "setup_hoisted_to_compile_wrapper": True,
                    "preparation_id": self.name,
                    "invocation_count": self.calls,
                    "closed": self.closed,
                }

            def execute(self, x, bias):
                if self.closed:
                    raise RuntimeError("compiled CUDA executor is closed")
                self.calls += 1
                return {
                    "executor": self.name,
                    "call_count": self.calls,
                    "x": x,
                    "bias": bias,
                }

            def close(self):
                self.closed = True
                return {"closed": True, "executor": self.name}

        prepared_executors = []
        prepare_calls = []

        def prepare_executor(*, required_cuda_visible_devices="0"):
            prepare_calls.append(required_cuda_visible_devices)
            executor = FakePreparedExecutor(
                f"fake-prepared-executor-{len(prepared_executors) + 1}",
            )
            prepared_executors.append(executor)
            return executor

        try:
            with unittest.mock.patch.object(
                _cuda_pointwise_reduce_workload,
                "prepare_h100_float32_pointwise_reduce_compiled_executor_device0",
                side_effect=prepare_executor,
            ):
                compiled = torch.compile(
                    benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                    backend="inductor",
                    fullgraph=True,
                    dynamic=False,
                )

                first = compiled("x", "bias")
                torch.compiler.reset()
                self.assertIs(prepared_executors[0].closed, True)
                second = compiled("x", "bias")

            self.assertEqual(prepare_calls, ["0", "0"])
            self.assertEqual(first["executor"], "fake-prepared-executor-1")
            self.assertEqual(second["executor"], "fake-prepared-executor-2")
            self.assertEqual(second["call_count"], 1)
            self.assertIs(
                compiled._torch_rs_cuda_compile_executor,
                prepared_executors[1],
            )
            self.assertEqual(
                compiled._torch_rs_cuda_compile_preparation["preparation_id"],
                "fake-prepared-executor-2",
            )
        finally:
            for executor in prepared_executors:
                executor.close()

    def test_private_cuda_buffer_pool_reuses_released_buffer_once(self):
        runtime = _FakeCudaRuntime()
        pool = _cuda_runtime_ownership.PrivateCudaBufferPool(
            runtime,
            (2,),
            name_prefix="unit_pool",
            owner_id="unit",
        )
        try:
            allocations = pool.preallocate(1)
            self.assertEqual(len(allocations), 1)
            self.assertEqual(pool.metadata()["available_buffers"], 1)

            first = pool.acquire()
            self.assertEqual(first.acquisition["source"], "preallocated")
            self.assertEqual(pool.metadata()["live_buffers"], 1)
            release = first.release()
            self.assertEqual(release["released_to_pool"], True)
            self.assertIsNone(first.release())

            second = pool.acquire()
            self.assertIs(second.buffer, first.buffer)
            self.assertEqual(
                second.acquisition["source"],
                "reused_released_allocation",
            )
            self.assertEqual(pool.metadata()["reuse_count"], 1)
            second.release()
        finally:
            close = pool.close()

        self.assertEqual(close["free_count"], 1)
        self.assertEqual(len(runtime.frees), 1)

    def test_private_cuda_buffer_pool_keeps_acquisition_and_release_compact(self):
        runtime = _FakeCudaRuntime()
        pool = _cuda_runtime_ownership.PrivateCudaBufferPool(
            runtime,
            (2,),
            name_prefix="unit_pool",
            owner_id="unit",
        )
        try:
            pool.preallocate(1)
            lease = pool.acquire()

            self.assertIsInstance(
                lease.acquisition_token,
                _cuda_runtime_ownership.PrivateCudaBufferPoolAcquisition,
            )
            self.assertNotIsInstance(lease.acquisition_token, dict)
            self.assertEqual(
                lease.acquisition["schema_version"],
                _cuda_runtime_ownership.CUDA_BUFFER_POOL_SCHEMA_VERSION,
            )
            self.assertEqual(lease.acquisition["source"], "preallocated")

            release = lease.release(compact=True)
            self.assertIsInstance(
                release,
                _cuda_runtime_ownership.PrivateCudaBufferPoolRelease,
            )
            self.assertNotIsInstance(release, dict)
            self.assertEqual(
                release["schema_version"],
                _cuda_runtime_ownership.CUDA_BUFFER_POOL_SCHEMA_VERSION,
            )
            self.assertIs(release["released_to_pool"], True)
            self.assertEqual(dict(release)["buffer_name"], lease.buffer.name)
        finally:
            pool.close()

    def test_prepared_executor_output_pool_reuses_released_buffers(self):
        executor, runtime = _fake_prepared_executor()
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        try:
            initial_malloc_count = len(runtime.mallocs)
            first = executor.execute(x, bias)
            second = executor.execute(x, bias)
            try:
                first_execution = first.metadata()["compile_execution"]
                second_execution = second.metadata()["compile_execution"]
                first_pool = first_execution["output_buffer_pool"]
                second_pool = second_execution["output_buffer_pool"]

                self.assertEqual(first_pool["source"], "preallocated")
                self.assertEqual(second_pool["source"], "preallocated")
                self.assertNotEqual(
                    first_pool["buffer_name"],
                    second_pool["buffer_name"],
                )
                self.assertEqual(len(runtime.mallocs), initial_malloc_count)

                first._torch_rs_close_private_cuda_buffer()
                with self.assertRaisesRegex(ValueError, "no longer live"):
                    first.metadata()

                third = executor.execute(x, bias)
                try:
                    third_execution = third.metadata()["compile_execution"]
                    third_pool = third_execution["output_buffer_pool"]
                    self.assertEqual(
                        third_pool["buffer_name"],
                        first_pool["buffer_name"],
                    )
                    self.assertEqual(
                        third_pool["source"],
                        "reused_released_allocation",
                    )
                    self.assertIs(
                        third_pool["reused_released_allocation"],
                        True,
                    )
                    self.assertIs(third_pool["allocated_in_execute"], False)
                    self.assertEqual(len(runtime.mallocs), initial_malloc_count)
                finally:
                    third._torch_rs_close_private_cuda_buffer()
            finally:
                second._torch_rs_close_private_cuda_buffer()

            pool = executor.metadata()["output_pool"]
            self.assertEqual(pool["initial_capacity"], 2)
            self.assertEqual(pool["allocation_count"], 2)
            self.assertGreaterEqual(pool["reuse_count"], 1)
            self.assertEqual(pool["live_buffers"], 0)
            self.assertEqual(pool["available_buffers"], 2)
        finally:
            close = executor.close()
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

        self.assertEqual(close["free_count"], 2)
        self.assertEqual(len(close["calls"]), 2)

    def test_prepared_executor_materializes_full_evidence_lazily(self):
        executor, runtime = _fake_prepared_executor()
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        try:
            output = executor.execute(x, bias)
            self.addCleanup(output._torch_rs_close_private_cuda_buffer)

            pending_execution = output._metadata["compile_execution"]
            self.assertIsInstance(
                pending_execution,
                _cuda_pointwise_reduce_workload._CompiledExecutionToken,
            )
            self.assertEqual(runtime.copies, [])
            self.assertEqual(runtime.syncs, [])

            metadata = output.metadata()
            execution = metadata["compile_execution"]

            self.assertIsInstance(execution, dict)
            self.assertEqual(execution["status"], "ok")
            self.assertEqual(
                execution["input_metadata"],
                [
                    _cuda_buffer.float32_metadata(
                        _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
                    ),
                    _cuda_buffer.float32_metadata(
                        (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
                    ),
                ],
            )
            self.assertEqual(
                execution["output_metadata"],
                _cuda_buffer.float32_metadata(
                    _cuda_pointwise_reduce_workload.OUTPUT_SHAPE,
                ),
            )
            self.assertEqual(execution["executor"]["invocation_index"], 1)
            self.assertIs(execution["executor"]["reused_prepared_executor"], False)
            self.assertEqual(
                execution["calls"]["cudaSetDevice_before_launch"]["result"],
                0,
            )
            self.assertEqual(
                execution["calls"]["cudaGetDevice_before_launch"]["value"],
                0,
            )
            self.assertEqual(execution["launch"]["blocks"], 1024)
            self.assertEqual(execution["launch"]["threads_per_block"], 32)
            self.assertIsNone(execution["launch"]["sync_error"]["result"])
            self.assertEqual(len(runtime.copies), 1)
            self.assertEqual(len(runtime.syncs), 1)
        finally:
            executor.close()
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

    def test_prepared_executor_output_pool_does_not_reuse_live_outputs(self):
        executor, runtime = _fake_prepared_executor()
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        outputs = []
        try:
            initial_malloc_count = len(runtime.mallocs)
            for _ in range(3):
                outputs.append(executor.execute(x, bias))

            executions = [
                output.metadata()["compile_execution"] for output in outputs
            ]
            pool_names = [
                execution["output_buffer_pool"]["buffer_name"]
                for execution in executions
            ]
            self.assertEqual(len(set(pool_names)), 3)
            self.assertEqual(
                executions[2]["output_buffer_pool"]["source"],
                "allocated_during_execute",
            )
            self.assertIs(
                executions[2]["output_buffer_pool"]["allocated_in_execute"],
                True,
            )
            self.assertEqual(len(runtime.mallocs), initial_malloc_count + 1)
            self.assertEqual(executor.metadata()["output_pool"]["live_buffers"], 3)
        finally:
            for output in outputs:
                output._torch_rs_close_private_cuda_buffer()
            close = executor.close()
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

        self.assertEqual(close["free_count"], 3)
        self.assertEqual(len(close["calls"]), 3)
        self.assertEqual(len(set(runtime.frees)), len(runtime.frees))

    def test_prepared_executor_close_defers_live_output_free(self):
        executor, runtime = _fake_prepared_executor()
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        try:
            output = executor.execute(x, bias)
            close = executor.close()

            self.assertEqual(close["free_count"], 1)
            self.assertEqual(close["deferred_live_buffers"], 1)
            self.assertEqual(len(close["calls"]), 1)
            self.assertIs(executor.metadata()["closed"], True)
            self.assertEqual(executor.metadata()["output_pool"]["live_buffers"], 1)
            with self.assertRaisesRegex(RuntimeError, "executor is closed"):
                executor.execute(x, bias)
            with self.assertRaisesRegex(RuntimeError, "executor is closed"):
                executor.synchronize(label="closed")
            execution = output.metadata()["compile_execution"]
            self.assertEqual(execution["status"], "ok")
            release = output._torch_rs_close_private_cuda_buffer()
            self.assertEqual(release["released_to_pool"], False)
            self.assertIs(release["freed_after_executor_close"], True)
            self.assertEqual(release["pool_free_count"], 2)
            self.assertEqual(executor.metadata()["output_pool"]["live_buffers"], 0)
            self.assertIs(executor.close()["already_closed"], True)
        finally:
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

        self.assertEqual(len(runtime.frees), 4)
        self.assertEqual(len(set(runtime.frees)), len(runtime.frees))

    def test_prepared_executor_lazy_readback_rejects_visibility_mask_change(self):
        with unittest.mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            executor, runtime = _fake_prepared_executor(
                required_cuda_visible_devices="0",
            )
            x = _fake_input_tensor(
                runtime,
                "x",
                _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
            )
            bias = _fake_input_tensor(
                runtime,
                "bias",
                (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
            )
            output = None
            try:
                output = executor.execute(x, bias)
                with unittest.mock.patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": "1"},
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        "CUDA_VISIBLE_DEVICES=0 is required",
                    ):
                        output.metadata()
            finally:
                if output is not None:
                    output._torch_rs_close_private_cuda_buffer()
                executor.close()
                x._torch_rs_close_private_cuda_buffer()
                bias._torch_rs_close_private_cuda_buffer()

    def test_prepared_executor_lazy_readback_rejects_device_change(self):
        executor, runtime = _fake_prepared_executor()
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        output = None
        try:
            output = executor.execute(x, bias)
            runtime.current_device = 1
            with self.assertRaisesRegex(RuntimeError, "CUDA device changed"):
                output.metadata()
        finally:
            runtime.current_device = 0
            if output is not None:
                output._torch_rs_close_private_cuda_buffer()
            executor.close()
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

    def test_prepared_executor_restores_device0_before_launch(self):
        runtime = _FakeCudaRuntime()

        class RecordingLibrary(_FakePointwiseReduceLibrary):
            def __init__(self, observed_runtime):
                self.observed_runtime = observed_runtime
                self.launch_devices = []

            def torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
                self,
                *arguments,
            ):
                self.launch_devices.append(self.observed_runtime.current_device)
                return super().torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
                    *arguments,
                )

        library = RecordingLibrary(runtime)
        executor, runtime = _fake_prepared_executor(
            runtime=runtime,
            kernel_library=library,
        )
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        output = None
        try:
            runtime.current_device = 1

            output = executor.execute(x, bias)
            execution = output.metadata()["compile_execution"]

            self.assertEqual(library.launch_devices, [0])
            self.assertEqual(runtime.set_devices[-1], 0)
            self.assertEqual(runtime.current_device, 0)
            self.assertEqual(
                execution["calls"]["cudaSetDevice_before_launch"]["result"],
                0,
            )
            self.assertEqual(
                execution["calls"]["cudaGetDevice_before_launch"]["result"],
                0,
            )
            self.assertEqual(
                execution["calls"]["cudaGetDevice_before_launch"]["value"],
                0,
            )
        finally:
            runtime.current_device = 0
            if output is not None:
                output._torch_rs_close_private_cuda_buffer()
            executor.close()
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

    def test_compiler_reset_closes_registered_cuda_executors(self):
        executor, runtime = _fake_prepared_executor()
        x = _fake_input_tensor(
            runtime,
            "x",
            _cuda_pointwise_reduce_workload.WORKLOAD_SHAPE,
        )
        bias = _fake_input_tensor(
            runtime,
            "bias",
            (_cuda_pointwise_reduce_workload.WORKLOAD_SHAPE[1],),
        )
        try:
            _compiler_state.register_native_cuda_compile_executor(executor)
            output = executor.execute(x, bias)

            torch.compiler.reset()

            self.assertIs(executor.metadata()["closed"], True)
            self.assertEqual(executor.metadata()["output_pool"]["live_buffers"], 1)
            with self.assertRaisesRegex(RuntimeError, "executor is closed"):
                executor.execute(x, bias)
            execution = output.metadata()["compile_execution"]
            self.assertEqual(execution["status"], "ok")
            release = output._torch_rs_close_private_cuda_buffer()
            self.assertEqual(release["released_to_pool"], False)
            self.assertIs(release["freed_after_executor_close"], True)
            self.assertEqual(release["pool_free_count"], 2)
            self.assertEqual(executor.metadata()["output_pool"]["live_buffers"], 0)
        finally:
            executor.close()
            x._torch_rs_close_private_cuda_buffer()
            bias._torch_rs_close_private_cuda_buffer()

        self.assertEqual(len(runtime.frees), 4)
        self.assertEqual(len(set(runtime.frees)), len(runtime.frees))

    def test_pytorch_reference_timing_excludes_checksum_materialization(self):
        events = []

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                synchronize=lambda index: events.append(f"sync:{index}"),
            ),
        )

        once_output = object()

        def compiled_once(*inputs):
            self.assertEqual(inputs, ("x", "bias"))
            events.append("compiled_once")
            return once_output

        once_times = iter([1000, 2000])

        def perf_counter_once():
            events.append("time")
            return next(once_times)

        def checksum_once(output):
            self.assertIs(output, once_output)
            events.append("checksum_once")
            return "once"

        with unittest.mock.patch.object(
            benchmark_compile_cuda.time,
            "perf_counter_ns",
            side_effect=perf_counter_once,
        ), unittest.mock.patch.object(
            benchmark_compile_cuda,
            "_checksum_tensor",
            side_effect=checksum_once,
        ):
            elapsed_ns, checksum, output = benchmark_compile_cuda._time_once(
                fake_torch,
                compiled_once,
                ("x", "bias"),
            )

        self.assertEqual(elapsed_ns, 1000)
        self.assertEqual(checksum, "once")
        self.assertIs(output, once_output)
        self.assertEqual(
            events,
            [
                "sync:0",
                "time",
                "compiled_once",
                "sync:0",
                "time",
                "checksum_once",
            ],
        )

        events.clear()
        repeated_outputs = []

        def compiled_repeated(*inputs):
            self.assertEqual(inputs, ("x", "bias"))
            output = object()
            repeated_outputs.append(output)
            events.append(f"compiled_repeated:{len(repeated_outputs)}")
            return output

        repeated_times = iter([3000, 4300])

        def perf_counter_repeated():
            events.append("time")
            return next(repeated_times)

        def checksum_repeated(output):
            self.assertIs(output, repeated_outputs[-1])
            events.append("checksum_repeated")
            return "repeated"

        with unittest.mock.patch.object(
            benchmark_compile_cuda.time,
            "perf_counter_ns",
            side_effect=perf_counter_repeated,
        ), unittest.mock.patch.object(
            benchmark_compile_cuda,
            "_checksum_tensor",
            side_effect=checksum_repeated,
        ):
            elapsed_ns, checksum = benchmark_compile_cuda._time_repeated(
                fake_torch,
                compiled_repeated,
                ("x", "bias"),
                3,
            )

        self.assertEqual(elapsed_ns, 1300)
        self.assertEqual(checksum, "repeated")
        self.assertEqual(
            events,
            [
                "sync:0",
                "time",
                "compiled_repeated:1",
                "compiled_repeated:2",
                "compiled_repeated:3",
                "sync:0",
                "time",
                "checksum_repeated",
            ],
        )

    def test_torch_rs_cuda_compile_repeated_closes_temporary_outputs(self):
        class FakeOutput:
            def __init__(self, index):
                self.index = index
                self.close_calls = 0

            def _torch_rs_close_private_cuda_buffer(self):
                self.close_calls += 1
                return {"result": 0, "index": self.index}

        outputs = []

        def compiled(*inputs):
            self.assertEqual(inputs, ("x", "bias"))
            output = FakeOutput(len(outputs) + 1)
            outputs.append(output)
            return output

        class FakeExecutor:
            def __init__(self):
                self.sync_labels = []

            def synchronize(self, *, label):
                self.sync_labels.append(label)
                return {"result": 0, "label": label}

        fake_executor = FakeExecutor()
        compiled._torch_rs_cuda_compile_executor = fake_executor

        output, elapsed_ns, timing_boundary = (
            benchmark_compile_cuda._time_torch_rs_cuda_compile_repeated(
                compiled,
                ("x", "bias"),
                3,
            )
        )

        self.assertGreaterEqual(elapsed_ns, 0)
        self.assertIs(output, outputs[-1])
        self.assertEqual([output.close_calls for output in outputs], [1, 1, 0])
        self.assertEqual(
            fake_executor.sync_labels,
            ["before_repeated_compiled_calls", "after_repeated_compiled_calls"],
        )
        self.assertTrue(timing_boundary["materialization_outside_timed_region"])
        self.assertEqual(timing_boundary["timed_call_count"], 3)

    def test_torch_compile_inductor_wrong_mask_rejects_before_cuda_probe(self):
        workload = benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32
        mask_attribute = "_torch_rs_cuda_compile_required_cuda_visible_devices"
        missing_attribute = object()
        previous_mask = getattr(workload, mask_attribute, missing_attribute)
        setattr(workload, mask_attribute, "0")
        try:
            with unittest.mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "1"},
            ), unittest.mock.patch.object(
                _cuda_driver_probe,
                "probe_cuda_driver_device0",
                side_effect=AssertionError("driver probe should not run"),
            ) as driver_probe, unittest.mock.patch.object(
                _cuda_driver_probe,
                "_load_shared_library",
                side_effect=AssertionError("CUDA runtime load should not run"),
            ) as load_shared_library, unittest.mock.patch.object(
                _cuda_pointwise_kernel,
                "_nvcc_provenance",
                side_effect=AssertionError("nvcc provenance should not run"),
            ) as nvcc_provenance:
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "CUDA_VISIBLE_DEVICES=0 is required",
                ):
                    torch.compile(
                        workload,
                        backend="inductor",
                        fullgraph=True,
                        dynamic=False,
                    )

                driver_probe.assert_not_called()
                load_shared_library.assert_not_called()
                nvcc_provenance.assert_not_called()
        finally:
            if previous_mask is missing_attribute:
                try:
                    delattr(workload, mask_attribute)
                except AttributeError:
                    pass
            else:
                setattr(workload, mask_attribute, previous_mask)

    def test_torch_compile_inductor_pointwise_reduce_reuses_executor_on_h100(self):
        self._require_h100_reference_torch()

        rows, columns = benchmark_compile_cuda.WORKLOAD_SHAPE
        x_host_bytes = b"\0" * (rows * columns * 4)
        bias_host_bytes = b"\0" * (columns * 4)
        expected_output_bytes = b"\0" * (rows * 4)
        expected_output_metadata = _cuda_buffer.float32_metadata(
            (rows,),
            device_index=0,
        )
        expected_checksum = (
            _cuda_pointwise_reduce_workload._checksum_tensor_metadata_values(
                expected_output_bytes,
                expected_output_metadata,
            )
        )
        input_bundle, input_evidence = (
            _cuda_pointwise_reduce_workload
            .make_h100_float32_pointwise_reduce_inputs_device0(
                x_host_bytes,
                bias_host_bytes,
            )
        )
        self.assertEqual(
            input_evidence["status"],
            "ok",
            msg=json.dumps(input_evidence, indent=2, sort_keys=True),
        )
        self.assertIsNotNone(input_bundle)
        self.addCleanup(input_bundle.close)

        compiled = torch.compile(
            benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        preparation = compiled._torch_rs_cuda_compile_preparation
        benchmark_compile_cuda._require_torch_rs_cuda_compile_prepared_executor(
            preparation,
            expected_invocation_count=0,
        )

        outputs = [
            compiled(*input_bundle.inputs),
            compiled(*input_bundle.inputs),
        ]
        for output in outputs:
            self.addCleanup(output._torch_rs_close_private_cuda_buffer)

        executions = []
        for output in outputs:
            metadata, execution = benchmark_compile_cuda._compile_execution_from_output(
                output
            )
            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                metadata
            )
            self.assertEqual(execution["status"], "ok")
            self.assertEqual(execution["device_output_checksum"], expected_checksum)
            self.assertEqual(execution["output_metadata"], expected_output_metadata)
            comparison = benchmark_compile_cuda._compare_compiled_output_bytes(
                output,
                execution,
                expected_output_bytes,
            )
            self.assertIs(comparison["exact_bytes_match"], True)
            self.assertEqual(comparison["mismatched_element_count"], 0)
            executions.append(execution)

        first_executor = executions[0]["executor"]
        second_executor = executions[1]["executor"]
        self.assertEqual(
            first_executor["preparation_id"],
            preparation["preparation_id"],
        )
        self.assertEqual(
            second_executor["preparation_id"],
            preparation["preparation_id"],
        )
        self.assertEqual(first_executor["invocation_index"], 1)
        self.assertIs(first_executor["reused_prepared_executor"], False)
        self.assertEqual(second_executor["invocation_index"], 2)
        self.assertIs(second_executor["reused_prepared_executor"], True)
        first_pool = executions[0]["output_buffer_pool"]
        second_pool = executions[1]["output_buffer_pool"]
        self.assertEqual(first_pool["source"], "preallocated")
        self.assertEqual(second_pool["source"], "preallocated")
        self.assertIs(first_pool["allocated_in_execute"], False)
        self.assertIs(second_pool["allocated_in_execute"], False)
        self.assertNotEqual(first_pool["buffer_name"], second_pool["buffer_name"])

        outputs[0]._torch_rs_close_private_cuda_buffer()
        third_output = compiled(*input_bundle.inputs)
        self.addCleanup(third_output._torch_rs_close_private_cuda_buffer)
        third_metadata, third_execution = (
            benchmark_compile_cuda._compile_execution_from_output(third_output)
        )
        benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
            third_metadata
        )
        self.assertEqual(third_execution["device_output_checksum"], expected_checksum)
        third_pool = third_execution["output_buffer_pool"]
        self.assertEqual(third_pool["buffer_name"], first_pool["buffer_name"])
        self.assertEqual(third_pool["source"], "reused_released_allocation")
        self.assertIs(third_pool["allocated_in_execute"], False)
        self.assertIs(third_pool["reused_released_allocation"], True)

        self.assertEqual(executions[0]["build"], preparation["build"])
        self.assertEqual(executions[1]["build"], preparation["build"])
        self.assertEqual(
            executions[0]["kernel_library"],
            preparation["kernel_library"],
        )
        self.assertEqual(
            executions[1]["kernel_library"],
            preparation["kernel_library"],
        )
        benchmark_compile_cuda._require_torch_rs_cuda_compile_prepared_executor(
            compiled._torch_rs_cuda_compile_executor.metadata(),
            expected_invocation_count=3,
        )
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_torch_compile_inductor_pointwise_reduce_reprepares_after_reset_on_h100(
        self,
    ):
        self._require_h100_reference_torch()

        rows, columns = benchmark_compile_cuda.WORKLOAD_SHAPE
        x_host_bytes = b"\0" * (rows * columns * 4)
        bias_host_bytes = b"\0" * (columns * 4)
        expected_output_bytes = b"\0" * (rows * 4)
        expected_output_metadata = _cuda_buffer.float32_metadata(
            (rows,),
            device_index=0,
        )
        expected_output_bytes_checksum = (
            _cuda_pointwise_reduce_workload._checksum_output_bytes(
                expected_output_bytes,
                rows,
                columns,
            )
        )
        expected_checksum = (
            _cuda_pointwise_reduce_workload._checksum_tensor_metadata_values(
                expected_output_bytes,
                expected_output_metadata,
            )
        )
        input_bundle = None
        compiled = None
        first_output = None
        second_output = None
        try:
            input_bundle, input_evidence = (
                _cuda_pointwise_reduce_workload
                .make_h100_float32_pointwise_reduce_inputs_device0(
                    x_host_bytes,
                    bias_host_bytes,
                )
            )
            self.assertEqual(input_evidence["status"], "ok")
            self.assertIsNotNone(input_bundle)

            compiled = torch.compile(
                benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )
            first_executor = compiled._torch_rs_cuda_compile_executor
            first_output = compiled(*input_bundle.inputs)
            first_metadata, first_execution = (
                benchmark_compile_cuda._compile_execution_from_output(first_output)
            )
            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                first_metadata,
            )
            self.assertEqual(first_execution["status"], "ok")
            self.assertEqual(
                first_execution["device_output_checksum"],
                expected_checksum,
            )

            torch.compiler.reset()
            self.assertIs(first_executor.closed, True)
            self.assertEqual(
                first_executor.metadata()["output_pool"]["live_buffers"],
                1,
            )
            self.assertEqual(first_output.checksum, expected_output_bytes_checksum)
            self.assertEqual(
                first_output.metadata()["compile_execution"]["status"],
                "ok",
            )

            second_output = compiled(*input_bundle.inputs)
            second_metadata, second_execution = (
                benchmark_compile_cuda._compile_execution_from_output(second_output)
            )
            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                second_metadata,
            )
            self.assertIsNot(
                compiled._torch_rs_cuda_compile_executor,
                first_executor,
            )
            self.assertIs(compiled._torch_rs_cuda_compile_executor.closed, False)
            self.assertEqual(second_execution["status"], "ok")
            self.assertEqual(
                second_execution["device_output_checksum"],
                expected_checksum,
            )
            self.assertEqual(
                second_execution["output_metadata"],
                expected_output_metadata,
            )
            self.assertEqual(second_execution["executor"]["invocation_index"], 1)
            comparison = benchmark_compile_cuda._compare_compiled_output_bytes(
                second_output,
                second_execution,
                expected_output_bytes,
            )
            self.assertIs(comparison["exact_bytes_match"], True)
        finally:
            if first_output is not None:
                first_output._torch_rs_close_private_cuda_buffer()
            if second_output is not None:
                second_output._torch_rs_close_private_cuda_buffer()
            if compiled is not None:
                compiled._torch_rs_cuda_compile_executor.close()
            if input_bundle is not None:
                input_bundle.close()

    def test_torch_compile_inductor_pointwise_reduce_restores_device0_before_launch_on_h100(
        self,
    ):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not cuda_visible_devices or "," not in cuda_visible_devices:
            self.skipTest("requires at least two visible CUDA devices")

        probe = _reference_cuda_probe(cuda_visible_devices=cuda_visible_devices)
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available") or probe.get("device_count", 0) < 2:
            self.skipTest(
                f"requires CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} "
                "to expose at least two GPUs"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )

        workload = benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32
        mask_attribute = "_torch_rs_cuda_compile_required_cuda_visible_devices"
        missing_attribute = object()
        previous_mask = getattr(workload, mask_attribute, missing_attribute)
        setattr(workload, mask_attribute, None)

        rows, columns = benchmark_compile_cuda.WORKLOAD_SHAPE
        x_host_bytes = b"\0" * (rows * columns * 4)
        bias_host_bytes = b"\0" * (columns * 4)
        expected_output_bytes = b"\0" * (rows * 4)
        expected_output_metadata = _cuda_buffer.float32_metadata(
            (rows,),
            device_index=0,
        )
        expected_checksum = (
            _cuda_pointwise_reduce_workload._checksum_tensor_metadata_values(
                expected_output_bytes,
                expected_output_metadata,
            )
        )
        input_bundle = None
        compiled = None
        output = None
        runtime = None
        try:
            input_bundle, input_evidence = (
                _cuda_pointwise_reduce_workload
                .make_h100_float32_pointwise_reduce_inputs_device0(
                    x_host_bytes,
                    bias_host_bytes,
                    required_cuda_visible_devices=None,
                )
            )
            self.assertEqual(input_evidence["status"], "ok")
            self.assertIsNotNone(input_bundle)

            compiled = torch.compile(
                workload,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )
            runtime = compiled._torch_rs_cuda_compile_executor._runtime
            set_device_1 = _cuda_pointwise_reduce_workload._runtime_call(
                runtime,
                "cudaSetDevice",
                1,
            )
            self.assertEqual(set_device_1["result"], 0)

            output = compiled(*input_bundle.inputs)
            metadata, execution = benchmark_compile_cuda._compile_execution_from_output(
                output,
            )

            benchmark_compile_cuda._require_public_cuda_tensor_wrapper_evidence(
                metadata,
            )
            self.assertEqual(execution["status"], "ok")
            self.assertEqual(execution["device_output_checksum"], expected_checksum)
            self.assertEqual(execution["output_metadata"], expected_output_metadata)
            self.assertEqual(
                execution["calls"]["cudaSetDevice_before_launch"]["result"],
                0,
            )
            self.assertEqual(
                execution["calls"]["cudaGetDevice_before_launch"]["result"],
                0,
            )
            self.assertEqual(
                execution["calls"]["cudaGetDevice_before_launch"]["value"],
                0,
            )
            comparison = benchmark_compile_cuda._compare_compiled_output_bytes(
                output,
                execution,
                expected_output_bytes,
            )
            self.assertIs(comparison["exact_bytes_match"], True)
        finally:
            if runtime is not None:
                _cuda_pointwise_reduce_workload._runtime_call(
                    runtime,
                    "cudaSetDevice",
                    0,
                )
            if output is not None:
                output._torch_rs_close_private_cuda_buffer()
            if compiled is not None:
                compiled._torch_rs_cuda_compile_executor.close()
            if input_bundle is not None:
                input_bundle.close()
            if previous_mask is missing_attribute:
                try:
                    delattr(workload, mask_attribute)
                except AttributeError:
                    pass
            else:
                setattr(workload, mask_attribute, previous_mask)

    def test_torch_compile_inductor_pointwise_reduce_signature_matches_bytecode(
        self,
    ):
        workload = benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32
        self.assertEqual(
            torch._normalized_h100_cuda_workload_instructions(workload),
            torch._COMPILE_H100_CUDA_WORKLOAD_INSTRUCTIONS,
        )
        self.assertIs(
            torch._is_h100_cuda_pointwise_reduce_compile_target(workload),
            True,
        )

    def test_torch_compile_inductor_pointwise_reduce_normalizes_python314_bytecode(
        self,
    ):
        def instruction(opname, arg=None, argval=None, argrepr=""):
            return SimpleNamespace(
                opname=opname,
                arg=arg,
                argval=argval,
                argrepr=argrepr,
            )

        python314_instructions = [
            instruction("RESUME", 0, 0),
            instruction(
                "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
                1,
                ("x", "bias"),
                "x, bias",
            ),
            instruction("BINARY_OP", 0, 0, "+"),
            instruction("LOAD_ATTR", 1, "sin", "sin + NULL|self"),
            instruction("CALL", 0, 0),
            instruction(
                "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
                1,
                ("x", "bias"),
                "x, bias",
            ),
            instruction("BINARY_OP", 10, 10, "-"),
            instruction("LOAD_ATTR", 3, "cos", "cos + NULL|self"),
            instruction("CALL", 0, 0),
            instruction("BINARY_OP", 5, 5, "*"),
            instruction("STORE_FAST", 2, "mixed", "mixed"),
            instruction(
                "LOAD_FAST_BORROW_LOAD_FAST_BORROW",
                32,
                ("mixed", "x"),
                "mixed, x",
            ),
            instruction("LOAD_ATTR", 5, "relu", "relu + NULL|self"),
            instruction("CALL", 0, 0),
            instruction("BINARY_OP", 0, 0, "+"),
            instruction("LOAD_ATTR", 7, "sum", "sum + NULL|self"),
            instruction("LOAD_SMALL_INT", 1, 1),
            instruction("LOAD_CONST", 1, ("dim",), "('dim',)"),
            instruction("CALL_KW", 1, 1),
            instruction("RETURN_VALUE", None, None),
        ]

        with unittest.mock.patch(
            "dis.get_instructions",
            return_value=python314_instructions,
        ):
            self.assertEqual(
                torch._normalized_h100_cuda_workload_instructions(
                    benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                ),
                torch._COMPILE_H100_CUDA_WORKLOAD_INSTRUCTIONS,
            )
            self.assertIs(
                torch._h100_cuda_workload_code_matches(
                    benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                ),
                True,
            )

    def test_torch_compile_inductor_pointwise_reduce_rejects_spoofed_body(self):
        def spoofed_workload(x, bias):
            del x, bias
            raise RuntimeError("spoofed body ran")

        spoofed_workload.__name__ = "h100_cuda_pointwise_reduce_float32"
        spoofed_workload._torch_rs_cuda_compile_workload_version = (
            benchmark_compile_cuda.WORKLOAD_VERSION
        )
        spoofed_workload._torch_rs_cuda_compile_workload_shape = (
            benchmark_compile_cuda.WORKLOAD_SHAPE
        )
        spoofed_workload._torch_rs_cuda_compile_output_shape = (1024,)
        spoofed_workload._torch_rs_cuda_compile_dtype = "torch.float32"

        compiled = torch.compile(
            spoofed_workload,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        with self.assertRaisesRegex(NotImplementedError, "only backend='eager'"):
            compiled(None, None)

    def test_torch_compile_inductor_pointwise_reduce_rejects_invalid_recompile_limit(
        self,
    ):
        invalid_type_values = (True, False, "1", 1.0, object())
        for value in invalid_type_values:
            with self.subTest(value=repr(value)):
                with self.assertRaisesRegex(TypeError, "recompile_limit"):
                    torch.compile(
                        benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                        backend="inductor",
                        fullgraph=True,
                        dynamic=False,
                        recompile_limit=value,
                    )

        with self.assertRaisesRegex(ValueError, "recompile_limit"):
            torch.compile(
                benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                recompile_limit=-1,
            )

    def test_torch_compile_inductor_pointwise_reduce_rejects_wrong_backend_on_h100(
        self,
    ):
        reference_torch = self._require_h100_reference_torch()

        args = SimpleNamespace(warmups=0, samples=1, repeats=1)
        _pytorch_reference, buffers = benchmark_compile_cuda._run_pytorch_reference(
            reference_torch,
            args,
        )
        input_bundle, input_evidence = (
            _cuda_pointwise_reduce_workload.make_h100_float32_pointwise_reduce_inputs_device0(
                buffers["x_host_bytes"],
                buffers["bias_host_bytes"],
            )
        )
        self.assertEqual(input_evidence["status"], "ok")
        self.assertIsNotNone(input_bundle)
        self.addCleanup(input_bundle.close)

        compiled = torch.compile(
            benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        with self.assertRaisesRegex(NotImplementedError, "CUDA compilation"):
            compiled(*input_bundle.inputs)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_torch_compile_inductor_pointwise_reduce_rejects_wrong_shape_on_h100(
        self,
    ):
        self._require_h100_reference_torch()

        rows = 512
        columns = benchmark_compile_cuda.WORKLOAD_SHAPE[1]
        input_bundle, input_evidence = (
            _cuda_pointwise_reduce_workload.make_h100_float32_pointwise_reduce_inputs_device0(
                b"\0" * (rows * columns * 4),
                b"\0" * (columns * 4),
                rows=rows,
                columns=columns,
            )
        )
        self.assertEqual(input_evidence["status"], "ok")
        self.assertIsNotNone(input_bundle)
        self.addCleanup(input_bundle.close)

        compiled = torch.compile(
            benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        with self.assertRaisesRegex(NotImplementedError, "x metadata mismatch"):
            compiled(*input_bundle.inputs)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_torch_compile_inductor_pointwise_reduce_rejects_cpu_inputs_on_h100(
        self,
    ):
        self._require_h100_reference_torch()

        compiled = torch.compile(
            benchmark_compile_cuda.h100_cuda_pointwise_reduce_float32,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        cpu_x = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
        cpu_bias = torch.tensor([0.25, -0.5], dtype=torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "CudaBenchmarkTensor"):
            compiled(cpu_x, cpu_bias)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_pointwise_kernel_honors_caller_visibility_mask(self):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not cuda_visible_devices:
            self.skipTest("requires caller-provided CUDA_VISIBLE_DEVICES")
        if "," in cuda_visible_devices:
            self.skipTest("requires a single caller-visible CUDA device")

        probe = _reference_cuda_probe(cuda_visible_devices=cuda_visible_devices)
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest(
                f"requires CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} "
                "to expose a GPU"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )

        script = rf"""
import importlib.util
import json
import sys

import torch_rs as torch

spec = importlib.util.spec_from_file_location(
    "_torch_rs_compile_cuda_benchmark_override_test",
    {str(BENCHMARK_SCRIPT)!r},
)
benchmark = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)

pointwise = benchmark._torch_rs_private_cuda_pointwise_kernel(
    {cuda_visible_devices!r}
)
print(json.dumps({{
    "status": pointwise["status"],
    "reason": pointwise["reason"],
    "cuda_visible_devices": pointwise["cuda_visible_devices"],
    "required_cuda_visible_devices": pointwise["required_cuda_visible_devices"],
    "cuda_visible_devices_match": pointwise["cuda_visible_devices_match"],
    "single_visible_cuda_device": pointwise["single_visible_cuda_device"],
    "device_type": pointwise["device_type"],
    "device_index": pointwise["device_index"],
    "device_output_checksum": pointwise["device_output_checksum"],
    "checksum_match": pointwise["checksum_match"],
    "gpu_name": pointwise["gpu"]["name"],
    "compute_capability": pointwise["gpu"]["compute_capability"],
    "public_cuda_is_available": torch.cuda.is_available(),
    "public_cuda_device_count": torch.cuda.device_count(),
    "public_cuda_is_initialized": torch.cuda.is_initialized(),
}}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices},
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

        pointwise = json.loads(completed.stdout)
        self.assertEqual(
            pointwise["status"],
            "ok",
            msg=json.dumps(pointwise, indent=2, sort_keys=True),
        )
        self.assertEqual(pointwise["reason"], "pointwise kernel checksum verified")
        self.assertEqual(pointwise["cuda_visible_devices"], cuda_visible_devices)
        self.assertEqual(
            pointwise["required_cuda_visible_devices"],
            cuda_visible_devices,
        )
        self.assertIs(pointwise["cuda_visible_devices_match"], True)
        self.assertIs(pointwise["single_visible_cuda_device"], True)
        self.assertEqual(pointwise["device_type"], "cuda")
        self.assertEqual(pointwise["device_index"], 0)
        self.assertEqual(
            pointwise["device_output_checksum"],
            _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM,
        )
        self.assertIs(pointwise["checksum_match"], True)
        self.assertIn("H100", pointwise["gpu_name"])
        self.assertEqual(pointwise["compute_capability"], [9, 0])
        self.assertIs(pointwise["public_cuda_is_available"], False)
        self.assertEqual(pointwise["public_cuda_device_count"], 0)
        self.assertIs(pointwise["public_cuda_is_initialized"], False)

    def test_private_cuda_pointwise_kernel_empty_override_skips_env_string_check(self):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not cuda_visible_devices:
            self.skipTest("requires caller-provided CUDA_VISIBLE_DEVICES")

        probe = _reference_cuda_probe(cuda_visible_devices=cuda_visible_devices)
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest(
                f"requires CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} "
                "to expose a GPU"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )

        pointwise = _cuda_pointwise_kernel.launch_float32_pointwise_device0(
            required_cuda_visible_devices=None,
        )
        self.assertEqual(
            pointwise["status"],
            "ok",
            msg=json.dumps(pointwise, indent=2, sort_keys=True),
        )
        self.assertEqual(pointwise["cuda_visible_devices"], cuda_visible_devices)
        self.assertIsNone(pointwise["required_cuda_visible_devices"])
        self.assertIs(pointwise["cuda_visible_devices_match"], True)
        self.assertEqual(pointwise["device_type"], "cuda")
        self.assertEqual(pointwise["device_index"], 0)
        self.assertEqual(
            pointwise["device_output_checksum"],
            _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM,
        )
        self.assertIs(pointwise["checksum_match"], True)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_private_cuda_pointwise_build_directory_rejects_symlinked_parent(self):
        target_root = REPOSITORY_ROOT / "target"
        target_root.mkdir(exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(dir=target_root))
        try:
            repository_root = temp_root / "repo"
            repository_root.mkdir()
            target = repository_root / "target"
            target.mkdir()
            symlink_target = temp_root / "outside-build"
            symlink_target.mkdir()
            pointwise_target = target / "torch_rs_private_cuda_pointwise"
            pointwise_target.symlink_to(symlink_target, target_is_directory=True)

            with self.assertRaisesRegex(
                RuntimeError,
                "refusing symlinked CUDA pointwise build directory",
            ):
                _cuda_pointwise_kernel._target_build_directory(
                    repository_root,
                    "abc123",
                )
        finally:
            shutil.rmtree(temp_root)

    def test_private_cuda_pointwise_build_rejects_symlinked_artifact_path(self):
        target_root = REPOSITORY_ROOT / "target"
        target_root.mkdir(exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(dir=target_root))
        original_target_build_directory = (
            _cuda_pointwise_kernel._target_build_directory
        )
        try:
            build_directory = temp_root / "build"
            build_directory.mkdir()
            source_target = temp_root / "outside.cu"
            source_target.write_text("// outside\n", encoding="utf-8")
            source_path = build_directory / "torch_rs_private_pointwise.cu"
            source_path.symlink_to(source_target)

            def fake_target_build_directory(repository_root, key):
                del repository_root, key
                return build_directory

            _cuda_pointwise_kernel._target_build_directory = (
                fake_target_build_directory
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "refusing symlinked CUDA pointwise artifact path",
            ):
                _cuda_pointwise_kernel._build_kernel(
                    repository_root=REPOSITORY_ROOT,
                    driver_probe={"device_0": {"compute_capability": [9, 0]}},
                    nvcc={"path": "/bin/true", "version": {"stdout": ""}},
                )
        finally:
            _cuda_pointwise_kernel._target_build_directory = (
                original_target_build_directory
            )
            shutil.rmtree(temp_root)

    def test_cpu_eager_compile_execution_is_not_eligible_cuda_evidence(self):
        def cpu_program(value):
            return value + value

        compiled = torch.compile(cpu_program, backend="eager", fullgraph=True)
        input = torch.tensor([1.0, -2.0], dtype=torch.float32)
        output = compiled(input)
        self.assertEqual(output.tolist(), [2.0, -4.0])
        self.assertEqual(str(output.device), "cpu")

        classification = benchmark_compile_cuda.classify_torch_rs_cuda_compile_evidence(
            {
                "implementation": "torch_rs",
                "status": "ok",
                "workload_version": benchmark_compile_cuda.WORKLOAD_VERSION,
                "compile_backend": "eager",
                "input_device_type": "cpu",
                "output_device_type": "cpu",
                "native_cuda_compile": False,
                "eager_fallback": True,
                "forwarded_to_pytorch": False,
            }
        )

        self.assertFalse(classification["eligible_cuda_compile_evidence"])
        self.assertEqual(classification["score_credit"], 0.0)
        self.assertIn(
            "compile backend is not the declared CUDA reference backend 'inductor'",
            classification["rejection_reasons"],
        )
        self.assertIn(
            "inputs did not execute on CUDA",
            classification["rejection_reasons"],
        )
        self.assertIn(
            "outputs did not materialize on CUDA",
            classification["rejection_reasons"],
        )
        self.assertIn(
            "eager fallback is not eligible CUDA compile evidence",
            classification["rejection_reasons"],
        )

    def test_cuda_compile_classifier_rejects_contradictory_compile_settings(self):
        classification = benchmark_compile_cuda.classify_torch_rs_cuda_compile_evidence(
            {
                "implementation": "torch_rs",
                "status": "ok",
                "workload_version": benchmark_compile_cuda.WORKLOAD_VERSION,
                "compile_backend": "inductor",
                "compile_fullgraph": False,
                "compile_dynamic": True,
                "compile_config": {
                    "backend": "inductor",
                    "fullgraph": True,
                    "dynamic": False,
                },
                "input_device_type": "cuda",
                "output_device_type": "cuda",
                "native_cuda_compile": True,
                "eager_fallback": False,
                "forwarded_to_pytorch": False,
            }
        )

        self.assertFalse(classification["eligible_cuda_compile_evidence"])
        self.assertEqual(classification["score_credit"], 0.0)
        self.assertIn(
            "compile fullgraph setting is not True",
            classification["rejection_reasons"],
        )
        self.assertIn(
            "compile dynamic setting is not False",
            classification["rejection_reasons"],
        )

    def test_cuda_compile_benchmark_empty_override_honors_visible_mask_on_h100(self):
        cuda_visible_devices = "1"
        probe = _reference_cuda_probe(cuda_visible_devices=cuda_visible_devices)
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest(
                f"requires CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} "
                "to expose a GPU"
            )
        if benchmark_compile_cuda._version_without_local(
            probe["version"],
        ) != benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION:
            self.skipTest(
                "requires PyTorch "
                f"{benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION}, "
                f"got {probe['version']}"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )

        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--warmups",
                "0",
                "--samples",
                "1",
                "--repeats",
                "1",
                "--required-cuda-visible-devices",
                "",
            ],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": cuda_visible_devices},
            text=True,
            timeout=180,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

        report = json.loads(completed.stdout)
        self.assertEqual(report["environment"]["cuda_visible_devices"], "1")
        self.assertEqual(report["candidate"]["status"], "ok")
        self.assertIs(
            report["candidate"]["eligibility"]["eligible_cuda_compile_evidence"],
            True,
        )
        self.assertIs(
            report["candidate"]["input_tensor_evidence"][
                "required_cuda_visible_devices"
            ],
            None,
        )
        self.assertIs(
            report["candidate"]["compile_execution"][
                "required_cuda_visible_devices"
            ],
            None,
        )
        self.assertIs(
            report["candidate"]["compile_execution"]["cuda_visible_devices_match"],
            True,
        )
        self.assertEqual(
            report["candidate"]["compile_execution"]["cuda_visible_devices"],
            "1",
        )
        self.assertEqual(
            report["candidate"]["compile_execution"]["device_output_checksum"],
            report["reference_workload"]["cold_checksum"],
        )

    def test_cuda_reference_benchmark_smoke(self):
        probe = _reference_cuda_probe()
        if not probe.get("imported"):
            self.skipTest("requires reference PyTorch")
        if not probe.get("available"):
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")
        if benchmark_compile_cuda._version_without_local(
            probe["version"],
        ) != benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION:
            self.skipTest(
                "requires PyTorch "
                f"{benchmark_compile_cuda.REFERENCE_PYTORCH_VERSION}, "
                f"got {probe['version']}"
            )
        if "H100" not in probe.get("device_name", ""):
            self.skipTest(
                f"requires an H100 CUDA device, got {probe['device_name']!r}"
            )

        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--warmups",
                "1",
                "--samples",
                "2",
                "--repeats",
                "1",
            ],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            text=True,
            timeout=180,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

        report = json.loads(completed.stdout)
        self.assertEqual(
            report["environment"]["benchmark_version"],
            benchmark_compile_cuda.BENCHMARK_VERSION,
        )
        self.assertEqual(
            report["environment"]["cuda_visible_devices"],
            "0",
        )
        self.assertIn("H100", report["environment"]["gpu"]["torch_cuda_device_name"])
        self.assertEqual(report["reference_workload"]["implementation"], "pytorch")
        self.assertEqual(report["reference_workload"]["status"], "ok")
        self.assertEqual(
            report["reference_workload"]["compile_config"]["backend"],
            "inductor",
        )
        self.assertEqual(
            report["reference_workload"]["output_metadata"]["device_type"],
            "cuda",
        )
        driver_probe = report["torch_rs_cuda_driver_probe"]
        self.assertEqual(
            driver_probe["schema_version"],
            _cuda_driver_probe.PROBE_SCHEMA_VERSION,
        )
        self.assertEqual(driver_probe["status"], "ok")
        self.assertIs(driver_probe["driver_initialized"], True)
        self.assertEqual(driver_probe["cuda_visible_devices"], "0")
        self.assertIn("H100", driver_probe["device_0"]["name"])
        self.assertEqual(driver_probe["device_0"]["compute_capability"], [9, 0])
        self.assertIsInstance(
            driver_probe["driver"]["version"]["version_text"],
            str,
        )
        self.assertIsInstance(driver_probe["runtime"]["runtime_version_text"], str)
        runtime_roundtrip = report["torch_rs_cuda_runtime_roundtrip"]
        self.assertEqual(
            runtime_roundtrip["schema_version"],
            _cuda_runtime_roundtrip.ROUNDTRIP_SCHEMA_VERSION,
        )
        self.assertEqual(
            runtime_roundtrip["buffer_schema_version"],
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
        )
        self.assertEqual(runtime_roundtrip["status"], "ok")
        self.assertIs(runtime_roundtrip["cpu_fallback"], False)
        self.assertEqual(runtime_roundtrip["device_type"], "cuda")
        self.assertEqual(runtime_roundtrip["device_index"], 0)
        self.assertEqual(runtime_roundtrip["cuda_visible_devices"], "0")
        self.assertEqual(
            runtime_roundtrip["host_input_checksum"],
            _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM,
        )
        self.assertEqual(
            runtime_roundtrip["device_roundtrip_checksum"],
            _cuda_runtime_roundtrip.DEFAULT_ROUNDTRIP_CHECKSUM,
        )
        self.assertIs(runtime_roundtrip["checksum_match"], True)
        self.assertIn("H100", runtime_roundtrip["device_0"]["name"])
        self.assertEqual(runtime_roundtrip["device_0"]["compute_capability"], [9, 0])
        pointwise = report["torch_rs_cuda_pointwise_kernel"]
        self.assertEqual(
            pointwise["schema_version"],
            _cuda_pointwise_kernel.POINTWISE_SCHEMA_VERSION,
        )
        self.assertEqual(
            pointwise["buffer_schema_version"],
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
        )
        self.assertEqual(pointwise["status"], "ok")
        self.assertIs(pointwise["cpu_fallback"], False)
        self.assertEqual(pointwise["device_type"], "cuda")
        self.assertEqual(pointwise["device_index"], 0)
        self.assertEqual(pointwise["cuda_visible_devices"], "0")
        self.assertEqual(
            pointwise["device_output_checksum"],
            _cuda_pointwise_kernel.DEFAULT_POINTWISE_CHECKSUM,
        )
        self.assertIs(pointwise["checksum_match"], True)
        self.assertIn("H100", pointwise["device_0"]["name"])
        self.assertEqual(pointwise["device_0"]["compute_capability"], [9, 0])
        self.assertIn("H100", pointwise["gpu"]["name"])
        self.assertEqual(pointwise["gpu"]["compute_capability"], [9, 0])
        self.assertEqual(pointwise["launch"]["sync_error"]["result"], 0)
        self.assertIs(pointwise["kernel_library"]["loaded"], True)
        pointwise_reduce = report["torch_rs_cuda_pointwise_reduce_workload"]
        self.assertEqual(
            pointwise_reduce["schema_version"],
            _cuda_pointwise_reduce_workload.POINTWISE_REDUCE_SCHEMA_VERSION,
        )
        self.assertEqual(
            pointwise_reduce["buffer_schema_version"],
            _cuda_buffer.BUFFER_SCHEMA_VERSION,
        )
        self.assertEqual(pointwise_reduce["status"], "ok")
        self.assertIs(pointwise_reduce["cpu_fallback"], False)
        self.assertEqual(pointwise_reduce["device_type"], "cuda")
        self.assertEqual(pointwise_reduce["device_index"], 0)
        self.assertEqual(pointwise_reduce["cuda_visible_devices"], "0")
        self.assertEqual(pointwise_reduce["workload_shape"], [1024, 1024])
        self.assertEqual(pointwise_reduce["output_shape"], [1024])
        self.assertEqual(
            pointwise_reduce["output_metadata"],
            report["reference_workload"]["output_metadata"],
        )
        prerequisite_tensor_wrapper = report[
            "torch_rs_cuda_prerequisite_tensor_wrapper"
        ]
        self.assertEqual(
            prerequisite_tensor_wrapper,
            pointwise_reduce["public_cuda_tensor_wrapper"],
        )
        tensor_wrapper = report["torch_rs_cuda_tensor_wrapper"]
        self.assertEqual(
            tensor_wrapper["schema_version"],
            _cuda_benchmark_tensor.CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION,
        )
        self.assertEqual(tensor_wrapper["status"], "ok")
        self.assertEqual(tensor_wrapper["shape"], [1024])
        self.assertEqual(tensor_wrapper["stride"], [1])
        self.assertEqual(tensor_wrapper["dtype"], "torch.float32")
        self.assertEqual(tensor_wrapper["device"], "cuda:0")
        self.assertEqual(tensor_wrapper["device_type"], "cuda")
        self.assertEqual(tensor_wrapper["device_index"], 0)
        self.assertIs(tensor_wrapper["is_cuda"], True)
        self.assertIs(tensor_wrapper["cpu_fallback"], False)
        self.assertIs(tensor_wrapper["readback"]["synchronized"], True)
        self.assertIs(tensor_wrapper["readback"]["payload_exposed"], False)
        self.assertIs(tensor_wrapper["native_cuda_compile"], True)
        self.assertEqual(tensor_wrapper["compile_backend"], "inductor")
        self.assertIs(tensor_wrapper["eager_fallback"], False)
        self.assertIs(tensor_wrapper["forwarded_to_pytorch"], False)
        self.assertIs(pointwise_reduce["output_metadata_match"], True)
        self.assertEqual(
            pointwise_reduce["device_output_checksum"],
            report["reference_workload"]["cold_checksum"],
        )
        self.assertEqual(
            pointwise_reduce["pytorch_reference_output_checksum"],
            report["reference_workload"]["cold_checksum"],
        )
        self.assertIs(pointwise_reduce["checksum_match"], True)
        self.assertEqual(pointwise_reduce["launch"]["sync_error"]["result"], 0)
        self.assertIs(pointwise_reduce["kernel_library"]["loaded"], True)
        self.assertIn("H100", pointwise_reduce["gpu"]["name"])
        self.assertEqual(pointwise_reduce["gpu"]["compute_capability"], [9, 0])
        self.assertIs(pointwise_reduce["nvcc"]["available"], True)
        self.assertIn(
            "Cuda compilation tools",
            pointwise_reduce["nvcc"]["version"]["stdout"],
        )
        self.assertGreater(report["reference_workload"]["cold_first_call_us"], 0.0)
        self.assertGreater(
            report["reference_workload"]["steady"]["median_us"],
            0.0,
        )
        self.assertEqual(report["candidate"]["implementation"], "torch_rs")
        self.assertEqual(report["candidate"]["status"], "ok")
        self.assertEqual(report["candidate"]["compile_backend"], "inductor")
        self.assertEqual(
            report["candidate"]["compile_config"],
            benchmark_compile_cuda.REFERENCE_COMPILE_CONFIG,
        )
        self.assertIs(report["candidate"]["native_cuda_compile"], True)
        self.assertIs(report["candidate"]["eager_fallback"], False)
        self.assertIs(report["candidate"]["forwarded_to_pytorch"], False)
        self.assertEqual(report["candidate"]["input_device_type"], "cuda")
        self.assertEqual(report["candidate"]["output_device_type"], "cuda")
        self.assertGreater(report["candidate"]["cold_first_call_us"], 0.0)
        self.assertGreater(report["candidate"]["steady"]["median_us"], 0.0)
        self.assertEqual(
            report["candidate"]["output_tensor_wrapper"],
            report["torch_rs_cuda_tensor_wrapper"],
        )
        self.assertIs(
            report["candidate"]["eligibility"]["eligible_cuda_compile_evidence"],
            True,
        )
        self.assertEqual(report["candidate"]["eligibility"]["score_credit"], 1.0)
        self.assertEqual(report["candidate"]["eligibility"]["rejection_reasons"], [])
        self.assertEqual(
            report["candidate"]["compile_execution"]["device_output_checksum"],
            report["reference_workload"]["cold_checksum"],
        )
        self.assertIs(
            report["candidate"]["compile_execution"]["output_comparison"][
                "exact_bytes_match"
            ],
            True,
        )
        self.assertIsNone(
            report["candidate"]["compile_execution"]["launch"]["sync_error"][
                "result"
            ],
        )
        self.assertIs(
            report["candidate"]["compile_execution"]["launch"]["sync_error"][
                "deferred_to_explicit_timing_boundary"
            ],
            True,
        )
        self.assertIs(
            report["candidate"]["compile_execution"]["readback_deferred"],
            True,
        )
        self.assertIs(
            report["candidate"]["prepared_executor_reuse"]["output_pool_enabled"],
            True,
        )
        self.assertIs(
            report["candidate"]["prepared_executor_reuse"][
                "steady_state_reused_output_buffer"
            ],
            True,
        )
        self.assertIs(
            report["candidate"]["timing_boundary"]["compiled_calls_only"],
            True,
        )
        self.assertIs(
            report["reference_workload"]["timing_boundary"]["compiled_calls_only"],
            True,
        )
        self.assertEqual(
            report["reference_workload"]["timing_boundary"][
                "materialization_outside_timed_region"
            ],
            report["candidate"]["timing_boundary"][
                "materialization_outside_timed_region"
            ],
        )
        self.assertGreater(
            report["aggregates"]["torch_rs_cuda_compile_score_percent"],
            0.0,
        )
        self.assertEqual(report["aggregates"]["zero_credit_unsupported_cell_count"], 0)


if __name__ == "__main__":
    unittest.main()
