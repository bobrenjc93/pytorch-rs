import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch_rs as torch
from torch_rs import (
    CudaBenchmarkTensor,
    _cuda_buffer,
    _cuda_benchmark_tensor,
    _cuda_driver_probe,
    _cuda_pointwise_kernel,
    _cuda_pointwise_reduce_workload,
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
    device_type="cuda",
    device_index=0,
    device="cuda:0",
    dtype="torch.float32",
):
    buffer = object.__new__(_cuda_buffer.PrivateCudaFloat32Buffer)
    buffer.runtime = None
    buffer.name = "synthetic"
    buffer.shape = (2,)
    buffer.stride = (1,)
    buffer.element_count = 2
    buffer.byte_count = 8
    buffer.device_index = device_index
    buffer._pointer = SimpleNamespace(value=1)
    buffer._closed = False
    buffer.malloc_call = {"result": 0}

    def metadata():
        return {
            "shape": [2],
            "stride": [1],
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


class CompileCudaBenchmarkTests(unittest.TestCase):
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
        tensor_wrapper = report["torch_rs_cuda_tensor_wrapper"]
        self.assertEqual(
            tensor_wrapper,
            pointwise_reduce["public_cuda_tensor_wrapper"],
        )
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
        self.assertEqual(report["candidate"]["status"], "zero_credit_unsupported")
        self.assertEqual(
            report["candidate"]["prerequisite_cuda_tensor_evidence"],
            tensor_wrapper,
        )
        self.assertIs(
            report["candidate"]["eligibility"]["eligible_cuda_compile_evidence"],
            False,
        )
        self.assertEqual(
            report["aggregates"]["torch_rs_cuda_compile_score_percent"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
