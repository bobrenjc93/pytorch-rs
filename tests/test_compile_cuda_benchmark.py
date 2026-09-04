import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch_rs as torch
from torch_rs import _cuda_driver_probe, _cuda_runtime_roundtrip


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


def _reference_cuda_probe():
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
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        text=True,
    )
    if completed.returncode != 0:
        return {"imported": False, "error": completed.stdout + completed.stderr}
    return json.loads(completed.stdout)


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

        probes = row["cuda_probes"]
        self.assertIs(probes["cuda_is_available"], False)
        self.assertEqual(probes["cuda_device_count"], 0)
        self.assertIs(probes["cuda_is_initialized"], False)
        self.assertIs(probes["accelerator_is_available"], False)
        self.assertEqual(probes["accelerator_device_count"], 0)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)

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
        self.assertGreater(report["reference_workload"]["cold_first_call_us"], 0.0)
        self.assertGreater(
            report["reference_workload"]["steady"]["median_us"],
            0.0,
        )
        self.assertEqual(report["candidate"]["implementation"], "torch_rs")
        self.assertEqual(report["candidate"]["status"], "zero_credit_unsupported")
        self.assertEqual(
            report["aggregates"]["torch_rs_cuda_compile_score_percent"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
