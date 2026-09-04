import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_compile_cpu.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_compile_cpu_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_compile_cpu = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_compile_cpu
spec.loader.exec_module(benchmark_compile_cpu)


class _FakeTensor:
    dtype = "fake.float32"
    device = "cpu"
    requires_grad = False
    shape = (3, 2)

    def __init__(self, *, alias_id, data_ptr=4096, storage_offset=0):
        self.alias_id = alias_id
        self._data_ptr = data_ptr
        self._storage_offset = storage_offset

    def stride(self):
        return (1, 3)

    def storage_offset(self):
        return self._storage_offset

    def is_contiguous(self):
        return False

    def tolist(self):
        return [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]

    def is_set_to(self, other):
        return (
            isinstance(other, _FakeTensor)
            and self.alias_id == other.alias_id
            and self.shape == other.shape
            and self.stride() == other.stride()
            and self.storage_offset() == other.storage_offset()
        )

    def numel(self):
        return 6

    def data_ptr(self):
        return self._data_ptr


class CompileBenchmarkArtifactTests(unittest.TestCase):
    def test_view_alias_oracle_rejects_payload_equivalent_copy(self):
        case = SimpleNamespace(name="cpu_float32_t_view")
        source = _FakeTensor(alias_id="source")
        expected = _FakeTensor(alias_id="source")
        actual_view = _FakeTensor(alias_id="source")
        copied = _FakeTensor(alias_id="copy")

        payload = benchmark_compile_cpu._view_alias_payload(
            case,
            actual_view,
            expected,
            (source,),
            cell_name="cpu_float32_t_view/case_default/torch_rs",
        )

        self.assertEqual(
            payload,
            {
                "kind": "output_is_set_to_eager_view",
                "input_index": 0,
                "is_set_to_eager_output": True,
                "storage_offset_equals_input": True,
                "data_ptr_equals_input": True,
            },
        )
        self.assertEqual(
            benchmark_compile_cpu._materialized_payload(copied),
            benchmark_compile_cpu._materialized_payload(expected),
        )
        with self.assertRaisesRegex(AssertionError, "view alias mismatch"):
            benchmark_compile_cpu._view_alias_payload(
                case,
                copied,
                expected,
                (source,),
                cell_name="cpu_float32_t_view/case_default/torch_rs",
            )

    def test_checked_in_raw_artifact_matches_markdown_summary(self):
        benchmark_compile_cpu.validate_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
            benchmark_compile_cpu.DEFAULT_MARKDOWN_REPORT_PATH,
        )


if __name__ == "__main__":
    unittest.main()
