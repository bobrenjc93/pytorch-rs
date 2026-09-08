import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_ID = "eval_b7f2a91c"
EXPECTED_BACKENDS = {
    "cuda_nvidia",
    "rocm_amd",
    "mps_apple",
    "xpu_intel",
    "xla_tpu_google",
    "hpu_gaudi_intel",
    "mtia_meta",
}


class HardwareHeterogeneityEvaluationTests(unittest.TestCase):
    def test_checked_in_evaluation_uses_the_versioned_matrix(self):
        config = json.loads(
            (REPOSITORY_ROOT / ".burner" / "evaluations.json").read_text(
                encoding="utf-8"
            )
        )
        evaluation = next(
            item for item in config["evaluations"] if item["id"] == EVALUATION_ID
        )

        self.assertEqual(evaluation["name"], "Hardware heterogeneity")
        self.assertEqual(evaluation["weight"], 4)
        self.assertTrue(evaluation["enabled"])
        self.assertEqual(
            evaluation["definitionVersion"], "evaldef_repo_b7f2a91c_v1"
        )
        self.assertIn("hardware-heterogeneity-matrix-v1.json", evaluation["prompt"])
        self.assertIn("real corresponding hardware", evaluation["prompt"])
        self.assertIn("CUDA_VISIBLE_DEVICES=0", evaluation["prompt"])

    def test_matrix_has_a_fixed_complete_denominator(self):
        matrix = json.loads(
            (
                REPOSITORY_ROOT
                / "docs"
                / "hardware-heterogeneity-matrix-v1.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(matrix["schema_version"], "hardware_heterogeneity_matrix_v1")
        self.assertEqual(matrix["evaluation_id"], EVALUATION_ID)
        self.assertEqual(matrix["reference"], {"framework": "PyTorch", "version": "2.13"})
        self.assertEqual(matrix["backend_aggregation"], "equal_arithmetic_mean")
        backend_ids = [backend["id"] for backend in matrix["backends"]]
        self.assertEqual(set(backend_ids), EXPECTED_BACKENDS)
        self.assertEqual(len(backend_ids), len(set(backend_ids)))
        self.assertNotIn("cpu", backend_ids)
        self.assertNotIn("meta", backend_ids)
        self.assertNotIn("privateuseone", backend_ids)

        capabilities = matrix["capabilities"]
        capability_ids = [capability["id"] for capability in capabilities]
        self.assertEqual(len(capability_ids), len(set(capability_ids)))
        self.assertEqual(sum(capability["weight"] for capability in capabilities), 100)
        self.assertTrue(
            set(matrix["breadth_required_capabilities"]).issubset(capability_ids)
        )
        self.assertIn("installed_pytorch_forwarding", matrix["zero_credit_evidence"])
        self.assertIn("missing_or_skipped_hardware_run", matrix["zero_credit_evidence"])


if __name__ == "__main__":
    unittest.main()
