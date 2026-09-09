"""Integrity checks for diagnostic timing and read-only campaign ingestion."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import rank2_sum_cuda_campaign as campaign
    try:
        import torch
    except ImportError:
        benchmark = None
    else:
        import benchmark_rank2_sum_cuda as benchmark
finally:
    sys.path.pop(0)


@unittest.skipIf(benchmark is None, "install the reference dependency group")
class OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = benchmark.torch.get_num_threads()
        benchmark.torch.set_num_threads(benchmark.torch_rs.get_num_threads())
        self.expected = benchmark.torch.tensor([1., 2.])

    def tearDown(self):
        benchmark.torch.set_num_threads(self.old_threads)

    def test_invalid_metadata_and_values_receive_zero_credit_without_timing(self):
        torch = benchmark.torch
        bad_outputs = {
            "dtype": self.expected.double(),
            "shape": self.expected.reshape(1, 2),
            "stride": torch.tensor([1., 0., 2., 0.])[::2],
            "storage_offset": torch.tensor([0., 1., 2.])[1:],
            "requires_grad": self.expected.clone().requires_grad_(),
            "layout": self.expected.to_sparse(),
            "device": torch.empty(2, device="meta"),
            "values": torch.tensor([1., 3.]),
        }
        for field, bad in bad_outputs.items():
            with self.subTest(field=field), patch.object(benchmark, "measure") as measure:
                result = benchmark.pair(lambda: bad, lambda: self.expected)
                self.assertFalse(result["valid"])
                self.assertEqual(result["credit"], 0)
                self.assertEqual(result["capped_parity"], 0)
                self.assertIn("validation_error", result)
                measure.assert_not_called()

    def test_scaling_has_no_parity_credit(self):
        with patch.object(benchmark.torch, "get_num_threads", return_value=4), patch.object(benchmark.torch_rs, "get_num_threads", return_value=1), patch.object(benchmark, "measure", return_value={"median_ns": 10}):
            result = benchmark.pair(lambda: self.expected, lambda: self.expected)
        self.assertTrue(result["valid"])
        self.assertEqual(result["comparison"], "scaling_diagnostic")
        self.assertIsNone(result["capped_parity"])
        self.assertEqual(result["credit"], 0)
        self.assertIsNone(benchmark.aggregate([result])["capped_geomean_parity"])

    def test_invalid_cells_remain_in_matched_denominator(self):
        valid = {"valid": True, "comparison": "matched_threads", "credit": 1.0}
        invalid = benchmark.invalid_cell("wrong output", 1, 1)
        scaling = benchmark.invalid_cell("wrong output", 1, 4)
        result = benchmark.aggregate([valid, invalid, scaling])
        self.assertEqual(result, {"matched_cells": 2, "invalid_matched_cells": 1,
                                  "scaling_diagnostic_cells": 1, "capped_geomean_parity": 0.0})

    def test_post_timing_validation_and_thread_drift_invalidate_cell(self):
        outputs = iter((self.expected, self.expected + 1))
        with patch.object(benchmark, "measure", return_value={"median_ns": 10}):
            self.assertFalse(benchmark.pair(lambda: next(outputs), lambda: self.expected)["valid"])
        with patch.object(benchmark, "measure", return_value={"median_ns": 10}), patch.object(benchmark.torch, "get_num_threads", side_effect=(1, 4)):
            result = benchmark.pair(lambda: self.expected, lambda: self.expected)
            self.assertFalse(result["valid"])
            self.assertIn("thread configuration changed", result["validation_error"])

    def test_missing_cuda_cell_is_retained_with_zero_credit(self):
        with patch.object(benchmark.torch.cuda, "is_available", return_value=False):
            result = benchmark.run_cell({"id": "gpu", "kind": "zeros", "elements": 2, "pytorch_threads": 1}, 9)
        self.assertFalse(result["valid"])
        self.assertEqual(result["credit"], 0)
        self.assertEqual(result["id"], "gpu")

    def test_real_sum_and_backward_metadata(self):
        for kind in ("sum", "backward_accumulate"):
            with self.subTest(kind=kind):
                cell = {"id": kind, "kind": kind, "shape": [13, 17], "layout": "contiguous", "axis": 1, "pytorch_threads": 1}
                result = benchmark.run_cell(cell, 901)
                self.assertTrue(result["valid"], result)
                self.assertEqual(result["output_metadata"]["native"], result["output_metadata"]["pytorch"])


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.public = json.loads(campaign.PUBLIC_CAMPAIGN.read_text())

    def test_public_matrix_is_preserved_and_quick_is_strict(self):
        full, _ = campaign.load_campaign(campaign.PUBLIC_CAMPAIGN, scoring=False, seed=campaign.PUBLIC_SEED, quick=False)
        quick, _ = campaign.load_campaign(campaign.PUBLIC_CAMPAIGN, scoring=False, seed=campaign.PUBLIC_SEED, quick=True)
        self.assertEqual(len(full), 115)
        self.assertEqual(sum(c["kind"] in ("sum", "backward_accumulate") for c in full), 100)
        self.assertLess(len(quick), len(full))
        by_id = {cell["id"]: cell for cell in full}
        for cell in quick:
            self.assertEqual(cell, by_id[cell["id"]])

    @unittest.skipIf(benchmark is None, "install the reference dependency group")
    def test_quick_and_full_execution_use_identical_held_out_inputs(self):
        first = {"id": "first", "kind": "sum", "shape": [3, 7], "layout": "contiguous", "axis": 0, "pytorch_threads": 1}
        selected = {**first, "id": "selected"}
        original_threads = benchmark.torch.get_num_threads()
        try:
            with patch.object(benchmark, "pair", return_value={}), patch.object(benchmark.torch_rs, "tensor", wraps=benchmark.torch_rs.tensor) as tensor:
                benchmark.run_cell(first, 38117)
                benchmark.run_cell(selected, 38117)
                full_input = tensor.call_args.args[0]
                benchmark.run_cell(selected, 38117)
                self.assertEqual(tensor.call_args.args[0], full_input)
                benchmark.run_cell(selected, 38118)
                self.assertNotEqual(tensor.call_args.args[0], full_input)
        finally:
            benchmark.torch.set_num_threads(original_threads)

    def test_rejects_invalid_membership_duplicates_and_workloads(self):
        mutations = [
            lambda c: c.update(schema_version=True),
            lambda c: c.update(unreviewed_cells=[]),
            lambda c: c.update(quick=[]),
            lambda c: c.update(quick=[cell["id"] for cell in c["full"]]),
            lambda c: c["quick"].append("unknown-id"),
            lambda c: c["quick"].append(c["quick"][0]),
            lambda c: c["full"].append(c["full"][0]),
            lambda c: c["full"][0].update(axis=2),
            lambda c: c["full"][0].update(shape=[-1, 8]),
            lambda c: c["full"][0].update(pytorch_threads=0),
            lambda c: c["full"][0].update(kind="not-supported"),
        ]
        for mutate in mutations:
            data = copy.deepcopy(self.public)
            mutate(data)
            with self.assertRaises(ValueError):
                campaign.validate_campaign(data)
        with self.assertRaises(ValueError):
            campaign.validate_campaign([])

    def test_local_campaign_cannot_be_scoring_input(self):
        with self.assertRaisesRegex(ValueError, "outside the candidate"):
            campaign.load_campaign(campaign.PUBLIC_CAMPAIGN, scoring=True, seed=91237, quick=False)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            link = Path(temp) / "alias.json"
            link.symlink_to(campaign.PUBLIC_CAMPAIGN)
            with self.assertRaisesRegex(ValueError, "outside the candidate"):
                campaign.load_campaign(link, scoring=True, seed=91237, quick=False)

    def test_external_contract_read_only_with_hermetic_fixture(self):
        # Simulate the ownership boundary entirely inside this worktree.
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            parent = Path(temp)
            candidate = parent / "candidate"
            candidate.mkdir()
            definition = parent / "reviewer-campaign.json"
            definition.write_text(json.dumps(self.public))
            before = definition.read_bytes()
            with patch.object(campaign, "ROOT", candidate):
                for seed in (None, -1, campaign.PUBLIC_SEED):
                    with self.assertRaisesRegex(ValueError, "held-out seed"):
                        campaign.load_campaign(definition, scoring=True, seed=seed, quick=True)
                full, full_meta = campaign.load_campaign(definition, scoring=True, seed=93281, quick=False)
                quick, quick_meta = campaign.load_campaign(definition, scoring=True, seed=93281, quick=True)
                self.assertTrue(full_meta["scoring"])
                self.assertEqual(full_meta["sha256"], quick_meta["sha256"])
                self.assertEqual(quick, [c for c in full if c["id"] in quick_meta["quick_ids"]])
            self.assertEqual(definition.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
