import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = REPOSITORY_ROOT / "scripts" / "evaluate_torch_compile_coverage.py"

spec = importlib.util.spec_from_file_location(
    "_torch_compile_coverage_evaluator_for_tests",
    EVALUATOR_SCRIPT,
)
evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)


def _program(x):
    return x


def _make_inputs(module):
    return (module.tensor([1.0], dtype=module.float32),)


def _case(name, category, *, recompile_limit=None):
    return SimpleNamespace(
        name=name,
        category=category,
        program=_program,
        make_inputs=_make_inputs,
        fullgraph=True,
        dynamic=None,
        mode=None,
        options=None,
        recompile_limit=recompile_limit,
    )


def _step(name, guard_change, expected_compile_count):
    return SimpleNamespace(
        name=name,
        make_inputs=_make_inputs,
        guard_change=guard_change,
        expected_compile_count=expected_compile_count,
        reset_before=False,
        expect_limit_error=False,
    )


def _scenario(name, case_name):
    return SimpleNamespace(
        name=name,
        case_name=case_name,
        steps=(_step("base", "initial", 1),),
    )


def _valid_fake_corpus():
    public_cases = (
        *(_case(f"tensor_{index}", "tensor_arithmetic") for index in range(5)),
        *(_case(f"broadcast_{index}", "broadcasting") for index in range(4)),
        *(
            _case(f"guard_{index}", "recompilation_guards", recompile_limit=4)
            for index in range(2)
        ),
        _case("guard_limit", "recompilation_guards", recompile_limit=2),
    )
    held_out_cases = (
        _case("heldout_broadcast_0", "broadcasting"),
        _case("heldout_broadcast_1", "broadcasting"),
        _case("heldout_guard_0", "recompilation_guards", recompile_limit=4),
        _case("heldout_guard_1", "recompilation_guards", recompile_limit=4),
    )
    return SimpleNamespace(
        COMPILE_CORPUS_VERSION=evaluator.EXPECTED_CORPUS_VERSION,
        CATEGORY_WEIGHTS={
            category: weight
            for category, weight in zip(
                evaluator.REQUIRED_CATEGORIES,
                (12, 8, 8, 6, 8, 8, 8, 8, 8, 6, 6, 6, 4, 4),
            )
        },
        COMPILE_CORPUS=public_cases,
        COMPILE_HELD_OUT_CORPUS=held_out_cases,
        COMPILE_RECOMPILATION_GUARD_SCENARIOS=(
            _scenario(
                "unary_shape_stride_requires_grad_guards",
                "guard_0",
            ),
            _scenario("binary_argument_metadata_guards", "guard_1"),
            _scenario("bounded_limit_then_reset", "guard_limit"),
        ),
        COMPILE_HELD_OUT_RECOMPILATION_GUARD_SCENARIOS=(
            _scenario("heldout_unary_rank3_metadata_mix", "heldout_guard_0"),
            _scenario("heldout_binary_broadcast_metadata_mix", "heldout_guard_1"),
        ),
    )


class TorchCompileCoverageEvaluatorTests(unittest.TestCase):
    def test_weighted_score_counts_missing_categories_as_zero(self):
        weights = {
            category: weight
            for category, weight in zip(
                evaluator.REQUIRED_CATEGORIES,
                (12, 8, 8, 6, 8, 8, 8, 8, 8, 6, 6, 6, 4, 4),
            )
        }
        verdicts = (
            evaluator.CaseVerdict("tensor_pass", "tensor_arithmetic", False, True),
            evaluator.CaseVerdict("tensor_fail", "tensor_arithmetic", False, False),
            evaluator.CaseVerdict("broadcast_pass", "broadcasting", True, True),
            evaluator.CaseVerdict("guard_pass", "recompilation_guards", False, True),
        )

        score, category_scores = evaluator._score_case_verdicts(weights, verdicts)

        self.assertEqual(score, 18.0)
        self.assertEqual(
            category_scores["tensor_arithmetic"],
            {
                "weight": 12,
                "eligible": 2,
                "passed": 1,
                "failed": 1,
                "weighted_score": 6.0,
            },
        )
        self.assertEqual(category_scores["broadcasting"]["weighted_score"], 8.0)
        self.assertEqual(category_scores["recompilation_guards"]["weighted_score"], 4.0)
        self.assertEqual(category_scores["modules_parameters_buffers"]["eligible"], 0)
        self.assertEqual(
            category_scores["modules_parameters_buffers"]["weighted_score"],
            0.0,
        )

    def test_malformed_corpus_metadata_rejected(self):
        corpus = _valid_fake_corpus()
        duplicate = _case("tensor_0", "tensor_arithmetic")
        corpus.COMPILE_HELD_OUT_CORPUS = (*corpus.COMPILE_HELD_OUT_CORPUS, duplicate)

        with self.assertRaises(evaluator.EvaluationFatalError) as raised:
            evaluator._validate_corpus_metadata(corpus)

        self.assertIn("duplicate case name 'tensor_0'", str(raised.exception))

    def test_candidate_output_mismatch_zeroes_case(self):
        corpus = _valid_fake_corpus()
        case = corpus.COMPILE_CORPUS[0]
        reference_case_results = {
            case.name: {
                "name": case.name,
                "category": case.category,
                "output": {"metadata": {"shape": [1]}, "values": [1.0]},
            }
        }
        worker_payload = {
            "ok": True,
            "cases": [
                {
                    "name": case.name,
                    "category": case.category,
                    "status": "passed",
                    "output": {"metadata": {"shape": [1]}, "values": [2.0]},
                }
            ],
            "guard_scenarios": [],
        }

        verdicts, _ = evaluator._compare_worker_to_reference(
            corpus,
            (case,),
            reference_case_results,
            {},
            worker_payload,
        )

        self.assertEqual(len(verdicts), 1)
        self.assertFalse(verdicts[0].passed)
        self.assertEqual(verdicts[0].failure_kind, "reference_mismatch")

    def test_candidate_skipped_case_gets_zero_credit(self):
        corpus = _valid_fake_corpus()
        case = corpus.COMPILE_CORPUS[0]
        reference_case_results = {
            case.name: {
                "name": case.name,
                "category": case.category,
                "output": {"metadata": {"shape": [1]}, "values": [1.0]},
            }
        }
        worker_payload = {
            "ok": True,
            "cases": [
                {
                    "name": case.name,
                    "category": case.category,
                    "status": "skipped",
                    "failure_kind": "unsupported",
                    "message": "unsupported case",
                }
            ],
            "guard_scenarios": [],
        }

        verdicts, _ = evaluator._compare_worker_to_reference(
            corpus,
            (case,),
            reference_case_results,
            {},
            worker_payload,
        )
        score, _category_scores = evaluator._score_case_verdicts(
            corpus.CATEGORY_WEIGHTS,
            verdicts,
        )

        self.assertEqual(score, 0.0)
        self.assertFalse(verdicts[0].passed)
        self.assertEqual(verdicts[0].failure_kind, "unsupported")

    def test_cli_fails_closed_when_reference_environment_is_missing(self):
        completed = subprocess.run(
            [sys.executable, "-S", str(EVALUATOR_SCRIPT), "--subset", "public"],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["score"], 0)
        self.assertIn("failed closed", payload["summary"])
        self.assertTrue(payload["evidence"])
        self.assertTrue(payload["suggestions"])

    def test_burner_evaluation_config_is_command_backed(self):
        with (REPOSITORY_ROOT / ".burner" / "evaluations.json").open(
            encoding="utf-8"
        ) as evaluations_file:
            config = json.load(evaluations_file)

        compile_eval = next(
            evaluation
            for evaluation in config["evaluations"]
            if evaluation["id"] == evaluator.EVALUATION_ID
        )

        self.assertEqual(
            compile_eval["command"],
            "bash scripts/evaluate_torch_compile_coverage.sh",
        )
        self.assertEqual(
            compile_eval["definitionVersion"],
            "evaldef_repo_a61c0e71_v2",
        )


if __name__ == "__main__":
    unittest.main()
