import importlib
import importlib.util
import json
import subprocess
import sys
import unittest
from contextlib import contextmanager
from dataclasses import replace
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


class _FakeTensor:
    dtype = "fake.float32"
    device = "cpu"
    requires_grad = False

    def __init__(self, values):
        self._values = list(values)
        self.shape = (len(self._values),)

    def stride(self):
        return (1,)

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return True

    def tolist(self):
        return list(self._values)


class _FakeGraphModule:
    def forward(self, *inputs):
        return inputs[0]


class _FakeCompiler:
    def reset(self):
        return None


class _FakeTorchModule:
    float32 = "fake.float32"

    def __init__(self):
        self.compiler = _FakeCompiler()
        self.compile_calls = []
        self.active_compile_counters = None

    def tensor(self, values, *, dtype=None, requires_grad=False):
        del dtype, requires_grad
        return _FakeTensor(values)

    def compile(self, program, **kwargs):
        del program
        self.compile_calls.append(kwargs)

        def compiled(*inputs):
            if self.active_compile_counters is not None:
                if self.active_compile_counters["lower_compile_graph"] == 0:
                    self.active_compile_counters["lower_compile_graph"] = 1
                self.active_compile_counters["execute_compile_trace_graph"] += 1
            backend = kwargs["backend"]
            if callable(backend):
                forward = backend(_FakeGraphModule(), inputs)
                return forward(*inputs)
            return inputs[0]

        compiled._torch_rs_compile_backend = kwargs["backend"]
        return compiled


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


def _real_corpus_namespace():
    try:
        corpus = evaluator._load_compile_corpus_module()
    except evaluator.EvaluationFatalError as error:
        if "No module named 'torch_rs'" in str(error):
            raise unittest.SkipTest("torch_rs is not installed in this environment")
        raise
    return SimpleNamespace(
        **{
            name: value
            for name, value in vars(corpus).items()
            if not name.startswith("__")
        }
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

    def test_current_v4_corpus_matches_pinned_manifest(self):
        evaluator._validate_corpus_metadata(_real_corpus_namespace())

    def test_pinned_manifest_rejects_case_program_replacement(self):
        corpus = _real_corpus_namespace()
        replacement = replace(
            corpus.COMPILE_CORPUS[0],
            program=corpus.cpu_float32_self_add,
        )
        corpus.COMPILE_CORPUS = (replacement, *corpus.COMPILE_CORPUS[1:])

        with self.assertRaises(evaluator.EvaluationFatalError) as raised:
            evaluator._validate_corpus_metadata(corpus)

        self.assertIn(
            "public v4 case cpu_float32_unary_abs_neg program changed",
            str(raised.exception),
        )

    def test_pinned_manifest_rejects_case_input_factory_replacement(self):
        corpus = _real_corpus_namespace()
        replacement = replace(
            corpus.COMPILE_CORPUS[0],
            make_inputs=corpus.cpu_float32_scalar_inputs,
        )
        corpus.COMPILE_CORPUS = (replacement, *corpus.COMPILE_CORPUS[1:])

        with self.assertRaises(evaluator.EvaluationFatalError) as raised:
            evaluator._validate_corpus_metadata(corpus)

        self.assertIn(
            "public v4 case cpu_float32_unary_abs_neg make_inputs changed",
            str(raised.exception),
        )

    def test_pinned_manifest_rejects_category_weight_shift(self):
        corpus = _real_corpus_namespace()
        corpus.CATEGORY_WEIGHTS = dict(corpus.CATEGORY_WEIGHTS)
        corpus.CATEGORY_WEIGHTS["tensor_arithmetic"] -= 1
        corpus.CATEGORY_WEIGHTS["broadcasting"] += 1

        with self.assertRaises(evaluator.EvaluationFatalError) as raised:
            evaluator._validate_corpus_metadata(corpus)

        self.assertIn("CATEGORY_WEIGHTS values changed", str(raised.exception))

    def test_pinned_manifest_rejects_guard_step_replacement(self):
        corpus = _real_corpus_namespace()
        scenario = corpus.COMPILE_RECOMPILATION_GUARD_SCENARIOS[0]
        bad_step = replace(scenario.steps[1], guard_change="shape")
        bad_scenario = replace(
            scenario,
            steps=(scenario.steps[0], bad_step, *scenario.steps[2:]),
        )
        corpus.COMPILE_RECOMPILATION_GUARD_SCENARIOS = (
            bad_scenario,
            *corpus.COMPILE_RECOMPILATION_GUARD_SCENARIOS[1:],
        )

        with self.assertRaises(evaluator.EvaluationFatalError) as raised:
            evaluator._validate_corpus_metadata(corpus)

        self.assertIn(
            "public v4 guard scenario "
            "unary_shape_stride_requires_grad_guards/same_metadata "
            "guard_change changed",
            str(raised.exception),
        )

    def test_subset_selection_uses_pinned_constants_not_corpus_helpers(self):
        corpus = _valid_fake_corpus()
        corpus.compile_corpus_cases = lambda include_held_out=False: (
            corpus.COMPILE_CORPUS[0],
        )
        corpus.compile_recompilation_guard_scenarios = (
            lambda include_held_out=False: ()
        )

        self.assertEqual(
            evaluator._select_subset(corpus, "public"),
            corpus.COMPILE_CORPUS,
        )
        self.assertEqual(
            evaluator._select_subset(corpus, "full"),
            corpus.COMPILE_CORPUS + corpus.COMPILE_HELD_OUT_CORPUS,
        )
        self.assertEqual(
            evaluator._select_guard_subset(corpus, "public"),
            corpus.COMPILE_RECOMPILATION_GUARD_SCENARIOS,
        )
        self.assertEqual(
            evaluator._select_guard_subset(corpus, "full"),
            corpus.COMPILE_RECOMPILATION_GUARD_SCENARIOS
            + corpus.COMPILE_HELD_OUT_RECOMPILATION_GUARD_SCENARIOS,
        )

    def test_guard_case_resolution_uses_pinned_constants_not_helper(self):
        corpus = _valid_fake_corpus()
        corpus.compile_corpus_case = lambda name: corpus.COMPILE_CORPUS[0]

        self.assertIs(
            evaluator._case_by_name(corpus, "guard_limit"),
            corpus.COMPILE_CORPUS[-1],
        )

    def test_compile_execution_uses_pinned_fields_not_case_helper(self):
        def forbidden_compile_kwargs(_backend):
            raise AssertionError("case.compile_kwargs must not be used by evaluator")

        case = SimpleNamespace(
            name="compile_kwargs_probe",
            category="tensor_arithmetic",
            program=_program,
            make_inputs=_make_inputs,
            fullgraph=True,
            dynamic=None,
            mode=None,
            options=None,
            recompile_limit=2,
            compile_kwargs=forbidden_compile_kwargs,
        )
        reference_torch = _FakeTorchModule()

        reference_result = evaluator._reference_case_result(reference_torch, case)

        self.assertEqual(reference_result["backend_call_count"], 1)
        self.assertEqual(len(reference_torch.compile_calls), 1)
        self.assertTrue(callable(reference_torch.compile_calls[0]["backend"]))
        self.assertIs(reference_torch.compile_calls[0]["fullgraph"], True)
        self.assertEqual(reference_torch.compile_calls[0]["recompile_limit"], 2)

        @contextmanager
        def fake_compile_counters():
            counters = {
                "lower_compile_graph": 0,
                "execute_compile_trace_graph": 0,
            }
            candidate_torch.active_compile_counters = counters
            try:
                yield counters
            finally:
                candidate_torch.active_compile_counters = None

        candidate_torch = _FakeTorchModule()
        original_counters = evaluator._candidate_compile_counters
        original_assert_no_pytorch = evaluator._assert_no_pytorch_modules
        try:
            evaluator._candidate_compile_counters = fake_compile_counters
            evaluator._assert_no_pytorch_modules = lambda context: None

            candidate_result = evaluator._candidate_case_result(
                SimpleNamespace(torch=candidate_torch),
                case,
            )
        finally:
            evaluator._candidate_compile_counters = original_counters
            evaluator._assert_no_pytorch_modules = original_assert_no_pytorch

        self.assertEqual(candidate_result["status"], "passed")
        self.assertEqual(len(candidate_torch.compile_calls), 1)
        self.assertEqual(candidate_torch.compile_calls[0]["backend"], "eager")
        self.assertIs(candidate_torch.compile_calls[0]["fullgraph"], True)
        self.assertEqual(candidate_torch.compile_calls[0]["recompile_limit"], 2)

    def test_candidate_worker_blocks_pytorch_imports_during_metadata_validation(self):
        class FakeTorchImporter:
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                if fullname == "torch":
                    return importlib.util.spec_from_loader(fullname, self)
                return None

            def create_module(self, spec):
                del spec
                return None

            def exec_module(self, module):
                module.__version__ = "review-fixture"

        fake_corpus = SimpleNamespace(
            torch=SimpleNamespace(__version__="0", __file__="<fake torch_rs>"),
            COMPILE_HELD_OUT_CORPUS=(),
        )
        validation_calls = []

        def fake_validate(corpus):
            self.assertIs(corpus, fake_corpus)
            validation_calls.append("called")
            with self.assertRaises(ImportError):
                importlib.import_module("torch")

        removed_torch_modules = evaluator._drop_pytorch_modules()
        fake_importer = FakeTorchImporter()
        original_load = evaluator._load_compile_corpus_module
        original_validate = evaluator._validate_corpus_metadata
        original_select = evaluator._select_subset
        original_select_guards = evaluator._select_guard_subset
        sys.meta_path.insert(0, fake_importer)
        try:
            evaluator._load_compile_corpus_module = lambda: fake_corpus
            evaluator._validate_corpus_metadata = fake_validate
            evaluator._select_subset = lambda corpus, subset: ()
            evaluator._select_guard_subset = lambda corpus, subset: ()

            payload = evaluator._run_candidate_worker(
                SimpleNamespace(subset="public")
            )
        finally:
            if fake_importer in sys.meta_path:
                sys.meta_path.remove(fake_importer)
            for name in list(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    sys.modules.pop(name)
            sys.modules.update(removed_torch_modules)
            evaluator._load_compile_corpus_module = original_load
            evaluator._validate_corpus_metadata = original_validate
            evaluator._select_subset = original_select
            evaluator._select_guard_subset = original_select_guards

        self.assertTrue(payload["ok"], payload)
        self.assertEqual(validation_calls, ["called"])

    def test_candidate_worker_rejects_cached_pytorch_during_metadata_validation(self):
        fake_corpus = SimpleNamespace(
            torch=SimpleNamespace(__version__="0", __file__="<fake torch_rs>"),
            COMPILE_HELD_OUT_CORPUS=(),
        )

        def fake_validate(corpus):
            self.assertIs(corpus, fake_corpus)
            sys.modules["torch"] = type(sys)("torch")

        removed_torch_modules = evaluator._drop_pytorch_modules()
        original_load = evaluator._load_compile_corpus_module
        original_validate = evaluator._validate_corpus_metadata
        original_select = evaluator._select_subset
        original_select_guards = evaluator._select_guard_subset
        try:
            evaluator._load_compile_corpus_module = lambda: fake_corpus
            evaluator._validate_corpus_metadata = fake_validate
            evaluator._select_subset = lambda corpus, subset: ()
            evaluator._select_guard_subset = lambda corpus, subset: ()

            payload = evaluator._run_candidate_worker(
                SimpleNamespace(subset="public")
            )
        finally:
            for name in list(sys.modules):
                if name == "torch" or name.startswith("torch."):
                    sys.modules.pop(name)
            sys.modules.update(removed_torch_modules)
            evaluator._load_compile_corpus_module = original_load
            evaluator._validate_corpus_metadata = original_validate
            evaluator._select_subset = original_select
            evaluator._select_guard_subset = original_select_guards

        self.assertFalse(payload["ok"], payload)
        self.assertIn(
            "candidate metadata validation imported installed PyTorch modules: torch",
            payload["fatal_error"],
        )

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

    def test_wrapper_invokes_the_command_backed_evaluator(self):
        wrapper = REPOSITORY_ROOT / "scripts" / "evaluate_torch_compile_coverage.sh"

        self.assertTrue(wrapper.exists())
        self.assertIn(
            "scripts/evaluate_torch_compile_coverage.py",
            wrapper.read_text(encoding="utf-8"),
        )

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
