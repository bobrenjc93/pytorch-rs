import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = REPOSITORY_ROOT / "scripts" / "evaluate_torch_compile_coverage.py"
VERIFIER_SCRIPT = REPOSITORY_ROOT / ".github" / "scripts" / "verify_native_extension.py"

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


def _case(name, category, *, recompile_limit=None, backward_through_sum=False):
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
        backward_through_sum=backward_through_sum,
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


def _write_executable(path, contents):
    if contents.startswith("#!"):
        shebang, _, body = contents.partition("\n")
        contents = shebang + "\n" + textwrap.dedent(body)
    else:
        contents = textwrap.dedent(contents).lstrip()
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


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

    def __init__(self, *, execute_trace_graph=True, resolved_backend=None):
        self.compiler = _FakeCompiler()
        self.compile_calls = []
        self.active_compile_counters = None
        self.execute_trace_graph = execute_trace_graph
        self.resolved_backend = resolved_backend

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
                if self.execute_trace_graph:
                    self.active_compile_counters["execute_compile_trace_graph"] += 1
            backend = kwargs["backend"]
            if callable(backend):
                forward = backend(_FakeGraphModule(), inputs)
                return forward(*inputs)
            return inputs[0]

        if self.resolved_backend is None:
            compiled._torch_rs_compile_backend = kwargs["backend"]
        else:
            compiled._torch_rs_compile_backend = self.resolved_backend
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
        _case("custom_public", "custom_functions"),
    )
    held_out_cases = (
        _case("heldout_broadcast_0", "broadcasting"),
        _case("heldout_broadcast_1", "broadcasting"),
        _case("heldout_guard_0", "recompilation_guards", recompile_limit=4),
        _case("heldout_guard_1", "recompilation_guards", recompile_limit=4),
        _case("heldout_custom", "custom_functions"),
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

    def test_current_v9_corpus_matches_pinned_manifest(self):
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
            "public v9 case cpu_float32_unary_abs_neg program changed",
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
            "public v9 case cpu_float32_unary_abs_neg make_inputs changed",
            str(raised.exception),
        )

    def test_pinned_manifest_rejects_custom_helper_replacement(self):
        corpus = _real_corpus_namespace()
        original_helper = corpus.cpu_float32_custom_helper_unary

        def replacement_helper(x):
            return x

        program_globals = corpus.cpu_float32_custom_function_unary.__globals__
        program_globals["cpu_float32_custom_helper_unary"] = replacement_helper
        try:
            with self.assertRaises(evaluator.EvaluationFatalError) as raised:
                evaluator._validate_corpus_metadata(corpus)
        finally:
            program_globals["cpu_float32_custom_helper_unary"] = original_helper

        self.assertIn(
            "public v9 case cpu_float32_custom_function_unary helper_sha256s changed",
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
            "public v9 guard scenario "
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
        guard_limit = next(
            case for case in corpus.COMPILE_CORPUS if case.name == "guard_limit"
        )

        self.assertIs(
            evaluator._case_by_name(corpus, "guard_limit"),
            guard_limit,
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

    def test_guard_scenario_requires_eager_backend_and_native_executor(self):
        case = _case(
            "guard_executor_probe",
            "recompilation_guards",
            recompile_limit=4,
        )
        scenario = SimpleNamespace(
            name="guard_executor_probe_scenario",
            case_name=case.name,
            steps=(_step("base", "initial", 1),),
        )

        def run_with_fake_torch(fake_torch):
            @contextmanager
            def fake_compile_counters():
                counters = {
                    "lower_compile_graph": 0,
                    "execute_compile_trace_graph": 0,
                }
                fake_torch.active_compile_counters = counters
                try:
                    yield counters
                finally:
                    fake_torch.active_compile_counters = None

            original_counters = evaluator._candidate_compile_counters
            original_assert_no_pytorch = evaluator._assert_no_pytorch_modules
            try:
                evaluator._candidate_compile_counters = fake_compile_counters
                evaluator._assert_no_pytorch_modules = lambda context: None
                return evaluator._candidate_guard_scenario_result(
                    SimpleNamespace(
                        torch=fake_torch,
                        COMPILE_CORPUS=(case,),
                        COMPILE_HELD_OUT_CORPUS=(),
                    ),
                    scenario,
                )
            finally:
                evaluator._candidate_compile_counters = original_counters
                evaluator._assert_no_pytorch_modules = original_assert_no_pytorch

        with self.assertRaises(AssertionError) as backend_error:
            run_with_fake_torch(_FakeTorchModule(resolved_backend="python_fallback"))
        self.assertIn("did not resolve backend='eager'", str(backend_error.exception))

        with self.assertRaises(AssertionError) as executor_error:
            run_with_fake_torch(_FakeTorchModule(execute_trace_graph=False))
        self.assertIn(
            "guard_executor_probe_scenario/base execute count 0 != 1",
            str(executor_error.exception),
        )

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

    def test_candidate_backward_gradient_mismatch_zeroes_case(self):
        case = _case(
            "training_probe",
            "training_autograd",
            backward_through_sum=True,
        )
        corpus = SimpleNamespace(COMPILE_HELD_OUT_CORPUS=())
        output = {"metadata": {"shape": [1]}, "values": [1.0]}
        reference_case_results = {
            case.name: {
                "name": case.name,
                "category": case.category,
                "output": output,
                "leaf_gradients": [
                    {"metadata": {"shape": [1]}, "values": [1.0]},
                ],
            }
        }
        worker_payload = {
            "ok": True,
            "cases": [
                {
                    "name": case.name,
                    "category": case.category,
                    "status": "passed",
                    "output": output,
                    "leaf_gradients": [
                        {"metadata": {"shape": [1]}, "values": [2.0]},
                    ],
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
        self.assertIn("leaf_gradients", verdicts[0].message)

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

    def test_native_extension_verifier_honors_virtualenv_override(self):
        spec = importlib.util.spec_from_file_location(
            "_torch_rs_native_extension_verifier_for_tests",
            VERIFIER_SCRIPT,
        )
        verifier = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(verifier)

        expected = REPOSITORY_ROOT / "target" / "custom-verify-venv"
        previous = os.environ.get("TORCH_RS_VERIFY_VIRTUALENV")
        try:
            os.environ["TORCH_RS_VERIFY_VIRTUALENV"] = str(expected)
            self.assertEqual(verifier.expected_virtualenv(), expected.resolve())
        finally:
            if previous is None:
                os.environ.pop("TORCH_RS_VERIFY_VIRTUALENV", None)
            else:
                os.environ["TORCH_RS_VERIFY_VIRTUALENV"] = previous

    def _make_wrapper_fixture(self):
        fixture_parent = REPOSITORY_ROOT / "target"
        fixture_parent.mkdir(exist_ok=True)
        fixture_root = Path(
            tempfile.mkdtemp(
                prefix="torch-compile-coverage-wrapper-",
                dir=fixture_parent,
            )
        )
        self.addCleanup(shutil.rmtree, fixture_root, ignore_errors=True)

        repo = fixture_root / "repo"
        scripts_dir = repo / "scripts"
        verifier_dir = repo / ".github" / "scripts"
        fake_bin = fixture_root / "bin"
        scripts_dir.mkdir(parents=True)
        verifier_dir.mkdir(parents=True)
        fake_bin.mkdir()

        shutil.copy2(
            REPOSITORY_ROOT / "scripts" / "evaluate_torch_compile_coverage.sh",
            scripts_dir / "evaluate_torch_compile_coverage.sh",
        )
        (scripts_dir / "evaluate_torch_compile_coverage.py").write_text(
            textwrap.dedent(
                """
                import json
                import os
                import sys

                with open(os.environ["FAKE_EVAL_LOG"], "a", encoding="utf-8") as log:
                    log.write(
                        "evaluate|"
                        f"uv_project={os.environ.get('UV_PROJECT_ENVIRONMENT', '')}|"
                        f"args={' '.join(sys.argv[1:])}\\n"
                    )
                print(json.dumps({"score": 0, "summary": "stub evaluator"}))
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (verifier_dir / "verify_native_extension.py").write_text(
            textwrap.dedent(
                """
                import os
                import sys

                expected = os.environ.get("TORCH_RS_VERIFY_VIRTUALENV", "")
                with open(os.environ["FAKE_EVAL_LOG"], "a", encoding="utf-8") as log:
                    log.write(f"verify|expected={expected}\\n")
                if not expected:
                    print("missing TORCH_RS_VERIFY_VIRTUALENV", file=sys.stderr)
                    raise SystemExit(1)
                """
            ).lstrip(),
            encoding="utf-8",
        )

        _write_executable(
            fake_bin / "uv",
            f"""#!{sys.executable}
            import os
            import pathlib
            import sys
            import time

            log_path = pathlib.Path(os.environ["FAKE_EVAL_LOG"])
            active_path = pathlib.Path(os.environ["FAKE_EVAL_ACTIVE"])
            repo = pathlib.Path(os.environ["FAKE_EVAL_REPO_ROOT"])
            workspace_venv = repo / ".venv"
            real_python = os.environ["FAKE_REAL_PYTHON"]
            delay = float(os.environ.get("FAKE_EVAL_DELAY", "0"))
            args = sys.argv[1:]


            def log(message):
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{{message}}|pid={{os.getpid()}}|"
                        f"uv_project={{os.environ.get('UV_PROJECT_ENVIRONMENT', '')}}|"
                        f"argv={{' '.join(args)}}\\n"
                    )


            def reject_workspace_venv_path():
                forbidden = str(workspace_venv)
                values = [*args, os.environ.get("UV_PROJECT_ENVIRONMENT", "")]
                if any(forbidden in value for value in values):
                    log("workspace-venv")
                    print(f"wrapper used repository .venv: {{forbidden}}", file=sys.stderr)
                    raise SystemExit(88)


            def guarded(name, callback):
                reject_workspace_venv_path()
                try:
                    fd = os.open(active_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    log(f"overlap:{{name}}")
                    print(f"concurrent setup detected in {{name}}", file=sys.stderr)
                    raise SystemExit(86)
                try:
                    os.write(fd, str(os.getpid()).encode("ascii"))
                    log(f"enter:{{name}}")
                    if delay:
                        time.sleep(delay)
                    callback()
                    log(f"exit:{{name}}")
                finally:
                    os.close(fd)
                    try:
                        active_path.unlink()
                    except FileNotFoundError:
                        pass


            def write_python_shim(path):
                path.write_text(
                    "#!/usr/bin/env bash\\n"
                    "{{\\n"
                    "  printf 'python|pid=%s|uv_project=%s|args=' "
                    "\\"$$\\" \\"${{UV_PROJECT_ENVIRONMENT-}}\\"\\n"
                    "  printf '%q ' \\"$@\\"\\n"
                    "  printf '\\\\n'\\n"
                    "}} >> \\"$FAKE_EVAL_LOG\\"\\n"
                    "exec \\"$FAKE_REAL_PYTHON\\" \\"$@\\"\\n",
                    encoding="utf-8",
                )
                path.chmod(0o755)


            def write_maturin_shim(path):
                path.write_text(
                    f"#!{{real_python}}\\n"
                    "import os\\n"
                    "import pathlib\\n"
                    "import sys\\n"
                    "import time\\n"
                    "\\n"
                    "log_path = pathlib.Path(os.environ['FAKE_EVAL_LOG'])\\n"
                    "active_path = pathlib.Path(os.environ['FAKE_EVAL_ACTIVE'])\\n"
                    "delay = float(os.environ.get('FAKE_EVAL_DELAY', '0'))\\n"
                    "args = sys.argv[1:]\\n"
                    "\\n"
                    "def log(message):\\n"
                    "    with log_path.open('a', encoding='utf-8') as log_file:\\n"
                    "        log_file.write(\\n"
                    "            f'{{message}}|pid={{os.getpid()}}|args={{\\\" \\\".join(args)}}\\\\n'\\n"
                    "        )\\n"
                    "\\n"
                    "try:\\n"
                    "    fd = os.open(active_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)\\n"
                    "except FileExistsError:\\n"
                    "    log('overlap:maturin-build')\\n"
                    "    print('concurrent setup detected in maturin-build', file=sys.stderr)\\n"
                    "    raise SystemExit(86)\\n"
                    "try:\\n"
                    "    os.write(fd, str(os.getpid()).encode('ascii'))\\n"
                    "    log('enter:maturin-build')\\n"
                    "    if delay:\\n"
                    "        time.sleep(delay)\\n"
                    "    out_dir = pathlib.Path(args[args.index('--out') + 1])\\n"
                    "    out_dir.mkdir(parents=True, exist_ok=True)\\n"
                    "    (out_dir / 'torch_rs-0.1.0-cp310-abi3-linux_x86_64.whl').write_text(\\n"
                    "        'fake wheel', encoding='utf-8'\\n"
                    "    )\\n"
                    "    log('exit:maturin-build')\\n"
                    "finally:\\n"
                    "    os.close(fd)\\n"
                    "    try:\\n"
                    "        active_path.unlink()\\n"
                    "    except FileNotFoundError:\\n"
                    "        pass\\n",
                    encoding="utf-8",
                )
                path.chmod(0o755)


            command = next((arg for arg in args if arg in {{"venv", "sync", "pip"}}), None)
            if command == "venv":
                def create_venv():
                    venv = pathlib.Path(args[-1])
                    bin_dir = venv / "bin"
                    bin_dir.mkdir(parents=True, exist_ok=True)
                    write_python_shim(bin_dir / "python")
                    write_maturin_shim(bin_dir / "maturin")

                guarded("uv-venv", create_venv)
            elif command == "sync":
                guarded("uv-sync", lambda: None)
            elif command == "pip":
                def install_wheel():
                    wheel = pathlib.Path(args[-1])
                    if not wheel.exists():
                        print(f"missing wheel {{wheel}}", file=sys.stderr)
                        raise SystemExit(1)

                guarded("uv-pip-install", install_wheel)
            else:
                print(f"unexpected uv args: {{args}}", file=sys.stderr)
                raise SystemExit(2)
            """,
        )

        log_path = fixture_root / "wrapper.log"
        active_path = fixture_root / "setup.active"
        env = os.environ.copy()
        env.update(
            {
                "FAKE_EVAL_ACTIVE": str(active_path),
                "FAKE_EVAL_DELAY": "0.15",
                "FAKE_EVAL_LOG": str(log_path),
                "FAKE_EVAL_REPO_ROOT": str(repo),
                "FAKE_REAL_PYTHON": sys.executable,
                "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
            }
        )
        return repo, env, log_path

    def test_wrapper_uses_target_venv_when_workspace_venv_already_exists(self):
        repo, env, log_path = self._make_wrapper_fixture()
        workspace_venv = repo / ".venv"
        workspace_venv.mkdir()
        (workspace_venv / "reviewer-created").write_text(
            "must not be touched",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                "bash",
                str(repo / "scripts" / "evaluate_torch_compile_coverage.sh"),
                "--subset",
                "public",
            ],
            cwd=repo,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertEqual(json.loads(completed.stdout)["summary"], "stub evaluator")
        self.assertTrue((workspace_venv / "reviewer-created").exists())
        self.assertFalse((workspace_venv / "bin").exists())

        dedicated_venv = repo / "target" / "torch-compile-coverage" / "venv"
        self.assertTrue((dedicated_venv / "bin" / "python").exists())
        log = log_path.read_text(encoding="utf-8")
        self.assertNotIn(str(workspace_venv), log)
        self.assertIn(f"uv_project={dedicated_venv}", log)
        self.assertIn(f"verify|expected={dedicated_venv}", log)

    def test_wrapper_serializes_concurrent_setup_and_installation(self):
        repo, env, log_path = self._make_wrapper_fixture()
        command = [
            "bash",
            str(repo / "scripts" / "evaluate_torch_compile_coverage.sh"),
            "--subset",
            "public",
        ]
        processes = [
            subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]

        results = [process.communicate(timeout=60) for process in processes]
        for index, process in enumerate(processes):
            stdout, stderr = results[index]
            self.assertEqual(
                process.returncode,
                0,
                f"process {index} stdout:\n{stdout}\nstderr:\n{stderr}",
            )
            self.assertEqual(json.loads(stdout)["summary"], "stub evaluator")

        log = log_path.read_text(encoding="utf-8")
        self.assertNotIn("overlap:", log)
        self.assertEqual(log.count("evaluate|"), 2)

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
