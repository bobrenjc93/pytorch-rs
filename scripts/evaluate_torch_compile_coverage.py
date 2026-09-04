#!/usr/bin/env python3
"""Emit Burner EvaluationOutput JSON for torch.compile coverage parity."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = REPOSITORY_ROOT / "tests" / "test_compile_corpus.py"
EVALUATION_ID = "eval_a61c0e71"
EVALUATOR_VERSION = "torch_compile_program_coverage_evaluator_v1"
EXPECTED_CORPUS_VERSION = "torch_compile_corpus_v4"
REFERENCE_PYTORCH_VERSION = "2.13.0"
REQUIRED_CATEGORIES = (
    "tensor_arithmetic",
    "broadcasting",
    "modules_parameters_buffers",
    "inference",
    "training_autograd",
    "python_control_flow",
    "graph_breaks_fullgraph",
    "dynamic_shapes_symbolics",
    "mutation_aliasing_views",
    "containers_pytrees",
    "decompositions",
    "custom_functions",
    "recompilation_guards",
    "dtype_device_transitions",
)
EXPECTED_PUBLIC_GUARD_SCENARIOS = (
    "unary_shape_stride_requires_grad_guards",
    "binary_argument_metadata_guards",
    "bounded_limit_then_reset",
)
EXPECTED_HELD_OUT_GUARD_SCENARIOS = (
    "heldout_unary_rank3_metadata_mix",
    "heldout_binary_broadcast_metadata_mix",
)


class EvaluationFatalError(RuntimeError):
    """Raised when the evaluator cannot produce a trustworthy score."""


@dataclass(frozen=True)
class CaseVerdict:
    name: str
    category: str
    held_out: bool
    passed: bool
    failure_kind: str | None = None
    message: str | None = None


def _load_compile_corpus_module():
    spec = importlib.util.spec_from_file_location(
        "_torch_rs_compile_corpus_for_evaluation",
        CORPUS_PATH,
    )
    if spec is None or spec.loader is None:
        raise EvaluationFatalError(f"could not load compile corpus from {CORPUS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise EvaluationFatalError(
            f"failed to import compile corpus {CORPUS_PATH}: {_exception_line(error)}"
        ) from error
    return module


def _version_without_local(version):
    return version.split("+", 1)[0]


def _exception_name(error):
    error_type = type(error)
    module = error_type.__module__
    name = error_type.__qualname__
    if module == "builtins":
        return name
    return f"{module}.{name}"


def _exception_line(error):
    return f"{_exception_name(error)}: {str(error).splitlines()[0]}"


def _package_version(distribution_name, module):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _all_cases(corpus_module, *, include_held_out):
    return tuple(corpus_module.compile_corpus_cases(include_held_out=include_held_out))


def _all_guard_scenarios(corpus_module, *, include_held_out):
    return tuple(
        corpus_module.compile_recompilation_guard_scenarios(
            include_held_out=include_held_out,
        )
    )


def _case_names(cases):
    return [case.name for case in cases]


def _guard_scenario_names(scenarios):
    return [scenario.name for scenario in scenarios]


def _validate_corpus_metadata(corpus_module):
    errors = []
    version = getattr(corpus_module, "COMPILE_CORPUS_VERSION", None)
    if version != EXPECTED_CORPUS_VERSION:
        errors.append(
            f"corpus version mismatch: {version!r} != {EXPECTED_CORPUS_VERSION!r}"
        )

    category_weights = getattr(corpus_module, "CATEGORY_WEIGHTS", None)
    if not isinstance(category_weights, dict):
        errors.append("CATEGORY_WEIGHTS must be a dict")
        category_weights = {}
    else:
        if tuple(category_weights) != REQUIRED_CATEGORIES:
            errors.append(
                "CATEGORY_WEIGHTS categories changed or reordered: "
                f"{tuple(category_weights)!r}"
            )
        if sum(category_weights.values()) != 100:
            errors.append(
                f"CATEGORY_WEIGHTS must sum to 100, got {sum(category_weights.values())}"
            )
        for category, weight in category_weights.items():
            if type(category) is not str or not category:
                errors.append(f"invalid category name {category!r}")
            if type(weight) is not int or weight <= 0:
                errors.append(f"{category!r} has invalid weight {weight!r}")

    public_cases = tuple(getattr(corpus_module, "COMPILE_CORPUS", ()))
    held_out_cases = tuple(getattr(corpus_module, "COMPILE_HELD_OUT_CORPUS", ()))
    if len(public_cases) != 12:
        errors.append(f"expected 12 public v4 cases, found {len(public_cases)}")
    if len(held_out_cases) != 4:
        errors.append(f"expected 4 held-out v4 cases, found {len(held_out_cases)}")

    seen_names = set()
    for case in (*public_cases, *held_out_cases):
        name = getattr(case, "name", None)
        category = getattr(case, "category", None)
        program = getattr(case, "program", None)
        make_inputs = getattr(case, "make_inputs", None)
        if type(name) is not str or not name:
            errors.append(f"case has invalid name {name!r}")
            continue
        if name in seen_names:
            errors.append(f"duplicate case name {name!r}")
        seen_names.add(name)
        if category not in category_weights:
            errors.append(f"{name} has unknown category {category!r}")
        if not callable(make_inputs):
            errors.append(f"{name} has non-callable input factory")
        code = getattr(program, "__code__", None)
        if code is None:
            errors.append(f"{name} program is not an exact Python function")
        elif code.co_argcount not in (1, 2):
            errors.append(f"{name} has unsupported arity {code.co_argcount}")
        if getattr(case, "fullgraph", None) is not True:
            errors.append(f"{name} must use fullgraph=True")
        for option_name in ("dynamic", "mode", "options"):
            if getattr(case, option_name, None) is not None:
                errors.append(f"{name} has unsupported {option_name!s} metadata")
        recompile_limit = getattr(case, "recompile_limit", None)
        if category == "recompilation_guards":
            if recompile_limit not in (2, 4):
                errors.append(f"{name} has invalid guard recompile_limit")
        elif recompile_limit is not None:
            errors.append(f"{name} has unexpected recompile_limit")

    public_scenarios = tuple(
        getattr(corpus_module, "COMPILE_RECOMPILATION_GUARD_SCENARIOS", ())
    )
    held_out_scenarios = tuple(
        getattr(corpus_module, "COMPILE_HELD_OUT_RECOMPILATION_GUARD_SCENARIOS", ())
    )
    if _guard_scenario_names(public_scenarios) != list(EXPECTED_PUBLIC_GUARD_SCENARIOS):
        errors.append("public recompilation guard scenarios do not match v4")
    if _guard_scenario_names(held_out_scenarios) != list(
        EXPECTED_HELD_OUT_GUARD_SCENARIOS
    ):
        errors.append("held-out recompilation guard scenarios do not match v4")
    for scenario in (*public_scenarios, *held_out_scenarios):
        scenario_name = getattr(scenario, "name", None)
        case_name = getattr(scenario, "case_name", None)
        steps = tuple(getattr(scenario, "steps", ()))
        if case_name not in seen_names:
            errors.append(f"{scenario_name} references unknown case {case_name!r}")
        if not steps:
            errors.append(f"{scenario_name} has no guard steps")
        last_compile_count = -1
        for step in steps:
            step_name = getattr(step, "name", None)
            guard_change = getattr(step, "guard_change", None)
            expected_count = getattr(step, "expected_compile_count", None)
            if type(step_name) is not str or not step_name:
                errors.append(f"{scenario_name} has invalid step name {step_name!r}")
            if type(guard_change) is not str or not guard_change:
                errors.append(f"{scenario_name}/{step_name} has invalid guard change")
            if type(expected_count) is not int or expected_count < last_compile_count:
                errors.append(
                    f"{scenario_name}/{step_name} has invalid expected compile count"
                )
            last_compile_count = expected_count if type(expected_count) is int else 0

    if errors:
        raise EvaluationFatalError(
            "malformed compile corpus metadata:\n" + "\n".join(errors)
        )


def _tensor_payload(tensor):
    return {
        "metadata": {
            "shape": list(tuple(tensor.shape)),
            "stride": list(tuple(tensor.stride())),
            "storage_offset": int(tensor.storage_offset()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "is_contiguous": bool(tensor.is_contiguous()),
        },
        "values": tensor.tolist(),
    }


def _inputs_payload(inputs):
    return [_tensor_payload(input) for input in inputs]


def _assert_payload_match(actual, expected, *, label):
    if actual != expected:
        raise AssertionError(
            f"{label} observable mismatch: actual={actual!r}, expected={expected!r}"
        )


def _make_recording_backend(calls):
    def backend(graph_module, example_inputs):
        calls.append((graph_module, example_inputs))
        return graph_module.forward

    return backend


def _reset_reference_compile_state(reference_torch):
    dynamo = getattr(reference_torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if reset is not None:
        reset()


def _reference_case_result(reference_torch, case):
    _reset_reference_compile_state(reference_torch)
    backend_calls = []
    backend = _make_recording_backend(backend_calls)
    inputs = case.make_inputs(reference_torch)
    before_inputs = _inputs_payload(inputs)
    expected = case.program(*case.make_inputs(reference_torch))
    compiled = reference_torch.compile(case.program, **case.compile_kwargs(backend))
    actual = compiled(*inputs)
    after_inputs = _inputs_payload(inputs)

    _assert_payload_match(
        _tensor_payload(actual),
        _tensor_payload(expected),
        label=f"{case.name}/reference",
    )
    _assert_payload_match(after_inputs, before_inputs, label=f"{case.name}/inputs")
    if len(backend_calls) < 1:
        raise AssertionError(f"{case.name} did not invoke the reference backend")
    return {
        "name": case.name,
        "category": case.category,
        "output": _tensor_payload(actual),
        "input_count": len(inputs),
        "backend_call_count": len(backend_calls),
    }


def _reference_guard_scenario_result(reference_torch, scenario):
    _reset_reference_compile_state(reference_torch)
    backend_calls = []
    backend = _make_recording_backend(backend_calls)
    case = scenario.case
    compiled = reference_torch.compile(case.program, **case.compile_kwargs(backend))
    steps = []

    for step in scenario.steps:
        if step.reset_before:
            _reset_reference_compile_state(reference_torch)
        inputs = step.make_inputs(reference_torch)
        before_inputs = _inputs_payload(inputs)
        if step.expect_limit_error:
            try:
                compiled(*inputs)
            except Exception as error:
                steps.append(
                    {
                        "name": step.name,
                        "status": "expected_error",
                        "guard_change": step.guard_change,
                        "error_type": _exception_name(error),
                        "error_message": str(error).splitlines()[0],
                    }
                )
                continue
            raise AssertionError(
                f"{scenario.name}/{step.name} did not raise the expected limit error"
            )

        expected = case.program(*step.make_inputs(reference_torch))
        actual = compiled(*inputs)
        after_inputs = _inputs_payload(inputs)
        _assert_payload_match(
            _tensor_payload(actual),
            _tensor_payload(expected),
            label=f"{scenario.name}/{step.name}/reference",
        )
        _assert_payload_match(
            after_inputs,
            before_inputs,
            label=f"{scenario.name}/{step.name}/inputs",
        )
        steps.append(
            {
                "name": step.name,
                "status": "ok",
                "guard_change": step.guard_change,
                "output": _tensor_payload(actual),
            }
        )

    if len(backend_calls) < 1:
        raise AssertionError(f"{scenario.name} did not invoke the reference backend")
    return {
        "name": scenario.name,
        "case": case.name,
        "backend_call_count": len(backend_calls),
        "steps": steps,
    }


def _reference_results(corpus_module, reference_torch, cases, scenarios):
    case_results = {}
    scenario_results = {}
    for case in cases:
        try:
            case_results[case.name] = _reference_case_result(reference_torch, case)
        except Exception as error:
            raise EvaluationFatalError(
                f"reference PyTorch rejected the eligible case {case.name}: "
                f"{_exception_line(error)}"
            ) from error
    for scenario in scenarios:
        try:
            scenario_results[scenario.name] = _reference_guard_scenario_result(
                reference_torch,
                scenario,
            )
        except Exception as error:
            raise EvaluationFatalError(
                "reference PyTorch rejected the eligible guard scenario "
                f"{scenario.name}: {_exception_line(error)}"
            ) from error
    return case_results, scenario_results


class _PytorchImportBlocker:
    def __init__(self, exception_type):
        self.exception_type = exception_type

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname == "torch" or fullname.startswith("torch."):
            raise self.exception_type(
                "torch import is blocked during torch_rs candidate execution"
            )
        return None


def _drop_pytorch_modules():
    return {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "torch" or name.startswith("torch.")
    }


@contextmanager
def _blocked_pytorch_imports(exception_type=ImportError):
    removed = _drop_pytorch_modules()
    blocker = _PytorchImportBlocker(exception_type)
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        for name in list(sys.modules):
            if name == "torch" or name.startswith("torch."):
                sys.modules.pop(name)
        sys.modules.update(removed)


def _assert_no_pytorch_modules(context):
    imported = sorted(
        name for name in sys.modules if name == "torch" or name.startswith("torch.")
    )
    if imported:
        raise AssertionError(
            f"{context} imported installed PyTorch modules: {', '.join(imported[:8])}"
        )


@contextmanager
def _candidate_compile_counters():
    from torch_rs import _compile_bytecode
    from torch_rs import _compile_trace

    original_lower_compile_graph = _compile_bytecode.lower_compile_graph
    original_execute_compile_trace_graph = _compile_trace.execute_compile_trace_graph
    counters = {
        "lower_compile_graph": 0,
        "execute_compile_trace_graph": 0,
    }

    def counting_lower_compile_graph(program, input_metadatas, *, name=None):
        counters["lower_compile_graph"] += 1
        return original_lower_compile_graph(program, input_metadatas, name=name)

    def counting_execute_compile_trace_graph(graph, *inputs):
        counters["execute_compile_trace_graph"] += 1
        return original_execute_compile_trace_graph(graph, *inputs)

    _compile_bytecode.lower_compile_graph = counting_lower_compile_graph
    _compile_trace.execute_compile_trace_graph = counting_execute_compile_trace_graph
    try:
        yield counters
    finally:
        _compile_bytecode.lower_compile_graph = original_lower_compile_graph
        _compile_trace.execute_compile_trace_graph = original_execute_compile_trace_graph


@contextmanager
def _program_call_counter(program):
    old_profile = sys.getprofile()
    code = program.__code__
    calls = {"count": 0}

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is code:
            calls["count"] += 1
        if old_profile is not None:
            old_profile(frame, event, arg)
        return profile

    sys.setprofile(profile)
    try:
        yield calls
    finally:
        sys.setprofile(old_profile)


def _candidate_case_result(corpus_module, case):
    torch_rs = corpus_module.torch
    torch_rs.compiler.reset()
    expected = case.program(*case.make_inputs(torch_rs))
    inputs = case.make_inputs(torch_rs)
    before_inputs = _inputs_payload(inputs)
    with _candidate_compile_counters() as counters:
        compiled = torch_rs.compile(case.program, **case.compile_kwargs("eager"))
        if getattr(compiled, "_torch_rs_compile_backend", None) != "eager":
            raise AssertionError(f"{case.name} did not resolve backend='eager'")
        with _program_call_counter(case.program) as program_calls:
            actual = compiled(*inputs)
        if program_calls["count"] != 0:
            raise AssertionError(
                f"{case.name} executed the original Python program during compiled call"
            )
        if counters["lower_compile_graph"] != 1:
            raise AssertionError(
                f"{case.name} compiled call lowered {counters['lower_compile_graph']} "
                "graphs, expected exactly 1"
            )
        if counters["execute_compile_trace_graph"] != 1:
            raise AssertionError(
                f"{case.name} executed {counters['execute_compile_trace_graph']} "
                "native trace graphs, expected exactly 1"
            )

        with _program_call_counter(case.program) as second_program_calls:
            second_actual = compiled(*case.make_inputs(torch_rs))
        if second_program_calls["count"] != 0:
            raise AssertionError(
                f"{case.name} executed the original Python program on cache hit"
            )
        if counters["lower_compile_graph"] != 1:
            raise AssertionError(f"{case.name} did not reuse the compiled cache")
        if counters["execute_compile_trace_graph"] != 2:
            raise AssertionError(
                f"{case.name} cache hit did not execute the native trace graph"
            )

    after_inputs = _inputs_payload(inputs)
    output = _tensor_payload(actual)
    _assert_payload_match(
        output,
        _tensor_payload(expected),
        label=f"{case.name}/torch_rs",
    )
    _assert_payload_match(
        _tensor_payload(second_actual),
        _tensor_payload(expected),
        label=f"{case.name}/torch_rs/cache_hit",
    )
    _assert_payload_match(after_inputs, before_inputs, label=f"{case.name}/inputs")
    _assert_no_pytorch_modules(case.name)
    return {
        "name": case.name,
        "category": case.category,
        "status": "passed",
        "output": output,
        "lower_compile_graph_count": counters["lower_compile_graph"],
        "execute_compile_trace_graph_count": counters["execute_compile_trace_graph"],
    }


def _candidate_guard_scenario_result(corpus_module, scenario):
    torch_rs = corpus_module.torch
    case = scenario.case
    torch_rs.compiler.reset()
    steps = []
    with _candidate_compile_counters() as counters:
        compiled = torch_rs.compile(case.program, **case.compile_kwargs("eager"))
        for step in scenario.steps:
            if step.reset_before:
                torch_rs.compiler.reset()
            inputs = step.make_inputs(torch_rs)
            before_inputs = _inputs_payload(inputs)

            if step.expect_limit_error:
                with _program_call_counter(case.program) as program_calls:
                    try:
                        compiled(*inputs)
                    except Exception as error:
                        if f"recompile_limit={case.recompile_limit}" not in str(error):
                            raise AssertionError(
                                f"{scenario.name}/{step.name} raised the wrong "
                                f"limit error: {_exception_line(error)}"
                            ) from error
                    else:
                        raise AssertionError(
                            f"{scenario.name}/{step.name} did not raise the "
                            "expected limit error"
                        )
                if program_calls["count"] != 0:
                    raise AssertionError(
                        f"{scenario.name}/{step.name} executed the original "
                        "Python program while rejecting a metadata miss"
                    )
                if counters["lower_compile_graph"] != step.expected_compile_count:
                    raise AssertionError(
                        f"{scenario.name}/{step.name} lower count "
                        f"{counters['lower_compile_graph']} != "
                        f"{step.expected_compile_count}"
                    )
                steps.append(
                    {
                        "name": step.name,
                        "status": "expected_error",
                        "guard_change": step.guard_change,
                        "lower_compile_graph_count": counters[
                            "lower_compile_graph"
                        ],
                    }
                )
                continue

            expected = case.program(*step.make_inputs(torch_rs))
            with _program_call_counter(case.program) as program_calls:
                actual = compiled(*inputs)
            if program_calls["count"] != 0:
                raise AssertionError(
                    f"{scenario.name}/{step.name} executed the original Python "
                    "program during compiled call"
                )
            after_inputs = _inputs_payload(inputs)
            _assert_payload_match(
                _tensor_payload(actual),
                _tensor_payload(expected),
                label=f"{scenario.name}/{step.name}/torch_rs",
            )
            _assert_payload_match(
                after_inputs,
                before_inputs,
                label=f"{scenario.name}/{step.name}/inputs",
            )
            if counters["lower_compile_graph"] != step.expected_compile_count:
                raise AssertionError(
                    f"{scenario.name}/{step.name} lower count "
                    f"{counters['lower_compile_graph']} != {step.expected_compile_count}"
                )
            steps.append(
                {
                    "name": step.name,
                    "status": "ok",
                    "guard_change": step.guard_change,
                    "output": _tensor_payload(actual),
                    "lower_compile_graph_count": counters["lower_compile_graph"],
                    "execute_compile_trace_graph_count": counters[
                        "execute_compile_trace_graph"
                    ],
                }
            )
    _assert_no_pytorch_modules(scenario.name)
    return {
        "name": scenario.name,
        "case": case.name,
        "status": "passed",
        "steps": steps,
    }


def _failed_candidate_result(name, category, error):
    return {
        "name": name,
        "category": category,
        "status": "failed",
        "failure_kind": _exception_name(error),
        "message": str(error).splitlines()[0],
    }


def _run_candidate_worker(args):
    try:
        with _blocked_pytorch_imports(ImportError):
            corpus_module = _load_compile_corpus_module()
        _validate_corpus_metadata(corpus_module)
        cases = _select_subset(corpus_module, args.subset)
        scenarios = _select_guard_subset(corpus_module, args.subset)
        case_results = []
        scenario_results = []
        with _blocked_pytorch_imports(RuntimeError):
            _assert_no_pytorch_modules("candidate worker startup")
            for case in cases:
                try:
                    case_results.append(_candidate_case_result(corpus_module, case))
                except Exception as error:
                    case_results.append(
                        _failed_candidate_result(case.name, case.category, error)
                    )
            for scenario in scenarios:
                try:
                    scenario_results.append(
                        _candidate_guard_scenario_result(corpus_module, scenario)
                    )
                except Exception as error:
                    scenario_results.append(
                        {
                            "name": scenario.name,
                            "case": scenario.case.name,
                            "status": "failed",
                            "failure_kind": _exception_name(error),
                            "message": str(error).splitlines()[0],
                        }
                    )
        torch_rs = corpus_module.torch
        return {
            "ok": True,
            "environment": {
                "torch_rs_version": _package_version("torch-rs", torch_rs),
                "torch_rs_path": getattr(torch_rs, "__file__", None),
                "python": sys.version.replace("\n", " "),
                "python_executable": sys.executable,
            },
            "cases": case_results,
            "guard_scenarios": scenario_results,
        }
    except Exception as error:
        return {
            "ok": False,
            "fatal_error": _exception_line(error),
            "traceback": traceback.format_exc(limit=8),
        }


def _run_candidate_subprocess(subset):
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--candidate-worker",
            "--subset",
            subset,
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationFatalError(
            "torch_rs candidate worker did not emit valid JSON: "
            f"exit={completed.returncode}, stdout={completed.stdout!r}, "
            f"stderr={completed.stderr!r}"
        ) from error
    if completed.returncode != 0:
        raise EvaluationFatalError(
            "torch_rs candidate worker failed before scoring: "
            f"exit={completed.returncode}, payload={payload!r}, "
            f"stderr={completed.stderr!r}"
        )
    return payload


def _select_subset(corpus_module, subset):
    if subset == "full":
        return _all_cases(corpus_module, include_held_out=True)
    if subset == "public":
        return _all_cases(corpus_module, include_held_out=False)
    raise ValueError(f"unknown subset {subset!r}")


def _select_guard_subset(corpus_module, subset):
    if subset == "full":
        return _all_guard_scenarios(corpus_module, include_held_out=True)
    if subset == "public":
        return _all_guard_scenarios(corpus_module, include_held_out=False)
    raise ValueError(f"unknown subset {subset!r}")


def _case_is_held_out(corpus_module, case_name):
    return case_name in {case.name for case in corpus_module.COMPILE_HELD_OUT_CORPUS}


def _score_case_verdicts(category_weights, verdicts):
    by_category = {
        category: {"eligible": 0, "passed": 0, "failed": 0}
        for category in category_weights
    }
    for verdict in verdicts:
        row = by_category[verdict.category]
        row["eligible"] += 1
        if verdict.passed:
            row["passed"] += 1
        else:
            row["failed"] += 1

    category_scores = {}
    total_score = 0.0
    for category, weight in category_weights.items():
        row = by_category[category]
        if row["eligible"] == 0:
            ratio = 0.0
        else:
            ratio = row["passed"] / row["eligible"]
        weighted_score = weight * ratio
        total_score += weighted_score
        category_scores[category] = {
            "weight": weight,
            "eligible": row["eligible"],
            "passed": row["passed"],
            "failed": row["failed"],
            "weighted_score": weighted_score,
        }
    return total_score, category_scores


def _compare_worker_to_reference(
    corpus_module,
    cases,
    reference_case_results,
    reference_scenario_results,
    worker_payload,
):
    if not worker_payload.get("ok"):
        return [
            CaseVerdict(
                case.name,
                case.category,
                _case_is_held_out(corpus_module, case.name),
                False,
                "candidate_import_or_environment_failure",
                worker_payload.get("fatal_error", "candidate worker failed"),
            )
            for case in cases
        ], []

    worker_cases = {row.get("name"): row for row in worker_payload.get("cases", [])}
    worker_scenarios = {
        row.get("name"): row for row in worker_payload.get("guard_scenarios", [])
    }
    guard_failure_by_case = defaultdict(list)
    for scenario_name, reference_scenario in reference_scenario_results.items():
        candidate_scenario = worker_scenarios.get(scenario_name)
        if candidate_scenario is None:
            guard_failure_by_case[reference_scenario["case"]].append(
                f"{scenario_name} missing from candidate worker output"
            )
            continue
        if candidate_scenario.get("status") != "passed":
            guard_failure_by_case[reference_scenario["case"]].append(
                f"{scenario_name} failed: {candidate_scenario.get('message')}"
            )
            continue
        candidate_steps = {
            step.get("name"): step for step in candidate_scenario["steps"]
        }
        for reference_step in reference_scenario["steps"]:
            candidate_step = candidate_steps.get(reference_step["name"])
            if candidate_step is None:
                guard_failure_by_case[reference_scenario["case"]].append(
                    f"{scenario_name}/{reference_step['name']} missing"
                )
                continue
            if candidate_step.get("status") != reference_step["status"]:
                guard_failure_by_case[reference_scenario["case"]].append(
                    f"{scenario_name}/{reference_step['name']} status "
                    f"{candidate_step.get('status')!r} != {reference_step['status']!r}"
                )
                continue
            if reference_step["status"] == "ok":
                try:
                    _assert_payload_match(
                        candidate_step.get("output"),
                        reference_step.get("output"),
                        label=f"{scenario_name}/{reference_step['name']}",
                    )
                except AssertionError as error:
                    guard_failure_by_case[reference_scenario["case"]].append(str(error))

    verdicts = []
    for case in cases:
        reference_case = reference_case_results[case.name]
        candidate_case = worker_cases.get(case.name)
        held_out = _case_is_held_out(corpus_module, case.name)
        if candidate_case is None:
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    "candidate_missing_case",
                    "candidate worker did not report this case",
                )
            )
            continue
        if candidate_case.get("status") != "passed":
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    candidate_case.get("failure_kind", "candidate_failure"),
                    candidate_case.get("message", "candidate failed"),
                )
            )
            continue
        try:
            _assert_payload_match(
                candidate_case.get("output"),
                reference_case.get("output"),
                label=case.name,
            )
        except AssertionError as error:
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    "reference_mismatch",
                    str(error),
                )
            )
            continue
        guard_failures = guard_failure_by_case.get(case.name)
        if guard_failures:
            verdicts.append(
                CaseVerdict(
                    case.name,
                    case.category,
                    held_out,
                    False,
                    "guard_scenario_failure",
                    "; ".join(guard_failures),
                )
            )
            continue
        verdicts.append(CaseVerdict(case.name, case.category, held_out, True))
    return verdicts, worker_payload.get("guard_scenarios", [])


def _category_line(category, row):
    return (
        f"{category} weight {row['weight']}: "
        f"{row['passed']}/{row['eligible']} eligible passed"
        if row["eligible"]
        else f"{category} weight {row['weight']}: 0 eligible, zero credit"
    )


def _summarize_output(
    corpus_module,
    reference_torch,
    worker_payload,
    verdicts,
    category_scores,
    score,
    subset,
):
    passed = sum(1 for verdict in verdicts if verdict.passed)
    failed = len(verdicts) - passed
    categories_with_cases = [
        category for category, row in category_scores.items() if row["eligible"]
    ]
    summary = (
        f"torch.compile coverage parity is {score:.1f}/100 from "
        f"{passed}/{len(verdicts)} reference-eligible {subset} corpus cases. "
        f"Covered categories: {', '.join(categories_with_cases)}; all other "
        "weighted categories remain zero-credit until they have passing "
        "reference-compilable torch_rs cases."
    )
    if failed:
        summary = (
            f"torch.compile coverage parity is {score:.1f}/100 with "
            f"{failed} failed reference-eligible {subset} corpus cases. "
            "Failed cases receive zero credit."
        )

    public_names = _case_names(corpus_module.COMPILE_CORPUS)
    held_out_names = _case_names(corpus_module.COMPILE_HELD_OUT_CORPUS)
    environment = worker_payload.get("environment", {}) if worker_payload else {}
    evidence = [
        (
            f"{EVALUATOR_VERSION} executed corpus {EXPECTED_CORPUS_VERSION} "
            f"with category weights summing to {sum(corpus_module.CATEGORY_WEIGHTS.values())}."
        ),
        (
            f"Reference PyTorch {_version_without_local(reference_torch.__version__)} "
            f"from {getattr(reference_torch, '__file__', None)} compiled and ran "
            f"all {len(verdicts)} selected eligible cases before scoring."
        ),
        (
            f"torch_rs from {environment.get('torch_rs_path')} ran in a subprocess "
            "with installed PyTorch imports blocked and native lowerer/executor "
            "calls instrumented."
        ),
        (
            f"Public cases: {', '.join(public_names)}. Held-out cases: "
            f"{', '.join(held_out_names)}."
        ),
        "; ".join(_category_line(category, row) for category, row in category_scores.items()),
        (
            "Recompilation guards validated: "
            f"{', '.join(_guard_scenario_names(_select_guard_subset(corpus_module, subset)))}."
        ),
    ]
    failures = [verdict for verdict in verdicts if not verdict.passed]
    for verdict in failures[:5]:
        evidence.append(
            f"{verdict.name} failed ({verdict.failure_kind}): {verdict.message}"
        )
    if len(failures) > 5:
        evidence.append(f"{len(failures) - 5} additional case failures omitted.")

    zero_credit_categories = [
        category for category, row in category_scores.items() if row["eligible"] == 0
    ]
    suggestions = []
    if zero_credit_categories:
        suggestions.append(
            "Add native compile support and reference-eligible corpus cases for "
            + ", ".join(zero_credit_categories[:6])
            + ("." if len(zero_credit_categories) <= 6 else ", and remaining categories.")
        )
    if failures:
        suggestions.append(
            "Fix the failing torch_rs compile cases before claiming coverage for "
            "their weighted categories."
        )
    suggestions.append(
        "Keep any quick screen as a strict subset of this corpus; run the default "
        "full evaluator for final scoring."
    )
    return {
        "score": round(score, 6),
        "summary": summary,
        "evidence": evidence,
        "suggestions": suggestions,
    }


def evaluate(subset):
    corpus_module = _load_compile_corpus_module()
    _validate_corpus_metadata(corpus_module)
    reference_torch = getattr(corpus_module, "reference_torch", None)
    if reference_torch is None:
        raise EvaluationFatalError("reference PyTorch is not importable")
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise EvaluationFatalError(
            "torch.compile coverage evaluation requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )

    cases = _select_subset(corpus_module, subset)
    scenarios = _select_guard_subset(corpus_module, subset)
    reference_case_results, reference_scenario_results = _reference_results(
        corpus_module,
        reference_torch,
        cases,
        scenarios,
    )
    worker_payload = _run_candidate_subprocess(subset)
    verdicts, _candidate_scenarios = _compare_worker_to_reference(
        corpus_module,
        cases,
        reference_case_results,
        reference_scenario_results,
        worker_payload,
    )
    score, category_scores = _score_case_verdicts(
        corpus_module.CATEGORY_WEIGHTS,
        verdicts,
    )
    return _summarize_output(
        corpus_module,
        reference_torch,
        worker_payload,
        verdicts,
        category_scores,
        score,
        subset,
    )


def _fatal_output(error):
    return {
        "score": 0,
        "summary": f"torch.compile coverage evaluation failed closed: {error}",
        "evidence": [
            f"{EVALUATOR_VERSION} could not produce a trusted score.",
            f"Python executable: {sys.executable}",
            f"Python version: {sys.version.replace(chr(10), ' ')}",
            f"Platform: {platform.platform()}",
        ],
        "suggestions": [
            "Run from an environment with the current torch_rs build installed.",
            f"Install the reference dependency group so torch=={REFERENCE_PYTORCH_VERSION} is importable.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=("full", "public"),
        default="full",
        help="run the full corpus or the public strict subset",
    )
    parser.add_argument(
        "--candidate-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.candidate_worker:
        print(json.dumps(_run_candidate_worker(args), sort_keys=True))
        return 0
    try:
        output = evaluate(args.subset)
    except EvaluationFatalError as error:
        print(json.dumps(_fatal_output(error), indent=2, sort_keys=True))
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
