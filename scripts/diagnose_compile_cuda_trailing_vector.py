#!/usr/bin/env python3
"""trailing_vector_v1: non-scoring CUDA compiler semantic diagnostic, no timings."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

from scripts import diagnose_compile_cuda_add as common
from scripts import diagnose_compile_cuda_neg_add as provenance
from tests.test_compile_cuda_trailing_vector import FORMS, SEEDS, inputs, shapes
from tests.test_compile_cuda_mul_scalar import call_without_python, make_program

ROOT = Path(__file__).resolve().parents[1]
CASE_SET = "trailing_vector_v1"


def cases(mask):
    if mask == "0,1":
        return [(expr, (5, 11), kind, kind == "ordinal1")
                for expr in (FORMS[0], FORMS[1], FORMS[-1])
                for kind in ("ordinal1", "mixed_ordinals")]
    result = [(expr, shape, "offset", True) for expr in FORMS for shape in shapes()]
    result += [(FORMS[0], (5, 11), kind, False) for kind in
               ("row", "column", "rank3", "scalar", "singleton", "strided", "mixed_cpu", "float64", "gradient")]
    result += [(expr, (5, 11), "offset", False) for expr in
               ("x.add(y, alpha=1)", "x.add(y, alpha=2)", "m.add(x, y)", "m.add(x, y, out=x)", "x * y")]
    return result


def arguments(module, shape, seed, kind):
    args = inputs(module, shape, seed, 3, "cuda:1" if kind == "ordinal1" else "cuda:0")
    x, y = args
    if kind == "row":
        y = y.reshape(1, shape[1])
    elif kind == "column":
        y = x[:, :1].cpu().contiguous().to("cuda:0")
    elif kind == "rank3":
        x = x.reshape(1, *shape)
    elif kind in ("scalar", "singleton"):
        y = y.select(0, 0) if kind == "scalar" else y[:1]
    elif kind == "strided":
        x = x.t().cpu().contiguous().to("cuda:0").t()
    elif kind == "mixed_cpu":
        y = y.cpu()
    elif kind == "mixed_ordinals":
        y = y.cpu().to("cuda:1")
    elif kind in ("float64", "gradient"):
        x = module.tensor(x.cpu().tolist(), dtype=common.torch.float64 if kind == "float64" else module.float32,
                          requires_grad=kind == "gradient").to("cuda:0")
        y = module.tensor(y.cpu().tolist(), dtype=common.torch.float64 if kind == "float64" else module.float32,
                          requires_grad=kind == "gradient").to("cuda:0")
    return x, y


def run_case(expression, shape, kind, supported, fullgraph):
    record = dict(expression=expression, shape=shape, kind=kind, expected_supported=supported,
                  reference_backend="eager", reference_options=dict(fullgraph=fullgraph, dynamic=None),
                  reference_eligible=False, credit=0, native_outcome="error")
    source = f"def program(x, y):\n    return {expression}\n"
    program = make_program(source)
    ref_program = make_program(source, common.torch)
    record.update(source=source, source_sha256=common.hashlib.sha256(source.encode()).hexdigest())
    expected = []
    common.torch._dynamo.reset()
    try:
        reference = common.torch.compile(ref_program, backend="eager", fullgraph=fullgraph, dynamic=None)
        for seed in SEEDS:
            args = arguments(common.torch, shape, seed, kind)
            eager = common.fingerprint(ref_program(*args))
            common.torch.cuda.synchronize(args[0].device)
            actual = common.fingerprint(reference(*arguments(common.torch, shape, seed, kind)))
            if actual != eager:
                raise AssertionError("reference compile differs from eager")
            expected.append(actual)
        record.update(reference_eligible=True, reference_outputs=expected)
    except Exception as error:
        record["reference_error"] = f"{type(error).__name__}: {error}"
    phase = "compile_configuration"
    try:
        compiled = common.native.compile(program, backend="eager", fullgraph=fullgraph)
        observed = []
        for seed in SEEDS:
            phase = "input_construction"
            args = arguments(common.native, shape, seed, kind)
            phase = "compiled_execution"
            actual = call_without_python(compiled, (program.__code__,), *args)
            common.torch.cuda.synchronize(actual.device.index)
            observed.append(common.fingerprint(actual))
        record["native_outputs"] = observed
        record["native_outcome"] = "pass" if record["reference_eligible"] and observed == expected else "mismatch"
    except (NotImplementedError, TypeError) as error:
        record.update(native_outcome="unsupported", rejection_phase=phase, native_error=f"{type(error).__name__}: {error}")
    except Exception as error:
        record.update(native_outcome="error", rejection_phase=phase, native_error=f"{type(error).__name__}: {error}")
    record["expectation_met"] = provenance.expectation_met(record)
    # Zero for every failure, unsupported case, or reference-ineligible case.
    # This local semantic indicator is never a corpus score or feature weight.
    record["credit"] = int(supported and record["expectation_met"])
    return record


def environment(build_record):
    result = provenance.environment(os.environ.get("CUDA_VISIBLE_DEVICES"))
    build_record.resolve().relative_to(ROOT)
    build = json.loads(build_record.read_text())
    if build["installed_native_sha256"] != result["native_extension_sha256"]:
        raise RuntimeError("native library does not match recorded release build")
    if build["profile"] != "release, thin LTO, codegen-units=1, extension-module, locked dependencies":
        raise RuntimeError("expected source-matched locked release build")
    # Validate every implementation and harness input against the immutable export.
    paths = [*sorted((ROOT / "src").rglob("*.rs")), *sorted((ROOT / "src").rglob("*.ptx")),
             *sorted((ROOT / "python/torch_rs").rglob("*.py")),
             *[ROOT / p for p in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "pyproject.toml", "uv.lock")],
             *sorted((ROOT / "scripts").glob("diagnose_compile_cuda*.py")),
             *sorted((ROOT / "tests").glob("test_compile_cuda*.py")), ROOT / "tests/test_cuda_add.py"]
    for path in paths:
        name = str(path.relative_to(ROOT))
        digest = common.sha(path)
        if build["source_files"].get(name) != digest:
            raise RuntimeError(f"release build/source mismatch: {name}")
        result["source_sha256"][name] = digest
    result.update(build_record=build, build_record_sha256=common.sha(build_record))
    result["environment"].update(python_full=sys.version, python_compiler=platform.python_compiler(),
                                 python_build=platform.python_build(), nvcc_available=common.command("nvcc", "--version"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", choices=[CASE_SET], required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.resolve().relative_to(ROOT)
    report = dict(schema=CASE_SET, purpose="Non-scoring bounded semantics only; no universal compile coverage or Inductor performance parity",
                  input_seeds=SEEDS, reference_backend="eager", cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report.update(environment(args.build_record))
        for case in cases(os.environ.get("CUDA_VISIBLE_DEVICES")):
            for fullgraph in (True, False):
                report["cases"].append(run_case(*case, fullgraph))
                save()  # Preserve completed cases if a later case is interrupted.
    except (Exception, KeyboardInterrupt) as error:
        report["setup_or_interruption_error"] = f"{type(error).__name__}: {error}"
    records = report["cases"]
    report["summary"] = dict(cases=len(records), reference_eligible=sum(r["reference_eligible"] for r in records),
                             native_pass=sum(r["native_outcome"] == "pass" for r in records),
                             native_unsupported=sum(r["native_outcome"] == "unsupported" for r in records),
                             expectation_failures=sum(not r["expectation_met"] for r in records))
    save()
    print(json.dumps(report["summary"]))
    return int(not records or "setup_or_interruption_error" in report or report["summary"]["expectation_failures"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
