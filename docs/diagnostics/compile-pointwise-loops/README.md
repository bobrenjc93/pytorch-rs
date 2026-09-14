# Bounded root literal-range normalization

Fresh [clean-commit captures](postcommit-dcfeb27a/README.md) now measure candidate
`dcfeb27a` against clean main `014fc0de`, including the two-H100 diagnostic and
unchanged fixed gates. The development record below and its original artifacts
are preserved as captured before the implementation commit.

This is development evidence on baseline
`014fc0de304375258056c5aeb3151cfdc2bcc72a`, not clean-head qualification or a
fixed-corpus score. The production change stays in the existing Python frontend;
native typed SSA, numerical admission, lowering, executors and cache/reset owners
are unchanged. The [contract](../../compile-pointwise-jit.md#bounded-root-literal-loops)
and [focused tests](../../../tests/test_compile_pointwise_loops.py) define the
bounded language. Only the old negative assertion rejecting a literal root loop
was removed; all existing helper regressions remain intact.

## Reproduction and ownership

Burner's read-only state and lock records identified this author as
`agent_910598c3` / `idea_0efad7e9`, holding both `gpu` and `cpu-heavy`. GPUs 0 and 1
were explicitly selected, with UUIDs
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` and
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Both are NVIDIA H100s, driver
580.82.07. Per-leg inventories preserve utilization and memory before/after.
Other owners' jobs were not interrupted.

The author copied Python 3.12.12 and the existing locked dependency environment
into this checkout, then installed a fresh locked offline release wheel from a
new local Cargo target. Other CPython distributions were copied locally for
3.10.19, 3.11.15, 3.13.13 and 3.14.5 hardware-free tests. The portable tests load
that wheel's abi3 extension and Python sources from a local extraction, without
reference PyTorch. The installed 3.12 environment contains PyTorch
`2.13.0+cu130`; no interpreter, dependency, cache or build output was written
outside the worktree. The retained environment sets `TMPDIR`, `XDG_CACHE_HOME`,
`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `CARGO_HOME`, `CARGO_TARGET_DIR`,
`CUDA_CACHE_PATH`, `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR` locally,
with bytecode writes disabled, four Cargo jobs and one framework host thread.

With those local paths and the canonical reservations established:

```bash
maturin build --release --locked --offline --out target/loop-work/wheels-labels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/loop-work/wheels-labels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p 'test_compile*.py'
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_loops tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=0,1 cargo test --locked --offline --lib pointwise
.venv/bin/python docs/diagnostics/compile-pointwise-loops/compare.py \
  --output target/loop-work/comparison-reproduction --devices 0,1 \
  --wheel target/loop-work/wheels-labels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
```

The diagnostic output directory must be new. It runs native/reference and
reference/native in fresh independent processes and caches on each reserved GPU.
Each ordinary default wrapper persists through five shape/value/scalar histories
for each of three generic programs. Inputs have fresh storage and offsets;
histories include IEEE values and empty outputs. Original-IR unequal-shape
rejection is checked separately on native. The unchanged native Rust numerical
boundary still applies, including to direct cached-kernel execution.

Five explicitly recorded warmups precede 17 samples, with the same local CUDA
runtime's real device synchronization around each timed call. Cold durations are
separate and native cold calls are instrumented to forbid root/helper execution.
Steady samples on both sides are unprofiled. An additional native warm call
checks body exclusion, absence of lowering and C-extension executor entry.
Default Inductor is unmodified; no reference-only resets, budget increases,
eager substitutions or favorable-repeat selection are used. Comparisons retain
`rtol=1e-5`, `atol=1e-6` and explicit NaN/infinity/signed-zero checks.

## Evidence interpretation

Every leg byte-compares wheel, imported extension/Python files and checkout
Python sources, and retains source hashes, generated CUDA/PTX, NVRTC options,
runtime paths, cache paths, framework order, inputs/outputs and raw timings.
NVRTC compiles the kernels; system `nvcc` 12.6.85 is recorded separately.
Rust is 1.92.0, using the repository's release thin-LTO/codegen-units=1 profile.
The native binary was freshly built once; subsequent Python-only revisions
repackage that same source-bound extension.

Native numerical matches, exclusion of Python bodies, C-extension executor
entries and generated CUDA/PTX are separate evidence. Executor profiling is not
a driver-level launch trace, and empty outputs do not imply a kernel launch.
Reference generated-kernel counters establish compilation where they increase;
warm/cache-hit calls without new kernel attribution are explicitly marked
unknown. These focused checks establish neither broad Inductor equivalence nor
an evaluation score or performance improvement.

The first hardware-free test run failed because constructing a deliberately
colliding hostile-key dictionary called equality before admission. The first GPU
run failed because its new restoration fixture used an unavailable native public
`current_device` API. Both fixtures were corrected and both original logs remain.
An audit then found ignored body instructions were counted only once during
normalization; the final frontend charges every repeated `NOP`/`PRECALL` too.
Comparison v1 predates that budget repair and closure-alias coverage. Its eight
legs and all 60 paired histories passed, but it is earlier-source evidence.
All v1 timings are retained, including reference/native ratios from 0.459 to
2.291; below one means native was slower. V2 added explicit executor attribution.
A subsequent guard audit found a global set to the private missing sentinel could
fall through to builtins; `sentinel-before-fix.log` retains the failing regression.
V3 measured its fix. A CPython 3.10 check then found valid jumps to a following
`pass` and to `EXTENDED_ARG` prefixes were rejected. The final frontend resolves
these transparent targets; the initial new test's indentation error and the
original rejection probe are also retained. V4 measures this final source. Each
rerun follows a production change; no earlier result is substituted or discarded.

## Recorded results

| Check | Result |
| --- | --- |
| Final CUDA-hidden full compiler sweep | 945 tests, 357 explicit skips, no failures |
| Final focused loop/helper/signature/static/operator GPU tests | 58 passed on reserved GPUs 0 and 1 |
| Final loop/helper tests on each CPython 3.10–3.14 | 47 tests, 7 hardware skips, no failures per interpreter |
| Fresh native Rust pointwise regressions | 34 passed; native source and binary unchanged across frontend revisions |
| Earlier broader GPU compiler sweep | 943 tests, 201 skips, no failures; precedes final admission fixes |
| Final independent v4 default-Inductor comparison | Eight successful legs, all 60 paired histories passed |
| Wheel/source/import provenance and dependency compatibility | Passed |

Final frontend SHA256:
`04fa201b3688bb9fb73db9305164c02f081951814e0d44dd3639bcd501536d64`.
Wheel SHA256:
`c9c9eef48b3c003d0588e1b74c629b8ab6354aa92b4da101b4382dfc36ff104d`.
Native extension SHA256:
`cded8225ded0f9a2fd776d579233ee8048038476b8c08e3f6a40d890a693443a`.
Final v4 per-history reference/native median ratios range from 0.578 to 1.369;
values below one mean native was slower. All samples remain in the export.
The recorded native compiler is NVRTC 13.0 targeting `compute_90`, with
`--fmad=true`, `--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`.
Both frameworks synchronize through the local CUDA 13.0 runtime
(`cudaRuntimeGetVersion=13000`); driver/runtime paths are retained per leg.

[comparisons.json.xz](comparisons.json.xz) exports all four campaigns' reports,
raw timing outcomes, input/output values, generated sources/PTX, logs and source
manifests. [validation.json.xz](validation.json.xz) exports the build/test logs,
original failures, reservation, dependencies, environment, source snapshots and
build identities. Each archive maps original worktree-relative paths to text and
SHA256; gzip files are decoded losslessly with their original hash retained.
Binary framework caches remain at the original raw locations.

Read a report without extracting files:

```python
import json, lzma
from pathlib import Path
archive = json.loads(lzma.decompress(Path(
    "docs/diagnostics/compile-pointwise-loops/comparisons.json.xz").read_bytes()))
summary = json.loads(archive["files"][
    "target/loop-work/comparison-v4/summary.json.gz"]["text"])
assert summary["passed"]
```

## Retention and qualification boundary

Raw author evidence lives at
`/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/loop-work/`.
The broad compiler sweep also writes `target/dispatch-smoke-*` reports. These
locations were surfaced to the operator and are outside the fixed observer's
`target/default-compile-eval` coverage. The exports here retain the focused
reports, original failures and build/source identities; they do not install or
replace a retention owner. No raw reports were deleted.

Clean candidate/main evidence, independent review, exact-head full qualification,
scoring, publication and merge remain with Burner's existing normal owner. This
worker did not create commits/branches, run a clean-main qualification, alter
frozen evaluators/corpora or change managed progress artifacts. The unchanged
canonical gates are the only source of evaluation credit.
