# Clean literal-loop evidence

These captures measured clean candidate
`dcfeb27a280078585ad00e8a0e9c9c527aae7f94` and clean main
`014fc0de304375258056c5aeb3151cfdc2bcc72a` on September 14, 2026 UTC.
Measurements finished before this evidence-only change. The earlier development
reports and original failures remain unchanged in the parent directory.

| Check | Candidate `dcfeb27a` | Main `014fc0de` |
| --- | --- | --- |
| Fixed public-default-compile-v2 validity | Valid | Valid |
| Coverage | 14.5%; 14/112 cells passed | 12.5%; 12/112 cells passed |
| CUDA performance | 19.740519%; 14/56 cells passed | 19.299361%; 12/56 cells passed |
| Common-success geometric reference/native latency ratio | 1.108433 | 1.153973 |
| Two-H100 generic-loop diagnostic | Eight successful legs; 60 paired histories passed | Loop domain unsupported; not repeated |
| Focused loop/helper tests, GPUs 0 and 1 | 47 passed | Not repeated |
| Focused loop/helper tests on each CPython 3.10–3.14 | 47 tests, seven hardware skips, no failures per interpreter | Not repeated |
| Fresh native pointwise regressions | 34 passed | Not repeated |

The only fixed-corpus pass/fail changes are the two CUDA `static_loop` cells.
Their reference/native latency ratios were **0.899019 and 1.112606**; the first
variant was slower on native. Unsupported cells keep the control-flow category's
geometric performance contribution at zero. The aggregate performance difference
comes from timing the already-supported arithmetic/broadcasting cells, not a
positive loop-category contribution. Common-success ratios use different success
sets and are not a paired speedup claim. All unsupported outcomes, slow results,
28 programs, 14 category weights, tolerances and denominators remain unchanged.
These scores describe the fixed corpus, not all Python programs.

The separate non-scoring diagnostic uses ordinary persistent default wrappers
for three generic programs: a recurrence, signed sequential loops with a direct
helper, and a zero-trip loop. Five histories per program cover fresh values and
storage offsets, shape/scalar changes, IEEE values and empty outputs. Native
unequal-shape rejection is checked separately. Both framework orders run in
independent processes and initially empty framework caches on both reserved GPUs.
Each history retains five individually timed warmups, 17 synchronized samples,
and a separate cold duration. No reference-only reset, compiler-budget change,
eager substitution or favorable-repeat selection was used. Tolerances remain
`rtol=1e-5`, `atol=1e-6`, with explicit NaN/infinity/signed-zero comparisons.
All 60 pairs passed; reference/native median ratios span **0.601131–1.140571**.

Native cold and extra warm probes exclude original root/helper execution and
record C-extension executor entry; warm probes perform no lowering. Steady
timings are unprofiled on both sides. The profiled native cold durations are
not directly comparable compiler-overhead measurements. CUDA/PTX and NVRTC
options are retained separately from numerical matches. Executor entry is not a
driver-level launch trace, and empty outputs do not imply a kernel launch.
Reference counters attribute generated Inductor kernels to 28 history calls;
the remaining 32 retain the explicit **unknown for this call** attribution.
This diagnostic does not establish broad compiler equivalence.

## Source, build and reproduction

Burner's read-only canonical `gpu` and `cpu-heavy` records identified this
author's owner, `idea_0efad7e9`. GPUs 0 and 1 were explicitly reserved:

- GPU 0: `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`
- GPU 1: `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`

Both are H100s on driver 580.82.07. Only GPU 0 was visible to the fixed gate.
Each leg retains before/after inventories. Python was 3.12.12, reference PyTorch
`2.13.0+cu130`, and Rust 1.92.0. Focused portable checks also used local Python
3.10.19, 3.11.15, 3.13.13 and 3.14.5. Native kernels used NVRTC 13.0 targeting
`compute_90`, with `--fmad=true`, `--ftz=false`, `--prec-div=true` and
`--prec-sqrt=true`. The diagnostic synchronized through its local CUDA 13.0
runtime (`cudaDeviceSynchronize`, version 13000). System nvcc 12.6.85 was recorded
but did not generate these JIT kernels.

The generic diagnostic and focused tests used a new locked offline release build
under `target/loop-postcommit/candidate-build`. Each unchanged fixed wrapper
independently built and installed its own source-bound release wheel. Local
dependency registries/environments were seeded from the author environment;
framework workers started with separate empty caches. Build/setup timings and
actual wheel, executable, import, source and cache identities are retained.
The three wheel hashes are distinct; the reports identify the wheel for each run.
Checkout/wheel/import bytes, all fixed source manifests and raw hashes were
verified. All writable interpreters, dependencies, builds and caches stayed
inside this author checkout. The detached main checkout and its Git metadata
are nested at `target/loop-postcommit/main`; no external worktree was modified.

With the canonical resources held and local environment settings from archived
`target/loop-work/env.sh`:

```bash
CARGO_TARGET_DIR="$PWD/target/loop-postcommit/candidate-build" \
  maturin build --release --locked --offline --out target/loop-postcommit/candidate-wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/loop-postcommit/candidate-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
.venv/bin/python docs/diagnostics/compile-pointwise-loops/compare.py \
  --output target/loop-postcommit/comparison --devices 0,1 \
  --wheel target/loop-postcommit/candidate-wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/loop-postcommit-candidate-fixed.json
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_loops tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR="$PWD/target/loop-postcommit/candidate-build" \
  cargo test --locked --offline --lib pointwise
```

The fixed command also ran from the nested clean-main root, with output
`target/loop-postcommit-main-fixed.json`. Use a new diagnostic output directory
when reproducing. Archived `run-fixed.py` and `run-focused.py` record the actual
commands, environment overrides, timestamps, clean status and exit codes,
including portable-wheel extraction for CPython checks. They are command records,
not changes to the committed measurement harness. The unchanged fixed evaluator,
corpus and dependency-lock hashes match between candidate and main.

## Retention and handoff

[candidate-fixed.json](candidate-fixed.json) and [main-fixed.json](main-fixed.json)
are byte-for-byte generated reports. [captures.json.xz](captures.json.xz) exports
focused reports, all raw timing outcomes and input/output values, generated
CUDA/PTX, source manifest, command/build/test logs, resource record and audit
procedure. Its `files` mapping stores text, original SHA256 and decoded SHA256;
gzip inputs are decoded losslessly. [audit.json](audit.json) records verified
identities and the exact two-cell delta. [SHA256SUMS](SHA256SUMS) covers this set.

[raw-retention-manifest.json](raw-retention-manifest.json) identifies all twelve
original fixed worker reports and their logs, with exact paths, sizes and hashes.
No raw reports, wheels or caches were deleted. The fixed raw directories are:

```text
/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/default-compile-eval/run-20260914T010308Z-9635ddf5/
/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/loop-postcommit/main/target/default-compile-eval/run-20260914T010746Z-c4c212b4/
```

**The nested clean-main directory is outside the fixed observer's coverage.**
Its exact location was reported to the operator before cleanup for separate
retention review. Candidate is under the observer's documented author path,
but its external export receipt is also unverified. The detailed fixed-worker
reports remain at these locations; the compact export does not contain their
large tensor-observation payloads. Generic raw files remain under
`target/loop-postcommit/comparison/`, with reports and generated CUDA/PTX exported
here. This step neither installs another retention owner nor claims observer
delivery. Independent review, exact-head qualification, publication and merge
remain with Burner.
