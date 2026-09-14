# Clean evidence after the zero-trip admission fix

These captures measured clean commit
`6ef71d3acc5cac5bb71b8928acb642993f81958d` on September 14, 2026 UTC, including
the review fix that validates skipped loop bodies. All measurements finished
before this evidence-only change. They supersede the previous candidate captures;
the earlier reports and original development failures remain unchanged.

The existing [clean-main report](../postcommit-dcfeb27a/main-fixed.json) still
measures main `014fc0de304375258056c5aeb3151cfdc2bcc72a`. Its clean nested checkout,
130-file source manifest, wheel, imports, interpreter, reference origins, and six
raw worker reports/logs were revalidated inside this worktree. Main and the fixed
evaluator/corpus are unchanged, so those original measurements are reused without
claiming a new baseline run.

| Check | Candidate `6ef71d3a` | Main `014fc0de` |
| --- | --- | --- |
| Fixed public-default-compile-v2 validity | Valid | Valid; existing capture revalidated |
| Coverage | 14.5%; 14/112 cells passed | 12.5%; 12/112 cells passed |
| CUDA performance | 19.849159%; 14/56 cells passed | 19.299361%; 12/56 cells passed |
| Common-success geometric reference/native latency ratio | 1.102739 | 1.153973 |
| Two-H100 generic-loop diagnostic | Eight successful legs; 60 paired histories passed | Loop domain unsupported; not repeated |
| Focused loop/helper suite with GPUs 0 and 1 visible | 53 passed, including eight hardware cases | Not repeated |
| Focused suite on each CPython 3.10–3.14 | 53 tests, eight hardware skips, no failures per interpreter | Not repeated |
| Native pointwise regressions | 34 passed | Not repeated |

The only fixed-corpus pass/fail differences against main are the two CUDA
`static_loop` cells, with reference/native latency ratios **1.046789 and 1.137011**.
The control-flow category still contributes zero to geometric performance because
its other cells remain unsupported. The aggregate performance difference comes
from already-supported arithmetic/broadcasting cells. Common-success ratios use
different success sets and are not a paired speedup claim. These are finite-corpus
scores, not percentages of all Python programs. All unsupported outcomes, slow
results, 28 programs, 14 category weights, tolerances and denominators are retained.

The unchanged non-scoring diagnostic uses persistent ordinary default wrappers
for a recurrence, signed sequential loops with a helper, and a zero-trip loop.
Five histories per program cover fresh values/storage offsets, shape/scalar
changes, IEEE values and empty outputs, with native unequal-shape rejection
checked separately. Both framework orders run in independent processes and fresh
framework caches on each reserved H100. Each history retains five individually
timed warmups, 17 synchronized samples and a separate cold duration. All 60 pairs
passed; reference/native median ratios span **0.398411–1.219403**, including every
slower result. Tolerances remain `rtol=1e-5`, `atol=1e-6`, with explicit
NaN/infinity/signed-zero comparisons. No reference-only reset, compiler-budget
change, eager substitution or favorable-repeat selection was used.

Native cold and extra warm probes exclude root/helper execution and record
C-extension executor entry; warm probes perform no lowering. Steady timings are
unprofiled. Profiled native cold durations are not directly comparable compiler
overhead measurements. Generated CUDA/PTX and NVRTC options are retained separately
from numerical matches. Executor entry is not a driver-level launch trace, and
empty outputs do not imply a kernel launch. Reference counters attribute generated
Inductor kernels to 28 history calls; the remaining 32 retain **unknown for this
call** attribution. This evidence does not establish broad compiler equivalence.

## Build, hardware and reproduction

Burner's read-only canonical `gpu` and `cpu-heavy` records identified owner
`idea_0efad7e9`. GPUs 0 and 1 remained explicitly reserved:

- GPU 0: `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`
- GPU 1: `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`

Both are H100s on driver 580.82.07. Only GPU 0 was visible to the fixed gate.
Before/after inventories are retained per leg. Python was 3.12.12, reference
PyTorch `2.13.0+cu130`, and Rust 1.92.0. Portable checks also used local Python
3.10.19, 3.11.15, 3.13.13 and 3.14.5. Native kernels used NVRTC 13.0 targeting
`compute_90`, with `--fmad=true`, `--ftz=false`, `--prec-div=true` and
`--prec-sqrt=true`. The diagnostic synchronized through its local CUDA 13.0
runtime (`cudaDeviceSynchronize`, version 13000). System nvcc 12.6.85 was recorded
but did not generate these JIT kernels.

The unchanged fixed wrapper built and installed a fresh locked release wheel.
Cargo/dependency/native build caches were warm; this was not a clean-target Rust
rebuild. Actual setup times/cache state remain in the report. The diagnostic and
focused checks used that same wheel, with checkout/wheel/import bytes verified.
Native regression tests reused the worktree-local test build of unchanged Rust
sources. All interpreters, dependencies, wheels and writable caches stayed inside
this author checkout. The baseline checkout and Git metadata remain nested at
`target/loop-postcommit/main`.

- Frontend SHA256: `1e5ca6378b7ddb737192b66ef4946d82853463b9d3c721f684a3b109224977c8`
- Evaluator source manifest SHA256: `6a3f8285a745152095d5d2361e6b62d9bb41b4fe0e422f08e49226f94b69c6b2`
- Wheel SHA256: `62a3655ce067a27479adc09b0d407fbe419c7b4cbc5a718ae441716523730764`

From the clean author checkout, with resources held and local settings in archived
`target/loop-work/env.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/loop-postcommit-6ef71d3a/candidate-fixed.json
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/default-compile-eval/wheels.mm84T1/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
.venv/bin/python docs/diagnostics/compile-pointwise-loops/compare.py \
  --output target/loop-postcommit-6ef71d3a/comparison --devices 0,1 \
  --wheel target/default-compile-eval/wheels.mm84T1/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_loops tests.test_compile_pointwise_helpers
```

Use the wrapper's newly produced wheel directory and a new diagnostic output
directory when reproducing. Archived command records retain actual environment
overrides, timestamps, clean status and exit codes, including portable extraction
and native regression commands. No implementation, test, dependency definition,
benchmark harness, evaluator or corpus changed in this evidence step.

## Retention and handoff

[candidate-fixed.json](candidate-fixed.json) is an unedited generated report.
[captures.json.xz](captures.json.xz) exports 183 files: focused reports and all
timing outcomes/input-output values, generated CUDA/PTX, source manifest,
command/build/test logs, resource records and audit procedure. Its `files` mapping
stores text, original SHA256 and decoded SHA256; gzip inputs are decoded losslessly.
[audit.json](audit.json) records verified identities and the exact two-cell delta.
[SHA256SUMS](SHA256SUMS) covers this evidence set.

[raw-retention-manifest.json](raw-retention-manifest.json) identifies six new
candidate worker reports and six unchanged main worker reports, plus their logs,
with exact paths, sizes and hashes. All original reports, wheels and caches remain:

```text
/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/default-compile-eval/run-20260914T012512Z-8c57dfda/
/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/loop-postcommit/main/target/default-compile-eval/run-20260914T010746Z-c4c212b4/
```

**The nested clean-main directory is outside the fixed observer's coverage.** Its
exact location was reported to the operator again for separate retention review
before cleanup. Candidate is under the documented author path, but the external
observer receipt is also unverified. The compact export does not include the
large fixed-worker tensor-observation payloads; they remain at these manifested
locations. Generic raw files remain in `target/loop-postcommit-6ef71d3a/comparison/`,
with reports and CUDA/PTX exported here. No cleanup or retention-owner change was
made. Independent review, exact-head qualification, publication and merge remain
with Burner.
