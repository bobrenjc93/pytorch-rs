# Predeclared bounded H100 comparison

Record this plan before hardware measurement. This is a seven-history diagnostic,
not a scoring gate; normal independent review and unchanged evaluations determine
qualification. Retain the actual `885264b` negative result and every new result,
including failures. Do not pick the best repeat or replace the negative result.

## Source and environment identity

Compare clean `885264b5319d76165e6be8d6df405450b3830c47` with the corrected source
only after Burner commits it. Build ordinary source-bound wheels from clean
checkouts rooted inside this worktree. Pin the corrected commit in the execution
receipt before starting, without altering this workload plan. Verify source
manifests against wheel contents and imported package paths; retain build logs,
compiler/runtime versions, wheel hashes, commands and timestamps. Do not attribute
development wheels or tests to a later clean commit. Existing historical captures
stay byte-identical; refresh current-candidate coverage/CUDA and dispatched-module
captures through the existing post-commit tooling after code changes.

Hold the shared GPU/CPU-heavy resources, with `CUDA_VISIBLE_DEVICES=0`, and record
the physical GPU 0 UUID before starting. Use that same H100 for every leg and
verify it again afterward. Record utilization/memory snapshots, driver, NVRTC,
CUDA runtime and locked PyTorch version. An idle snapshot is not a reservation.
Every interpreter, wheel, dependency, build and cache directory stays inside the
worktree. Set one host thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and the
framework's thread setting). Use fresh processes and empty, distinct caches per
leg, with identical warmup and sampling work.

## Programs and histories

Use ordinary public `torch_rs.compile(fn)` and untouched ordinary default
`torch.compile(fn)` with no compiler options. The seven independent non-corpus
histories mirror the preserved CPU diagnostic. Inputs are contiguous float32 on
logical `cuda:0`, with deterministic finite values; reuse input identities and
aliasing within each wrapper's history. Seed input generation identically for
both frameworks and builds. Keep each compiled wrapper alive for its complete
history, including warmup and sampling; no reference-only resets.

| History | Program | Input/history |
| --- | --- | --- |
| Literal unary | `x * 1.25 + 0.75` | `(31, 7)` |
| Two-tensor arithmetic | `x * y + x` | two distinct `(31, 7)` tensors |
| Broadcast | `x + y` | `(31, 7)` and `(7,)` |
| Repeated alias | `x + y` | both arguments are the same `(31, 7)` tensor |
| Shape revisit | `x * y + y` | cycle distinct pairs at `(31, 7)`, `(47, 7)`, revisiting each |
| Promoted capture | `x * scale + x` | `(31, 7)`; cycle captured `scale=1.25, 2.5, 0.75` |
| Eight Boolean entries | `x*a + x*b + x*c` | `(31, 7)`; cycle all eight captured Boolean masks in ascending order |

Before steady sampling, traverse each whole history twice to establish its
specializations and record outputs. Separate processes prevent inline reference
comparison: the final verifier compares every setup, warmup and sample output
against the persistent default wrapper's matching observation. Then perform five whole
history warmups and 17 whole-history samples. Synchronize the relevant framework
immediately before and after each timed history traversal, and record all samples
and the number of calls per sample. Keep cold setup time separate from steady
samples. Materialize and compare every output outside timing using the existing
pointwise regression tolerances and metadata/alias assertions; verify unchanged
inputs and fresh outputs. Do not weaken numerical checks, drop unsupported
outcomes or reset either wrapper within a history. Preserve failures and continue
recording other independent legs when feasible.

## Balanced order and reporting

Run the following four rounds once, retaining each round as a separate result.
Here B is the unchanged `885264b` native wheel, C the committed corrected native
wheel, and R ordinary default PyTorch. Each pair has fresh processes/caches and
executes all seven histories in the fixed table order. The two R legs in a round
are separate measurements associated with their respective native build.

1. B then R; C then R.
2. R then C; R then B.
3. C then R; B then R.
4. R then B; R then C.

This balances both native/reference order and before/corrected build order.
Report every cold duration, all 17 samples, medians and failures for all 56 paired
history legs (7 histories × 2 builds × 4 rounds), with both native and reference
observations. Summarize each build across all planned rounds rather than selecting
a favorable result. A setup or infrastructure failure remains a recorded failure;
any necessary rerun must be separately labeled with its reason and must not erase
the first attempt. CPU profiles, this bounded hardware comparison and ordinary
regression tests cannot award fixed-corpus credit or override normal qualification.


## Version 2 synchronization amendment

The first clean v1 attempt is retained in the
[ab6a3acd record](../postcommit-ab6a3acd/README.md#balanced-comparison-blocker):
every native history stopped at the absent public synchronization method before
timing. The v2 consumer uses the same locked CUDA runtime's `cudaSetDevice(0)`
and `cudaDeviceSynchronize()` for both frameworks and records their shared
runtime identity. All programs, histories, orders, traversals and tolerances
above remain fixed. After Burner commits this correction, rerun all 16 legs
with fresh output directories and source-bound wheels; do not combine v1 and v2
legs. Development smoke tests do not satisfy that clean comparison.

## Reproduction with the committed consumer

The [consumer](gpu-dispatch.py) uses no evaluator imports or altered reference
settings. Its `build` command records a locked release build's clean source
manifest before and after compilation, wheel bytes, compiler identities and log.
Its `leg` command verifies the selected wheel against every installed native
Python/extension member and the clean source. It records all 24 traversals per
history (two setup, five warmup, 17 samples), full outputs and input invariance,
metadata and fresh-output checks outside timing. `verify` checks ordering, source,
GPU, all 56 pairs and every recorded output using the existing pointwise JIT
regression tolerance (`abs(actual-reference) <= 1e-6 + 1e-5*abs(reference)`),
equal-NaN policy and matching signs when both values are zero. Tensor leaves are
hexadecimal float strings, preserving strict JSON serialization even for unexpected
NaN/infinity observations; comparison decodes them without losing zero signs. Both legs remain available on
failure. The verifier records all input paths/hashes before comparisons and
retains failures from every independent history/output instead of stopping at
the first mismatch. Its final result fails if any comparison or contract fails.
A successful leg alone does not assert paired numerical parity. No-replay checks
are supplied by the separate CPU diagnostic and unchanged compiler regressions;
the hardware timing loops are not profiled or given extra probe traversals.

After Burner commits the correction, prepare clean before/corrected checkouts
inside the current worktree and a worktree-local Python 3.12 environment with the
locked dev/reference groups. Use already installed Rust/CUDA toolchains; all
writable Cargo, uv, runtime and framework caches must stay inside the worktree.
The consumer selects per-command cache paths automatically, including explicit
`CUDA_CACHE_PATH` and `TORCH_HOME` overrides before framework imports. The source checkouts
must have no untracked or modified files; use their ignored `target/` directories
for builds. Run the current committed consumer for both source checkouts. Record
resource ownership outside these commands; the script does not allocate GPUs.

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$PWD/target/uv-cache"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
consumer="$PWD/docs/diagnostics/compile-pointwise-positional/warm-dispatch/gpu-dispatch.py"
python="$PWD/.venv/bin/python"
before_source="$PWD/target/warm-dispatch/before-source"
corrected_commit="$(git rev-parse HEAD)"
# before_source is an independently initialized detached checkout of 885264b,
# contained here; never mutate the parent repository to create it.
"$python" "$consumer" build --source-root "$before_source" \
  --commit 885264b5319d76165e6be8d6df405450b3830c47 \
  --output "$PWD/target/warm-dispatch/build-B/report.json"
"$python" "$consumer" build --source-root "$PWD" --commit "$corrected_commit" \
  --output "$PWD/target/warm-dispatch/build-C/report.json"
```

Inspect and record GPU 0's full physical UUID before starting. The following loop
implements the exact 16-leg order. Each iteration installs the corresponding
recorded wheel into the local environment before launching a fresh interpreter;
reference legs retain their association with that same build. Replace the UUID
placeholder with the recorded value. The `uv` executable may be read-only outside
the checkout; its environment, wheel writes and cache remain local.

```bash
gpu_uuid=GPU-REPLACE-WITH-RECORDED-GPU-0-UUID
builds=(B B C C C C B B C C B B B B C C)
implementations=(native reference native reference reference native reference native native reference native reference reference native reference native)
reports=()
for index in "${!builds[@]}"; do
  build_record="$PWD/target/warm-dispatch/build-${builds[index]}/report.json"
  wheel="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel"])' "$build_record")"
  uv --no-config pip install --python "$python" --force-reinstall --no-deps "$wheel"
  report="$PWD/target/warm-dispatch/leg-$index/report.json.gz"
  reports+=("$report")
  "$python" "$consumer" leg --build-record "$build_record" \
    --build-label "${builds[index]}" --implementation "${implementations[index]}" \
    --leg-index "$index" --gpu-uuid "$gpu_uuid" --output "$report" || true
  # A failed leg remains recorded. Continue the predeclared remaining legs.
done
"$python" "$consumer" verify "${reports[@]}" \
  --output "$PWD/target/warm-dispatch/verification/report.json"
```

Each invocation requires a new output directory and refuses overwrites. Preserve
all build/leg/verifier reports and logs with the execution receipt. Failed setup
cannot become an omitted workload; `verify` fails if any leg is missing, failed,
out of order, numerically wrong or bound to a different source/device. Do not
rerun for a faster sample. This predeclared comparison supplements, and never
replaces, the clean candidate/main evaluator refresh owned by Burner.
