# Native GELU: staged development evidence

The verified direct-libNVVM provider passed the private native compiler
prerequisite before the bounded public/eager implementation was added. The
[supported surface](../../supported-surface.md) and
[compiler contract](../../compile-pointwise-jit.md#functional-gelu) describe
what is implemented. This directory records development measurements, not a
clean-commit qualification or general numerical/performance certificate.

| Capture | Result | Essential raw record |
| --- | --- | --- |
| Private actual graph/planner/executor, 44 formulas and 113 calls | 335/335 output comparisons pass, including all six prior cancellation witnesses | [Private archive](attempt-002-essential.tar.xz), [member identities](attempt-002-manifest.json.xz) |
| Ordinary eager versus PyTorch eager | 33/33 comparisons pass | [Public archive](public-attempt-002-essential.tar.xz), [member identities](public-attempt-002-manifest.json.xz) |
| Public default compile versus public default compile | 236/236 comparisons pass across uninterrupted scalar/shape/constant/threshold histories | Same public archive |
| Native eager with deliberately absent NVRTC | 33/33 comparisons pass; no NVRTC library mapped | Same public archive |
| Unchanged full fixed corpus, diagnostic mode | 22/112 coverage cells and 22/56 CUDA cells pass; diagnostic aggregates 21.0 and 33.536072 | [Supporting record](supporting-records.tar.xz), `fixed/report.json` |

The private and public comparisons have zero finite-bit or signed-zero
differences. Each compiled comparison has 1,924 NaN-word differences; payload
equivalence is not claimed. Raw float32 bytes, post-call inputs, pointers,
retained outputs with current/prior call IDs, declarations, observed exits,
loaded extension identities, executor PTX and selected passive cache files are
retained. Broad compiler caches are excluded. The original comparator, fixed
corpus, tolerances and weights are unchanged.

All GPU captures use physical H100 GPU0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07,
PyTorch 2.13.0+cu130 and native CUDA runtime 13.0 (NVRTC 13.0 for compiled
captures; none for the no-NVRTC eager leg). Complete loaded-library
hashes and before/after inventory snapshots are in the raw records. No foreign
jobs were interrupted.

The separate one-order timing diagnostic uses five warmups and retains 17
synchronized samples. At 257/131072 elements, eager GELU medians were
10.025/10.276 us versus PyTorch eager 10.556/10.406 us; compiled GELU was
35.173/35.574 us versus Inductor 26.420/32.008 us. The first small compiled call
was 120.079 ms versus 2167.016 ms; later first calls inherit process warmup.
Affine and compiled trig controls, changed-input checks and every slow result
remain in `comparison.json`. These measurements establish no performance gain.

Supporting records include generation API/process receipts, complete provider
PTX, the exact pre-public implementation patch, bounded source/raw audits,
test/build logs and packaging checks. The first private attempt failed in the
diagnostic's pooled-allocation context query before kernels; the first public
attempt used unsupported CUDA `narrow` in an offset fixture. Their original
sources, failed reports and process receipts are retained under `failures/`.
They are harness failures, not discarded numerical failures. Earlier stock
CUDA13 numerical failures (4/335 leaves, 24 elements) and the unsuccessful NVCC
provider selection remain unchanged in the historical
[stage1](../compile-gelu-stage1/README.md) and
[vendor](../compile-gelu-vendor/README.md) records at commits `934719a` and `1bc7add2`.

Archive and source digests are listed in [SHA256SUMS](SHA256SUMS); the
[supporting member index](supporting-records-manifest.json.xz) binds each
retained file without rewriting its measured provenance. The
[archive audit](archive-audit.md) verified all 4,442 members; its two wording
corrections are applied here.

## Reproduction and generation

Run from the checkout root, keeping caches and outputs inside it. Use GPU0
only under the canonical lease. Set `CUDA_VISIBLE_DEVICES=0`,
`PYTHONDONTWRITEBYTECODE=1`, `TMPDIR="$PWD/target/gelu-linked/tmp"`,
`TORCH_RS_CUDART=/usr/local/cuda-13.0/lib64/libcudart.so.13` and
`TORCH_RS_NVRTC=/usr/local/cuda-13.0/lib64/libnvrtc.so.13`; create that temporary
directory first. The existing `scripts/evaluate_torch_compile_default.sh
--setup-only` prepares a local release wheel/interpreter. Unset `PYTHONPATH`
when using `target/default-compile-eval/venv/bin/python -B` below.

- Obtain the official archive identified by
  [its immutable input record](../compile-gelu-vendor/official-input.json).
  Verify its archive hash, extract only the declared bitcode member to
  `target/gelu-linked/input/libdevice.10.bc`, and verify SHA256
  `1fc1bc8d4131d5a59a91fc26f4908886e231f842c9e83ae951979ff3df82bdb5`.
  Do not execute the downloaded compiler. Run
  `docs/diagnostics/compile-gelu-linked/generate_provider.py --run` with the
  local Python. It pins installed CUDA12.8 libNVVM and records both official
  and no-library controls. Only the official leg defines the complete device
  function; its PTX SHA256 is
  `d669b53d4f529a03c40d4b7eda3d0def5437e9a4edc52a130ed2d180075129de`.
  Complete commands, submitted-buffer hashes and API statuses are in
  `generation/` in the supporting archive. The historical input record's
  failed NVCC selection is not attributed to this successful libNVVM run.
- Eager generation uses installed CUDA13 nvcc on `src/cuda/gelu.cu`,
  `--ptx --gpu-architecture=compute_75 --fmad=true --ftz=false
  --prec-div=true --prec-sqrt=true --std=c++17 --compiler-bindir=/usr/bin/g++`.
  The exact command, intermediates and tool hashes are under `eager-generation/`.
  Write a regeneration to a fresh local output and compare with
  `src/cuda/gelu.ptx`. This is separate from the compiled provider.
- For the private matrix, invoke `probe.py declare target/gelu-linked/private-rerun
  --provider-input src/cuda/erf_provider.ptx --provider-input src/cuda/erf_provider.ll`;
  also pass the new generation receipt using `--provider-input`. Then invoke
  its `native`, `reference`, `compare` and `pack` legs with that directory.
- For public evidence, invoke `public_probe.py declare target/gelu-linked/public-rerun`,
  then its `native-eager`, `reference-eager`, `native-compiled`,
  `reference-compiled`, `native-no-nvrtc`, `compare` and `pack` legs with the same
  directory. Each execution creates fresh local caches; do not reuse a run
  directory or change compile policy between calls.

The two Python probes are in this directory. Provider generation is an offline
maintainer operation, not a runtime compiler dependency. Generated vendor math
is covered by the separately included [NVIDIA terms](../../../src/cuda/NOTICE.md).

## Delivery boundary

These captures record commit `1bc7add2f6144ad6bd52343659926090d152985c` **plus
uncommitted implementation changes**, with actual source/build hashes. The
private capture predates the public implementation. Public capture 002
predates final lint/documentation and Python-only build-guard changes; the
fixed diagnostic predates those last build-guard changes. None is relabeled
as measuring the eventual clean implementation commit. Recorded paths and
identities describe the actual executions, including the private stage's
read-only external Python interpreter.

Author checks passed 257 Rust library tests, 15 new Python tests, 48 focused
regressions (one multi-device skip under the GPU0-only lease), 17 generator
harness tests, both default/Python-binding clippy configurations and formatting.
Wheel and source-package vendor/source inclusion was checked. Exact logs,
earlier failed checks and limits accompany the supporting record.

**Outstanding after Burner commits:** regenerate current public eager/compiled/
no-NVRTC and timing evidence from the clean code commit, run the unchanged
fixed coverage and CUDA-performance scoring gates, and complete canonical full
qualification/independent review. The present fixed run uses `--diagnostic`
and records `valid=false`; it is not a final score. All worker output hashes,
logs and the full 112-cell report are retained; bulky fixed raw worker tensors
remain at the recorded worktree-local run directory, not in this archive.
Historical stage records remain pinned. Burner alone owns commits, evaluation
and progress stamping; this evidence does not approve delivery.
