# Public-default compile baseline — 2026-09-12

Initial **pre-adoption** capture, superseded by public-default corpus v2. Review
found an asymmetric bfloat16 tolerance; the corrected gate applies the same
allowance to both frameworks and is remeasured separately. This original report
is retained unchanged, not used as calibration for the revised definitions.
See the [final v2 baseline](../default-compile-v2/README.md) for the corrected run.

This is a **metric correction**, not a compiler implementation change. The
measurements are pinned to clean evaluator commit
`c05b1dcc8eda0ba621de48184cf52b5f727dec77`, whose native implementation is unchanged
from main at `a8ea1225406d3aaa0412d74091ecd8aef97ea1a7`. The evidence-only publication
commit follows the measured commit. Campaign adoption is pending human review.

| Gate | Passing candidate cells | Score |
| --- | ---: | ---: |
| Default compile coverage, CPU + CUDA | 0 / 112 | 0 / 100 |
| Default compile CUDA performance | 0 / 56 | 0 / 100 |

All 28 reference programs passed default PyTorch Inductor on CPU and CUDA,
including the reversed-order CUDA runs. There were **no common successful CUDA
cells**, so a relative latency ratio is undefined (`null`), not 1× or “parity.”
The failures expose the missing general default compiler path, missing module
and custom-autograd APIs, and unsupported CUDA gradients. Existing bounded
`backend="eager"` graph execution and the private specialized CUDA kernel are
not credited as default Inductor equivalence.

## Evidence and reproduction

- [Measured baseline](baseline-2026-09-12.json): both versioned command results,
  every fixed-denominator cell, reference timings, compiler counters, failure
  reasons, source/build/library hashes, and environment metadata.
- [Evaluation contract](../../torch-compile-default-evaluator.md): corpus,
  aggregation, numerical tolerances, and scope limitations.

The two commands ran sequentially through Burner's command-evaluation client at
`ea8866960b1553f7390d66b929a8969836751d85`, from 15:29:52 to 15:35:24 UTC. This did
not start a scheduler, admit feature work, merge PRs, or rewrite campaign history.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric coverage
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric cuda-perf
```

Hardware: NVIDIA H100, compute capability 9.0, GPU UUID
`8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. This is one of eight
H100s on the development machine, not a simulated or CPU-only CUDA result.
Both implementations used logical `cuda:0`, the same CUDA 13 runtime, and one
host compute thread. Python was 3.12.14; reference PyTorch was 2.13.0+cu130.

Both gates built and installed a new release wheel from the measured checkout.
Dependency and native build caches were warm; each reference worker's
Inductor/Triton disk caches started empty. The report separates setup, factory,
cold-call, and steady timings. It does not claim cold-machine setup costs.

Large full-value observations remain in the original worktree under
`target/default-compile-eval/run-20260912T152953Z-9065c0db/` and
`target/default-compile-eval/run-20260912T153320Z-e1c28483/`; their paths and SHA-256
hashes are recorded in the checked-in report. They are retained locally, not
checked into Git. The committed report contains all scoring inputs and timing
samples; the full-value comparisons were executed before those results were
accepted. Reproduce with the frozen corpus to regenerate the large observations.

## Validation history

The 142 focused evaluator/legacy-benchmark/documentation tests passed, as did
Ruff formatting/lint checks. Burner's 224 tests, server/web builds, and type
checks passed separately.

Development-only runs caught a bfloat16 reference-versus-eager rounding
difference and a non-leaf-gradient error in input setup. The evaluator was
corrected before freezing: bfloat16 has the documented dtype-specific reference
tolerance, and gradient inputs become leaves on their target device. Candidate
comparison remains at the tighter tolerance. Input setup uses explicit
`cuda:0` and framework-neutral real CUDA synchronization, avoiding unrelated
factory/synchronization API gaps. No reference case was dropped. Earlier
development observations are retained under `target/default-compile-eval/` and
are not scored evidence.

The old 100 scores describe different measurements and are not comparable to
these new definition versions. Do not relabel this branch's capture as a future
main-branch baseline; adopt the reviewed definitions and use Burner's normal
baseline workflow at the actual adopted revision.
