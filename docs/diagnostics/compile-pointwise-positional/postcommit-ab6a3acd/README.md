# Clean warm-dispatch evidence

This record measures clean candidate `ab6a3acd82c681d6e02ad56f4458decb1384ae42`,
actual main `a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4`, and the unchanged
`885264b5319d76165e6be8d6df405450b3830c47` comparison build. The fixed gates and
three dispatched-module captures completed; the separate warm-dispatch comparison
is blocked by a committed diagnostic API error. This record does not qualify the
branch or establish general Inductor parity.

## Fixed measurements

| Build | Commit | Weighted coverage | CUDA performance | Common-success reference/native ratio |
| --- | --- | ---: | ---: | ---: |
| [main](main.json.gz) | `a281503f3391` | 8% (6/112) | 12% (6/56) | 1.7822267 |
| [candidate](candidate.json.gz) | `ab6a3acd82c6` | 9% (8/112) | 12% (8/56) | 1.25727301 |

The [candidate](candidate.json.gz) and [main](main.json.gz) reports retain all
112 coverage cells and 56 CUDA cells, unsupported outcomes, both variants and
changed-value checks, both CUDA implementation orders, five warmups, 17 samples,
synchronization, cold/steady timings and setup provenance. Both are valid,
non-diagnostic runs of the unchanged `public-default-compile-v2` gate. The
common-success sets differ, so their ratios are not a paired speedup claim.
No evaluator, corpus, tolerance, denominator or reference configuration changed.

## Balanced comparison blocker

The committed [plan](../warm-dispatch/measurement-plan.md) was executed once in
its full 16-leg order on physical H100 GPU 0
(`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`). Fresh source-bound release wheels
came from clean [885](comparison/build-B.json) and
[corrected](comparison/build-C.json) sources; every leg used a fresh process and
separate empty runtime/compiler caches. All eight reference legs completed their
seven histories, retaining two setup traversals, five warmups and all 17 samples.
All eight native legs failed each history before timing because the committed
consumer calls `torch_rs.cuda.synchronize()`, which the native public module does
not provide. No compiled native call or timing sample was reached.

The [verification result](comparison/verification.json) retains all 56 paired
history attempts and every missing-observation/contract failure. The 16 input
reports are preserved beside it as `leg-00.json.gz` through `leg-15.json.gz`.
There is no native timing or paired speedup result. Repairing synchronization in
the diagnostic and repeating the bounded comparison requires a separately
committed tooling correction; it was not patched during this evidence step.
The [original negative result and CPU profiles](../warm-dispatch/README.md)
remain unchanged. Neither this failed comparison nor the refreshed fixed scores
replace normal independent qualification.

## Captures and checks

Three committed consumers captured the executor actually dispatched by their
non-corpus program with the candidate gate's installed wheel:

- Positional [CUDA](positional-codegen/kernel.cu), [PTX](positional-codegen/kernel.ptx.gz), [provenance](positional-codegen/provenance.json) and [source manifest](positional-codegen/source-manifest.json.gz).
- Bounded broadcast [CUDA](bounded-codegen/kernel.cu), [PTX](bounded-codegen/kernel.ptx.gz) and [provenance](bounded-codegen/provenance.json).
- Pointwise JIT [CUDA](jit-codegen/kernel.cu), [PTX](jit-codegen/kernel.ptx.gz) and [provenance](jit-codegen/provenance.json).

All 36 focused cache, persistent-scalar and capture-consumer tests passed.
Three of those tests exercise the diagnostic verifier with synthetic records.
These checks retain persistent ordinary-default wrappers, unchanged assertions,
NaN/signed-zero histories, promotion, cache recency/failure/reset behavior,
structural sharing and dispatch selection. The synthetic diagnostic-verifier
tests do not test the missing native synchronization API.
The [documentation checks](artifact-checks.json) verify 12 smoke tests, affected
and inbound links, manifest relationships and unchanged historical artifacts.

The [verification receipt](verification.json) checks clean source identities,
wheel/imported bytes, raw worker hashes, full matrices and selected CUDA/PTX
hashes. [Lossless logs and command receipts](logs.json.gz) retain setup,
measurements and focused checks, including failures. An initial main invocation
rejected an output path outside its detached source root before any workloads;
its invalid internal report is retained. The corrected invocation wrote under
that root and ran the full unchanged gate. No measured workload was repeated to
select a favorable timing.

## Reproduction and provenance

Exact commands, timestamps, paths and build logs are archived. Candidate sources
are this worktree; independent detached `885` and main checkouts are under
`target/postcommit-ab6a3acd/before` and `target/postcommit-ab6a3acd/main`.
Interpreters, dependencies, builds, temporary files and writable caches remain
inside this worktree. Dependency/build caches may be warm; every source build
produced a fresh locked release wheel. The fixed evaluator created fresh worker
Inductor/Triton caches. GPU measurements ran sequentially on the same H100.
Python 3.12.12, PyTorch 2.13.0+cu130, Rust 1.92.0, driver 580.82.07,
selected runtime/NVRTC versions and installed nvcc 12.6 are recorded separately.

At each clean revision, run the existing gate with an output under that revision's
checkout:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/new-clean-report.json
```

Use the three existing capture scripts with the freshly installed candidate
wheel; their exact invocations are in the command receipt. The balanced consumer
commands and order are in the linked predeclared plan. Preserve its failure until
the committed consumer is repaired. All earlier snapshots, raw artifacts and
historical failures remain byte-identical and retain their original attribution.
Burner owns commits, independent review, full qualification and delivery.
