# Clean evidence after the skipped-local repair

These captures measure clean commit
`18e94de5d0177c8f67ace8d330018c5e0da3bcc2` on September 14, 2026 UTC, including
the [skipped-local repair](../review-skipped-locals.md). All measurements and
checks finished before this evidence-only edit. They refresh the current
candidate evidence; earlier captures, failures, scores and the documented
late-raw retention gap remain unchanged. This step does not replace independent
review, exact-head qualification or Burner's canonical score publication.

| Check | Result |
| --- | --- |
| Unchanged public-default-compile-v2, `--metric both` | Valid |
| Coverage | 14.5%; 14/112 cells passed |
| CUDA performance | 20.0; 14/56 cells passed |
| Common-success geometric reference/native latency ratio | 1.1550354001155279 |
| Two-H100 generic-loop diagnostic, both framework orders | Eight legs; 60 paired histories passed |
| Focused loop/helper/signature/diagnostic tests on H100 | 66 passed; no skips |
| Same focused tests on CPython 3.10.19, 3.11.15, 3.13.13, 3.14.5 | 66 each; ten hardware skips each |

[candidate-fixed.json](candidate-fixed.json) is the unedited generated report.
All 28 programs, 14 categories, weights, tolerances, unsupported outcomes and
denominators remain intact. Against historical main `014fc0de`, only the two
CUDA `static_loop` pass/fail cells differ. Their category still contributes zero
to geometric performance because its other cells remain unsupported. Arithmetic
and broadcasting contribute 12 and 8 respectively in this run. These are finite
corpus measurements, not broad compiler equivalence or feature percentages.

The prior `bdcfb051` capture also measured 20.0. The earlier canonical CUDA
regression (19.613404424665234 versus 20) and polish result remain preserved in
the [operator recovery record](../operator-recovery.md). No unchanged candidate
was rerun to select favorable results, and no timing difference is attributed
to the warm-validation or skipped-local changes without causal evidence.

## Numerical and compiler observations

The unchanged diagnostic keeps each ordinary default wrapper across five
shape/value/scalar histories for three generic programs. Both framework orders
run independently on each reserved H100, with fresh values/storage offsets,
IEEE values and empty outputs. Native original-IR unequal-shape rejection still
passes. Each history records a separate cold duration, five warmups and 17
real-device-synchronized samples. Reference/native median ratios range from
**0.5706331565395365 to 1.5037925770441751**; native was slower in **42/60**
histories. Every sample remains retained. Tolerances stay `rtol=1e-5, atol=1e-6`,
including the reference-zero sign check.

Native cold and extra warm probes observed no original root/helper execution,
recorded C-extension executor entry, and observed no warm lowering. Generated
CUDA/PTX bytes from the actual observed receivers were hash-verified separately
from numerical matches. Executor entry is not a driver-level launch trace;
empty outputs do not imply a kernel launch. Reference counters attribute new
generated kernels to 28 histories; the other 32 remain **unknown for this call**.
Initial and last available output/input observations are retained, not every
normal warmup/sample tensor return. Padding/prior-input storage-history coverage
remains nonexhaustive.

The focused hardware tests additionally passed the exact skipped-local
reproducer against ordinary default `torch.compile`, executed-unbound rejection,
helper and shape histories, input immutability, warm binding rejection/recovery,
reset and device restoration. The prior full 245-test GPU sweep and unchanged
native Rust suite were not repeated in this evidence step.

The clean hardware-free warm probe retained all five warmup batches and 17
sample batches of 1,000 mocked-native calls, with a separate 100-call profile.
Arithmetic/literal-loop medians were 20.689937/21.187840 microseconds per call.
Each profile recorded 100 signature, namespace and range checks, and zero root
code rescans. These host/mock measurements do not establish GPU speedup.

## Provenance and reproduction

Canonical `gpu` and `cpu-heavy` locks identified this session's owner
`agent_910598c3-retry`, PID 3092842. Only the required H100s 0 and 1 were selected:
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` and
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. The fixed evaluator used GPU 0;
the diagnostic used both independently. Inventories retain utilization and
memory before/after. The runs used driver 580.82.07, CPython 3.12.12, PyTorch
`2.13.0+cu130`, Rust 1.92.0, NVRTC 13.0 targeting `compute_90`, and CUDA runtime
13.0 synchronization. System nvcc 12.6.85 was recorded separately.

The unchanged wrapper built/installed a fresh locked release wheel using warm
worktree-local dependency/native build caches; this was not a clean-target Rust
rebuild. Wrapper setup recorded 0.489384766 seconds for the native build/package
step and 0.821340588 seconds total. All checks used that wheel. All 130 evaluator
source hashes, checkout/wheel/import Python bytes, extension bytes, actual
receiver artifacts and worker report/log hashes were verified.

- Frontend SHA256: `a70b2059fc15c98878f3ac53ee09b53e5f427be5670bc1e7f9c7babfad59e40e`
- Evaluator source manifest SHA256: `087609fb0684ac533c9864a79cd78b0b4b7110031189f4bf53b26c431e0d2f7e`
- Wheel SHA256: `61ff5d4a7701c10cf84779746933bf09d7f20a5aaaab57cfee57b806d7cafdfc`

Actual diagnostic/test argv, timestamps, clean status and exit codes are retained
in `commands.json` in the archive. From the reserved checkout with local settings
in `target/operator-recovery/env.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 UV_SYSTEM_CERTS=true \
  bash scripts/evaluate_torch_compile_default.sh --metric both \
  --output target/postcommit-18e94de5/candidate-fixed.json
target/default-compile-eval/venv/bin/python -B \
  docs/diagnostics/compile-pointwise-loops/compare.py \
  --output target/postcommit-18e94de5/comparison \
  --durable-raw-output "$APPROVED_RAW_ATTEMPT/comparison" \
  --wheel target/default-compile-eval/wheels.M8BnUR/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl \
  --devices 0,1
```

Reproduction requires new attempt/workspace paths and the newly generated wheel.
Interpreters, dependencies, builds, wheels and writable framework/CUDA caches
remain inside the checkout; only the explicitly approved raw destination is
external. Portable checks extracted the same wheel locally with the existing
`typing_extensions` dependency.

The unchanged [historical main report](../postcommit-dcfeb27a/main-fixed.json)
was revalidated against all 130 files at `014fc0de` and all 12 retained original
worker report/log byte streams. [main-revalidation.json](main-revalidation.json)
records that audit. Its original checkout is gone; no new main build, import or
workload measurement is claimed. Its historical manual timing comparison does
not replace the canonical performance baseline of 20.

## Retention

[audit.json](audit.json) records provenance and outcomes.
[captures.json.xz](captures.json.xz) exports **172 raw files**, byte-verified
against the originals, including compressed diagnostic reports, actual receiver
CUDA/PTX, command/build/test logs and the fixed report.
[raw-retention-manifest.json](raw-retention-manifest.json) records paths, sizes
and hashes. Large fixed worker payloads are manifested rather than embedded.

New diagnostic originals were written directly under:

```text
/tmp/burner-approved-default-compile.UA2M6I/static-loop-operator-recovery.iJS0Kv/diagnostic-outputs/postcommit-18e94de5-01/
```

The original fixed worker reports/logs remain at the exact location reported to
the operator before cleanup:

```text
/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/default-compile-eval/run-20260914T024649Z-42cdacb0/
```

The unchanged `f0747ea1` observer covers this canonical author path and its
full-leaf path; the final precleanup archive receipt is pending. Coverage alone
is not a completed-copy claim. The observer does not cover generic diagnostic
or historical nested-main paths. No retention owner, cleanup, frozen evaluator,
benchmark harness, dependency definition or implementation changed here.
Earlier missing compressed raw files remain missing; historical decoded exports
are not recovered original bytes. Burner owns independent review, qualification
and delivery.
