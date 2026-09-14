# Clean evidence after the operator-requested repair

These measurements used clean implementation commit
`bdcfb051600783bb8f7550d42f36409b9ed2891d` on September 14, 2026 UTC. Every
measurement completed before this evidence-only edit. They refresh the candidate
evidence after the comparator/retention/timeout repairs and warm-validation
change. Earlier captures, failures, the CUDA regression and the late-raw
retention gap remain unchanged. This is not independent review or full-gate
qualification; Burner owns those decisions and canonical score publication.

| Check | Recorded result |
| --- | --- |
| Unchanged public-default-compile-v2, `--metric both` | Valid |
| Coverage | 14.5%; 14/112 cells passed |
| CUDA performance | 20.0; 14/56 cells passed |
| Common-success geometric reference/native latency ratio | 1.0718673076278857 |
| Generic loops on two reserved H100s, both framework orders | Eight legs and 60 paired histories passed |
| Focused loop/helper/signature/diagnostic tests | 63 passed; no skips |

[candidate-fixed.json](candidate-fixed.json) is the unedited generated report.
The 28 programs, 14 categories, tolerances, unsupported outcomes and denominators
are unchanged. The only pass/fail differences from historical main `014fc0de`
are the two CUDA `static_loop` cells. Their category still contributes zero to
geometric performance because other cells remain unsupported. The arithmetic
and broadcasting categories contribute 12 and 8 respectively in this run.
This is a finite-corpus measurement, not broad Python/compiler coverage.

The earlier canonical CUDA result, 19.613404424665234 versus baseline 20, remains
recorded in the [operator recovery notes](../operator-recovery.md). Its original
polish result and unqualified status also remain historical facts. This new
single run is not proof that redundant validation caused that earlier loss.
For example, the new affine variant-0 candidate/reference medians were
69.856/69.546 microseconds in reference-first order and 62.896/72.990 in
candidate-first order. All 17 samples in both orders remain in the report;
no favorable repeat or sample selection was used.

## Focused observations

The committed diagnostic retained ordinary persistent native/default-Inductor
wrappers across five shape/value/scalar histories per program, fresh values and
storage offsets, IEEE values and empty outputs. Native original-IR unequal-shape
rejection remained intact. Each history has a separate cold duration, five
warmups and 17 real-device-synchronized samples. Ratios span
**0.5951179820992677–1.2107631627709983**; native was slower in **42/60** histories.
All outcomes remain retained at the unchanged `rtol=1e-5, atol=1e-6`, including
the corrected sign check whenever the reference output is zero.

Cold and extra warm native probes observed no original root/helper execution,
recorded `_PointwiseKernel.run` entry, and observed no warm lowering. Actual
receiver CUDA/PTX hashes were verified independently of numerical matches.
Executor entry is not a driver-level kernel trace, and empty outputs do not prove
a launch. Reference counters attribute new generated kernels to 28 histories;
the other 32 retain **unknown for this call** attribution. Initial and last
available output/input observations are saved, not every normal warmup/sample
tensor return. Padding/prior-input storage-history coverage remains nonexhaustive.

The clean hardware-free warm probe retained all five warmup batches and 17 sample
batches of 1,000 mocked-native calls, followed by a separate 100-call profile.
Arithmetic/literal-loop medians were 21.100366/21.369934 microseconds per call.
Each profile recorded 100 signature, namespace and range validations, and zero
root-code rescans. This confirms the narrow warm-path behavior; mocked host
timings do not establish GPU speedup or explain the earlier CUDA result. The
original development before/after pair remains unchanged.

The prior full Python, native Rust and CPython-version sweeps were not repeated:
this evidence step changed no implementation or test. The 63 focused checks
exercise the new wheel and diagnostic, including hostile bindings, zero-trip
admission, recovery/reset, device restoration and failure-side retention.

## Provenance and reproduction

The canonical `gpu` and `cpu-heavy` locks identified owner
`agent_910598c3-retry`, PID 3092842. The existing reservation selected only H100s
0 and 1, UUIDs `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1` and
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. The fixed evaluator used GPU 0;
the generic diagnostic used both devices independently. Inventories retain
before/after utilization and memory. Driver 580.82.07, CPython 3.12.12,
PyTorch `2.13.0+cu130`, Rust 1.92.0, NVRTC 13.0 (`compute_90`) and CUDA runtime
13.0 synchronization were used. System nvcc 12.6.85 was recorded separately.

The unchanged wrapper built a fresh locked release wheel in the new canonical
worktree-local target. Dependencies were copied from this checkout's existing
environment and verified by locked sync; dependency caches were warm. Recorded
native build time was 49.641699441 seconds, total wrapper setup 50.002419527
seconds. The diagnostic and focused checks used that exact wheel and evaluator
environment. All 130 evaluator source hashes, checkout/wheel/import Python
bytes, extension bytes and worker report/log hashes were verified.

- Frontend SHA256: `2309690925aef68bcc2890202087190dfb83bb96e9d57431d5331fa2fc85302e`
- Evaluator source manifest SHA256: `5c73ba2fca2b54ac527c40201e946a34e57483c42e1af6d98b1d84b3b8838bd2`
- Wheel SHA256: `f09685fc8fe52c243c1a4403791d7d36f85baebee290d977bd3583fc2d884d57`

Actual diagnostic/test argv, timestamps, clean status and return codes are in
the archived `commands.json`. From the reserved clean checkout, with local
environment/cache settings in `target/operator-recovery/env.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 UV_SYSTEM_CERTS=true \
  bash scripts/evaluate_torch_compile_default.sh --metric both \
  --output target/postcommit-bdcfb051/candidate-fixed.json
target/default-compile-eval/venv/bin/python -B \
  docs/diagnostics/compile-pointwise-loops/compare.py \
  --output target/postcommit-bdcfb051/comparison \
  --durable-raw-output "$APPROVED_RAW_ATTEMPT/comparison" \
  --wheel target/default-compile-eval/wheels.0vnABb/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl \
  --devices 0,1
```

Reproduction requires new attempt/workspace paths and the wrapper's newly
produced wheel path. Interpreters, dependencies, builds, wheels and writable
CUDA/Inductor/Triton caches stay inside this checkout. Only the operator's
explicitly authorized raw-output destination is external.

The unchanged [historical main report](../postcommit-dcfeb27a/main-fixed.json)
was revalidated against all 130 source files at `014fc0de` and all 12 retained
original worker report/log bytes. [main-revalidation.json](main-revalidation.json)
records this read-only audit. Its original checkout is gone; no new main import,
build or workload capture is claimed. That historical manual comparison does
not replace the canonical performance baseline of 20.

## Retention and handoff

[audit.json](audit.json) records verified provenance and outcomes.
[captures.json.xz](captures.json.xz) exports **164 raw files**, with exact original
bytes verified against their sources, including compressed diagnostic reports,
actual receiver CUDA/PTX, command/build/test logs and the generated fixed report.
[raw-retention-manifest.json](raw-retention-manifest.json) gives paths, sizes and
hashes, including all fixed worker payloads; those large payloads are not embedded
in the compact archive.

New diagnostic originals were written directly under:

```text
/tmp/burner-approved-default-compile.UA2M6I/static-loop-operator-recovery.iJS0Kv/diagnostic-outputs/postcommit-bdcfb051-01/
```

The frozen evaluator's original worker reports/logs remain at:

```text
/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/agent_910598c3/target/default-compile-eval/run-20260914T022744Z-f47ceb3a/
```

This exact path was reported to the operator before cleanup. The read-only
recovery-owner contract installs the unchanged `f0747ea1` observer for canonical
author/full-leaf cleanup. Its final archive receipt is still pending; coverage
does not itself prove a completed copy. That observer does not cover the generic
diagnostic or historical nested-main paths. No new retention owner, late copy,
cleanup or evaluator modification was introduced. The earlier missing original
compressed files remain missing; decoded historical exports are not recovered
original bytes. Independent review, exact-head qualification and delivery remain
with Burner.
