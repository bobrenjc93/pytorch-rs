# Default input-rooted transpose validation

**Current validation:** the [committed-source QA capture at `95bdbf2`](diagnostics/default-alias-mutation-qa-postcommit-95bdbf2.md)
covers the current alias-mutation and input-transpose regressions. It records
focused correctness checks, not a performance measurement or canonical score.
The [default compiler guide](compile-pointwise-jit.md) defines the current capability.

## Historical capability capture at `8ac9d0e`

Clean commit `8ac9d0e768e2cd6faab558dc6838240231ccf585` passed the focused
transpose and public CUDA scalar `add_` checks. [Clean-commit verification](cuda-add-inplace.md#clean-commit-verification)
links the generated report and raw receipts: 39 focused tests passed, 18 portable
tests passed with 21 hardware skips, and 23 native tests passed. No implementation
or test changed during capture. This capture and the older scored artifacts below
are historical: they measure their named revisions, not the current candidate.
Their original receipts and measurements remain unchanged.

Implemented from `60202557b4f110d07777f585e804ab5f55e1ff7b`. This is bounded
metadata-compilation support, not general Inductor or performance parity.
The default frontend retains one immutable view recipe with original binding
sources, construction identities and active alias order. Native metadata
preflight includes retained inactive/zero-trip constructions on warm calls;
only active constructions enter the existing alias bridge. Numerical SSA,
output ordering, preparation and launch remain with their existing owners.

## Review correction and measurement status

Independent review found that negating a runtime boolean discarded its input
provenance and admitted it as an integer transpose axis. The correction retains
the original binding through unary negation and helper/container forwarding,
and rejects parameter-derived axes before native execution or cache publication.
Literal, local, global and closure constants remain admitted, and view-free
boolean arithmetic retains its existing behavior.

Before the correction was committed, verification used a freshly packaged release wheel in the worktree-local `.venv`
on GPU0: 24 focused tests passed (including 11 hardware tests); 161 helper,
control-flow, input/result and cache/guard regressions passed with three existing
skips. The hardware-free run passed 25 tests with 11 explicit CUDA skips.
The new regressions cover both boolean values, direct parameters and selected
tree leaves, double negation, helper/container forwarding, unused/inactive/zero-trip
instructions, failed-publication recovery, admitted constants and unchanged
view-free arithmetic. Focused source review reported no remaining findings.

[Review-fix receipts](diagnostics/default-input-transpose-review-fix.tar.gz)
retain the initial reproducer failures, a corrected helper fixture, final logs,
exact commands and uncommitted source/build/import identity against base
`a2332c9deff957b4b09c3e50149d4cfe1c94d2e1`. They are development verification,
not a clean-commit measurement. Native Rust sources and the evaluation harness
were unchanged by this correction.

Burner committed the correction as `48c7b82`; the clean-commit capture below
measured that corrected candidate before the subsequent author revision. The earlier `d22c5ac` measurement and dirty
repair receipts remain unchanged and supply no current-candidate performance
credit. No scoring run was performed during the uncommitted repair.

## Historical scored capture at `48c7b82`

Captured on 2026-09-17 from clean implementation commit
`48c7b82cd55318e8be455d792f2410d3a0f2ff95`, before these evidence-only updates.
The inherited Burner task declares the canonical `gpu` and `cpu-heavy` resources;
all device execution used GPU0 through `CUDA_VISIBLE_DEVICES=0`.

The repository-supported paired command ran once:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric both --output target/postcommit-48c7b82/default-compile-report.json
```

The [complete unmodified report](diagnostics/default-input-transpose-postcommit-48c7b82.json)
records `public-default-compile-v2`, untouched compiler defaults, five warmups,
17 samples, both CUDA implementation orders and fresh worker compiler caches.
All 84 reference program executions passed across CPU and the two CUDA rounds.
The fixed denominator, unsupported outcomes and slow samples are unchanged.

| Measurement/check | Clean-commit outcome |
| --- | --- |
| Weighted default-compile coverage | 19.5 / 100; 20 / 112 cells pass |
| Weighted default-compile CUDA performance | 33.28903094923071 / 100; 20 / 56 cells pass |
| Uncapped common-success latency ratio | 1.0971804306957396, limited to the 20 passing CUDA cells |
| Focused transpose checks against the freshly installed release wheel | 24 passed, including 11 real-GPU tests and native-only execution |
| Hardware-free transpose and branch admission | 25 passed, 11 explicit CUDA skips |
| Native planner/bridge and wrapping failure | 17 passed; three PyO3 test-only deprecation warnings retained |
| Source, build, import and raw-artifact provenance audit | Passed: 132 source hashes, all 12 worker output/log hashes and exported copies |

In that `48c7b82` capture, the canonical `transpose_view` cells received zero:
their external mutation check called the then-missing `Tensor.add_` API. Those focused checks exercised alias
mutation through the existing raw-bit helpers; they do not substitute for that
canonical outcome. These results establish neither a gain over main nor general
Inductor/performance parity, and do not replace independent review or merge gates.

The [compact clean-commit archive](diagnostics/default-input-transpose-postcommit-48c7b82.tar.gz)
contains exact commands, setup/build output, focused test logs, native-only
runtime identity, source/import/wheel hashes, the provenance audit and worker
logs. Full-value worker outputs remain in
`target/postcommit-48c7b82/export/`, exported through
`BURNER_EVALUATION_ARTIFACT_DIR`; the report and audit retain their paths and hashes.
The original run directory is
`target/default-compile-eval/run-20260917T211038Z-b51cde18/`.
Large value arrays are not checked into Git.

The evaluator environment uses the worktree-local Python 3.12.12 and locked
PyTorch `2.13.0+cu130`. Native NVRTC is 13.0, loaded CUDA runtime is 13000, and
installed `nvcc` is 12.6.85; Rust is 1.92.0. Setup took 0.987751929 seconds,
including 0.532835713 seconds for the cached native build and fresh wheel packaging.
The local environment and dependency/native-build caches were reused; the release
wheel was newly created and installed. GPU0 is H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07; focused-check
before/after snapshots show 0%/2% utilization and 4 MiB allocated in both.
No implementation, test, dependency, harness or evaluation definition changed
during this capture. No post-commit measurement or check failed.

The [earlier report](diagnostics/default-input-transpose-postcommit.json) and
[archive](diagnostics/default-input-transpose-postcommit.tar.gz) remain pinned to
`d22c5acbf83be77a789d32839ee57a6e5d4b41f8`, before the runtime-axis correction.
Their original setup, timings, paths and hashes are preserved; the new audit
verifies that those files and the development archives are unchanged.

## Development checks before the implementation commit

The following records describe the earlier dirty development checkout. Their
original archive and first failures are preserved unchanged. The `48c7b82`
scored capture above predates the historical `8ac9d0e` capability capture.

All GPU checks used `CUDA_VISIBLE_DEVICES=0`. Both frameworks were invoked as
`compile(fn)` without compiler overrides. Numerical tolerances remained
`rtol=1e-5, atol=1e-6`; view payload checks compare raw uint32 bits exactly.
Independent shape histories use public compiler resets to stay within the
unchanged reference recompile limit; no fallback warning occurred in the final run.

| Check | Outcome | Raw receipt |
| --- | --- | --- |
| Focused transpose plus helper/loop/branch/input/result/cache/guard regressions | 184 tests, 181 passed, 3 existing skips | `final-tests.log` |
| Final transpose tests, including nine real-GPU cases | 20 passed | `transpose-final.log` |
| Hardware-free transpose and existing branch admission | 32 tests, 23 passed, 9 explicit CUDA skips | `portable-wheel-final.log` |
| Native CUDA graph planner/bridge, metadata query and wrapping failure | 17 passed | `native-reviewed.log` |
| Release wheel, `cargo fmt --check`, Clippy with `-D warnings` | Passed | `build-reviewed.log`, `fmt-reviewed.log`, `clippy-reviewed.log` |
| Installed frontend equals current source bytes; native-only process blocks PyTorch imports | Passed | `identity.json`, `transpose-native-only.log` |

Coverage includes scalar/vector/matrix/empty/singleton inputs, changing shapes,
values and offsets, repeated/distinct/chained/equal-axis constructions, helper
and loop composition, nested result identity, external alias mutation, retained
storage after source wrappers disappear, current ABI rebinding and thread
ownership. Mixed results exercise sensitive FMA cancellation/overflow, trig,
scalar promotion and numerical output order. Portable tests verify late and
inactive rejection before preparation/launch, current inactive geometry without
helper rescans, original-input provenance, conditional method guards, and
wrapping/reconstruction rollback on old cache entries and a new alias ABI.
Native tensors currently expose float32 only; invalid-input cases also reject
foreign float64 tensors, CPU/grad inputs and noncontiguous CUDA inputs. This does
not claim additional native dtype or multi-device support.

## Development identity and retained evidence

GPU0 was NVIDIA H100, UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver
580.82.07. Before/after snapshots report 0% utilization and 4 MiB allocated.
The local interpreter was Python 3.12.12; reference PyTorch was `2.13.0+cu130`.
The native JIT reported NVRTC 13.0, including in the process blocking PyTorch
imports; loaded CUDA runtime reported 13000. The separately installed `nvcc`
was 12.6. Rust was 1.92.0; the release abi3 wheel used the repository's thin LTO
and single codegen unit. Full compiler options, loaded library paths, source,
wheel and extension hashes are in `identity.json`.

Raw output was directed through `BURNER_EVALUATION_ARTIFACT_DIR=target/evidence`
and is preserved in [the validation archive](diagnostics/default-input-transpose-validation.tar.gz).
`commands.txt` and `env.sh` preserve commands and local cache/build paths;
`identity-command.py` records import/build/device identity without changing
compiler defaults. The archive retains earlier failures rather than replacing
them with successful reruns:

- Initial interpreter download failed through the proxy. A read-only copy of an
  installed interpreter supplied the local environment.
- Early test fixtures used unsupported factory/mutation conveniences and were
  corrected to existing transfer and raw-bit helpers.
- Inactive-continuation numerical contamination, warm inactive geometry, and
  shared branch-node accounting were corrected. The earlier 409-test broad
  development run had 399 passes, 9 skips and the one branch-budget failure;
  its unchanged original test passes in final validation.
- Clippy's unnecessary-lifetime finding was corrected. An initial long reference
  shape history hit the default eight-recompile limit; final bounded histories
  cover the same shape/offset categories without changing that limit.

**Setup containment incident:** the first `maturin develop` inherited
`CONDA_PREFIX=/home/bobren/local/a/pytorch-env` and installed an editable package
there, violating the requested worktree-only boundary. `build.log` records that
selection. No external cleanup was attempted. Subsequent builds unset inherited
environment selection, installed wheels explicitly into this worktree's `.venv`,
and verified local source/import identity. The external installation is not used
as validation evidence and may require owner cleanup.

Development design and focused source reviews reported no remaining findings.
The clean-commit paired capture above completes the deferred measurement;
canonical review, evaluation adoption and delivery remain Burner's responsibility.
No evaluator, corpus, tolerance, dependency lockfile, managed progress artifact,
branch, commit or PR was changed or created by this evidence step.
