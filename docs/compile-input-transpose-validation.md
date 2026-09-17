# Default input-rooted transpose validation

Implemented from `60202557b4f110d07777f585e804ab5f55e1ff7b`. This is bounded
metadata-compilation support, not general Inductor or performance parity.
The default frontend retains one immutable view recipe with original binding
sources, construction identities and active alias order. Native metadata
preflight includes retained inactive/zero-trip constructions on warm calls;
only active constructions enter the existing alias bridge. Numerical SSA,
output ordering, preparation and launch remain with their existing owners.

## Clean-commit evidence

Captured on 2026-09-17 from clean implementation commit
`d22c5acbf83be77a789d32839ee57a6e5d4b41f8`, before these evidence-only updates.
The inherited Burner task declares the canonical `gpu` and `cpu-heavy` resources;
all device execution used GPU0 through `CUDA_VISIBLE_DEVICES=0`.

The repository-supported paired command ran once:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh --metric both --output target/postcommit-evidence/default-compile-report.json
```

The [complete unmodified report](diagnostics/default-input-transpose-postcommit.json)
records `public-default-compile-v2`, untouched compiler defaults, five warmups,
17 samples, both CUDA implementation orders and fresh worker compiler caches.
All 84 reference program executions passed across CPU and the two CUDA rounds.
The fixed denominator, unsupported outcomes and slow samples are unchanged.

| Measurement/check | Clean-commit outcome |
| --- | --- |
| Weighted default-compile coverage | 19.5 / 100; 20 / 112 cells pass |
| Weighted default-compile CUDA performance | 33.768731260903884 / 100; 20 / 56 cells pass |
| Uncapped common-success latency ratio | 1.1578276300529724, limited to the 20 passing CUDA cells |
| Focused transpose checks against the freshly installed release wheel | 20 passed, including nine real-GPU tests and native-only execution |
| Hardware-free transpose and branch admission | 23 passed, nine explicit CUDA skips |
| Native planner/bridge and wrapping failure | 17 passed; three PyO3 test-only deprecation warnings retained |
| Source, build, import and raw-artifact provenance audit | Passed: 132 source hashes, all 12 worker output/log hashes and exported copies |

The canonical `transpose_view` cells still receive zero: their external mutation
check calls the missing `Tensor.add_` API. The focused checks exercise alias
mutation through the existing raw-bit helpers; they do not substitute for that
canonical outcome. These results establish neither a gain over main nor general
Inductor/performance parity, and do not replace independent review or merge gates.

The [compact clean-commit archive](diagnostics/default-input-transpose-postcommit.tar.gz)
contains exact commands, setup/build output, focused test logs, native-only
runtime identity, source/import/wheel hashes, the provenance audit and worker
logs. Full-value worker outputs remain in
`target/postcommit-evidence/export/`, exported through
`BURNER_EVALUATION_ARTIFACT_DIR`; the report and audit retain their paths and hashes.
The original run directory is
`target/default-compile-eval/run-20260917T205502Z-de59baff/`.
Large value arrays are not checked into Git.

The fresh evaluator environment uses the worktree-local Python 3.12.12 and locked
PyTorch `2.13.0+cu130`. Native NVRTC is 13.0, loaded CUDA runtime is 13000, and
installed `nvcc` is 12.6.85; Rust is 1.92.0. Setup took 53.753295384 seconds,
including a 52.444668009-second release build. Dependency caches were warm;
the evaluator environment and release wheel were newly created. GPU0 is H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07; focused-check
before/after snapshots both show 0% utilization and 4 MiB allocated.
No implementation, test, dependency, harness or evaluation definition changed
during this capture. No post-commit measurement or check failed.

## Development checks before the implementation commit

The following records describe the earlier dirty development checkout. Their
original archive and first failures are preserved unchanged; the clean-commit
capture above supplies current-candidate evidence.

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
