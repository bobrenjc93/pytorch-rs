# Default input-rooted transpose validation

Implemented from `60202557b4f110d07777f585e804ab5f55e1ff7b`. This is bounded
metadata-compilation support, not general Inductor or performance parity.
The default frontend retains one immutable view recipe with original binding
sources, construction identities and active alias order. Native metadata
preflight includes retained inactive/zero-trip constructions on warm calls;
only active constructions enter the existing alias bridge. Numerical SSA,
output ordering, preparation and launch remain with their existing owners.

## Final checks

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

## Identity and retained evidence

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

Independent design and focused source reviews reported no remaining findings.
Canonical fixed-corpus evaluation, paired scoring measurements and delivery
remain Burner's responsibility. No evaluator, corpus, tolerance, dependency
lockfile, managed progress artifact, branch, commit or PR was changed or created.
