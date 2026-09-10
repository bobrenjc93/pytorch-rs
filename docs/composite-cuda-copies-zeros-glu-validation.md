# CUDA copies, matrix zeros, and functional GLU integration

The [final clean-commit transfer capture](diagnostics/composite-cuda-copies-zeros-glu/postcommit-dc236672/README.md)
measured `dc2366727f7839e8710af48fd63b08268d40f987`: all six fixed cases passed
at all three evaluator-selected seeds using a fresh local release build and
environment. The deferred capture is complete; independent review and delivery
gates remain required.

The integration findings and regression records below describe earlier validation
in `composite_433d72e4`, based on merge HEAD `cef8b64`. Documentation and test
repairs were uncommitted during those checks. They remain precommit regression
records, separate from the final clean-commit capture and any performance claim.

## Integration findings

The merged runtime already preserves all three source capabilities without
additional implementation changes:

- Native float32 CUDA zeros accept rank 1 and rank 2 on explicit indexed devices.
- Contiguous rank-1 CUDA tensors with `requires_grad=False` support same-device
  `clone()` and `to(copy=True)` with preserve format, including empty and offset
  views. Ordinary same-device `to()` preserves object identity. Cross-device,
  noncontiguous, scalar, and higher-rank copies and CUDA autograd remain excluded.
- Functional GLU composes existing CPU chunk, sigmoid, and multiplication for
  exact native float32 ranks 1 through 3. It retains its inference-only boundary,
  empty/strided inputs, override dispatch, and tracked-input use under `no_grad`.

Both quoted source-evaluation documentation regressions were reproducible by
inspection of the merged tree. `ARCHITECTURE.md`, both `FEATURES.md` coverage
tables and its CUDA narrative now describe both CUDA increments. The
supported-surface summary table also retained the old factory restriction and
was repaired. `METHOD_DOC` now matches the native `Tensor.to` documentation;
the quickstart test expects the bounded copy contract and explicitly checks
copy support and ordinary identity. Strict documentation equality, all
supported-surface checks, and every differential assertion remain in place.

The detailed supported-surface CUDA paragraphs, runtime implementation, zeros
error messages, and focused CUDA/GLU tests had already integrated successfully.
They did not need runtime expansion or duplicated implementations.

## Historical source evidence

The [vector-copy source capture](cuda-same-device-copy-validation.md) remains
pinned to `78310d9277dcb54bb328c24cbf445f7d4e500cd5`: 5/6 slots, with matrix
zeros unsupported. The [matrix-zero source capture](diagnostics/cuda-rank2-zeros/postcommit-6b0f68b0/README.md)
remains pinned to `6b0f68b0d2408bd8bda5073ec332fd88f02a67fb`: 5/6 slots, with
same-device copies unsupported. Their raw measurements, receipts, hashes,
seeds, failures, and original worktree identities are unchanged. Their summaries
now explicitly identify them as historical source-PR observations. Neither
capture measures the composite or supplies its performance/evaluation credit.

## Environment and commands

All environments, caches, wheels, build outputs, and test temporary files were
created inside this worktree. Python dependencies use the existing `uv.lock`;
there are no dependency or frozen-evaluator changes. The initial environment
selected the host Python 3.12 interpreter, then was recreated with a managed
worktree-local Python before building or testing.

The integration uses Python 3.12.14 and a separate Python 3.14.7 environment,
PyTorch 2.13.0+cu130, Rust/Cargo 1.92.0, Maturin 1.14.1, and GCC 11.5.0.
The host has NVIDIA H100 GPUs (97,871 MiB, capability 9.0), driver 580.82.07.
`TORCH_RS_CUDART` selects worktree-local `nvidia/cu13/lib/libcudart.so.13`
(runtime 13000). Installed nvcc is 12.6.85 and is unused by the copy/zero build.
Release builds use `extension-module`, thin LTO, one codegen unit, and stripping.
The Python 3.12 Cargo cache and target were initially empty; Python 3.14 uses
its own fresh target and the now-populated local dependency cache.

Builds use `maturin build --release --locked`, followed by installing the wheel
with `uv pip install --force-reinstall --no-deps`. The second build also uses
`--offline`. Tests import the installed wheel, verified by the repository's
`.github/scripts/verify_native_extension.py`; the transfer evaluator requires a
byte-identical wheel extension extracted to `python/torch_rs/torch_rs.abi3.so`.
Exact commands, environment variables, timestamps, status, and exit codes are
recorded in the integration receipts described below.

## Regression results

The repaired documentation and Tensor.to tests ran first: **22 passed**.
After strengthening the copy-boundary snippets, the same 22 tests passed again.
The focused H100 copy/zero/host-transfer, factory, Tensor.to, and GLU run had
**63 passed, 4 skipped**; those four two-device checks then passed explicitly
with `CUDA_VISIBLE_DEVICES=0,1`.

| Command (common local environment recorded in receipts) | Outcome |
| --- | --- |
| `cargo fmt --check` | Passed |
| `cargo clippy --locked --offline --all-targets -- -D warnings` | Passed |
| Same Clippy command with `--features python-bindings` | Passed |
| `CUDA_VISIBLE_DEVICES='' cargo test --locked --offline --all-targets -- --nocapture` | 364 reported successes; 19 hardware bodies skipped explicitly |
| Same CPU command with `--features python-bindings` | 375 reported successes; 19 hardware bodies skipped explicitly |
| `CUDA_VISIBLE_DEVICES=0 cargo test --locked --offline --all-targets -- --nocapture` | 364 reported successes; one two-device guard skipped and subsequently passed with two GPUs |
| `CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v` | 5,464 tests, 18 skips, no failures |

Python 3.12's 18 skips comprise 15 two-GPU checks, two legacy pre-3.12
recursion cases, and one macOS libc++ case. All 15 two-GPU Python checks passed separately (3 copy/zero checks, 1 host-transfer
check, and 11 eager/compiled CUDA regressions), as did the native two-device
addition guard. Platform-specific skips remain explicit.

The initial Python 3.14 full run had one environment-layout failure among
5,464 tests (179 skips). The existing stack-validator smoke test requires
imports physically beneath `REPOSITORY_ROOT/.venv`, while the first environment
was at `target/composite-integration/python314/venv`. A current-source snapshot
under `target/composite-integration/python314-checkout` supplies its own real
`.venv` using locked dependencies and the same source-matched Python 3.14 wheel.
The validator and all assertions are unchanged. Its previously failing test
passed in the corrected layout, then the full suite passed: **5,464 tests,
179 skips, no failures**. Those skips retain GPU-mask and platform/version
requirements in the portable lane. The snapshot
retains the current test/doc repairs and original committed files; it has no
new branch or commit and is not claimed as a clean Git checkout.

The unchanged six-case transfer evaluator also passed all 18 precommit trials
at its three selected seeds (`398888062646314880`, `8190242462381804573`,
`8899749379339317329`). Accounting was recomputed with the existing evaluator;
source, extension, and evaluator hashes matched, both worker roles loaded local
CUDA runtime 13000, and candidate workers imported no PyTorch. Raw diagnostic
output, build/run logs, receipts, and verification remain under
`target/composite-integration/`; they are **not** the final clean-commit evidence
and confer no current-candidate performance or evaluation credit.

## Preserved regression records

These records describe the precommit commands actually run during integration;
they do not replace final clean-commit evaluation evidence:

- [Exact command/environment/time/exit receipts](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/commands.json).
- [Python 3.12 full log](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/python-full.log) and [corrected Python 3.14 full log](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/python-full-314-canonical.log).
  The [initial Python 3.14 layout failure](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/python-full-314.log) is preserved too.
- [Native H100 log](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/native-h100.log),
  [focused differentials](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/focused.log),
  [copy/zero device guards](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/two-device.log),
  and [other two-device regressions](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/two-device-regressions.log).
- [Python 3.12 build receipt](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/build-record.json),
  [Python 3.14 build receipt](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/build-record-314.json),
  and [source/extension/snapshot audit](diagnostics/composite-cuda-copies-zeros-glu/integration-tests/source-audit.json).

Both release extensions have SHA-256
`90b435d50e8612270fc0a1ba43447c92639c79f6d6a9b2a968342426d155ab11`;
the production-source fingerprint is
`d27159403a544fc5a4148be3fd8d493f3fc39481e198c68656125de91ba4a454`.
The audit verifies the Python 3.14 snapshot's production sources, tests, and
scripts match this composite and the frozen evaluator hashes remain unchanged.
Raw regression logs and build receipts were copied without rewriting them.

## Final clean-commit transfer capture

The [postcommit evidence bundle](diagnostics/composite-cuda-copies-zeros-glu/postcommit-dc236672/README.md)
preserves the full six-case report, build/run receipts, raw logs, actual CUDA
provenance, source/extension/evaluator hashes, and clean status before and after
measurement at `dc2366727f7839e8710af48fd63b08268d40f987`.

All six cases passed at all three evaluator-selected seeds (18/18 trials).
The new release build also passed 19 focused CUDA/GLU checks and three two-GPU
device-restoration checks, with no skips. Earlier broad regressions were not
repeated or attributed to that capture, and historical leaf records are unchanged.

Independent review, complete current evaluation gates, no-regression checks,
and required remote CI remain Burner delivery gates. No branch, commit, push,
PR, frozen evaluator, or Burner-managed progress artifact was changed here.
