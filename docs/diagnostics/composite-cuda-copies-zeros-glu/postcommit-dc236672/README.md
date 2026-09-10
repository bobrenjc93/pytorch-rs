# Clean-commit composite CUDA transfer evidence

Measured implementation: **`dc2366727f7839e8710af48fd63b08268d40f987`**.
The unchanged six-case CUDA transfer evaluator passed **6/6 cases at all three
selected seeds (18/18 trials)** on H100. This records correctness for that fixed
capability, with no throughput, overall hardware score, or performance-parity claim.

The complete worktree was clean before setup, build, evaluator execution, and
focused checks, and remained clean after every command. Only this evidence and
its accompanying integration-report update were added after measurement.
The capture used a new source-matched release build and a new worktree-local
Python environment; it did not reuse the precommit extension as build input.

| Fixed case | Outcome |
| --- | --- |
| `cuda_f32_vector_zero_roundtrip` | 3/3 passed |
| `cuda_f32_matrix_zero_roundtrip` | 3/3 passed |
| `cuda_f32_contiguous_cpu_upload` | 3/3 passed |
| `cuda_f32_strided_cpu_upload` | 3/3 passed |
| `cuda_f32_view_to_cpu_copy` | 3/3 passed |
| `cuda_f32_same_device_copy` | 3/3 passed |

Seeds selected by the evaluator, without `--seed` arguments: `4004991550803647476`, `4769562916386723985`, `4106961968642668418`.
Capture interval: `2026-09-10T10:06:05.788213+00:00` to `2026-09-10T10:07:12.361673+00:00`.
The denominator, workload matrix, reference, and evaluator are unchanged.
No trials, unsupported outcomes, or slow results were removed or rewritten.

## Raw evidence and verification

- [Full evaluator report](transfers.json) retains all reference/candidate worker
  observations, values, pointer/aliasing checks, runtime provenance, and accounting.
  [Evaluator stdout/stderr](transfers.log) is empty because output goes to JSON.
- [Build receipt](build-record.json), [build log](build.log), and
  [exact command receipts](commands.json) retain commands, selected environment,
  timestamps, exit codes, clean status, source stability, and log hashes.
- [Preflight](preflight.json) and [native provenance check](provenance.log) identify
  the actual interpreter, installed extension, PyTorch, GPU, and loaded CUDA runtime.
- [Verification](verification.json) recomputes accounting with the committed
  evaluator and independently revalidates all 18 trials. Source, wheel,
  extension, and evaluator hashes match; worker imports/executables are local to
  this composite and every loaded CUDA runtime is in the fresh environment.
  Candidate workers have no blocked imports or loaded PyTorch modules.
- [Pre-capture diff audit](candidate-audit.json) inventories all 82 changed files
  against `main`, verifies 35 precommit regression log hashes, and verifies both
  historical leaf source hashes against their original Git trees.
- [Capture helpers](capture-tools.json) preserve the exact receipt, extraction,
  environment, preflight, and verification helpers used under `target/`.
  They do not replace or modify the repository evaluator.
- [Publication audit](publication.json) checks byte-identical evidence copies and
  confirms that the resulting changes are evidence/documentation only.

## Build and runtime

- Python `3.12.14`, PyTorch `2.13.0+cu130`, CUDA
  `13.0`; all 31 dependencies installed from the existing lockfile
  into `target/composite-postcommit-dc236672/venv`.
- NVIDIA H100, capability 9.0, 97,871 MiB, driver 580.82.07.
  GPU 0 UUID: `8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`.
  Evaluation and single-device checks use `CUDA_VISIBLE_DEVICES=0`.
- Both worker roles actually load fresh-environment `libcudart.so.13`, reported
  runtime version `13000`. `TORCH_RS_CUDART` explicitly selects that library;
  the raw reports include its absolute path and the verification includes its hash.
- Rust/Cargo 1.92.0, Maturin 1.14.1, GCC 11.5.0. Available nvcc is CUDA 12.6.85;
  it is unused by this build and capture. Existing PTX is embedded; the CUDA
  allocation/copy path dynamically loads the runtime.
- Fresh `target/composite-postcommit-dc236672/build` Cargo target, existing
  worktree-local Cargo/uv dependency caches, locked offline build. Release
  profile: `extension-module`, ABI3 Python 3.10+, thin LTO, one codegen unit,
  and stripping. All generated files/caches stayed in this worktree.

| Identity | SHA-256 |
| --- | --- |
| Production source | `d27159403a544fc5a4148be3fd8d493f3fc39481e198c68656125de91ba4a454` |
| Native extension (wheel, installed package, evaluator source layout) | `90b435d50e8612270fc0a1ba43447c92639c79f6d6a9b2a968342426d155ab11` |
| Release wheel | `985b028700f24a01caf67ed9dd4bb4f1e909f3bccb7d7f959f7d0a2d96280494` |
| Raw transfer report | `78be3406fb0b515c9d15dd1018456c904ffe11dd9c65ae5817643696ad2019fb` |
| Build receipt | `807bf7dca42016570ccc88a947ff7d01e82eb80ba22a24b4c5cdc7a9f085238e` |

The fresh extension is byte-identical to the earlier source-matched build; the
new compilation and its timestamps are preserved in the build log/receipt.

## Checks on this build

[Focused tests](focused.log): **19 passed, no skips**, covering generated vector
copies, empty/offset/singleton layouts, exact values, independent allocations,
source preservation, reuse, unsupported copy boundaries, generated matrix zeros,
zero-factory restrictions, CPU GLU inference/layout/numerical/binding/override
behavior, and GLU's real-CUDA rejection boundary.

[Two-device tests](two-device.log): **3 passed, no skips** with
`CUDA_VISIBLE_DEVICES=0,1`, covering copy-device restoration in both directions,
cross-device rejection, and rank-1/rank-2 allocation/CPU-copy restoration.

Earlier full Python/native regression runs are retained in the
[integration report](../../../composite-cuda-copies-zeros-glu-validation.md).
They were not rerun or attributed to this postcommit capture. Historical leaf
captures remain pinned to their original commits and original 5/6 outcomes.

## Reproduction

From this code commit and a clean worktree, follow the
[repository evaluator guide](../../../hardware-heterogeneity-evaluator.md)
with a fresh local environment and target. Exact absolute commands/environment
are in `commands.json`; the build and evaluator commands were:

```sh
target/composite-postcommit-dc236672/venv/bin/maturin build \
  --release --locked --offline \
  --interpreter "$PWD/target/composite-postcommit-dc236672/venv/bin/python" \
  --out target/composite-postcommit-dc236672/wheels
CUDA_VISIBLE_DEVICES=0 target/composite-postcommit-dc236672/venv/bin/python \
  scripts/evaluate_cuda_transfers.py \
  --build-record target/composite-postcommit-dc236672/build-record.json \
  --output target/composite-postcommit-dc236672/transfers.json
```

The Maturin wheel was installed into the new environment, and its native member
was extracted byte-for-byte to `python/torch_rs/torch_rs.abi3.so`, as required
by the evaluator. The build receipt uses the existing `source_provenance` and
`sha256` functions from `scripts/evaluate_cuda_math.py`. A future code build
requires its own fresh receipt and capture. An ensuing artifact-only commit may
retain this measured code identity only while intervening changes contain
reports/evidence and no implementation or benchmark-harness changes.

This capture completes the deferred measurement requirement. It does not approve
the branch or replace independent review, current evaluation gates,
no-regression checks, required CI, or Burner delivery.
