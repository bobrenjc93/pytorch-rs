# Native CUDA contiguous materialization: H100 correctness

The required clean-commit capture measured implementation commit
`cbb0df3489b6a9546aafa2926a40da201bf38a23` on 2026-09-11 UTC. Git status was
empty before and after the build and every test command. This report and the new
evidence were added afterward; implementation, tests, dependencies, harnesses and
supported behavior are unchanged in this evidence step.

The [release receipt](diagnostics/cuda-contiguous/postcommit-cbb0df3/build-record.json),
[tracked input manifest](diagnostics/cuda-contiguous/postcommit-cbb0df3/tracked-inputs.json),
and [capture inventory](diagnostics/cuda-contiguous/postcommit-cbb0df3/inventory.json)
bind the measured commit, source, tests, commands and native binary. The
[final audit](diagnostics/cuda-contiguous/postcommit-cbb0df3/evidence-audit-final.log)
verified those hashes, clean capture status, local import/build paths and unchanged
historical artifacts. Raw logs and receipts were copied byte-for-byte from the
current worktree's `target/cuda-contiguous-postcommit-cbb0df3`; recorded paths
identify their actual capture locations.

| Artifact | SHA-256 |
| --- | --- |
| Production source | `7c4d4950f5c99d24513f184ee596e5aed2e72b02181e0288be09f792b72f6f05` |
| Production diff against measured commit (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Native extension, installed wheel and source package | `cc4bf845fece98275489f9d9665c4dc3ae68a74d7687a8d0f7ad9a8851c6eb9e` |

## Environment and build

A fresh canonical worktree-local `.venv` was installed with
`uv sync --locked --no-install-project --group dev --group reference --python /usr/bin/python3.12`.
The previous environment was preserved under this worktree's `target` directory.
The existing local UV and Cargo dependency caches were reused; the Cargo release
target and CUDA JIT cache were fresh. All generated files remained in this worktree.
The unchanged repository build tool ran `maturin build --release --locked --offline`
without `--allow-dirty`, installed the wheel, and copied identical extension bytes
to the source package. Release profile: thin LTO, one codegen unit, `extension-module`.

[Preflight](diagnostics/cuda-contiguous/postcommit-cbb0df3/preflight.log) records
Python 3.12.13, NumPy 2.5.1, PyTorch `2.13.0+cu130`, and Rust/Cargo 1.92.0.
Native CUDA runtime was explicitly selected from the new `.venv`'s
`nvidia/cu13/lib/libcudart.so.13`, reporting version 13000; PyTorch reports CUDA 13.0.
Installed nvcc is 12.6.85 and was **not used**: the driver JIT compiles embedded PTX.

Ordinary GPU checks used `CUDA_VISIBLE_DEVICES=0`: NVIDIA H100, compute capability
9.0, 97,871 MiB reported by `nvidia-smi`, driver 580.82.07, UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`. Restoration checks alone used `0,1`,
adding UUID `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Every command receipt
contains before/after GPU inventory, utilization and memory snapshots.

## Clean-commit results

| Check | Result | Raw log |
| --- | --- | --- |
| Rust CUDA primitive/unit checks | 12 passed | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/rust-cuda-unit.log) |
| Selected Rust CUDA integration suites | 19 passed | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/rust-cuda-integration.log) |
| New H100 contiguous/reshape differentials | 6 passed; two-device case skipped under mask `0` | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/python-contiguous.log) |
| Two-device contiguous and existing copy restoration | 2 passed | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/python-device-context.log) |
| Existing CPU layout/view/reshape/clone differentials | 85 passed | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/python-cpu-layout.log) |
| Existing CUDA storage, transfer, arithmetic and compile regressions | 137 passed; 19 two-device cases skipped under mask `0` | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/python-cuda-regressions.log) |
| Hardware-test portability with CUDA hidden | All 7 skipped clearly; no correctness credit | [log](diagnostics/cuda-contiguous/postcommit-cbb0df3/python-no-device.log) |

These unchanged tests cover non-square transposes, offset slices, strided columns,
seeded shapes and a grid-stride tail beyond 1M elements; values, dtype/device/index,
strides/offsets, source nonmutation, alias/copy mutations and reshape with and
without copying. Raw device byte comparisons preserve signed zero, subnormals,
infinities and quiet/signaling NaN payloads. Scalar/singleton/empty identity,
checked extent/offset overflow, format/rank/dtype/gradient boundaries, lifetimes,
blocked PyTorch imports and device/driver-context restoration are included.
The existing regression shapes, slow cases and unsupported outcomes were retained.
Unrelated full Rust suites and Clippy were not repeated or attributed to this capture.

## Reproduction and preserved attempts

From the clean measured commit, use the local-cache exports in
[environment.sh.txt](diagnostics/cuda-contiguous/postcommit-cbb0df3/environment.sh.txt)
and a fresh canonical `.venv`, then run:

```sh
.venv/bin/python scripts/capture_depth_concat_build.py --output target/cuda-contiguous-postcommit-cbb0df3/release
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v tests.test_cuda_contiguous
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v tests.test_cuda_contiguous.ContiguousDeviceGuardTests tests.test_cuda_same_device_copy.CopyDeviceGuardTests
```

Choose an unused output directory for each capture. Exact setup, Rust, CPU and
existing CUDA regression commands, timestamps, exit statuses and hashes are in
the per-command `*.receipt.json` files. The capture wrapper only records and
checks provenance; it invokes the committed repository build tool and tests.

The first post-commit integrity audit incorrectly resolved the canonical venv
interpreter symlink as though it were a package path. The corrected audit uses
lexical interpreter identity, following `BENCHMARKING.md`, while package/runtime
paths must resolve inside `.venv`. The failed [audit log](diagnostics/cuda-contiguous/postcommit-cbb0df3/evidence-audit.log),
receipt and original `audit-first.py.txt` remain preserved. No build, test,
implementation or measured receipt was changed to obtain the passing audit.

The original [precommit inventory](diagnostics/cuda-contiguous/precommit-b8622bc/inventory.json)
and all development artifacts remain unchanged, including both unsuccessful
fixture attempts and their source snapshots. Those dirty-source runs are
separate development evidence; the clean capture above fulfills the outstanding
post-commit requirement.

This is bounded native layout correctness evidence, with no performance,
compiler-parity or evaluator-score claim. Campaign evaluators, fixed cases/weights,
benchmark evidence, `.burner` definitions and managed progress are unchanged.
PR1970/1971 remain separate unadopted evaluator campaigns. This capture does not
replace independent review or merge gates.
