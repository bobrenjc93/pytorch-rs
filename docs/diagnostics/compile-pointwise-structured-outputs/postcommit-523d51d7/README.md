# Clean structured-output capture at 523d51d7

Measured implementation: `523d51d73bb90237f8675ee584bfb890d39c84df`, with `main` at `30a3b504ef4d43bf2958998cc39545996cc09970`. This supplies the clean-commit capture deferred in the [realization repair record](../review-realization/README.md). Every measured command began and ended with a clean checkout; all 6,787 tracked files retained their original bytes throughout measurement.

## Results

The unchanged committed compiler tests ran against a fresh source-bound release wheel. They compare ordinary `torch_rs.compile(fn)` with ordinary stock `torch.compile(fn)`, preserving existing tolerances, zero-sign/nonfinite checks, persistent wrappers, aliases and no-original/helper-replay assertions.

| Check | Measured result |
| --- | --- |
| Pointwise suite on H100, device 0 | 304 total: 296 passed, 8 two-device skips |
| Required two-device checks, devices 0 and 1 | 8 passed, no skips |
| Release native pointwise/ownership checks | 58 passed |
| Python conversion-failure ownership | 1 passed |
| Portable checks, each CPython 3.10–3.14 | 145 passed, 159 hardware skips |
| Clippy, formatting and native extension verification | Passed |
| Installed wheel identities on all five interpreters | Passed |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware skips are not GPU passes. The native maximum test exercised 4,096 nodes and 64 outputs in both orders, recording 4,225 instructions, 104 registers and 106,496 scratch bytes for 256 workers.

For `p=x*y; q=p-x; r=p.sin().sin().sin().sin().sin().sin().sin(); return(q,r)`, shape `(1,)`, `x=1e10` and `y=1.0000001192092896`, both implementations return `q=1192.0928955078125`. Sizes 13 and 257 also pass.

The permanent 2,117-node order regression and its duplicate-product variant retain the same native executor across topology changes. At shape `(2,)` with those finite inputs:

| Return order | Native q | Default PyTorch q |
| --- | ---: | ---: |
| `(q,r)` | 1024.0 | 1024.0 |
| `(r,q)` | 1192.0928955078125 | 1192.0928955078125 |

Overflow, subnormal, signed/shared/competing-product, constant, nested-output and retained-prior-output cases pass the existing strict comparisons. These bounded tests do not establish general Inductor parity or approve the branch.

## Provenance and retention

Wheel SHA256: `6750726ba3364930be98fe9ca414a18986b7f38b64f68540fdd899aacf7f32b7`.

The locked offline release build used a fresh Cargo target. Reference, Triton and CUDA caches were initially empty and remained inside this worktree; the later runtime probe reused these report-local caches. Build/runtime records identify Rust 1.92.0, Maturin 1.15.0, PyTorch `2.13.0+cu130`, H100/driver 580.82.07, NVRTC 13.0 and CUDA runtime 13000. PATH `nvcc` reports 12.6.85; native kernels were compiled by NVRTC. The runtime probe executed native compilation before importing Torch, verified structured aliases and compared materialized outputs with the ordinary default reference.

[Measurements](measurements.json.gz) retain timestamped commands, source hashes, clean status, environments and return codes. [Raw captures](raw-captures.tar.gz) retain numerical JSON, generated native CUDA/PTX and instruction plans, logs, wheel/import/runtime/GPU identities and capture orchestration. Identical files use tar hard links while retaining every original path. The [raw manifest](raw-manifest.json.gz) and [verification](verification.json) verify all 8,073 packaged files and archive hashes.

The canonical root `target/default-compile-eval/structured-outputs-postcommit-523d51d7/` additionally retains the wheel, byte-verified committed-source archive, original reference compiler caches and `retention-manifest.json.gz` covering 41,770 files. It is covered by Burner's canonical report-root observer. No cleanup or external write was performed; no external archival path is claimed. Every earlier report, development failure and historical measurement remains unchanged.

Only this evidence and its compiler-guide link change. Implementation, tests, dependencies, benchmark harnesses, evaluation definitions and managed progress artifacts remain unchanged. No official scoring corpus or unrelated repository suite was rerun. Native scalar-program dispatch/upload/scratch costs were not benchmarked; this capture awards no performance or coverage score.
