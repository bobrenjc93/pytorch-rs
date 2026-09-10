# Clean composite capture: CUDA matmul, rsqrt, and linear modes

Measured implementation commit: **`cc42fdb265af856d8110404e91ff6cf8dd70a5e7`**, based on
`96205cb01e85f6961a4d8ea40aa792b485289d84`. This is a fresh clean-code
capture of the repaired composite, separate from the historical PR #1952 and
uncommitted author runs. All build, test, correctness, and timing commands
finished with empty git status before these files were published. No measured
implementation, dependency, test, harness, evaluator, or corpus changed.

## Results and receipts

| Check | Result | Receipt / raw output |
| --- | --- | --- |
| Fresh release extension | Passed; initially absent Cargo target | [Build receipt](build-record.json), [log](build.log) |
| Unchanged CUDA math corpus | 6/6 cases; 18/18 trials across all three required seeds | [Report](evaluation.json), [run receipt](run-record.json) |
| Public matmul timings | All 22 shape/offset cases validated; 91.6218% capped geometric parity | [Report](public-timings.json), [receipt](public-timings.receipt.json), [log](public-timings.log) |
| Matmul, rsqrt, and linear Python differentials | 43 passed; one expected two-device mask skip | [Receipt](python-focused.receipt.json), [log](python-focused.log) |
| Separate two-GPU matmul | 1 passed on GPUs 0,1 | [Receipt](python-two-device.receipt.json), [log](python-two-device.log) |
| Native Rust matmul | 2 passed | [Receipt](rust-matmul.receipt.json), [log](rust-matmul.log) |
| Rust checked bounds | 1 passed | [Receipt](rust-bounds.receipt.json), [log](rust-bounds.log) |
| Rust rsqrt saved storage | 1 passed | [Receipt](rust-rsqrt-storage.receipt.json), [log](rust-rsqrt-storage.log) |
| Rust rsqrt backward | 3 passed, plus 1 grad-mode check | [Backward receipt](rust-rsqrt-autograd.receipt.json), [grad-mode receipt](rust-rsqrt-grad-mode.receipt.json) |
| Existing evaluator integrity tests | 19 passed | [Receipt](evaluator-tests.receipt.json), [log](evaluator-tests.log) |
| Independent numerical reproductions | 40 passed | [Actual values](numerical-probes.log), [command receipt](numerical-probes.receipt.json) |
| Evidence verification | Passed; 855 code/test/harness files and 20 historical files checked | [Checks](evidence-checks.json), [log](verification.log) |

The fixed CUDA seeds remain `8503945240872567646`, `8613321571747136749`, and
`4480763905421893394`. All six cases remain in the denominator. Each reference
and candidate runs in a separate process; candidate workers block PyTorch
imports. The reports retain device-pointer attributes, materialized outputs,
input-preservation checks, and actual selected runtime paths. The additional
matmul tests cover wide generated rectangles, offsets, aliases, empty/zero-K
products, storage lifetimes, nonfinite classifications and subnormal bits,
plus gradient and compilation rejection. Linear tests include omitted versus
explicit empty kwargs, saved-original callable identity, the wrapper returning
13 rather than 23, nested modes, exception recovery, metadata, and CUDA rejection.
Rsqrt checks retain float32 VJP order and the CUDA/higher-order boundaries.

No failed measurement attempts occurred in this post-commit capture. Earlier
source and development records remain unchanged and do not supply combined
proof. The full Rust/Python suites already run during authoring were not
repeated; this step runs the checks needed to bind and validate fresh evidence.
Independent review, all ten current-definition gates (including performance 97),
and exact-head CI remain required. The timing aggregate here is a diagnostic,
not a Burner gate score or approval to merge.

## Measured provenance

- Source fingerprint: `03c2563f6ef1532b2279e778012f434f67fd002ed0c30da03183dfcc0699331e`.
- Native extension SHA-256: `1faf970c2684071aa7475a227ff972085a260f3081cd931fa361dbb09c95cada`.
- CPython 3.12.12, PyTorch 2.13.0+cu130, NumPy 2.5.1; worktree-local interpreter
  and packages. Native imports use this checkout's `PYTHONPATH=python` extension.
- NVIDIA H100, compute capability 9.0, driver 580.82.07. Actual loaded CUDA
  runtime: 13000 (13.0); cuBLAS: 13.1.1. Library paths and SHA-256 identities
  are in the timing report. Rust/Cargo 1.92.0 builds release `extension-module`,
  abi3-py310, thin LTO, one codegen unit. Available nvcc 12.6.85 is unused;
  matmul calls native cuBLAS and the other CUDA kernels use driver-JIT PTX.
- Cargo build target and capture cache/temporary directories were initially
  absent. The worktree-local Cargo registry and Python environment were reused.
  Evaluator workers additionally use their own fresh temporary CUDA JIT cache.
- [Source inventory](source-record.json), [command receipts](commands.json),
  [completion](completion.json), and [verification](evidence-checks.json) bind
  code, tests, harness, source ancestry, clean status, paths, and log/report hashes.
  The preflight receipt observes the pre-existing extension before the fresh
  build; qualification uses the new build receipt and subsequent checks.

## All public-API timing cases

Both implementations use `a @ b`, equal precreated float32 inputs, TF32 disabled,
ten warmups per execution order, two reversed orders, and 31 five-call blocks
per order. Every call synchronizes before and after timing. Outputs and original
inputs are checked before/after timing. Raw samples are retained; no slow cells
or samples were dropped. Each of the 11 fixed/generated shapes has offsets 0
and 3. The fixed equal-weight aggregate is
`exp(mean(log(min(1, PyTorch_median / native_median))))` over all 22 cells.
Values below are microseconds per call; dispersion is the recorded P10–P90.

| M × K × N | Offset | Native median | Native P10–P90 | PyTorch median | PyTorch P10–P90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 × 65539 × 1 | 0 | 19.944 | 19.620–21.639 | 16.659 | 16.351–17.741 |
| 1 × 65539 × 1 | 3 | 19.954 | 19.666–21.653 | 16.559 | 16.271–17.064 |
| 64 × 64 × 64 | 0 | 20.284 | 20.072–22.326 | 17.609 | 17.408–19.475 |
| 64 × 64 × 64 | 3 | 20.150 | 20.048–21.180 | 17.620 | 17.462–18.933 |
| 128 × 128 × 128 | 0 | 21.379 | 21.230–23.191 | 18.825 | 18.700–20.721 |
| 128 × 128 × 128 | 3 | 21.357 | 21.192–21.925 | 18.735 | 18.556–18.999 |
| 512 × 512 × 512 | 0 | 30.408 | 30.184–31.700 | 27.809 | 27.644–29.340 |
| 512 × 512 × 512 | 3 | 30.828 | 30.548–32.249 | 28.219 | 28.034–29.140 |
| 1024 × 1024 × 1024 | 0 | 71.827 | 71.367–74.641 | 70.334 | 69.886–71.318 |
| 1024 × 1024 × 1024 | 3 | 72.630 | 71.911–73.882 | 71.871 | 70.641–73.469 |
| 2048 × 2048 × 2048 | 0 | 368.081 | 365.753–371.792 | 354.540 | 351.996–361.122 |
| 2048 × 2048 × 2048 | 3 | 365.380 | 364.258–367.964 | 352.844 | 351.595–354.694 |
| 127 × 1025 × 65 | 0 | 23.845 | 23.656–24.941 | 20.705 | 20.517–21.839 |
| 127 × 1025 × 65 | 3 | 24.045 | 23.790–25.268 | 21.134 | 20.779–22.095 |
| 800 × 800 × 800 | 0 | 55.283 | 54.645–56.669 | 53.160 | 52.393–54.222 |
| 800 × 800 × 800 | 3 | 55.739 | 55.241–57.258 | 53.108 | 52.605–54.220 |
| 1379 × 1379 × 1379 | 0 | 158.302 | 157.063–162.029 | 157.508 | 155.956–161.582 |
| 1379 × 1379 × 1379 | 3 | 156.195 | 155.232–157.264 | 153.976 | 153.215–155.186 |
| 222 × 221 × 770 | 0 | 24.761 | 24.357–25.741 | 22.083 | 21.827–23.479 |
| 222 × 221 × 770 | 3 | 24.559 | 24.371–25.160 | 22.064 | 21.837–22.546 |
| 956 × 1117 × 206 | 0 | 38.493 | 38.224–39.169 | 35.806 | 35.600–36.742 |
| 956 × 1117 × 206 | 3 | 38.886 | 38.414–40.507 | 36.183 | 35.746–37.795 |

## Root numerical counterexamples

All results below are newly measured native/PyTorch values, with the original
`rtol=1e-5, atol=1e-4`. The raw numerical report also retains positive/negative
decimals from K=4096 through 1,000,000. The committed tests cover generated
multi-row/multi-column products and adversarial nonfinite/subnormal cases.

| Input | K | Native | PyTorch |
| --- | ---: | ---: | ---: |
| full(0.1) @ full(0.1) | 65536 | 655.3600463867188 | 655.3600463867188 |
| full(0.1) @ full(1.0) | 65539 | 6553.900390625 | 6553.900390625 |
| full(0.1) @ full(1.0) | 262144 | 26214.40234375 | 26214.40234375 |
| full(0.1) @ full(1.0) | 1000000 | 100000.0078125 | 100000.0078125 |
| [2^127, 2^127, -2^127, -2^127, zeros] @ ones (two repetitions) | 1031 | 0.0 | 0.0 |
| [2^127, 2^127, -2^127, -2^127, zeros] @ ones (two repetitions) | 2048 | 0.0 | 0.0 |

## Reproduction and publication

Use the [focused guide](../../../cuda-matmul.md#reproduction) at the measured
implementation commit, a clean checkout rooted inside the current worktree,
local packages/runtime/cache paths, and a new absent Cargo target. The exact
orchestration is preserved as [capture.py.txt](capture.py.txt); it invokes the
unchanged committed build/evaluator helper, benchmark harness, and tests.
[environment.sh.txt](environment.sh.txt) overrides the copied
[base environment](environment-base.sh.txt); actual final variables appear in
the build/source/timing receipts. Choose new output/build/cache directories for
a rerun. The independent numerical command is preserved verbatim in its receipt.
[verify.py.txt](verify.py.txt) records the read-only verification performed
before publication. No production or benchmark harness was modified here.

These files were first generated under `target/postcommit-cc42fdb2/`, then copied
byte-for-byte after clean measurements and verification finished. The
[publication manifest](publication-manifest.json) hashes each published receipt,
report, log, and reproduction record. Accompanying documentation was added only
after capture. A later artifact-only commit may follow the measured code commit;
implementation or benchmark changes require new measurements.
