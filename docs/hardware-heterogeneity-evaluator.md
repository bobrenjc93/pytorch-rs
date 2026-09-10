# Hardware Heterogeneity Evaluator

This Burner evaluation measures how much of PyTorch's accelerator breadth and
feature depth `torch_rs` actually supports. CPU is intentionally excluded: it
is the host baseline, not an accelerator family.

The versioned source of truth is
[`hardware-heterogeneity-matrix-v1.json`](hardware-heterogeneity-matrix-v1.json).
Version 1 gives equal weight to seven stable PyTorch accelerator families:
NVIDIA CUDA, AMD ROCm, Apple MPS, Intel XPU, Google TPU/XLA, Intel Gaudi HPU,
and Meta MTIA. `PrivateUse1`, `meta`, and Vulkan are not counted as accelerator
families: the first is an extension mechanism, the second has no hardware
execution, and the last is not a stable general training backend in the
reference release.

## Score

Each backend is evaluated against stable PyTorch 2.13 programs in eight
capability groups:

| Capability | Weight within each backend |
| --- | ---: |
| Device tensors, placement, and host/device transfers | 15% |
| Dtypes, layouts, views, and indexing | 15% |
| Elementwise math, reductions, and linear algebra | 15% |
| Autograd and training | 15% |
| Neural networks and optimizers | 15% |
| RNG, serialization, and checkpointing | 10% |
| Compilation | 10% |
| Multi-device and distributed execution | 5% |

For backend `b` and capability `c`, let `p(b,c)` be the fraction of fixed
reference-eligible cases that `torch_rs` passes. A case is reference-eligible
only when the declared stable PyTorch stack compiles or runs it correctly on
that accelerator. Each backend score is the weighted sum of its eight
capability fractions. The headline score is the arithmetic mean of all seven
backend scores, multiplied by 100. A backend with no qualifying evidence has a
score of zero; it is never removed from the denominator because the evaluator
machine does not contain that hardware.

The evaluator must also report an unweighted *backend breadth* percentage. A
family counts toward breadth only when real-hardware evidence passes at least
one representative case in all three foundational groups listed by
`breadth_required_capabilities`: device tensors/transfers, core math, and
autograd/training. This supplementary number prevents a single specialized
kernel from being described as general support. The feature-depth matrix,
not breadth alone, is the Burner score.

## Evidence rules

A case earns credit only when it:

- executes on real hardware from the named accelerator family;
- matches PyTorch outputs and required observable semantics;
- uses native `torch_rs` storage and execution rather than installed-PyTorch
  forwarding, CPU fallback, emulation, or a metadata-only shim; and
- is covered by an executable differential test whose workload and result are
  bound to the evaluated commit.

When hardware is unavailable on the evaluator host, a reproducible
hardware-CI or release artifact may provide credit if it records the exact
commit, device, driver/runtime, dependency versions, test matrix, raw results,
and pass/fail accounting. Cross-compilation and mocked availability do not earn
runtime credit. Missing, skipped, unsupported, stale, or unverifiable cases
remain zero; they are not silently excluded.

API probes such as `is_available()`, device parsing, and backend preference
flags are useful compatibility work but do not prove accelerator execution. A
private benchmark-only path earns credit only for the exact cases it executes,
not for the rest of that backend or capability group.

## Local CUDA requirement

The Burner host has eight NVIDIA H100 GPUs. For changes that could affect CUDA
coverage, run the relevant differential cases on one real device with
`CUDA_VISIBLE_DEVICES=0`; reserve more devices only for an explicit
multi-device case. Record the H100 model, compute capability, driver, CUDA
runtime and compiler, PyTorch version, selected device, and memory pressure.
Portable tests must still skip clearly on hosts without matching hardware,
without weakening the hardware-backed acceptance criteria.

Changing the backend set, capability weights, eligibility rules, or zero-credit
rules requires a new matrix schema version and a matching Burner evaluation
definition version. Cases may be added within this contract, but existing
failing cases must not be deleted merely to raise the score.

## Fixed CUDA float32 device/transfer denominator

The `cuda_float32_transfers_v1` case set fixes six equally weighted cases within
CUDA's existing `device_tensors_and_transfers` capability. The seven backend
weights and all eight capability weights remain unchanged.

| Case ID | Required public behavior |
| --- | --- |
| `cuda_f32_vector_zero_roundtrip` | `zeros((17,), device="cuda:0")` and `zeros((0,), device="cuda:0")`, each downloaded with both `.cpu()` and `.to("cpu")` |
| `cuda_f32_matrix_zero_roundtrip` | Direct `zeros((7, 13), device="cuda:0")`, downloaded with both `.cpu()` and `.to("cpu")` |
| `cuda_f32_contiguous_cpu_upload` | Contiguous CPU `(7, 13)` tensor uploaded with `.to("cuda:0")` |
| `cuda_f32_strided_cpu_upload` | CPU `(7, 13)` base viewed as `[:, 1:12].transpose(0, 1)`, uploaded with `.to("cuda:0")` |
| `cuda_f32_view_to_cpu_copy` | The same offset, non-dense view of a CUDA base, copied with both `.cpu()` and `.to("cpu")` |
| `cuda_f32_same_device_copy` | CUDA `(17,)` tensor copied with both `.clone()` and `.to("cuda:0", copy=True)`; ordinary `.to("cuda:0")` must preserve object identity |

Every tensor is float32, with no gradient tracking. The view shape is `(11, 7)`,
its strides are `(1, 13)`, and its storage offset is 1. Copies have strides
`(1, 11)` and storage offset 0. Other tensors and copies are contiguous.
Nonzero values are drawn independently of either framework using Python
`random.Random(seed).uniform(-2, 2)`, rounded to float32. At least two distinct
nonnegative evaluator-selected seeds are required; omitting `--seed` generates
and records three seeds. Shapes and required operations are fixed by this
version. All subchecks, including empty input, must pass for a case to earn
credit; subchecks and seeds do not add denominator slots. Comparisons are exact.

[`scripts/evaluate_cuda_transfers.py`](../scripts/evaluate_cuda_transfers.py)
runs every reference and candidate trial in separate isolated processes with
`CUDA_VISIBLE_DEVICES=0`. The reference must be NVIDIA PyTorch 2.13.0 on real
hardware. Candidate processes load this checkout's native extension, block
installed `torch` imports, and record even swallowed forwarding attempts.
Evaluator-owned driver queries check CUDA pointer memory type and device,
synchronize, and read logical elements using their strides. CPU storage reads
must also agree with public `.tolist()`. Empty allocations have no pointer to
query; the vector slot additionally requires the nonempty hardware case.

The runner verifies source values and metadata before and after copying,
distinct objects and allocations, and copy contents after an independent source
mutation. View-download and same-device-copy fixtures start with a public
rank-one CUDA zero allocation, filled by an evaluator-owned driver upload and
reshaped. This identical setup on both sides avoids requiring public upload or
matrix allocation for those slots. Only the declared public operations earn
credit; driver fixture initialization and observation earn none. Pointer checks
and the import blocker are not a sandbox for hostile native code: published
credit still requires review of the commit-bound native transfer path.

For this capability only, `p(cuda_nvidia, device_tensors_and_transfers) = passed / 6`.
Its contribution within the CUDA backend is `15 * passed / 6` percentage points.
Each case therefore contributes at most 2.5 points within that backend. Missing,
skipped, unsupported, forwarded, incorrect, malformed, and unbound results earn
zero and retain their slots. Reference failure also retains a zero slot and
makes the command exit 2 (incomplete reference run). A complete reference run
exits 0 even if all candidate cases are unsupported. The JSON records reference
eligibility and candidate verdicts separately. It emits no overall heterogeneity,
backend breadth, performance, or other capability score. These specific view
checks do not establish additional `dtypes_layout_views` credit.

### Run the transfer evaluator

Build a fresh local extension and create an evaluator build receipt using the
[native build instructions below](#reproduce-and-bind-a-native-build). The same
receipt format is used: commit, production-source digest, extension SHA-256,
build command, Rust/Cargo compiler versions, and `nvcc` selection (record
`unused` when the build does not compile CUDA). Never attest a copied or stale
binary as a fresh build. Use the installed pinned toolchain and keep all build
outputs, downloads, caches, and temporary files inside the worktree. Burner jobs
must reserve the shared `gpu` resource.

```bash
mkdir -p target/cuda-transfers
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B scripts/evaluate_cuda_transfers.py \
  --seed 9173 --seed 260909 \
  --build-record target/cuda-math/build-record.json \
  --output target/cuda-transfers/evidence.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B -m unittest discover \
  -s tests -p test_cuda_transfers_evaluator.py -v
```

Interpreter overrides `--reference-python` and `--candidate-python` are
available. Without a matching build receipt, candidate credit is zero. The JSON
records commit/source, evaluator/helper/matrix and native-extension hashes,
build configuration, interpreter/package versions, seeds, process IDs, GPU
model/UUID/compute capability, driver and memory pressure, and paths/versions of
the CUDA runtimes actually mapped by each worker. The evaluator itself does not
compile CUDA or select `nvcc`; the build receipt records the compiler actually
used, rather than inferring it from the driver's compatibility version. Source,
extension, evaluator, or matrix changes during a run invalidate candidate
credit. Hardware unit tests skip clearly when NVIDIA CUDA is absent; the runner's
six-slot denominator is retained on such hosts. Synthetic accounting fixtures
are not evidence of hardware support. This evaluator-only addition claims no
production implementation gain.

## Fixed CUDA float32 math denominator

The `cuda_float32_math_v1` case set within `math_reductions_linalg` fixes six
equally weighted cases. It adds concrete cases without changing the v1 backend
set, capability weights, existing cases, or breadth requirements.

| Case ID | Public operation | Input shapes | Output shape |
| --- | --- | --- | --- |
| `cuda_f32_add_same_shape` | `a + b` | `(7, 13)`, `(7, 13)` | `(7, 13)` |
| `cuda_f32_add_trailing_vector` | `a + b` | `(7, 13)`, `(13,)` | `(7, 13)` |
| `cuda_f32_neg` | `-a` | `(7, 13)` | `(7, 13)` |
| `cuda_f32_mul_scalar` | `a * -1.75` | `(7, 13)` | `(7, 13)` |
| `cuda_f32_sum_axis` | `sum(a, dim=1, keepdim=False)` | `(7, 13)` | `(7,)` |
| `cuda_f32_matmul` | `matmul(a, b)` | `(7, 13)`, `(13, 5)` | `(7, 5)` |

All inputs and outputs are contiguous float32 tensors on logical `cuda:0`.
The evaluator generates nonzero, mixed-sign inputs independently of either
framework, using Python `random.Random(seed).uniform(-2, 2)` rounded to float32.
At least two distinct, nonnegative evaluator-selected integer seeds are required;
negative seeds are rejected because Python aliases them to their absolute values.
Without `--seed`, the runner selects three random seeds and records them for
reproduction. Each
case must pass every seed to earn one of the six slots. TF32 is disabled in the
reference; output tolerances are `atol=1e-6`, `rtol=1e-5`. Shapes, operations,
tolerances, and input generation are versioned; changing these requires a new
case-set version rather than silently replacing this denominator.

[`scripts/evaluate_cuda_math.py`](../scripts/evaluate_cuda_math.py) runs every
reference and candidate trial in a fresh, separate Python process. Reference
eligibility requires NVIDIA PyTorch 2.13.0 and successful execution on real CUDA
hardware. Candidate workers import this checkout's `python/torch_rs` and local
native extension, block `torch` imports before importing the candidate, and
retain even swallowed forwarding attempts as zero-credit evidence. Inputs are
created through public APIs. An evaluator-owned CUDA driver probe checks input
and output pointer memory type and device ordinal, synchronizes execution, and
copies the output into host memory. The public `.cpu().tolist()` observation
must also agree. Every input is re-materialized afterward and must match its
original snapshot, since these operations must preserve their operands. Missing
or changed post-operation snapshots earn zero. This setup does not award
transfer, layout, compilation, autograd, or performance credit.

Missing, failed, skipped, unsupported, forwarded, malformed, incorrect, and
unbound candidate trials earn zero. Reference failure also leaves the slot in
the denominator as zero and makes the command exit with status 2 to distinguish
an incomplete hardware run. Successful reference runs exit 0 even when every
candidate fails. All raw failures and materialized values remain in the JSON.
The output reports this cell's `passed / 6` only; it does not overwrite previous
evidence or emit a new overall heterogeneity or breadth score. Other cells and
backends retain their existing accounting.

### Reproduce and bind a native build

Build the extension from the evaluated checkout, then place its ABI3 library
at `python/torch_rs/torch_rs.abi3.so`. Use an interpreter with PyTorch 2.13.0 and
the CUDA runtime available. Burner jobs running this evaluator should reserve
the shared `gpu` resource; only GPU 0 is used. The following Linux example confines build outputs,
Cargo downloads, and temporary files to `target/cuda-math`. It uses the already
installed pinned Rust toolchain; it does not install or update a toolchain.

```bash
mkdir -p target/cuda-math/tmp target/cuda-math/cargo-home
export CARGO_HOME="$PWD/target/cuda-math/cargo-home"
export CARGO_TARGET_DIR="$PWD/target/cuda-math/build"
export TMPDIR="$PWD/target/cuda-math/tmp"
export RUSTC="$(rustup which --toolchain 1.92.0 rustc)"
export RUSTDOC="$(rustup which --toolchain 1.92.0 rustdoc)"
cuda_math_cargo="$(rustup which --toolchain 1.92.0 cargo)"
export PYO3_PYTHON="$PWD/.venv/bin/python"
"$cuda_math_cargo" build --locked --release --features extension-module
cp target/cuda-math/build/release/libpytorch_rs.so python/torch_rs/torch_rs.abi3.so
```

Only after the build succeeds with unchanged source, the evaluator records its
build receipt. A receipt is an evaluator attestation of that build, not a
candidate-reported claim. Preserve it with the output when publishing hardware
CI evidence; a missing or mismatched receipt cannot earn credit.

```bash
export CUDA_MATH_CARGO="$cuda_math_cargo"
.venv/bin/python -B - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from evaluate_cuda_math import source_provenance, sha256
receipt = {
    **source_provenance(),
    "extension_sha256": sha256("python/torch_rs/torch_rs.abi3.so"),
    "build_command": "cargo build --locked --release --features extension-module",
    "rustc": subprocess.check_output([os.environ["RUSTC"], "--version"], text=True).strip(),
    "cargo": subprocess.check_output([os.environ["CUDA_MATH_CARGO"], "--version"], text=True).strip(),
    "nvcc": "unused",
}
Path("target/cuda-math/build-record.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B scripts/evaluate_cuda_math.py \
  --seed 9173 --seed 260909 \
  --build-record target/cuda-math/build-record.json \
  --output target/cuda-math/evidence.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B -m unittest discover \
  -s tests -p test_cuda_math_evaluator.py -v
```

`--reference-python` and `--candidate-python` can select different environments;
the candidate still loads this checkout's package and extension. Each result
records the commit, production source digest, evaluator and matrix hashes,
extension path and hash, build receipt, interpreter, package versions, seeds,
process IDs, GPU UUID/model/compute capability, driver version, memory pressure,
and the paths and versions of CUDA runtimes actually loaded in each process.
`nvcc` is unused by the runner. The receipt records the native build compiler
and configuration separately. Production source, evaluator, or matrix changes
during a run invalidate candidate credit. Portable tests clearly skip the hardware check when NVIDIA
CUDA is unavailable; the command's six-case denominator never shrinks.

The import blocker and pointer checks reject ordinary forwarding and CPU-storage
fallbacks; they are not a sandbox for hostile native code. Published credit also
requires reviewing the commit-bound native execution path under the evidence
rules above: copying a CPU-computed answer into GPU storage is not native CUDA
math. Synthetic accounting fixtures in `tests/test_cuda_math_evaluator.py` are
unit tests of zero-credit rules and do not constitute hardware evidence.

## Fixed CUDA float32 inference compilation denominator

`cuda_float32_inference_compilation_v1` adds six equally weighted cases to the
existing CUDA `compilation` capability (weight 10). Both frameworks use exactly
`compile(program, backend="eager", fullgraph=True, dynamic=False)` on ordinary
public, contiguous float32 CUDA tensors without gradients.

| Case ID | Program body | Input shapes | Output shape |
| --- | --- | --- | --- |
| `cuda_f32_compile_add_same_shape` | `a + b` | `(7, 13)`, `(7, 13)` | `(7, 13)` |
| `cuda_f32_compile_add_trailing_vector` | `a + b` | `(7, 13)`, `(13,)` | `(7, 13)` |
| `cuda_f32_compile_neg` | `-a` | `(7, 13)` | `(7, 13)` |
| `cuda_f32_compile_mul_scalar` | `a * -1.75` | `(7, 13)` | `(7, 13)` |
| `cuda_f32_compile_sum_rows` | `a.sum(dim=1, keepdim=False)` | `(7, 13)` | `(7,)` |
| `cuda_f32_compile_matmul` | `a @ b` | `(7, 13)`, `(13, 5)` | `(7, 5)` |

The [separate runner](../scripts/evaluate_cuda_compilation.py) defines these
programs independently of production code, historical diagnostics, the 38-case
compiler corpus, and the four-shape performance corpus. It reuses the CUDA math
evaluator's source/build hashing, PyTorch import blocker, deterministic float32
input generation, CUDA driver observation, runtime provenance, and execution
validation helpers. Each framework/case/seed gets its own isolated `-I -B`
process. Candidate workers load only the checkout's Python package and local
native extension; reference workers require NVIDIA PyTorch 2.13.0.

At least two distinct nonnegative evaluator-selected seeds are required. The
runner defaults to two randomly selected seeds and records them. For each seed,
one compiled wrapper executes the initial data and then changed data generated
with `seed XOR 0x9E3779B97F4A7C15`. Both outputs must match the reference with
`atol=1e-6`, `rtol=1e-5`; reference TF32 is disabled. Every execution validates
CUDA storage with driver pointer queries and synchronized host readback, checks
public `.cpu().tolist()` agreement, and re-materializes all inputs to require
unchanged contents. Shapes, strides, storage offsets, dtype, device, and
`requires_grad=False` must match the fixed metadata.

An evaluator-owned profiler blocks execution of the candidate's original Python
program body during wrapper creation and both calls, retaining even swallowed
body/import attempts. It observes the actual bytecode lowerer returning the
expected operation, graph executor entry, and successful native extension
executor return. Changed-input execution must use the existing graph: lowering
is blocked on that call, while graph and native execution must happen again.
Missing evidence, eager-body execution, copied constant outputs, forwarding,
mutation, malformed metadata, and unbound native builds cannot earn credit.
These observations are deliberately tied to the current native compiler hooks;
a future compiler architecture needs independently reviewed observer changes.
They are not a sandbox against hostile native code or proof that arbitrary
programs run on CUDA. Published evidence still requires native-path review.

Accounting iterates the six manifest slots, never the successful results.
Every seed and both executions must pass to earn one case. Missing, unsupported,
failed, skipped, duplicated, malformed, stale, or unbound results retain zero
slots. An unavailable or failed reference is recorded separately, also retains
its slot, and makes the runner exit 2. A complete reference run exits 0 even
when candidates are unsupported. Missing or malformed build receipts still
produce all six slots with zero credit. Changes to production source, native
extension, evaluator, shared helper, or matrix during a run invalidate candidate
credit. Synthetic accounting tests are not hardware evidence.

For this bounded capability, `p(cuda_nvidia, compilation) = passed / 6`.
Five verified cases represent `10 * (5/6)` points within CUDA and approximately
`10 * (5/6) / 7 = 1.190476` points of the seven-backend feature-depth formula.
This is a bounded accounting contribution, **not an automatic score increase**:
reconcile it with existing calibration without double-counting prior compilation
evidence. This evaluator-only change adds no production behavior. It establishes
no performance, dynamic-shape, backward, general compiler, other-capability, or
backend-breadth credit. Existing backend/capability weights, transfer/math sets,
compiler/performance corpora, historical diagnostics, and Burner-managed
progress artifacts are unchanged.

### Run and reproduce compilation accounting

Use the [fresh native build and receipt procedure](#reproduce-and-bind-a-native-build)
above. The receipt must include `commit`, `source_sha256`,
`production_diff_sha256`, `extension_sha256`, `build_command`, `rustc`, `cargo`,
and `nvcc` (explicitly `unused` when appropriate). Keep dependencies, build
outputs, compiler caches, and temporary files inside the worktree. Burner jobs
must reserve the shared `gpu` resource; the runner requires physical GPU 0.

```bash
mkdir -p target/cuda-compilation
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B scripts/evaluate_cuda_compilation.py \
  --seed 1927 --seed 83719 \
  --build-record target/cuda-math/build-record.json \
  --output target/cuda-compilation/evidence.json
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -B -m unittest \
  tests.test_cuda_compilation_evaluator -v
```

Set `TORCH_RS_CUDART` and `TORCH_RS_CUBLAS` to the selected local runtime
libraries when they are not on the library search path. The JSON records mapped
CUDA runtime paths/versions and cuBLAS/driver libraries, GPU model/UUID/compute
capability, driver and memory pressure, interpreter/package versions, source
and evaluator hashes, build configuration, raw values, failures, and accounting.
It records an installed `nvcc` version separately from actual compiler use:
this eager-backend comparison does not invoke `nvcc` or Inductor. Native
pointwise kernels use driver-JIT embedded PTX; native matmul uses cuBLAS.
Portable unit tests run without CUDA, and hardware tests skip clearly when the
reference GPU or local extension is unavailable.

The [H100 evidence](evaluation-data/cuda-inference-compilation-v1-h100.json)
was regenerated from clean implementation commit
`372ebc1661f485729bfffd8b891a1480bda95377`. Its production sources match
main `46db0021e8db4b563327ac4b8290eb7eab4318f4`. The committed evaluator and matrix
were used without changes. A fresh empty Cargo target directory was built with
locked, offline Rust 1.92.0, reusing only the worktree-local dependency environment
and Cargo registry. No dependencies were installed or changed.

The [capture receipt](evaluation-data/cuda-inference-compilation-v1-h100-receipt.json)
records the actual command, timestamps, environment, artifact hash, and clean
Git status before and after measurement. The artifact embeds the fresh build
receipt, full build log, source/evaluator/extension hashes, and every trial.
Build, import, executable, and runtime dependency paths belong to this worktree;
the installed compiler and system NVIDIA driver paths are recorded separately.
Evidence and documentation were updated only after the clean capture finished.

That run selected seeds `6521260218851170669` and `4271397935971695806`:
all 12 reference trials passed, and the candidate earned **5/6** slots. Row sum
failed bytecode lowering (`KW_NAMES`) at both seeds and earned zero. The device
was NVIDIA H100, compute capability 9.0, driver 580.82.07. Both processes mapped
CUDA runtime 13.0 from the local PyTorch dependency environment; PyTorch was
2.13.0+cu130. The fresh release build took 45.89 seconds wall time
with thin LTO and one codegen unit. Installed `nvcc` was 12.6.85 and was unused.
Native cuBLAS 13 and driver-JIT PTX paths are recorded separately in the worker
observations. No inference performance was measured. All 13 focused accounting
and isolation tests passed; their output is preserved in the capture receipt.
