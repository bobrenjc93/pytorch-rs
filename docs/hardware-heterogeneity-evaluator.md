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
