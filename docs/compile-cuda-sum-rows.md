# Native compiled CUDA row sums

The native eager compiler accepts `x.sum(1)`, `x.sum(1, True)`,
`x.sum(1, keepdim=True)` and keyword forms such as
`x.sum(dim=-1, keepdim=False)`. Inputs must be exact native, contiguous,
rank-two CUDA float32 tensors without gradients. Dimensions are exact constant
integers `1` or `-1`; `keepdim` is an exact constant boolean, defaulting to
`False`. Constant locals and guarded module globals follow the existing
compiler constant rules. Other dimensions, option types, `dtype`, `out`,
argument unpacking and top-level `torch.sum` remain outside this compiler
surface. CPU compilation is unchanged.

`KW_NAMES` on Python 3.11/3.12 and the keyword-call tuple on Python 3.10 and
3.13/3.14 are decoded without invoking Python callables. Keywords are bound
only for the supported Tensor sum method; helper and other keyword calls
remain rejected. The live Tensor method guards include `sum` on cold calls
and cache hits.

The graph contains `call_reduction`, target `sum`, with separate normalized
`(dim, keepdim)` reduction options. Output metadata is `(M,)` or `(M, 1)`,
canonical contiguous strides, CUDA float32 on the input ordinal, offset zero,
and no gradients. It is a shape-changing reduction, not a unary elementwise
operation. A single node uses `_compile_trace_reduction`; composed graphs
use the native `SumRows` operation in `_compile_trace_cuda_graph`. Both call
`Tensor::sum_rank_two_dimension(1, keepdim)` and reuse the existing CUDA kernel,
storage bounds, synchronization, completion and device-restoration guards.
The complete Python graph and native composed plan are validated before
execution. No original Python body or reference PyTorch is executed by the
candidate.

The six-case hardware evaluator changes only its observed native hook for
`sum`, from the old unary hook to the real reduction entry point. Its programs,
input generation, seeds policy, options, tolerances, denominator and weights
are unchanged. Negative controls retain rejection of Python forwarding,
counterfeit hooks, body execution, re-lowering, malformed evidence and incorrect
results. The 38-case CPU compile corpus and four-workload private CUDA
performance suite are unchanged.

## Development validation and delivery boundary

Evidence under `diagnostics/compile-cuda-sum-rows/development/` records this
uncommitted worktree, including failed development attempts. Build receipts
bind the parent commit, production source and diff hashes, and native extension
hash; these are **development receipts, not clean implementation-commit
receipts**. The first independent hardware capture preserved the existing five
passes and passed the previously zero row-sum slot on both evaluator-selected
seeds. This is six bounded inference correctness cases, not universal compiler,
accelerator, training or performance parity.

The independent regressions use held-out shapes and normally distributed
inputs, changed values through the same wrapper, nonzero offsets, singleton
strides, zero rows/columns, keepdim metadata, composed reductions, live method
mutation, invalid options/metadata, late graph rejection, stream completion,
and GPU 0/GPU 1 device restoration. Python 3.10–3.14 each exercise their own
real keyword bytecode and native CUDA execution. Hardware-only checks skip
explicitly when CUDA is unavailable.

The copied Python distributions, virtual environment, dependencies, Cargo
registry, build outputs and all caches were confined to this worktree. The
reference is PyTorch 2.13.0+cu130. GPU inventory and runtime paths are recorded
in the evidence. Rust 1.92.0 builds the release ABI3 extension; the reduction
uses existing driver-JIT PTX, so nvcc is unused for this native operation.
The host nvcc version is separately recorded.

The final development wheel passed the following checks:

| Check | Result |
| --- | --- |
| Independent reduction suite | 14 tests, one expected two-device skip; separate two-device test passed |
| Python 3.10, 3.11, 3.12, 3.13, 3.14 | Six lowering/metadata/native-CUDA tests passed on each interpreter |
| Existing compiler and evaluator regressions | 139 tests passed |
| Existing CUDA compiler regressions | 62 tests, six expected two-device skips |
| Rust default / Python bindings | 389 / 401 tests passed |
| Rust formatting and both Clippy configurations | Passed |
| Installed wheel provenance | Passed; source and installed extension hashes agree |
| Fixed hardware compilation | 6/6 in both development captures, two evaluator-selected seeds per capture |
| Frozen compiler evaluator | 38/38 |
| Unchanged private performance suite | 4/4 in each of two runs |

The hardware seeds were `1136685960689225517`, `4173613457994413490`
(first capture) and `1713849872941315970`, `1407868679530581933`
(final wheel). Every case also reused its wrapper on the evaluator's changed
input dataset. All unsupported cells outside these fixed cases retain their
existing status; no larger capability score is inferred.

The performance runs retained five warmups, 17 samples, three repetitions,
synchronization, output materialization, all four shapes and equal weights.
The first used fresh CUDA/Inductor/Triton caches and built the private kernels;
the second reused those caches. Their common-success geometric-mean ratios
were 1.3339x and 1.2911x, respectively (both reports capped at 100% for this
private suite). Raw samples, cold/factory accounting and GPU snapshots are
retained. These timings do not measure the new generic row-sum graph and are
not a clean-commit performance non-regression claim. The private kernels used
nvcc 12.6.85; native reduction and reference workers used CUDA runtime 13.0.
Python-version-only native checks selected the read-only system CUDA 13.0.96
runtime; their exact paths and empty reference-import sets are recorded.

Failed attempts remain in the evidence bundle. Early test setup assumed
unsupported CUDA factories, strided slicing, gradient enabling and dtype
exports; reference wrappers also hit their recompilation limit, and a dynamic
cache test initially ignored the existing concrete-stride guards. These were
corrected without changing the implementation or reference tolerances. Two
old tests expected compiled sum to be unsupported and now assert the bounded
success while retaining their rejection checks. An initially copied editable
install pointer caused the isolated wheel regression to import the parent
checkout and fail; the pointer was removed and a fresh local wheel installed
before final validation. Missing `typing_extensions` in standalone interpreter
attempts was corrected locally (the final Python 3.12 test uses the local
virtual environment). No failed attempt was discarded or represented as a
passing gate.

Burner must create its final implementation commit, rebuild and recapture from
that clean commit, then perform independent review and the ordinary ten
no-regression gates before managed publication/merge. This implementation
session does not commit, publish a merge score, or modify Burner's managed
progress artifacts.

Reproduce after preparing a genuinely local environment and local cache paths:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v \
  tests.test_compile_sum_lowering tests.test_compile_cuda_sum_rows
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m unittest -v \
  tests.test_compile_cuda_sum_rows.CompileCudaSumDeviceTests
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/evaluate_cuda_compilation.py \
  --build-record target/row-sum/build-record.json \
  --output target/row-sum/hardware-compilation.json
.venv/bin/python scripts/evaluate_torch_compile_coverage.py
```

Use the existing [build receipt procedure](hardware-heterogeneity-evaluator.md#reproduce-and-bind-a-native-build)
for the actual revision and extension. Omitting `--seed` lets the hardware
evaluator select two independent seeds; preserve every case and outcome.
