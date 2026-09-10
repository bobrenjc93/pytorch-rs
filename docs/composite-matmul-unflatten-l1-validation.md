# Compiled CUDA matmul, unflatten and L1 integration

The combined implementation preserves captured native CUDA rank-2 contiguous
float32 matmul, public `torch.unflatten`, and matching-shape finite CPU float32
unreduced L1 backward. The detailed contracts remain in
[supported surface](supported-surface.md) and
[compiled matmul](compile-cuda-matmul.md).

The root-reported unflatten defect persisted in the imported source: the public
binding looked up `Tensor.unflatten` after dispatch. A fresh wheel built in this
composite reproduced the raising-method failure. Both bindings now use shared
Rust view/error logic after their respective schema and scope checks. The
public path converts integers once and never looks up either replaceable Tensor
entry point. It retains public dimension/empty-size error ordering, legitimate
override dispatch, storage sharing, offsets, strides and first-order gradients.
Permanent PyTorch 2.13 differentials cover method replacement, nested-mode
exception restoration, and unflatten views feeding weighted unreduced L1
backward into both original leaves.

The source compiler already includes native graph execution, captured-input and
callable guards, numerical regressions through K=1,000,000, overflow cases,
offsets, overlap, empty and zero-K products, exception recovery and separate
GPU-device restoration checks. The L1 source already checks both operands'
logical finite values and preserves fused inference paths. Those implementations
and their supported boundaries are retained. The separate timing diagnostic now
attaches each raw call and completed sample to the cell immediately, retaining
partial blocks and output hashes if a later check fails. An injected-check
regression covers that failure path; the matrix, warmups, sample counts, timing
intervals and aggregation are unchanged. Feature summaries now include the
public unflatten operation and the supported L1 gradient subset.

## Review repair: unflatten conversion order

Independent review reproduced a second binding defect: `dim` conversion ran
before `sizes` unpacking. The public binding now fully converts sizes first,
matching PyTorch 2.13 when dimension conversion mutates the sizes list or both
arguments have conversion errors. Schema validation and override dispatch still
precede conversion; dimension-range and empty-size checks retain their order.
Permanent positional/keyword differentials cover list replacement and clearing,
competing overflow/type errors, and dimension hooks skipped after sizes errors.

Both new regressions failed against the source-matched pre-repair wheel.
With a fresh local release wheel, all 76 focused unflatten/L1 tests passed,
including real H100
CUDA rejection, aliases, gradients and nested-mode restoration. Rustfmt and the
isolated wheel verifier passed; installed Python/native sources, interpreter,
reference imports and CUDA libraries resolved locally and matched the build.
Build/provenance records, actual command receipts and the failing pre-repair log
remain under `target/review-unflatten-order/`. These are development checks of
the uncommitted repair, not final clean-commit evidence.

## Clean committed validation before the review repair

Burner committed the integrated implementation as
`208e9bff072bc9354ca0aec3e4a5330c1bde53ca`. The
[post-commit capture](diagnostics/composite-matmul-unflatten-l1/postcommit-208e9bf/README.md)
built a fresh release wheel from an absent target and verified the installed
Python/native sources and local runtime identities. The 74 CPU-surface tests,
compiled/eager matmul checks, independent isolated-program proof, GPUs 0,1
restoration, focused Rust checks, 18 fixed math trials, 38-case corpus and all
four fixed scoring shapes passed. The separate 12-cell diagnostic retained all
raw samples and slower composed results, with 77.46% capped geometric parity.
Its clean source, build, runtime, compiler and command receipts are published;
the failed scoring import attempt is retained alongside the isolated retry.
These results measured the pre-repair composite. They are now stale for the
current candidate and remain unchanged pending the required clean-commit refresh.

## Development validation (preserved)

Development validation uses a copied local CPython 3.12.12 distribution, local
reference PyTorch 2.13.0+cu130 and NumPy, local CUDA 13 libraries, local caches,
Rust 1.92.0 and a release abi3 wheel built from this combined worktree. The
copied editable-package pointer was removed before candidate imports; the
installed wheel and isolated `-I` subprocesses resolve inside this worktree.
Available nvcc 12.6 is unused: native matmul calls cuBLAS and pointwise kernels
use embedded PTX through driver JIT. GPU checks use H100 with driver 580.82.07,
GPU 0 except the separate restoration checks on GPUs 0,1. The timing run
selected local CUDA runtime 13000 and Triton's local PTX assembler 12.8.93; its
exact compiler version, path and hash are recorded separately from unused nvcc.

Development results and command receipts are retained under
`target/integration/`. They describe uncommitted integration validation and
cannot qualify the final candidate. The checked-in matmul source reports and
all their failed attempts remain unchanged, pinned to their recorded source
and build identities. They do not measure this composite.

Completed development checks:

| Check | Result | Local log/report under `target/integration/` |
| --- | --- | --- |
| Full portable Python suite | 5,560 run, 5,339 passed, 221 skips | `python-suite.log` |
| Unflatten and L1 public/reference tests | 74 passed | `cpu-surfaces.log` |
| Compiler, CUDA and CPU-surface regressions on GPU 0 | 214 run, 208 passed, six device-mask skips | `compiler-gpu.log` |
| Separate compiled/eager GPU 0,1 restoration | Both passed | `two-gpu.log` |
| Rust default / Python bindings, CUDA hidden | 381 / 393 passed | `rust-default.log`, `rust-bindings.log` |
| Real-GPU native bridge / matmul | One / two passed | `rust-gpu.log`, `rust-matmul.log` |
| Six fixed CUDA math cases, three prescribed seeds | 18/18 trials passed | `fixed-cuda-math.json` |
| Unchanged compiler corpus | 38/38 reference-eligible cases passed | `frozen-compiler.log` |
| Separate compiled timing diagnostic | All 12 cells passed correctness; all slower composed cells retained | `compiled-timings.json` |
| Timing diagnostic accounting and partial failure retention | Both passed | `diagnostic-tests.log` |
| Clippy with Python bindings, Rustfmt and docs smoke | Passed | `rust-clippy.log`, `docs.log` |
| Installed wheel and every installed Python source | Matched this worktree | `provenance.log`, `wheel-verifier.log` |

The timing diagnostic started with empty Triton and Inductor caches and retained
them across orders. Each of the 12 cells retains two execution orders, 31
five-call samples per backend/order, 155 raw call timings and output-check
counts per backend/order, first-call costs and input/output checks. Its observed
capped geometric parity was 78.87%, including all six slower composed cells;
this is a development diagnostic, not a score for the final composite.

The development native extension SHA-256 is
`f640082cbbc767c127e4b771844c4c62d90201e38bca4282335c9f9eba9922a2`.
The initial target was empty; the repaired extension rebuilt the native crate
in that target. This reused build directory is another reason these checks
are development validation rather than the required final fresh build.
Each recorded test/capture command has a `.receipt.json` containing its actual
argv, timestamps, exit code, log hash, source manifest and environment.
`repro-before.log` retains the two expected pre-repair failures; setup/build
logs also remain available. No failed attempt was overwritten.

## Remaining managed handoff

Burner must commit the conversion-order repair before final evidence is
regenerated with the existing clean-build and capture procedure. The prior raw
records retain their measured identities; they do not supply current-candidate
performance credit after this source change. Independent exact-head review,
all ten non-regressing current-definition gates, exact-head CI, confirmed source
PR delivery, continued dispatch pause and managed merge remain Burner-owned
requirements. Development checks do not replace these requirements.
