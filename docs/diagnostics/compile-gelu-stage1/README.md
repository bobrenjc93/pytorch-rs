# Native GELU prerequisite: failed integrated Erf experiment

**Stage 1 failed; GELU remains unsupported.** The private Erf candidate ran
through the existing native planner and CUDA executor, but four of 335 output
comparisons failed the unchanged pointwise numerical policy. Per the staged
contract, public eager/frontend implementation stopped. Production sources were
restored to accepted main `60202557b4f110d07777f585e804ab5f55e1ff7b`; the
experimental implementation is retained as a patch, not an active private API.

The failing expression is ordinary composition:

```python
g = torch.nn.functional.gelu(x)
p = g * g
q = x * y
return p, p + q  # also tested in reverse observable order
```

For y=-x, six elements fail in each order and each of two calls. At x=5, native
returns -9.5367431640625e-6 versus default Inductor -1.33514404296875e-5, an
absolute error of 3.814697265625e-6 against an allowance of
1.0001335144042968e-6. One-ULP upstream Erf differences become observable through
cancellation. The native plan and associated reference PTX agree on the final
contraction direction. Uniform input-subnormal normalization cannot explain or
repair differences at these normal positive arguments. No tolerance, polynomial,
shape/value exception, realization rule, or evaluator was changed to hide them.

## Evidence index

| File | Contents |
| --- | --- |
| [implementation.patch](implementation.patch) | Tested private Erf changes to six existing owners, malformed-program test, and five private Python tests; applies to accepted main. |
| [probe.py](probe.py) | Explicit SSA fixtures using the existing private bridge, separate default-Inductor/native processes, and the unchanged existing comparator. No fake public callable or proxy execution. |
| [raw-evidence.tar.gz](raw-evidence.tar.gz) | Complete attempts 001/002: predeclared matrices, raw float32 buffers, reports, native CUDA/PTX/plans, reference wrappers and final cache artifacts. |
| [raw-manifest.json](raw-manifest.json) | Every archived path, byte count and SHA256; all archived bytes were checked before removing loose copies. |
| [failure-elements.json](failure-elements.json) | All 24 violating element comparisons, including words, inputs, absolute errors and unchanged allowances. |
| [independent-audit.md](independent-audit.md) | Independent raw-byte, fixture-mapping and generated-code audit with attribution limits. |
| [build-and-checks.json](build-and-checks.json) | Exact commands, source/extension hashes, compiler provenance, lease observation and post-process GPU snapshots. |
| [checks](checks) | Unedited build, test and run logs. |
| [prerequisites.json](prerequisites.json) | Hashes of the external staged contract, designs and all required phase reports/audits read before implementation. |

Attempt 001 is a retained harness failure: it used the unsupported direct native
CUDA tensor constructor, so no numerical result was obtained. Attempt 002 fixes
only construction to use the existing CPU-to-CUDA transfer. Its matrix was
written before execution: 44 cases, 113 calls and 335 output leaves. All calls
completed, 331 leaves passed, four failed, and 2,324 words differed overall.
The matrix includes private Erf, ordinary GELU decomposition, affine consumers,
shared/nested/recomputed calls, competing products in both orders, integer/
Boolean/floating zero constants, signed zeros/subnormals/nonfinites, mixed trig,
28/30/32/98/100/102-operation chains, retained outputs and shape/scalar histories.
Constant repetitions do not imply distinct input coverage.

The private Erf implementation extends validation, operand traversal/remapping,
shape/dependency admission, region counts/locality/CSE, call ranks, program
encoding/disassembly/validation and CUDA execution. Live Erf requires equal
actual input shapes, including unused inputs; dead shape validation and accepted
broadcast-trig support remain intact. It uses float32 `erff` with uniform signed
input-subnormal normalization for both runtime and constant arguments, without
adding sin/cos's constant-double path. These are tested hypotheses that failed
the composition gate, not supported numerical semantics.

## Checks and limits

The experiment passed `cargo check --offline --features python-bindings`, the
release extension build, five Python private-admission tests, and 73 Rust
pointwise tests on physical GPU0 (including actual CUDA execution, preparation,
retained outputs, bounded maximum graphs and failure/ownership checks). The
**335-leaf numerical gate failed**. These results concern the archived patch;
no production feature is claimed by the evidence-only final tree.

Both runtime legs used physical H100 GPU0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07, under the idea's
existing `cpu-heavy`/`gpu` leases. Native used CUDA 13.0 NVRTC 13.0.88 with
`compute_90`, FMA enabled and global FTZ disabled. Reference used PyTorch
2.13.0+cu130/default Inductor. Available nvcc 12.6.85 was inventoried, not used to
compile these kernels. Mapped runtime/library hashes and native flags are in
the reports. After child exit GPU0 returned to 4 MiB/0% and the compute-app list
was empty. No foreign jobs were interrupted; lease release remains Burner's job.

Native and reference ran in separate processes with fresh cache directories
inside this worktree. Wrappers and native executors persist within each case.
The native diagnostic supplies shape hints and scalar promotion through existing
private owners; this does not establish public frontend histories or guards.
Generated reference artifacts are source/group evidence, not a selected-kernel
or SASS trace. No public GELU eager route, no-NVRTC deployment, or arbitrary-value
parity was tested. The earlier offline eager-image reports remain finite external
prerequisites, not production evidence for this branch.

Unrelated performance controls, fixed coverage/CUDA-performance evaluations and
full qualification were not run: the contract places them after correctness,
and the prerequisite failed. There is no score or speed claim. A future attempt
needs a general compatible primitive lowering and fresh integrated evidence;
this result does not justify expression-specific corrections or another
numerical subsystem.

To inspect the raw evidence, list or extract the archive into a fresh directory
inside this worktree. To reproduce the experiment, apply the archived patch in
an isolated checkout of the stated base, use the recorded build commands, and
run `probe.py declare`, `native`, `reference`, then `compare` against a new
worktree-local directory under Burner's resource ownership. Do not overwrite
these failed-attempt records.
