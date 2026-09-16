# Clean integration evidence at 5bf3e9b0

Measured C is implementation/tooling commit
`5bf3e9b0edfe9c3f06fb8a5cf55905307c0b0960`; B is accepted
`60202557b4f110d07777f585e804ab5f55e1ff7b`. Both source inventories were clean
and revalidated after execution. These are new combined-source measurements,
separate from the immutable development and predecessor archives.

All eight serial ordinary-default legs passed the unchanged [protocol](protocol.md)
and verifier: 96 cells, 1,632 samples, 480 warmups, 96 first calls and 96 additional
churn calls. Exact B/C bits and metadata/alias checks passed; finite public-reference
checks passed at the frozen 1e-4/1e-4 tolerance. Recorded raw output bit differences
were zero across all compared pairs. No sample or workload was removed, and no
timing leg was rerun.

The [measured summary](postcommit-5bf3e9b0.json) contains every cell's median,
Q1/Q3/MAD/min/max, factory/first-call timing, all churn timings, all 72 comparison
ratios and the unchanged verifier result. The [raw archive manifest](postcommit-5bf3e9b0-manifest.json)
binds every retained path/size/hash and its ordered compressed parts, including
all samples, inputs/outputs, selected receipts, commands/streams, fresh B/C wheels,
the actual C sdist, native bytes and source/build/runtime records.
The [seal receipt](postcommit-5bf3e9b0-seal-receipt.json.gz) records the archive
command's actual successful exit and complete-member re-read.

## Fixed non-Erf results

Ratios above one favor the denominator. The two orders are independent observations,
not a statistical guarantee. Size labels follow the frozen consumer; broadcast
uses (37, 7) and (9363, 7) with a width-seven operand.

| Family | Size | B/C forward | B/C reverse | Ref/B forward | Ref/B reverse | Ref/C forward | Ref/C reverse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affine | 257 | 1.110 | 1.082 | 0.831 | 0.837 | 0.909 | 0.885 |
| affine | 65537 | 1.148 | 1.068 | 0.978 | 1.015 | 1.107 | 1.096 |
| trig | 257 | 1.072 | 1.053 | 0.750 | 1.042 | 0.808 | 0.796 |
| trig | 65537 | 1.136 | 1.121 | 0.901 | 0.922 | 0.935 | 0.974 |
| shared | 257 | 1.303 | 1.272 | 0.618 | 0.614 | 0.784 | 0.756 |
| shared | 65537 | 1.462 | 1.521 | 0.562 | 0.551 | 0.780 | 0.839 |
| broadcast | 257 | 1.111 | 1.038 | 0.727 | 0.728 | 1.207 | 0.754 |
| broadcast | 65537 | 1.104 | 1.056 | 0.829 | 0.811 | 0.856 | 0.865 |
| structured | 257 | 1.046 | 1.030 | 0.530 | 0.531 | 0.538 | 0.539 |
| structured | 65537 | 1.124 | 1.116 | 0.715 | 0.727 | 0.770 | 0.719 |
| over_cap | 257 | 1.015 | 0.997 | 0.240 | 0.252 | 0.255 | 0.262 |
| over_cap | 65537 | 1.205 | 1.189 | 0.228 | 0.237 | 0.285 | 0.271 |

C improved 23 of 24 native steady cells. The small over-cap reverse cell regressed
from 126.792 to 127.182 microseconds; its forward medians were 128.013 to 126.181.
Large shared arithmetic improved from 62.294 to 42.605 microseconds forward and
64.047 to 42.114 reverse.

Cold and churn costs remain visible. Small shared first calls increased from
13.600/12.613 ms in B to 30.628/29.093 ms in C (forward/reverse). Small over-cap
first calls increased from 17.186/15.687 to 100.181/97.061 ms. The final negation
shape-five preparation revisit increased from 72.620/76.806 microseconds to
75.965/90.546. Other cold and history observations remain in the measured summary.

C was slower than default PyTorch in 21 of 24 paired steady cells. These native
B/C improvements do not establish public-reference parity or a scoring gain.
None of the six timed families contains Erf, so this matrix does not measure GELU
link cost or GELU performance. GELU alone adds no CUDA decomposition-category
credit while LayerNorm remains unsupported.

## Capability, controls and provenance

The fresh C wheel passed 28 focused GELU/Erf, identity, compiler/link failure and
capture-retention controls. The frozen consumer passed 33 hardware-free controls.
Twelve separate untimed selected GELU invocations passed, covering ordinary,
nested/shared, over-cap and integer-zero-erased expressions through 13/0/13
elements. Their actual identity, selected source, pre-link PTX, compiler options,
input/output bits and metadata are retained; Python-body/eager-GELU execution was
rejected during capture. Native same-Program direct/VM and ownership tests from
development remain separately labelled in the existing development archive.
No unrelated full suite was rerun.

Preflight and both C post-timing ownership-control phases passed: retained hits
avoided host-plan construction/accounting, equivalent negation kept one executor
through preparation eviction, distinct broadcast addresses respected eviction,
and reset retained valid external owners. All over-cap matrix Programs selected
VM with 325 instructions; the other families selected direct execution.
Release-unobservable compiler/load/upload/link-destruction counters remain unknown.

Both release wheels were built from fresh target/Cargo directories and checked
against their own source and installed native/Python bytes. Actual C wheel/sdist
provider and NOTICE/license inclusion passed. Dependencies were copied into owned
B/C environments; wheel installations and setup durations are separately recorded.
The documented CUDA layout remedy preserved and hash-checked the original
directories before linking identical bytes to the common worktree-local provider.

Actual JIT NVRTC was 13.0, CUDA runtime 13000, driver 580.82.07 / API 13000,
Rust 1.92.0 and Python 3.12.13; the separately recorded system nvcc was 12.6.
Reference PyTorch was unchanged 2.13.0+cu130. Every leg used only H100 GPU0,
UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, one host thread, fresh per-leg
caches, five warmups, 17 retained results and the same completion barrier.
Actual loaded library paths/hashes, including NVRTC builtins, are retained.
There was no concurrent GPU profiling or diagnostic during the timing sequence.

All executed post-commit required commands succeeded; historical/development
failures remain untouched. The manifest binds the 24bb integration protocol
alongside the unchanged d6e8 protocol, 1861 consumer and b907 controls.
Outer-controller receipts and canonical exports remain Burner's separate records.
This evidence does not replace independent review, canonical scoring/full
qualification, exact-head CI or merge approval.
