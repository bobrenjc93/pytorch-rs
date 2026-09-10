# Compiled CUDA matmul development validation

This is **uncommitted development evidence** on base
`046b7a21e4e2fb7b59b56ea8a9679a9d9c5b0981`, not a clean candidate commit or
Burner merge qualification. No commit, push, PR or managed progress update was
performed. The [implementation manifest](implementation-manifest.json) records
the tested changes; the [build record](build-record.json) binds production
sources and the installed release wheel/extension. The build record was assembled
from the completed build command and its retained log; the initial Cargo target
was absent, and the final build reused dependencies and recompiled Rust sources.

The local CPython 3.12.14 interpreter and packages live inside this worktree.
PyTorch 2.13.0+cu130 and native code used H100 GPU 0, driver 580.82.07, with
local CUDA runtime 13000 and cuBLAS 13 shared libraries. Rust 1.92.0 built
`release`, `extension-module`, abi3-py310, thin LTO and one codegen unit.
Available nvcc 12.6 was unused. The timing report includes exact mapped runtime
and cuBLAS paths/hashes, compiler/cache configuration, GPU memory pressure,
interpreter/package paths, all installed Python-source hashes, and source-before/
source-after identities. The [wheel verifier](installed-wheel-verification.log)
and isolated `-I` tests checked current-worktree imports, blocked PyTorch imports,
and rejected original Python-body execution, including changed input data.

| Check | Result and retained evidence |
| --- | --- |
| New compiled H100 regressions | [13 tests](compiled-tests-attempt3.log): 12 passed; the separate two-device test skipped under GPU 0 |
| Broader compiler/CUDA/docs regression run | [189 tests](python-regressions.log): 182 passed, six hardware-mask skips, one documentation navigation failure; no compiler/CUDA failures |
| Corrected navigation | [All 12 docs checks passed](docs-final.log); the new guide link moved from historical evidence to contributor guides |
| Separate GPUs 0,1 | [Both eager and compiled device tests passed](two-gpu.log) |
| Rust suite with CUDA hidden | [393 passed](rust-all-cpu.log); hardware paths return early without devices |
| Real-GPU Rust bridge/matmul | [Bridge passed](rust-gpu.log); [two eager CUDA tests passed](rust-cuda-matmul.log) |
| Portable Python behavior | [13 tests](portable-skips.log): two passed, eleven clear hardware skips |
| Rustfmt/Clippy | Formatting passed; [Clippy with warnings denied passed](clippy-attempt1.log) |
| Unchanged compiler corpus | [38/38 eligible programs passed](frozen-corpus.log), bounded existing score 100 |
| Unchanged CUDA math corpus | [All six cases at all three existing seeds passed](fixed-cuda-math.json), 18 successful trials |
| Separate compiled timing diagnostic | [All 12 cells passed](compiled-timings.json), **76.45% capped geometric parity** against PyTorch Inductor |
| Independent review | [Findings, fixes and final read-only review](independent-review.txt) |

The timing result is a scoped diagnostic, not a new Burner score. All six pure
matmul cells reached the per-cell cap in this run; composed cells were slower
and remain included. The report retains both orders, first-call/wrapper costs,
31 five-call samples per implementation per order, dispersion, changed-data
proofs, input preservation and materialized output checks. All 12 cells have equal
weight; failed or missing cells would force the fixed geometric aggregate to
zero. The aggregate and sample counts were independently recomputed from raw
samples after completion. Compiler disk caches are disclosed and reused across
orders; first-call costs are not universal cold-cache measurements.

## Retained unsuccessful attempts

- [First compiled attempt](compiled-tests-attempt1.log) exceeded the test
  wrapper's default 16-specialization limit when enumerating generated shapes
  and offsets. It was interrupted; the [focused failure](compiled-first-error.log)
  identifies the exception. Only that test wrapper's limit was raised to 128.
- [Second compiled attempt](compiled-tests-attempt2.log) used an unsupported
  direct CUDA `ones` factory in the new tests. The tests now use supported CPU
  construction followed by upload; the eager factory boundary was not changed.
- [Broader run](python-regressions.log) caught the navigation placement error
  described above. Its failure and corrected focused rerun are both retained.

All original logs/receipts are copied byte-for-byte from `target/evidence`.
[Artifact hashes](artifact-hashes.json) verify those copies. Commands with
`.receipt.json` companions retain exact argv, exit status, duration and relevant
environment; earlier setup/build/focused attempts retain their original logs.
The [reproduction guide](../../../compile-cuda-matmul.md) describes clean delivery.
Burner must commit implementation/tests/harness first, then recapture fresh
clean-commit evidence, independent review, all ten non-regressing gates and
exact-head CI before merge. None of those requirements is waived by this run.
