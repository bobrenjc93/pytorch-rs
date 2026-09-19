# Full public-call CUDA experiment

**No executable optimization was retained.** A small-register shared-memory
experiment avoided one scratch allocation, but did not establish a reliable
full-call improvement over incoming `9a436482`. The native/compiler sources were
restored byte-for-byte. This author output adds diagnostic tooling and navigation;
it is not a basis for another unchanged-code performance evaluation.

The [diagnostic](../../scripts/diagnose_compile_full_call.py) compares untouched
public `compile(fn)` calls in separate processes. Its eight independent programs
cover arithmetic, runtime scalars, trigonometry, broadcasting, nested results and
input views, across scalar/empty/vector/matrix/tail inputs. All inputs use dense
offset subspans. Each shape has reused and fresh-input modes, five warmups and
17 individually synchronized calls. Input construction and complete readback are
outside timing; wrapper creation and first-shape calls are recorded separately.
Both framework orders were captured for accepted `60202557`, incoming `9a436482`
and the **uncommitted, subsequently removed** experiment. No canonical evaluator
was run and no score is inferred.

Example full-call medians, microseconds, in native-first / native-second order:

| Fresh-input case | Accepted main | Incoming | Removed experiment |
| --- | ---: | ---: | ---: |
| Affine, 257 elements | 46.81 / 46.57 | 45.54 / 43.50 | 43.97 / 44.74 |
| Affine, 262147 elements | 207.88 / 179.42 | 135.23 / 145.93 | 140.06 / 199.37 |
| Polynomial, 4097 elements | 65.45 / 68.21 | 62.11 / 63.84 | 63.28 / 63.34 |

The complete observations retain slow cases and rejections. In each order,
accepted main supported 58/76 cells and rejected 18 view cells; incoming and the
experiment supported 76/76. Every supported cell passed complete value, shape,
stride, offset, dtype, device, gradient, wrapper-identity and input-alias checks
against default PyTorch, including signed-zero revalidation. Native processes
blocked PyTorch imports; separate untimed probes checked original-body replay.
These finite developer examples do not establish general parity or explain a
canonical performance result.

The host was GPU0 H100, UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver
580.82.07, runtime 13.0; Python 3.12.12, PyTorch 2.13.0+cu130 and Rust 1.92.0.
Builds, imports and caches were worktree-local. NVCC inventory was 12.8/V12.8.93.
NVRTC unloaded before the end-of-process inventory; separately timestamped,
later untimed same-build receipts report NVRTC 13.0 and retained module options.
Those receipts are not retroactively attributed to the timing runs. The current
script records retained executor compiler receipts directly for future captures.

[Compact receipts](default-compile-full-call-20260918.tar.xz) contain every timing
sample, comparisons, identities, commands, build/test logs and the removed patch.
`raw-archive.json` identifies the complete raw archive, including all saved arrays,
retained under `target/full-call/evidence/` as `BURNER_EVALUATION_ARTIFACT_DIR`.
Raw member hashes were verified after archiving; prior evidence was untouched.
The measured script is preserved separately from subsequent receipt-validation
improvements. Initial import-name, test-type, Clippy-cast and README-link failures
remain alongside their successful retries.

Final retained-code checks: CUDA-hidden compiler suite **403 tests, 195 skipped**;
GPU transpose/add_/preparation/core compiler regressions **71 tests, 2 multi-device
skips**; diagnostic integrity and README checks **18 passed**. The removed
experiment also passed 76 native pointwise tests and Linux Clippy with warnings
denied. These Linux checks do not reproduce macOS or Python 3.14 CI.

The requested clean-commit, release-build, three-way H100 evidence remains
outstanding at this author boundary. A justified executable improvement and
separate continuation admission are needed before advancing that work; this
rejected experiment supplies neither qualification nor permission to resample.
