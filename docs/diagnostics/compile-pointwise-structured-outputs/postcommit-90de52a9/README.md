# Clean structured-output evidence at 90de52a9

Measured implementation: `90de52a97559b7596d8d64661904e3db59112d2a`, against
`main` at `30a3b504ef4d43bf2958998cc39545996cc09970`. This fresh clean-commit
capture includes the returned-product contraction repair. It completes the
capture deferred in the [development repair record](../review-returned-products.md).
The [initial 36954660 capture](../postcommit-36954660/README.md) and all development
measurements remain unchanged under their original provenance.

## Checks

The unchanged committed tests ran against a newly built release wheel:

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| Complete Python pointwise suite on H100 | 284 | 8 device tests |
| Dedicated two-physical-device checks | 8 | 0 |
| Release native pointwise and allocation/launch/completion ownership | 41 | 0 |
| Release Python-conversion failure ownership | 1 | 0 |
| Portable pointwise suite, each CPython version | 143 | 149 hardware tests |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5.
Hardware skips are not GPU passes. The conversion test's original interpreter
startup output is retained in its complete log.

The raw capture includes 157 materialized native/reference comparisons and their
dispatched CUDA/PTX, including the 91 competing-product regression combinations.
Every captured PTX has one kernel entry. Existing assertions cover return order,
signed/shared products, constants, zeros/nonfinite values, aliases, current metadata,
retained-prior-output freshness, identical-graph/different-topology executable reuse,
helper/loop/branch composition, failure atomicity and maximum output/scalar ABI.
The isolated child-process check passed with PyTorch imports and original/helper
body execution forbidden. Both frameworks used ordinary `compile(fn)` defaults;
existing tolerances were unchanged.

## Provenance and retention

[Measurements](measurements.json.gz) record clean before/after status, all tracked
source hashes, timestamped commands, cache state, interpreter/import paths, wheel
members, loaded runtime libraries, GPU snapshots and test outcomes. All tracked
files stayed byte-identical during capture. Every interpreter's installed Python
package and extension matched the wheel, without a source overlay.

Wheel SHA256: `506b5fb991037587e3c83ef423e047ba6f82111d8169d9b006dfdc27b531037b`.

The locked offline release build used Rust 1.92.0 and Maturin 1.15.0, with a fresh
Cargo target and initially absent reference compiler caches. H100 runs used
`CUDA_VISIBLE_DEVICES=0`; device tests used only `0,1`. The driver was 580.82.07,
reference PyTorch was 2.13.0+cu130, and native kernels used NVRTC 13.0 with loaded
CUDA runtime 13000. PATH `nvcc` reported 12.6.85; NVRTC compiled these kernels.
Full GPU UUIDs and actual local library paths are retained.

The [raw archive](raw-captures.tar.gz) retains commands, logs, numerical outputs,
CUDA/PTX and the orchestration scripts for the existing repository build/tests.
The [manifest](raw-retention-manifest.json) verifies archive members and identifies
the wheel, committed source, reference-cache and nested-diagnostic archives under
`target/default-compile-eval/structured-outputs-postcommit-90de52a9/`.
All archives were verified byte-for-byte. The 3,821 earlier raw evidence files,
including original failures and reviewer archives, were verified unchanged.
No disposable-worktree cleanup occurred. All writes stayed within this worktree;
Burner's canonical report-root observer owns external archival, and no external
archive path is claimed here.

No official score evaluation or unrelated full suite was rerun. This capture
changes only evidence and its documentation; implementation, tests, dependencies,
harnesses, evaluator definitions and managed progress files remain unchanged.
Independent review and merge gates are still required.
