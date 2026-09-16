# GELU in selected CUDA Programs

Ordinary `torch_rs.compile(fn)` now executes bounded functional GELU in the
selected direct or VM Program module. The [compiler guide](../../compile-pointwise-jit.md#functional-gelu)
defines the public boundary; the [numerical contract](../../compile-pointwise-numerics.md)
defines rounding. Erf stays private. CPU compilation, gradients, LayerNorm,
reductions, approximation arguments and general Inductor parity are unsupported.

The integration preserves the frozen NVIDIA provider and one Kernel/module
owner per selected executable. Canonical live-Erf dependency is checked against
actual Program words before publication, independently of original equal-shape
admission. Empty live-Erf VM execution still links. Exposed NVRTC PTX is the
linker's input, not its final cubin.

## Source and evidence identities

| Role | Immutable source / record |
| --- | --- |
| Accepted B | `60202557b4f110d07777f585e804ab5f55e1ff7b` |
| Reused GELU source | `bc1a44ac408399efd130291bf684ff425ea4d88b`; [historical evidence](../compile-gelu-linked/README.md), including failed provider/capture attempts |
| Reused selected Program source | `cef34b14d865ee6c207120b1ee3701a8e5e26bd4`; implementation `4e291f5272b4b872ddc1704398b088b1917affc8`; [historical evidence](../program-identity/README.md) |
| New integration contract | [Unmodified 24bb protocol](protocol.md), alongside [d6e8 protocol and 1861 consumer](../program-identity/protocol.md) and b907 consumer controls |
| Development evidence | [Archive and raw-part manifest](development-manifest.json) |

Reused diagnostic files remain byte-identical at their original paths. Their
native builds, failures, timings and scores describe those historical sources,
not this combination. The development archive contains their identity audit,
the seven reviewed contracts, command streams/receipts and source snapshots,
actual release wheel/sdist bytes, native binaries and selected input/output
captures. Failed attempts remain included. These are development measurements
from an uncommitted combination, not clean-C public timing or qualification.
Concatenate the manifest's ordered `.part-*` files to reconstruct its verified
gzip archive; each tracked part stays below GitHub's file-size limit.

## Development validation

Rust all-target tests passed (486 tests); native-bridge tests passed (300), with
an additional native eager-image separation control. Clippy and formatting passed.
The complete pointwise suite ran 401 tests: 391 passed, nine multi-GPU cases
were skipped, and one smoke test failed before GPU execution because its
interpreter symlink resolved outside the worktree. That unchanged smoke test
passed after installing a byte-identical local interpreter, giving 392 passing
cases across the run and targeted rerun. Final GELU/identity/failure/docs controls
passed 40 tests; portable controls passed 44 with CUDA hidden. Wheel/sdist checks verified embedded provider
bytes, licenses and installed/source identity. The untimed capture records 12
selected invocations, actual compiler/runtime library hashes and the reserved
H100 GPU0 UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`.

Same-Program direct/VM controls compare finite bits, signed zeros, subnormals,
cancellation, constants, nested/shared consumers, runtime scalars and output/hint
histories. Non-finite comparisons require matching classification; original NaN
payload/sign transport is not guaranteed. Public reference tolerance is unchanged.
Release-unobservable counters remain unknown. No multi-GPU execution is claimed.
The surplus `test_compile*.py` legacy module-capture run was interrupted (exit
-15); it is retained as incomplete, not reported as a passing suite. Both initial
Rust fixture/check failures and the immutable-file whitespace findings are also
retained. Authored-source whitespace checks pass.

## Burner evidence handoff

Burner creates the implementation/tooling commit. After that commit, use fresh
source-bound B/C release builds and the unchanged consumer's `check`, `freeze`,
`build`, `preflight`, eight ordered `leg` commands and `verify`. The complete
[protocol](protocol.md) owns arguments, order, counts, fresh caches and controls;
the clean evidence phase is limited to 30 minutes and evidence/docs changes.
Run [capture.py](capture.py) against C's actual wheel/sdist to bind the 24bb
contract alongside the unchanged consumer bindings and selected GELU receipts.
[run.py](run.py) records inner commands; [archive.py](archive.py) seals and
re-reads supplied worktree-local raw paths with complete path/size/SHA256 records.
Include **all** attempts and both source-bound wheels before handoff. Outer
controller receipts and canonical evaluation exports remain Burner's records.

The eight clean B/C/reference legs, canonical scores, qualification and exact-head
CI are not development results. Six non-Erf timing families do not measure GELU
link cost. Coverage, native B/C speed and public-reference performance remain
distinct; GELU alone gives no CUDA decomposition-category credit while LayerNorm
is unsupported. No timing improvement or score gain is claimed.
