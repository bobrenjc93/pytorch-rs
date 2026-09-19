# Default compiled alias mutation: developer validation

Dirty author sources based on `7424e19513a7d31427cb13c9d71a769acd0a72c0`,
measured on 2026-09-18. These are capability checks, not clean-commit evidence,
a canonical evaluation, or a performance/parity claim. The
[compiler contract](../compile-pointwise-jit.md#alias-only-views-and-scalar-mutation)
is authoritative for supported syntax and failure semantics.

One ordered native alias/effect sequence now handles input-rooted `view`,
`transpose` and scalar `add_`. Mutation-bearing programs reject numerical Tensor
operations, including required inactive/unused bodies, before writes. The native
bridge preflights every node, shares original storage, and preserves construction
and receiver identities. Alias-only calls create no numerical preparation or
executable. Runtime failure after effects begin can leave mutations visible;
success-only cache publication is not rollback.

## Checks and observed limits

| Check | Outcome |
| --- | --- |
| Final release wheel, source/import identity and 65 wheel RECORD entries | Passed |
| Native library suite, single-threaded, GPU0 visible | 306 passed |
| Selected public-default pointwise/identity/view/guard/in-place and explicit-eager Python suites | 582 run, OK, 15 skipped |
| Same selection plus quickstart/replay checks, CUDA hidden | 607 run, OK, 300 skipped |
| Additional nested receiver identity test against independent eager semantics | 1 passed |
| Final quickstart and alias suite, CUDA hidden, after the added identity test | 45 run, OK, 20 skipped |
| All-targets Python-binding Clippy, warnings denied; formatting; diff whitespace | Passed |

Tests cover current scalar values, empty/scalar/offset layouts, overlapping
original input aliases, discarded sequential writes, literal shape inference,
helpers/control flow, cold/warm binding guards, rejection before writes, failed
publication, external mutation, retained owners and exceptional float bits.
Native-only subprocesses block PyTorch imports; alias-only execution also succeeds
with an invalid NVRTC pin. Linux/GPU0 results do not establish macOS or multi-device
behavior. Native fault tests use controlled injection, not device faults.

The non-scoring capture used untouched public defaults in four separate workers
(two framework orders), two independent programs, two shapes, and fresh/reused
inputs: 32 worker cells, 3 warmups and 5 synchronized single-call samples per cell
(160 samples). Wrapper creation and first-shape costs are separate. Input reset
and full readback are outside timing. All workers completed, but **strict paired
comparison failed** for the ordered-view program in 8 of 16 paired cells:
PyTorch 2.13 returned distinct wrappers for two references to the same `add_`
receiver after another alias write. Native preserves the required exact receiver
identity and matches independent eager semantics. Saved input/output bits and all
other metadata match exactly; the original failed comparison remains unchanged.
The other 8 paired cells passed. No speedup or general compiled parity is claimed.

An earlier exceptional-value test also exposed reference warm reuse of `+0.0`
for a later `-0.0` scalar. The retained investigation contrasts that history with
fresh default compilation and eager semantics. Exact native bit assertions remain;
the reference special-value test uses fresh compilation per scalar. Other first
failures are preserved: local Maturin/interpreter setup, POP_TOP and inactive-return
admission, tuple RETURN_CONST/identity, unused input rank and bool-to-int scalar
provenance, obsolete tuple fixtures, a test budget lookup, and Clippy. Corrective
runs and their source snapshots are distinct receipts, not rewritten successes.
The final sequential Git-blob verification stalled and was terminated; the saved
batched blob-identity and exact production-source verification passed.

## Reproducible source and raw receipts

Only the worktree-owned Python 3.12.14+meta environment was used. Release builds
used explicit interpreters and source snapshots with local Cargo output reuse;
installation and loaded native paths were verified inside this worktree. NumPy
was 2.5.1, reference PyTorch 2.13.0+cu130, Rust 1.92.0. Physical GPU0 was H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07, with selected CUDA
runtime/NVRTC 13.0. Inventory nvcc was 12.8 V12.8.93, not the pointwise compiler.
Pre/post utilization snapshots are recorded; snapshots do not establish exclusivity.

The raw archive listed below contains exact argv/exits/logs, failed and successful
source snapshots/wheels, final source manifest/patch, installed RECORD verification,
loaded library hashes, all timing/output snapshots, the unchanged strict comparator
and the separately labeled untimed identity analysis. Historical repository
artifacts and protected evaluator inputs were verified unchanged. Raw files remain
inside the worktree because the request's final filesystem boundary conflicts with
its external `/tmp` retention instruction; Burner must preserve this archive before
removing the worktree. No external collection is claimed.

| Artifact | SHA256 |
| --- | --- |
| `target/alias-mutation/default-alias-mutation-author.tar.gz` (14,915,422 bytes; 712 verified payloads) | `34e070d95a99e96705e1f4e0b1a3bc84caa9f2e6730edb5619e40e5ec7708506` |
| Archive `MANIFEST.json` | `ed94e8120b89363f5f6106d4fd634b1dca6d66200dc362f8a67f949cf6c8df6a` |
| `target/alias-mutation/evidence/final-source-verification.json` | `db91c51447052fc73987febf72c3aa5c23af6b7505e04b25f9aecf4217e3fe27` |
| Release wheel, `target/alias-mutation/wheels/checked/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl` | `2d60bc19d282c67e6766c249c4cf9804f36c6776314a9ee08cf278199aa4787a` |
| Loaded `torch_rs.abi3.so` | `fff68d21024741d291d851db651d6336a92185069f34c373435e828ced616f23` |

Archive member paths are worktree-relative. `MANIFEST.json` verifies every payload
byte; `evidence/` below `target/alias-mutation/` contains the command receipts.
The raw `capability-diagnostic.py` contains the original strict comparator;
`identity-analysis.py` documents the separate untimed investigation. Neither
changes the failed paired result. No canonical evaluator was invoked.
