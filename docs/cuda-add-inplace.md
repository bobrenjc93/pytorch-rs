# CUDA scalar `Tensor.add_`

`x.add_(other, *, alpha=1)` adds a real Python/NumPy scalar to native CUDA
float32 storage and returns the exact receiver. `other` may be a keyword.
Numeric `alpha` must equal one before float32 conversion; Python boolean alpha
is rejected, while NumPy boolean alpha retains the existing scalar-parser
distinction (`np.bool_(True)` is one). Existing scalar overflow, mode dispatch
and ordered `__torch_function__`/`NotImplemented` rules apply to the real native
`add_` descriptor. Positional alpha and `x2` are unsupported.

The receiver must have no gradients. Nonempty layouts must be non-overlapping
and dense, independently of rank, dimension order or singleton strides.
Dense offset subspans are supported. Empty tensors perform no pointer arithmetic
or kernel launch, including views whose offset exceeds the allocation extent.
CPU/meta receivers, tensor operands and nonempty holey/overlapping layouts are
unsupported. This does not add compiler-side mutation support: an `add_` inside
a default-compiled function still rejects. Public `add_` can mutate an admitted
input-rooted view returned by that function; materialized outputs stay independent.

## Ownership and completion

This is a CUDA-only exception to immutable shared numerical storage. CPU storage
and autograd saved values retain their existing contract. The mutation follows
the Python → Tensor → Storage → CUDA allocation ownership path, preserving the
same allocation, storage Arc, aliases and metadata. It uses a separate mutation
executor, outside the out-of-place `BinaryOperation` executor. Existing
`launch_add` and `unary_output` contracts are unchanged.

Argument conversions and their callbacks finish before mutable native borrowing.
The existing density predicate and separate checked allocation-interval bounds
precede launch. A dedicated 64-bit grid-stride entry in the existing context-keyed
primitive module performs `add.rn.f32`, without FTZ or a zero-add shortcut. There
is no result allocation, CPU/PyTorch round trip, new allocator or module cache.

The allocation owner and device guard survive explicit legacy-stream completion
after every launch attempt. Success means completion. Launch errors take precedence
over completion errors; either failure quarantines the existing allocation cache.
This native operation issues no writes on pre-launch rejection. That guarantee
does not undo user callback effects or concurrent work. An uncertain post-launch
failure promises neither rollback nor usable contents. Concurrent mutations have
no additional ordering guarantee.

## Author verification

This revision is author-only, based on published source
`41499a26015380a5d6c09dae80e0a0b9f7354dbf`. The accompanying
[developer receipts](diagnostics/cuda-add-inplace-author.tar.gz) record dirty
author sources, commands, hashes, local wheel/import identity and complete logs,
including first failures. They are not clean-commit measurements, an independent
review, canonical evaluation or publication approval. Prior transpose measurements
remain unchanged and do not measure this revision. Any subsequent clean-commit
evidence refresh requires separate operator admission after Burner commits.

The release wheel was built with an explicit interpreter and installed with
`pip --isolated` into a fresh worktree-owned Python 3.12 environment after checking
its prefix and installation destinations. Its base interpreter is a read-only
Python 3.12.12 installation; the venv prefix, packages, wheel, native extension,
build outputs and writable caches are inside this worktree. The shared Python
3.10 environment mentioned in earlier receipts was neither installed into nor
repaired. The repository import verifier passed.

| Check | Local outcome |
| --- | --- |
| Focused public `add_` suite, final release wheel | 15 passed (10 GPU, 5 portable) |
| Existing add/sub, CUDA add/mul and compiled-transpose regressions | 72 passed, 6 expected skips |
| CUDA-hidden `add_`/add/sub tests | 17 passed, 10 explicit GPU skips |
| Native scalar-add geometry, ownership, faults and GPU tests | 6 passed |
| Native graph metadata/conversion/refcount rollback tests | 17 passed |
| Linux all-targets Clippy, warnings denied; `cargo fmt --check` | Passed |
| Release wheel build, isolated install and repository import verifier | Passed |

Initial lint failures (scoped unsafe FFI and test lints), two Python fixture
errors (unsupported CUDA factory spelling and meta error type), and one native
fixture error (setting a leaf flag instead of the view's effective gradient flag)
are retained alongside their corrected runs. PyTorch's reference compilation
emitted its existing `torch.jit.script_method` deprecation warning; it did not fail
the tests and is distinct from the corrected native Clippy failure.

GPU checks use only GPU0 under the existing gpu/cpu-heavy lease. They cover
rank/permutation/offset aliases and sentinels, exact receiver and shared storage,
retained lifetimes, repeated calls and long tails, empty offsets, IEEE signed zero,
subnormals, NaNs and rounding, invalid-call bitwise nonmutation, callback ordering,
cold threads, independent-stream visibility and default-compiled returned views.
Native-only subprocesses block PyTorch imports and Python copy/materialization
methods; Rust counters independently assert no new device-storage allocation or
host/device transfer during mutation. Portable native tests inject launch and
completion errors without device faults or OOM, checking owner lifetime, error
precedence and quarantine.

The refcount correction uses `pyo3::ffi::Py_REFCNT` on live borrowed pointers,
preserving the conversion/rollback test's ownership and refcount semantics.
Linux all-targets Clippy with PyO3 0.29.2 and warnings denied passes. This addresses
the supplied macOS CI deprecation failure (run 35277569306, job 105391744185);
local Linux checks do not reproduce or establish the macOS CI result.

Device-state checks cover GPU0 success, rejection, empties and cold threads.
Cross-device restoration was not exercised because no second GPU was reserved.
Fault ordering uses portable injection rather than real device failures. NaN
classification is compared; NaN payload equality is not claimed. No performance
measurement, speedup claim or canonical evaluator run is part of this author step.
