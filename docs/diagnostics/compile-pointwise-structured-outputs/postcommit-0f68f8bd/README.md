# Clean structured-output capture at 0f68f8bd

Measured implementation: `0f68f8bde413605572ebe1aa878b474dbb90a5d9`, against main `30a3b504ef4d43bf2958998cc39545996cc09970`. This refreshes the current-candidate evidence after the producer-identity, predicate-provenance and optional diagnostic-listing changes. Earlier captures remain pinned to their original revisions.

Clean status was recorded before and after each measured command. All 6,803 tracked-file fingerprints matched the initial snapshot after every command; source-subset fingerprints also matched before and after. These are boundary observations, not continuous monitoring.

## Results

The committed tests used a fresh source-bound release wheel and ordinary `torch_rs.compile(fn)` versus ordinary stock `torch.compile(fn)`. Existing tolerances, signed-zero/nonfinite assertions, persistent histories, alias checks and no-original/helper-replay checks were unchanged.

| Check | Final result |
| --- | --- |
| Pointwise suite on H100, physical device 0 | 311 total: 303 passed, 8 two-device skips |
| Required two-device checks, physical devices 0 and 1 | 8 passed, no skips |
| Release native pointwise/ownership checks | 63 passed |
| Python conversion-failure ownership | 1 passed |
| Portable suite, each CPython 3.10–3.14 | 149 passed, 162 hardware skips |
| Clippy, formatting, native-extension verification | Passed |
| Installed-wheel identities, all five interpreters | Passed |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware skips are not GPU passes. Four initial portable runs and the first interpreter-identity probe failed to import the declared `typing-extensions` dependency after a no-dependency wheel install into fresh environments. Installing the same locked dependency version fixed setup; complete reruns passed. All five original failure logs and command records remain in the archive. No source, dependency definition or test was changed.

All 720 duplicate-product comparisons and all 16 new independent/reused-producer comparisons passed. The latter keep native executors alive across return-order changes, exercise two input histories and repeated calls, and check prior-output freshness. With `x=1e10`, `y=1.0000001192092896` and the committed long nonlinear chain producing `r`, the observed `q` values were:

| Product relationship | Return order | Native q | Stock default q |
| --- | --- | ---: | ---: |
| Independent `q=x*y-1e10` | `(p,r,q)` | 1192.0928955078125 | 1192.0928955078125 |
| Independent `q=x*y-1e10` | `(q,r,p)` | 1024.0 | 1024.0 |
| Reused `q=p-1e10` | `(p,r,q)` | 1024.0 | 1024.0 |
| Reused `q=p-1e10` | `(q,r,p)` | 1192.0928955078125 | 1192.0928955078125 |

`numerical-summary.json` links every producer comparison to its original raw file and hash. The existing large-graph, shape-history, helper/loop/branch, metadata, storage, cache and failure-atomicity checks also passed. These bounded results do not establish general Inductor equivalence or approve the branch.

## Provenance and retention

Wheel SHA256: `224abab9e70ab0b6c27041ce11eb6b7d60e7cddd037d8b3a36df288147fdb1ba`.

Local environments and the Cargo registry were recreated inside this worktree from read-only installed inputs. `uv sync --locked --no-install-project --group dev --group reference` validated the main environment against the existing lock. The locked offline release build used a fresh Cargo target. Reference, Triton and CUDA caches were initially empty; later checks reused the report-local caches. Records identify the actual executables and imports, Rust 1.92.0, Maturin 1.14.1, PyTorch `2.13.0+cu130`, H100 UUIDs, driver 580.82.07, NVRTC 13.0 and CUDA runtime 13000. PATH `nvcc` reports 12.6.85; native kernels used NVRTC. The runtime probe executed native compilation before importing PyTorch and verified the installed wheel's members against both the wheel and current Python sources.

[Measurements](measurements.json.gz) preserve commands, timestamps, source fingerprints, status, environments and exit codes, including initial setup failures. [The full raw archive](raw-captures.tar.xz) preserves all 45,469 capture files: wheel, committed compiler/test sources, compiler caches, numerical JSON, generated native CUDA/PTX, diagnostic plan listings, logs and provenance. Plan listings are reconstructed through the validated lowering; they are not captured GPU instruction streams. Identical file contents use TAR hard links. [The manifest](raw-manifest.json.gz) verifies every archived file's bytes; [verification](verification.json) records archive hashes and completed checks.

The original report tree remains at `target/default-compile-eval/structured-outputs-postcommit-0f68f8bd/`. Its retention manifest covers 45,468 files; the checked-in archive additionally contains that manifest. Only rebuildable Cargo output is excluded. This capture is included in the candidate diff for Git retention: no external copy or automatic observer coverage is assumed, and no cleanup was performed. The [historical custody limits](../README.md#provenance-and-retention-limits), including the review7 gap, remain unchanged.

Only evidence and its index/guide links change in this step. Implementation, dependencies, tests, benchmark harnesses, evaluation definitions and managed progress artifacts remain unchanged. No official scoring corpus or unrelated repository suite was rerun; no performance or coverage score is claimed.
