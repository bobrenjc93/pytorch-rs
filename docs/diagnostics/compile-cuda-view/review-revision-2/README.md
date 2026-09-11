# CUDA view second review revision diagnostics

This development capture measures the unary-expression and element-count fixes
on top of `d634f3670d6365157e8a854231c02ae81d80a46d`. The revision is uncommitted:
Burner must commit it before a separate fresh clean-commit capture. The prior
[clean capture](../postcommit-e426d1f6/README.md) remains pinned to `e426d1f6`;
all historical measurements and failed attempts are unchanged.

The [reproduction](reproduced-findings.log) records all five reported mismatches
before edits on clean HEAD. Native/reference eager return TypeError or
RuntimeError, while all four compiler policies reject capture prematurely.
The baseline installed wheel matches the preceding clean implementation;
HEAD's intervening changes contain evidence and documentation only.

Unary negation now retains bounded exact integer values solely for public-error
validation, without invoking user conversions. Computed dimensions, including
unused or overwritten results, still reject capture. Frontend metadata uses the
existing native checked resolver to diagnose known element-count and ambiguous
inference errors before exact-integer/rank admission. Exact later booleans are
normalized only for validation. Valid boolean/higher-rank shapes remain
unsupported; the native whole-graph payload restrictions are unchanged.

Regressions compare exact exception types and shape diagnostics against both
eager implementations under every policy. They repeat invalid calls, verify
empty caches and prohibit original-body, native-graph and per-node Python
execution. Hardware-free tests cover frontend metadata, native shape validation
and a user object whose negation/index conversions must never run.

| Check | Result | Log |
| --- | --- | --- |
| Complete current-source `test_compile*.py` / `test_top_level_compile.py` sweep | 624 passed; 13 device skips | [compiler](compiler-current-source.log) |
| Seeded H100 view suite and exact-error regressions | 21 passed; 1 device skip | [view](view-focused.log) |
| CUDA-hidden view/reshape/reference and metadata | 80 passed; 16 hardware skips | [portable](portable.log) |
| View/reshape context restoration on GPUs 0,1 | 2 passed | [devices](two-device.log) |
| Rust graph planning / CUDA storage boundaries | 13 / 7 passed | [graph](rust-graph.log), [storage](rust-storage.log) |
| Rust default all-targets / Python bindings library, CUDA hidden | 395 / 205 passed | [default](rust-default.log), [bindings](rust-bindings.log) |
| Clippy default / Python bindings, warnings denied; formatting | passed | [default](clippy-default.log), [bindings](clippy-bindings.log), [format](format.log) |
| Native import/source verification; guide examples | passed | [imports](native-verification.log), [examples](guide-examples.log) |
| README/docs smoke after evidence publication | 12 passed | [docs](docs-final.log) |

The [release record](release/build-record.json) and [audit](audit.json) bind the
source, native extension, wheel, interpreter, inputs and exact commands. Source
SHA-256: `92c6837dd4dd350390e37e047971efe146bcc5db1823646c9d86ad60937f5756`.
The [source/test patch](source-and-tests.patch) reconstructs this revision from
HEAD. The complete compiler sweep follows the final source and test edits.
This bundle makes no performance or scoring claim.

The [environment](env.sh.txt) reuses the canonical `.venv`, worktree-local
uv-managed CPython 3.12.14 and locked dev/reference dependencies and caches.
The repository's `scripts/capture_depth_concat_build.py --allow-dirty --output target/view-review2/release`
builds and installs the exact-source release wheel with a fresh empty target,
Rust/Cargo 1.92.0, locked offline compilation, thin LTO and one codegen unit.
All build/import/interpreter/cache paths are inside this worktree; installed
system compilers and the NVIDIA driver are inspected read-only.

[Preflight](preflight.log) records PyTorch 2.13.0+cu130, NumPy 2.5.1, native and
reference CUDA runtime 13.0, and driver 580.82.07. Ordinary tests use H100 GPU 0,
UUID `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; restoration checks add only GPU 1,
UUID `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. nvcc 12.6.85 is installed but
unused; execution uses driver-JIT PTX and runtime cuBLAS. Receipts record GPU
identity, utilization and memory before/after execution. These snapshots do not
reserve devices.

The [recorder](capture.py.txt) preserves command output and checks unchanged
source, inputs, native binaries and git status during each command. Exact
[GPU](validate-gpu.sh.txt) and [Rust](validate-rust.sh.txt) commands and the
[publication audit](publish.py.txt) accompany the receipts. The audit verifies
all 59 installed Python files and all 249 historical artifacts. The separate
docs receipt follows publication with the same production sources and inputs.

The [input index](input-manifest-index.json) reuses the existing baseline delta
and records only new replacements/removals against the
[original manifest](../measured-inputs.json). Apply the indexed delta and serialize
with `json.dumps(mapping, indent=2, sort_keys=True) + '\n'` to reproduce each
receipt's manifest SHA-256. The [inventory](inventory.json) hashes this bundle.
All commands in this revision passed; the reproduction deliberately asserts the
pre-fix failures. Earlier failed attempts remain in the historical bundles.
