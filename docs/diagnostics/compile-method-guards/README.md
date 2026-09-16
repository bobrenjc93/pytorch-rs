# Method-guard grouping check

The default compiler still checks the same 19 direct bindings on each of the
original Tensor classes, in class-major order. Grouping obtains two live class
dictionaries per successful call instead of 38. This is an operation-count
reduction, not a demonstrated end-to-end speedup. Native math, admission,
cache transactions and fixed evaluators are unchanged.

## Checks and measured sources

The new guard suite passed seven tests: all 38 mutable owner/name pairs,
282 mutation/state cases, 564 rejection checks and 38 retained-exception
lifetime cases. No pair was skipped. The focused native/default suites passed
87 tests with two explicit multi-device skips under the GPU0-only lease.
An initial broadcast-trig import failure was resolved by supplying the existing
tests-directory import path; both invocation logs remain indexed. Native-only
setup installed no PyTorch and passed all 12 README smoke tests. An initial
Maturin invocation rejected inherited `CONDA_PREFIX` alongside `VIRTUAL_ENV`;
the successful build unset that variable for the command.

The original is the retained, clean X4BGHb build of `162abef6f497341327e4ac3f5e412f92a4e3bf81`.
The changed wheel was built normally from that commit **plus the uncommitted
guard and onboarding edits**. It remains development evidence. The required
[clean-commit refresh for 9009c8ea](../compile-gelu-linked/postcommit-9009c8ea/README.md)
now records the fixed gates, public GELU and frozen guard comparison separately.
No final score is claimed for this development run.
The exact paths, commands, exits, source and installed-file hashes are in the
[declaration](declaration.json) and retained records indexed by [raw-index.json](raw-index.json).

| Identity | Original | Changed |
| --- | --- | --- |
| Wheel SHA256 | `1f3998184e4e5a542ff480c111e00a241b9167ff35917e9c4825a12fe0f55bf8` | `c21540d964bee7ee02eeb62c207119fb3c96d291353af4b372aa905f177a6fbb` |
| Native SHA256 | `b9db6f35d6129c9681b6d47e190558c5b1897ce997ad1c6c7dbe011249f420c2` | `d335da08693751bc75c187016cae210d45b789903f7d52a3310be68ea55b1ca0` |

Both interpreters are Python 3.12.13, SHA256
`202c17d1671602a4ef1d43e9b2fdbef0769443f37bf5e51f6b603e0b2c27d9d8`.
Third-party dependency metadata matches; native binaries and torch-rs package
metadata differ. This build-level confound prevents grouping-only attribution.
The changed build used Rust 1.92.0, locked release Maturin and the repository's
`evaluate_torch_compile_default.sh --setup-only` flow, which produces no score.
Execution used H100 GPU0 `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver
580.82.07, CUDA runtime 13000 and explicit CUDA 13 NVRTC. Actual library hashes,
versions and before/after GPU inventories are retained per worker.

## Frozen public-call result

The [driver](measure.py) ran the prescribed original/changed/changed/original
sequence once, without profiling. Each worker kept one ordinary public compile
wrapper over five shapes for `x.sin() + extra`, with five warmups and 17 samples
per shape. All 340 samples, first-call costs, MADs and ranges are in
[timing-comparison.json](timing-comparison.json) and the indexed worker records.
All 20 warm cache snapshots were unchanged. Separate uninterrupted
original/changed/default-PyTorch correctness workers passed
[56 output/retained comparisons](correctness-comparison.json); the timing run's
40 first/last output comparisons also passed the unchanged numerical policy.
Intermediate timed outputs were not numerically recaptured.

| Matrix shape (history order) | Original 1 median µs | Changed 1 | Changed 2 | Original 2 | Changed/original paired ratios |
| --- | ---: | ---: | ---: | ---: | --- |
| 128×256 | 37.476 | 34.372 | 35.423 | 36.325 | 0.9172 / 0.9752 |
| 193×256 | 39.439 | 36.435 | 36.475 | 37.246 | 0.9238 / 0.9793 |
| 202×118 | 38.468 | 36.425 | 42.965 | 36.916 | 0.9469 / 1.1639 |
| 207×292 | 38.478 | 35.864 | 36.645 | 37.156 | 0.9321 / 0.9862 |
| 128×256 return | 38.128 | 36.766 | 35.874 | 37.176 | 0.9643 / 0.9650 |

Most paired medians decreased, but 202×118 reversed by order, including a
16.4% slower changed leg. Two workers per version and differing native bytes
do not establish a causal speedup or a precise confidence interval. There was
no favorable-order selection or resampling. Full unchanged canonical gates
still decide qualification; the earlier rejected performance scores stand.

## Reproduction and retention

Use the pinned original source/interpreter/wheel identified by the declaration,
read-only. Build the changed checkout with the locked setup-only command above.
Set the following paths to those actual installations and a fresh worktree-local
run directory; do not copy changed files into the original package:

```bash
python docs/diagnostics/compile-method-guards/measure.py declare "$run" \
  --original-source "$original_source" --original-python "$original_python" \
  --original-wheel "$original_wheel" --changed-source "$PWD" \
  --changed-python "$changed_python" --changed-wheel "$changed_wheel" \
  --cudart /usr/local/cuda-13.0/lib64/libcudart.so.13 \
  --nvrtc /usr/local/cuda-13.0/lib64/libnvrtc.so.13
python docs/diagnostics/compile-method-guards/measure.py correctness "$run"
python docs/diagnostics/compile-method-guards/measure.py timing "$run"
```

Run guard, native and GELU checks before timing, with no concurrent GPU work.
The driver rejects changed source/package identities within and across workers.
The checked-in comparison/declaration files are byte-for-byte copies of the
generated records. The digest index retains references to raw float32 bits,
complete worker reports, process receipts and build/test logs under
`target/guard-grouping`; the canonical worktree and builds remain retained.
Cache trees and duplicate raw tensors are not added to the tracked tree.
Prior GELU archives remain unchanged. This bounded audit is not independent
full code review, fixed evaluation or merge approval.
