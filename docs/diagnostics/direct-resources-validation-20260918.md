# Committed-source direct-resource validation

Developer validation, not a score, release approval or new runtime optimization.
This pass changes only regression tests and live documentation. Production code
remains equivalent to held **D `781d9b1ffa3ac8fae76087e479bbaa905c431b55`**;
`production-equivalence.json` and the test-only patch record that boundary.
The [earlier dirty four-variant experiment](direct-resources-20260918.md) remains
unchanged and is not pooled into these observations.

Separate fresh release wheels were built from byte-verified Git exports of
**B `60202557b4f110d07777f585e804ab5f55e1ff7b`**, **H
`a9045e8b14dd5ee63ff19537860514227c379692`**, and **D**. These are committed
source captures; the enclosing worktree had the separately recorded test/docs
edits. Each wheel's RECORD, exported frontend and loaded native identity were
verified. No shared Python installation was changed.

The unchanged public full-call diagnostic ran its eight programs/all shapes,
fresh and reused inputs, both framework orders in separate processes/caches,
five warmups and 17 synchronized single-call samples. Input creation/readback
were excluded; wrapper creation and first-shape calls were recorded separately.
Native processes blocked PyTorch imports; warm body-replay probes and selected
owner/preparation histories were outside timing. All six strict comparisons
passed: **420 paired cells**, **876 successful capture cells**, **14,892 samples**,
and **36 retained B view rejections**. There were no capture errors. Saved arrays
contain no zero, NaN or infinity values; exceptional-value coverage comes from
the separate tests, not these arrays or their vacuous signed-zero checks.

GPU0 was H100 `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07,
under the owner's lease with `CUDA_VISIBLE_DEVICES=0`. CUDA runtime/NVRTC were
13.0; nvcc 12.8/V12.8.93 was inventory, not the pointwise compiler. Python was
3.12.14+meta, Rust 1.92.0 and reference PyTorch 2.13.0+cu130. Exact commands,
statuses, providers, pre/post device snapshots and source/build/import hashes
are retained in the bundle. Idle snapshots do not establish exclusivity.

## Finite observations

Unweighted geometric means of baseline/D per-cell median latency ratios over
common successes in both orders (greater than one means lower D latency):

| Comparison | Fresh | Reused | Slower D cells, fresh / reused |
| --- | ---: | ---: | ---: |
| B / D | 1.107× | 1.138× | 2/58 / 4/58 |
| H / D | 1.015× | 1.006× | 26/76 / 30/76 |

D's matched reference/native summaries were 1.475× fresh and 1.385× reused.
These finite descriptive summaries establish neither general parity nor a cause
of an earlier score change. The small H differences do not establish a reliable
broad speedup. All slower and cold rows remain in `all-relative-cells.json`:
D was slower on first-shape calls in 75/116 common B cells and 104/152 H cells.
For example, the empty polynomial reused cell in native-second order took
35.123 µs steady and 115.843 ms on its first-shape call, versus H's 25.048 µs
and 0.455 ms. First-shape calls include required compilation/preparation; they
are not isolated compiler timings. Actual executable objects and identical PTX
hashes are counted separately; graph-free receipts remain `None`. Receipts do
not independently trace device execution.

## Checks and replay

- Native: **301 passed**. The first run had 300 passes and one failure because
  the new child test used the wrong Rust module filter; its reaching assertion
  caught zero selected tests. The corrected full rerun passed. Both logs remain.
- Real GPU NVRTC default, soname and absolute-pin configurations: **8 tests
  passed each**, retaining all compiler-failure/no-fallback/cache assertions.
- Public-default GPU regressions: **213 passed, 4 skipped**. CUDA-hidden
  pointwise: **232 passed, 197 skipped**. Portable fixture/docs/replay:
  **28 passed, 5 skipped**.
- Formatting, Linux Clippy with warnings denied and diff checks passed. This
  does not reproduce macOS CI. An initial source-audit assertion looked for the
  test include in `lib.rs` instead of `tensor_pointwise.rs`; that failed audit
  and correction are retained separately.

The [bundle](direct-resources-validation-20260918.tar.xz) contains all three
source/wheel sets, arrays, timings, owner histories, test sources and complete
receipts. The [verification receipt](direct-resources-validation-20260918-checks.json)
records its hash and relocated replay with the original capture directory
unavailable. Use the existing archive-local helper and unchanged comparator:

```sh
mkdir -p target/direct-validation-replay
tar -xJf docs/diagnostics/direct-resources-validation-20260918.tar.xz -C target/direct-validation-replay
.venv/bin/python target/direct-validation-replay/profiled-call/bundle.py restore \
  target/direct-validation-replay/profiled-call \
  docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz
.venv/bin/python scripts/diagnose_compile_full_call.py --compare \
  target/direct-validation-replay/profiled-call/held-reference-second.json \
  target/direct-validation-replay/profiled-call/held-native-first.json
```

The immutable base archive remains required by the existing helper. The receipt
reports actual embedded/mapped array counts. Original absolute paths remain
provenance, not replay dependencies. Independent review and any release or
canonical evaluation remain the owner's separately admitted workflow.
