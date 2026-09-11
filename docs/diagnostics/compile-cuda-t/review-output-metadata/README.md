# Repeated-output metadata review repair

The reviewer reproduced a cached output-validation bypass in
`y = [x.t()]; return y, y`: the second declaration could contain
`shape=(True, 1)` and still reach native execution. Output identity reuse now
has separate validation bookkeeping keyed by output/metadata object identities.
Every distinct pairing is checked; valid repeated containers retain identity.

These are **development diagnostics**, measured from modified sources on
`48d88b94c328336517c261653d131994c874452a`. The [release receipt](release/build-record.json)
and [audit](audit.json) bind the actual source/diff hashes, installed Python,
wheel and native extension. The earlier [clean capture](../postcommit-e01f1d0f/README.md)
measures `e01f1d0f` before this repair and does not validate it. A fresh clean-code
capture remains required after Burner commits the repair; this agent does not
commit. Earlier baseline, development and clean-capture records remain unchanged.

## Reproduction and verification

- [Failing reproduction](repeated-output-red.log): all eight static/dynamic and
  list/tuple/nested-container variants accepted malformed second-occurrence
  metadata on unchanged production sources. This expected failure is preserved.
- [CUDA t suite](compiled-t.log): 11 passed, one two-device skip. The added
  cached regression checks 48 malformed declarations (boolean/float shape,
  boolean stride, float offset, integer gradient flag and wrong device),
  asserting zero native bridge calls and no recapture. Independently allocated
  valid metadata trees preserve container/tensor identity and match PyTorch.
- [Existing compiler regressions](compiler-regressions.log): 201 passed, eight
  two-device skips. CPU/CUDA arithmetic,
  packing, metadata, static analysis, cache guards and the unchanged corpus.
- [CUDA-hidden portability](no-device.log): one passed, 11 hardware skips.
- [Installed extension verification](imports.log), [preflight](preflight.log)
  and [docs regressions](docs.log) (12 tests) passed.

The existing build helper produced a fresh locked/offline release wheel in an
empty target, using the canonical worktree-local `.venv` and its locked
Python 3.12.14, NumPy 2.5.1 and PyTorch 2.13.0+cu130 dependencies. Dependency
caches were reused. Runtime/JIT/Python caches started empty for this capture.
H100 checks used only `CUDA_VISIBLE_DEVICES=0`; hidden-device checks used an
empty mask. Native and reference CUDA runtime versions were 13.0, driver
580.82.07; nvcc 12.6.85 was installed but unused (embedded PTX uses driver JIT).
Receipts preserve commands, timestamps, physical GPU snapshots and native
hashes. Concurrent correctness runs are not performance measurements.

[Capture](capture.py.txt) reuses the existing repository provenance helper;
[publication audit](publish.py.txt) verifies source/build/import consistency,
retains the expected failure and checks earlier evidence inventories unchanged.
[Measured inputs](measured-inputs.json) serve every successful receipt. Overlay
[red-inputs-delta.json](red-inputs-delta.json) on that mapping and serialize with
`json.dumps(mapping, indent=2, sort_keys=True) + '\n'` to reconstruct the red
receipt's input manifest exactly. Only the compiler source hash differs.
[Environment](environment.sh.txt) and build logs retain actual worktree paths;
wheels, environments, binaries and caches stay untracked under this worktree.

No Rust/native code, dependencies, workload matrices, scoring definitions,
unsupported outcomes or Burner-managed progress artifacts changed. Unrelated
Rust, multi-device and performance suites were not repeated for this Python
validation repair. These diagnostics do not replace independent review or
Burner's clean-commit evidence and merge gates.
