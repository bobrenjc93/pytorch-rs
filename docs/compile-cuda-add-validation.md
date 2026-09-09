# CUDA addition capture validation history

The durable usage and reproduction contract is in the
[CUDA addition capture guide](compile-cuda-add.md). The records below describe
specific validation runs; they are not performance or coverage scores.

## Composite evidence

The retained reports measure committed composite implementation
`606e1ff9d71033b425fd281c56a276bfbd31377c`, which includes both
marker-free CUDA addition capture and `torch.hstack`. They were regenerated
with the unchanged `scripts/diagnose_compile_cuda_add.py` in a clean detached
checkout at `target/validation` inside the composite worktree. Its HEAD and
`git status --porcelain=v1 --untracked-files=all` stayed unchanged and clean
through the release build and complete diagnostic batch. Reports were written
under ignored `target/` before being copied into `docs/diagnostics/`.

| Report | Cases | Reference eligible | Native passes | Explicit unsupported |
| --- | ---: | ---: | ---: | ---: |
| [Python 3.12.13, GPU 0](diagnostics/compile-cuda-add-h100-python312.json) | 82 | 80 | 56 | 26 |
| [Python 3.14.5, GPU 0](diagnostics/compile-cuda-add-h100-python314.json) | 82 | 80 | 56 | 26 |
| [Python 3.12.13, GPUs 0/1](diagnostics/compile-cuda-add-h100-multidevice.json) | 6 | 4 | 4 | 2 |

Every case, output fingerprint, reference eligibility, rejection phase, and
unsupported error matches the compiler source-branch reports exactly. Compared
with those source reports, only provenance changed: the measured commit,
combined Rust source hash, rebuilt extension hash, and local runtime paths.
All recorded source hashes match both the committed implementation
and the integrated files. All runtime paths resolve inside this composite
worktree; no leaf-worktree runtime paths remain in these reports.

The release abi3 wheel was built using Rust 1.92.0 and
`maturin build --release --locked`; its SHA-256 is
`3555fdd68a093aee9ba889f90033a3462c876f04c6a44547d2ca3a8aee338a09`.
All 59 packaged Python files matched the source and installed packages, and
the installed extension matched the wheel and all three reports. Canonical
native-extension provenance checks passed on both Python versions. The H100
driver was 580.82.07; reference PyTorch was `2.13.0+cu130`, and the selected
native runtime reported CUDA 13.0. Native addition uses NVIDIA driver JIT of
embedded PTX 6.0/sm_50; the available nvcc 12.6 compiler is not used by that path.

The frozen 38-program corpus, historical marked four-shape CUDA benchmark,
evaluation definitions and weights, and Burner-managed progress artifacts
are unchanged. These refreshed correctness diagnostics add no CUDA performance
claim and preserve every unsupported outcome.

## Post-commit evidence checks

After Burner committed the integration, the native release artifacts were
cleared and rebuilt in the clean measurement checkout, and the new wheel was
installed for both Python versions. All three diagnostics ran before any
retained evidence changed. Their JSON differs from the preceding composite
reports only in `base_commit`: every source digest, extension digest, case,
output fingerprint, reference eligibility, rejection, and runtime path is
unchanged. The new wheel includes the committed README; its metadata, build
SBOM, and archive record changed, while all packaged Python files and the native
extension are identical to the author-validated wheel.

On both Python versions, the focused CUDA boundary suite passed 11 tests with
one two-device skip under `CUDA_VISIBLE_DEVICES=0`; the device/cache/restoration
test then passed with `CUDA_VISIBLE_DEVICES=0,1`. Wheel/source/extension hashes,
canonical import provenance, and clean-checkout status also passed. This step
reran only evidence-related checks; the full-suite and Rust results below
remain the earlier author validation, including its baseline failure caveats.

## Composite validation checks

These checks were completed during author integration using wheel SHA-256
`1a2c669ecc55c91746f0d52354c394348947356541aef17ee8c767412dbbe05b`.
They are retained as historical validation, separate from the post-commit
measurement batch above.

- Python 3.14.5: the final full suite passed all 5,242 tests with nine skips.
- Python 3.12.13: the final full suite ran 5,242 tests with nine skips and only
  two failing noncanonical boolean-buffer subcases. Both failures also occurred
  with a release wheel rebuilt from clean baseline `e5a8a9c` on this GCC-built
  interpreter. With `PYTHONHASHSEED=6`, that baseline additionally reproduced
  the three factory-order failures described below; factory ordering passed
  in the final composite full run. No buffer or factory implementation or
  assertion was changed.
- Both Python versions passed the focused compiler/corpus, CUDA boundary,
  hstack, cat/vstack, and README suites: 174 tests with one two-device skip under
  `CUDA_VISIBLE_DEVICES=0`. All five selected compiler/addition/transfer device
  guard tests then passed on both interpreters with `CUDA_VISIBLE_DEVICES=0,1`.
- Rust formatting and Clippy with warnings denied passed. All-target tests
  passed without Python bindings (352 tests) and with them (363 tests).
  The seven standalone native CUDA tests also passed with the explicitly
  selected local CUDA 13 runtime under both GPU masks `0` and `0,1`.
- The README first-success example, CUDA guide example, supported-surface
  hstack examples, diagnostic CLI help, and 160 local Markdown file links passed.

The first Python 3.12 run also found obsolete documentation assertions and a
stack benchmark smoke-test provenance mismatch through the local virtualenv's
`lib64 -> lib` alias. The documentation assertions now require the bounded CUDA
contract and guide link, and the validation-history index is separate from
timing evidence. Removing that local environment alias gave canonical import
paths; all 15 documentation/provenance smoke checks and the final full-suite
stack check then passed. The benchmark harness and validators were unchanged.

## Source validation before composite integration

The following narrative was retained from compiler source commit `6c741cd`.
It describes the source branch wheels and reports before integration with
`torch.hstack`; its wheel hashes and test counts are historical, not provenance
for the composite wheel. The composite reports are regenerated separately.

The reports were regenerated from clean implementation commit
`278c9b1eb90f6d4d2bcdc518572c2d3992b41261` after it was committed by Burner.
The release wheel was rebuilt and installed for Python 3.12.13 and 3.14.5;
all three diagnostics ran before any retained report or documentation changed.
`git status --porcelain=v1 --untracked-files=all` was empty before the build and
after the complete diagnostic batch, with HEAD unchanged. Every recorded source
hash was checked against that commit. All 59 packaged Python files also matched
the committed source and installed package, and the installed native extension
matched the rebuilt wheel. The refreshed wheel's SHA-256 is
`87e0488be1bfd6103d0e7ac68a36188759562d1bd764d27abe46a2de9e7fee05`.
Case outcomes and output hashes are unchanged from the initial diagnostics;
the reports now reference the commit containing the measured implementation
and harness. No implementation or harness changes accompany this evidence
refresh.

The H100 differential reports contain 82 single-device cases per interpreter:
56 supported passes and 26 explicit unsupported outcomes, with 80 cases eligible
on reference PyTorch. The additional reference-eligible mixed-device case uses a
CPU scalar tensor and a CUDA scalar tensor, which PyTorch permits and this
bounded compiler rejects. The six-case two-device report has four eligible
passes and two mixed-ordinal rejections that are also reference-ineligible.
These are diagnostic counts, not a coverage score or an expanded denominator.

A host-specific baseline issue was also checked without changing this feature:
on GCC-built CPython 3.12.13, two noncanonical boolean-buffer subcases fail in
both this wheel and a release wheel rebuilt from unmodified base `e5a8a9c`.
The existing native buffer decoder uses the low bit; that matches Clang-built
CPython 3.12.12's memoryview behavior but not this GCC build's nonzero-byte
behavior. CUDA graph diagnostics pass on the GCC interpreter too. The stack
benchmark smoke test additionally requires canonical package paths and an
interpreter under `.venv`; managed validation environments were placed at
`.venv/compat312` and `.venv/compat314` to meet that existing contract. No buffer
implementation, benchmark validator, or scoring corpus was changed.

The managed Python 3.12 full run also exposed existing `nn.factory_kwargs`
dictionary-order comparisons: both implementations iterate a set of keys, and
three ordering assertions failed in that run. The unchanged baseline wheel
also reproduces an ordering mismatch with `PYTHONHASHSEED=6`; isolated reruns
can pass. The `nn.factory_kwargs` source and its tests are unchanged by this
branch; the ordering issue remains a separate baseline failure.

Final checks on the release abi3 wheel:

- Python 3.14.5: full suite, 5,230 tests, passed with nine skips.
- Python 3.12.12: full suite, 5,230 tests, nine skips; only the three baseline
  `nn.factory_kwargs` ordering assertions described above failed. All compiler
  tests passed. The focused compiler/unchanged-corpus run also passed (73 tests,
  one two-device skip under the single-device mask).
- Rust: formatting and Clippy with warnings denied passed; all-target tests
  passed both without Python bindings (352 tests) and with them (363 tests).
- All three H100 differential reports passed their declared supported/rejected
  outcomes. Six CUDA/compiler multi-device tests passed on Python 3.12; the
  compiler multi-device test also passed on Python 3.14 with the final wheel.
- Native-extension provenance passed for both managed environments. All 59
  packaged Python files matched the working tree and installed wheel; the
  installed extension matched the wheel. The wheel's SHA-256 is
  `d17b73a381e65ec95a7bc58f5d48fad7d352190ec2a2a6ed3a797f42cac4121c`.
