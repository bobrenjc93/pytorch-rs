# Validation composite integration

This composite preserves the three source PRs based on
`28fb6b923843989f9536608a75f3d240ce4c8a7e`:

| Source PR | Included tip | Intent |
| --- | --- | --- |
| #1926 | `78a9589c006c675496d0910c63c92bd57e520891` | [Factory parity across cache states](factory-kwargs-cache-parity.md) |
| #1927 | `a7facaede036d9aba72f7b417aa9db5bce15c818` | [Six fixed CUDA transfer cases](hardware-heterogeneity-evaluator.md#fixed-cuda-float32-devicetransfer-denominator) |
| #1928 | `4450c1a0c43fdc6c71054a41d7850b00f548be64` | [Original #1924 validation and paired timing history](diagnostics/compile-cuda-neg/pr1924-history.md) |

All three tips are ancestors of integrated HEAD
`030a11a7f2c85e39b6fbb0471130fc2208e0706e`. Before the documentation clarification,
every included file matched its source PR byte for byte. Production Python and
Rust code, dependency manifests, and lockfiles match the starting main revision.
No source or test integration repair was needed. Independent review identified
one documentation ambiguity: #1926's validation narrative needed attribution to
its original PR rather than the composite. That attribution is now explicit;
the original measurements and all 151 archived #1924 files remain unchanged.
Burner retains ownership of delivery and source-PR reconciliation upon merge.

## Combined-tree checks

Checks ran on 2026-09-10 UTC from clean integrated HEAD above, rooted at
`/data/users/bobren/a/pytorch-rs-burner/.burner/worktrees/composite_4b8e7f80`.
The attribution clarification and this report were written afterward; no
implementation, test, or benchmark-harness changes followed the checks.

The canonical `.venv` used CPython 3.14.5 and locked PyTorch 2.13.0+cu130.
`scripts/test-python.sh` built and installed a fresh release ABI3 wheel with
Rust/Cargo 1.92.0, `extension-module`, thin LTO, and one codegen unit. Its
provenance check resolved both `torch_rs` and the native extension inside this
worktree's `.venv`. Downloads, caches, builds, and temporary files stayed inside
the worktree. The full suite had no `PYTHONHASHSEED` override.

| Check | Result |
| --- | --- |
| `./scripts/test-python.sh` | 5,363 tests; OK with 13 skips; 688.972 seconds |
| Three factory test modules on system CPython 3.12.13 | 15 tests passed, including seeds 0–11 and all nine installed/source/bytecode function pairs |
| `tests.test_pr1924_historical_evidence` | All seven preservation, attribution, receipt, accounting, and link checks passed |
| `scripts/evaluate_cuda_transfers.py --seed 9173 --seed 260909` with a matching fresh build receipt | All six reference cases eligible at both seeds; four native cases passed, two remained unsupported |
| `cargo fmt --check` | Passed |
| Independent review | No source integration defects; documentation attribution finding addressed. Reviewer independently passed 16 preservation/accounting tests |

GPU checks used only `CUDA_VISIBLE_DEVICES=0`: NVIDIA H100, driver 580.82.07.
The transfer workers both mapped this worktree's
`.venv/lib/python3.14/site-packages/nvidia/cu13/lib/libcudart.so.13`, reporting
runtime version 13000. System `nvcc` was 12.6.85; the native extension build and
transfer evaluator did not invoke it. The evaluator's local extension was
verified byte for byte against the freshly built wheel, and source/evaluator/
matrix/extension hashes remained stable throughout the run.

The unsupported transfer cases were direct matrix zero allocation and
same-device copying, which stops at unsupported CUDA `clone()`. Both retained
their slots in the six-case denominator. These checks make no additional
capability or performance claim and do not replace Burner's ten evaluations.
Historical timing reports grant no current-candidate performance credit.
Local logs, environment/build records, and raw transfer observations are in
`target/integration/`; they are ignored working artifacts, not published timings.

## Separate baseline follow-ups

The same wheel was also installed in a worktree-local Python 3.12.13 environment.
All factory tests passed there. A separate 13-test buffer/stack-validator probe
with that environment temporarily at canonical `.venv` reproduced three known
failures:

- Two noncanonical boolean-buffer cases in `test_tensor_buffer_reference`.
- One stack-validator failure comparing unnormalized `lib64` paths with resolved
  `lib` paths. Native-extension provenance itself passed.

An earlier probe from the secondary environment additionally hit the validator's
canonical-`.venv` requirement; the canonical rerun removed that setup mismatch.
Both probe logs are retained locally. Python 3.14.5 was restored to `.venv`
afterward, and the Python 3.12 environment remains at
`target/integration/python312`. The affected production code, tests, and validator
are unchanged from main. These two baseline issues remain separate follow-ups;
neither assertions nor implementation were changed to suppress them.
