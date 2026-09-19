# Compiled alias mutation: QA correction

Developer validation of dirty author sources based on
`23316fba1f1fbf5df9a4b88ede0771a6f734b2f7` (2026-09-18), not a clean-commit
capture or canonical evaluation. The [compiler guide](../compile-pointwise-jit.md)
remains the capability contract. [Earlier alias evidence](default-alias-mutation-20260918.md)
is unchanged, including its 8/16 strict reference identity failures and warm
signed-zero reference divergence; this correction does not waive those limits.

The existing lowerer now records unary conversion independently of sign parity
and the old scalar type, then applies bool-to-int conversion only to a current
exact bool. Python 3.10 `CALL_FUNCTION_KW` uses the existing keyword stack path.
Result recipes preserve original immutable integer tuples from `co_consts`,
while dynamic containers remain fresh. Numerical specialization policy and
native execution are unchanged. New native witnesses distinguish late valid
packing from alias/no-op nodes before writes, and assert hostile conversion
callbacks are never invoked.

## Validation

The archive contains commands, statuses, logs, source snapshots and hashes,
release wheels and RECORD/import checks, both Python-version receipts, and
untimed reference characterization. Tests ran against the source-owned release
wheel. A byte-verified test/source copy under the QA directory confines existing
fixtures' hard-coded output paths there. Unchanged timing diagnostic consumers
were not rerun; no timing campaign or canonical evaluator was invoked.

Pre-fix checks used the prior local wheel, whose installed Python package was
verified byte-identical to held F; that reused wheel is not called a fresh F
build. Its failures reproduce the scalar and tuple defects and missing keyword
opcode. Initial regression fixtures also had unsupported zero-argument helpers,
closure branch layout and cache assumptions; those failed commands are retained.
Zero-trip bodies retain their existing observed-source guards, so changed values
may legitimately require admission. A later fixture expected the wrong category
for boolean alpha; the canonical `RuntimeError` assertion was corrected without
changing production behavior. Python 3.10 initially lacked `typing_extensions`;
the local dependency repair and successful reruns are separate receipts.

| Check | Outcome |
| --- | --- |
| Native unit suite, GPU0 visible; Rust inputs identical to release snapshot | 308 passed |
| Selected GPU/default numerical, alias, guard/cache/result and explicit-eager suites | 579 run, OK, 15 skipped |
| CUDA-hidden selection plus quickstart checks | 591 run, OK, 302 skipped |
| Actual Python 3.10 keyword execution on GPU0 | 9 valid calls; 5 invalid forms rejected before writes |
| Python 3.10 portable alias admission suite | 6 passed |
| Release wheel RECORD/import/source verification; all-targets Clippy with warnings denied; formatting | Passed |

The untimed public-default tuple characterization records equal tuple values on
both frameworks. Native preserves original `co_consts` identity across warm calls,
reset and code replacement. PyTorch 2.13 preserves within-call sharing and warm
specialization reuse, but does not return the original constant-pool object and
changes tuple identity after reset. This is one observed program, not general
compiled parity. No timing samples were collected in this correction.

The owned environments used Python 3.12.14+meta and 3.10.20, NumPy 2.5.1 and
2.2.6 respectively, reference PyTorch 2.13.0+cu130, and Rust 1.92.0. GPU0 was H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. Selected CUDA runtime
and NVRTC were 13.0; nvcc inventory was 12.8 V12.8.93, not the pointwise compiler.
Loaded library hashes and pre/post GPU snapshots are retained. Linux/GPU0 checks
do not establish macOS, Python 3.14 or multi-device results; idle snapshots do not
establish exclusivity.

## Environment boundary incident

I set `UV_PYTHON_INSTALL_DIR` but omitted `UV_PYTHON_BIN_DIR` for the Python 3.10
download. `uv python install` also wrote `/home/bobren/.local/bin/python3.10`,
pointing into this QA directory, violating the worktree-only boundary. Its prior
state is unknown. The command, observed link and notification are preserved in
`evidence/setup-boundary-incident.txt`; no external repair/removal was attempted.
Operator cleanup is required before the worktree is removed. Subsequent commands
set both directories and use explicit local interpreters. No shared Python
installation was installed into or repaired.

## Artifact custody

All new raw receipts are under `target/alias-mutation-qa`. The archive manifest
records member paths, sizes and SHA256s; it excludes environments/build caches
and does not nest the prior archive. Main must copy and verify this archive
outside the disposable worktree. Dirty author validation does not approve the
candidate or replace independent review and a separately admitted release.

| Artifact | SHA256 |
| --- | --- |
| `target/alias-mutation-qa/alias-mutation-qa-author.tar.gz` (8,347,457 bytes; 111 verified payloads) | `a83434670c49aa517af46583db136bc8885e144fe4d110eb185c76b9551179df` |
| Archive `MANIFEST.json` (member paths, sizes, content hashes) | `e25e6382077be8c19a5f9340ddfa3e737c2d81f33e40a4a4b9d8f54e51b17030` |
| `target/alias-mutation-qa/evidence/source-verification.json` | `2b72ffe7b48d504b0cfd7cbfa5cad8dfc36166b45e898d26937f51416330182b` |
| Release wheel in `target/alias-mutation-qa/wheels/final/` | `efbc80ca2ea922ebce7efabdeacb55d0c94670eadb6e3c1e03d5b97d6c517528` |
| Loaded `torch_rs.abi3.so` | `e6fe9b6f3b4eb232dc61b5c515a5c956828bb78723673972596f24b8b171e1df` |

Archive paths are worktree-relative. `target/alias-mutation-qa/evidence/` contains
the named command receipts; `author-final.patch`, the source snapshots and the
hash-verified failed-test fixture snapshots retain the actual tested revisions.
