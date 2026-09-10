# Stack benchmark package-path provenance

The generated-shape stack validator accepts package aliases only when both the
recorded path and current import origin resolve to the same existing file inside
the current worktree's `.venv`. Fresh validator artifacts record canonical paths
for NumPy, PyTorch, `torch_rs`, and its native extension. An internal
`.venv/lib64 -> lib` alias therefore passes in either spelling.

Both lexical and resolved containment are checked. Parent/sibling/global
packages, aliases entering from outside, symlinks escaping the environment,
symlinked environments, parent traversal, missing files, directories, and
different files with identical bytes are rejected. The interpreter retains its
lexical `.venv` path: its symlink to an existing base Python is allowed, while a
different executable spelling is not substituted. The validator also checks the
current interpreter prefix, native loader, ABI suffix, `_C` file identity, and
package versions. Portable fixtures exercise these checks under `target/` and
skip clearly if the platform cannot create symlinks.

Only package-path handling changed. Script hashes, artifact versions, git
provenance, generated workloads, sampling/accounting, output/value checks, and
timing formulas retain their existing rules. Native build/provenance tooling and
production code are unchanged. No historical artifact was rewritten and these
smoke checks establish no performance credit.

## Same-code reproduction

On 2026-09-09, this job started at
`03075466b077c69ed1f7a4f047ebb33004d3306b`. Before edits, the benchmark, validator,
stack/buffer tests, production Python/Rust code, manifests, and lockfiles were
verified identical to main `28fb6b923843989f9536608a75f3d240ce4c8a7e`.
No prior worker environment was reused.

System `/usr/bin/python3.12` is CPython **3.12.13**, GCC **11.5.0**. A new canonical
`.venv` installed the locked dev/reference groups: NumPy **2.5.1**, PyTorch
**2.13.0+cu130**, and a freshly built `torch-rs` release wheel. Package origins
used `.venv/lib64/python3.12/site-packages`; resolving them traversed `lib64 -> lib`.
Native-extension provenance passed. The unchanged stack/buffer run executed
13 tests and reported three failures: the generated smoke artifact's four
`lib64`/`lib` path mismatches, and two noncanonical boolean-buffer subcases.

## Validation

Each interpreter is tested at the canonical `.venv` path with locked dependencies
and a worktree-local release build (Rust/Cargo **1.92.0**, `extension-module`,
`abi3-py310`, thin LTO, one codegen unit). Commands after environment setup:

```bash
uv sync --locked --no-install-project --group dev --group reference
.venv/bin/maturin build --release --locked --out target/wheels
uv pip install --python .venv/bin/python --force-reinstall --no-deps target/wheels/torch_rs-*.whl
.venv/bin/python .github/scripts/verify_native_extension.py
.venv/bin/python -m unittest tests.test_top_level_stack_benchmark_artifact -v
.venv/bin/python -m unittest tests.test_readme_quickstart -v
./scripts/test-python.sh
```

Cargo/uv caches, downloaded managed Python, compiler caches, temporary fixtures,
and logs stay under `target/`; installed packages stay in `.venv`. User site
packages and bytecode writes are disabled; `PYTHONPATH`, `CONDA_PREFIX`, and
`PYTHONHASHSEED` are unset. The full suite sees only GPU 0: NVIDIA H100, driver
**580.82.07**. Its loaded CUDA libraries are this environment's
`nvidia/cu13/lib/libcudart.so.13` and `libnvrtc.so.13` (CUDA **13.0**); system
`nvcc` reports **12.6.85**. The stack smoke workloads themselves remain CPU-only.

Python 3.12 was created with `uv venv --python /usr/bin/python3.12 .venv`.
Managed **3.14.5** (Clang **22.1.3**) was downloaded with
`uv python install 3.14.5 --no-bin`, with `UV_PYTHON_INSTALL_DIR` pointing into
`target/`. Its wheel used a separate staging environment and Cargo build directory;
the locked reference dependencies and wheel were then installed at canonical
`.venv` for testing. Python 3.12 is restored there afterward.

| Check | Result |
| --- | --- |
| Python 3.12 canonical `scripts/test-python.sh` during implementation | 5,372 tests in 531.893 s; three failures, 14 skips |
| Python 3.12 final stack-validator + README/documentation tests | All 26 passed in 10.043 s |
| Python 3.14 generated stack-validator suite at canonical `.venv` | All 14 passed in 19.242 s |
| Python 3.14 portable path fixtures without PyTorch or `torch_rs` installed | All nine passed |
| Native-extension provenance and installed bytes versus fresh wheel, both interpreters | Passed |
| `git diff --check` | Passed |

The full run's generated stack smoke test passed. Besides the two baseline
boolean-buffer failures, that run found a new documentation link incorrectly
placed inside the exact historical-report index. The link was moved to the
general provenance policy; all 12 documentation checks subsequently passed
without changing their assertions. The final 26-test run covers the completed
path/fixture refinements and documentation repair. The only unresolved failures
are the two boolean-buffer subcases reproduced before edits; their tests and
production behavior remain unchanged. The full suite was not repeated on 3.14.

Local logs and environment/build identities are retained in `target/evidence/`,
including `baseline-312.log`, `canonical-312.log`, `final-focused-312.log`, and
`validator-314.log`. These are validation records, not published measurements.
