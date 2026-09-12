# Pointwise JIT review fixes

The four review findings are covered by implementation changes and independent
regressions:

- `tensor::cuda_graph` again requires `python-bindings`; the pointwise module
  has its own `python-bindings`/test guard. Both default and bindings builds
  exercise the native JIT tests without dead-code warnings.
- Two-product add/subtract graphs select an explicit FMA operand. Tests cover
  overflow, cancellation, operand reversal, negative and held-out coefficients,
  changed values and generated PTX. Addition normalizes a negative coefficient
  on the left before selecting contraction.
  A liveness regression added to the worktree during validation exposed dead
  consumers affecting this choice. Dead expressions are now removed after full
  IR validation and before counting live uses; invalid dead nodes still fail.
- Negated products contract with positive zero, preserving the negative sign
  of underflowed positive products and the reference's positive sign for exact
  zero products. Remaining uses of a reused product are counted after that
  contraction. Standalone negation retains its existing zero-sign behavior.
- Function globals must be an exact dictionary before any lookup. Custom
  mappings are rejected on both valid and invalid graph attempts, without
  running lookup hooks or entering code generation.
  Additional admission regressions cover defaults, keyword defaults and closure
  containers: exact types are checked before any truthiness or iteration hooks.
  Signature mutation is checked on warm calls as well as initial admission.

[Development validation bundle](review-fixes.json.gz) retains build/source/wheel
identities, commands, all compiler-file results, generated CUDA/PTX, output
observations and logs. It is explicitly dirty-source regression evidence, not
a clean-commit score. The original Rust build error, failing new regressions,
intermediate Clippy/contraction failures and original probes are preserved.
Reference-only contraction-orientation probes record which FMA order matches
the reference; their `left`/`right` booleans are not candidate pass/fail scores.
The initial Rust test commands inherited the host's CUDA cache setting. Both
complete Rust suites were rerun with explicit worktree-local paths and
`CUDA_CACHE_DISABLE=1`; their `rust-liveness-*.log` files are the final checks.
The initial logs are retained, and no external cache cleanup was attempted.

Checks on the repaired release wheel:

- Focused H100 JIT/liveness/signature selection: 21 tests, one explicit two-device skip.
- The skipped restoration/module-ownership test passes separately with
  `CUDA_VISIBLE_DEVICES=0,1`; compilation failure and cache reset restore the
  caller's device as well as successful execution.
- Python 3.14 without CUDA/reference availability: eight hardware-free tests
  pass, 13 hardware tests explicitly skip.
- All 49 entrypoint tests pass again after the signature guard changes.
- Rust all-target tests: 401 pass without Python bindings, 428 with bindings.
- All-target Clippy with warnings denied passes in both feature configurations.
- `cargo check --locked --lib --no-default-features` passes.
- Documentation: 12 Python quickstart checks pass; rustdoc completes with zero
  doctests. Formatting and whitespace checks pass.
- All 13 before/after arithmetic probe expressions agree with default Inductor
  under the existing finite-value tolerance and exact zero-sign checks.

All original 67 compiler files passed (801 test cases). The newly added
liveness and signature files are also checked, and all three pointwise test
modules run on the final wheel. Rust suites were rerun after the liveness fix;
the subsequent admission changes affect only Python. This covers all 69 current
compiler files (805 unique cases) without repeating the unchanged eager/backend
suites. The compiler selection is
`sorted(Path('tests').glob('test*compile*.py'))`, with one fresh
`.venv/bin/python -m unittest tests.<file_stem>` process per file. The bundle
preserves the driver and every process status; failed files are never omitted.
Single-device execution uses `CUDA_VISIBLE_DEVICES=0`. Code generation selected
NVRTC 13.0, CUDA runtime 13000 and `compute_90`, with FMA enabled and fast math
disabled, on the H100 identified in the captured provenance.

Reproduce with worktree-local `CARGO_HOME`, `CARGO_TARGET_DIR`, `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, `TMPDIR`, `XDG_CACHE_HOME`, `CUDA_CACHE_PATH`,
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`:

```bash
.venv/bin/maturin build --release --locked --out target/review-wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/review-wheels/torch_rs-*.whl
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest tests.test_compile_pointwise_jit \
  tests.test_compile_pointwise_liveness tests.test_compile_pointwise_signature_guards -v
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets
CUDA_VISIBLE_DEVICES=0 cargo test --locked --all-targets --features python-bindings
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --all-targets --features python-bindings -- -D warnings
```

Burner committed these fixes as `60abd863`. The subsequent
[post-commit capture](README.md) now supplies fresh clean coverage/performance,
generated-code evidence and all three pointwise regression modules. The older
`5b93c983` measurements and this dirty-source development bundle remain unchanged
at their original source identities. This bundle itself claims no score; no
evaluator, corpus, dependency, tolerance, denominator or managed progress
artifact changed.
