# Contributing

`pytorch-rs` is an experimental Rust tensor engine with a PyTorch-compatible
Python API. Keep contributions small, semantic, and backed by focused checks.

## Locked Setup

Install [uv](https://docs.astral.sh/uv/) and [rustup](https://rustup.rs/), then
build from the repository root:

```bash
uv venv --clear --python 3.12
uv sync --locked --no-install-project --group dev
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
```

The lockfiles are part of the contract: `uv.lock` pins Python dependencies,
`Cargo.lock` pins Rust dependencies, and `rust-toolchain.toml` pins the Rust
toolchain. Use locked installs and builds by default. Change a lockfile only
when the patch intentionally changes dependencies, and call that out in review.

## Environment Expectations

- Import the installed package as `torch_rs`, usually aliased to `torch` in
  examples and tests.
- Use the repository virtual environment for Python-facing builds and tests.
  Set `PYO3_PYTHON="$PWD/.venv/bin/python"` when invoking Cargo or Maturin with
  `python-bindings`.
- The current native backend is CPU `float32` focused. Portable tests should
  skip hardware-only cases when an accelerator is unavailable rather than
  weakening the assertion.
- If a change touches devices, CUDA, dispatch, transfers, or accelerator
  performance, run a real accelerator check when available. Prefer
  `CUDA_VISIBLE_DEVICES=0` for single-GPU validation and record the GPU,
  driver, CUDA, PyTorch, and build settings used.
- Keep build and test artifacts inside the worktree. Do not depend on local
  user configuration, parent checkouts, or globally installed packages.

## Contributor Preflight

Before README/docs or Python smoke tests, verify the environment and install the current release wheel:

```bash
uv sync --locked --no-install-project --group dev --group reference
mkdir -p target
wheel_dir="$(mktemp -d "$PWD/target/preflight-wheel.XXXXXX")"
TMPDIR="$PWD/target" VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin build --release --locked --out "$wheel_dir"
uv pip install --python .venv/bin/python --force-reinstall --no-deps "$wheel_dir"/torch_rs-*.whl
.venv/bin/python - <<'PY'
import importlib.metadata as meta, pathlib, sys
import numpy, torch, torch_rs
root = pathlib.Path.cwd().resolve()
assert pathlib.Path(sys.prefix).resolve() == (root / ".venv").resolve()
assert torch.__version__.split("+", 1)[0] == "2.13.0"
for dist, module in (("torch-rs", torch_rs), ("numpy", numpy), ("torch", torch)):
    print(f"{dist} {meta.version(dist)}: {pathlib.Path(module.__file__).resolve()}")
print(f"python {sys.version.split()[0]}: {sys.executable}")
PY
rustc --version && cargo --version
.venv/bin/python .github/scripts/verify_native_extension.py
```

## Choosing Tests

Start with the narrowest checks that exercise the changed behavior, then expand
when a patch touches shared parsing, tensor layout, autograd, or packaging.

- Documentation-only changes: run the docs smoke test.

  ```bash
  .venv/bin/python -m unittest tests.test_readme_quickstart
  ```

- Rust tensor-core changes: run formatting and focused Rust tests first.

  ```bash
  cargo fmt --check
  cargo test --all-targets
  ```

- Python API changes: rebuild the extension and run the matching public and
  reference tests side by side.

  ```bash
  VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
    .venv/bin/maturin develop --release --locked
  .venv/bin/python -m unittest tests.test_<area> tests.test_<area>_reference
  ```

- Before sending a broad or risky patch, use
  `./scripts/test-python-exact-head.sh` to validate a fresh exact-HEAD wheel.

## Draft PR Workflow

Burner develops each increment in an isolated branch and evaluates it against a
fixed base revision. Keep PRs and Phabricator updates as drafts until the
maintainer explicitly publishes them or marks them ready for review.

Do not use `Changes Planned` or `--plan-changes` as a substitute for draft
updates; review-status transitions are maintainer-owned.

Preserve machine-readable review metadata such as `Pull Request resolved:`,
`Differential Revision:`, ghstack headers, and import/export tags when editing
descriptions. When updating an existing stack, repair missing linkage before
exporting; do not create replacement PRs for an update.

## Documentation Ownership

User-visible API changes should update the durable docs that describe that API:
`README.md` for entry points, `FEATURES.md` for weighted coverage,
`docs/supported-surface.md` for the exhaustive contract, `ARCHITECTURE.md` for
source maps, and `BENCHMARKING.md` for performance policy.

Burner owns the managed README progress section and the generated evaluation
artifacts under `docs/burner-evaluation-history.json` and
`docs/burner-evaluation-progress.svg`. Do not edit those files or add
repository-side progress generators, validators, tests, or workflows.
