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

Native-only work does not require PyTorch. For reference comparisons, add
`--group reference` to the `uv sync` command above before the Maturin build.

The lockfiles are part of the contract: `uv.lock` pins Python dependencies,
`Cargo.lock` pins Rust dependencies, and `rust-toolchain.toml` pins the Rust
toolchain. Use locked installs and builds by default. Change a lockfile only
when the patch intentionally changes dependencies, and call that out in review.

## Contributor Preflight

Before README examples or Python smoke tests, confirm the active checkout environment, packages, and Rust channel:

```bash
. .venv/bin/activate
python - <<'PY'
import importlib.metadata as md, pathlib, sys
if pathlib.Path(sys.prefix).resolve() != pathlib.Path(".venv").resolve():
    raise SystemExit(sys.prefix)
import numpy, torch_rs
print("python", sys.executable)
print("torch-rs", md.version("torch-rs"), torch_rs.__file__)
print("numpy", numpy.__version__, numpy.__file__)
PY
rustc --version && cargo --version
```

After a manual release wheel install, run `.venv/bin/python .github/scripts/verify_native_extension.py`; `./scripts/test-python.sh` runs that check before the suite. See [docs/troubleshooting.md](docs/troubleshooting.md) for setup recovery steps.

## Environment Expectations

- Import the installed package as `torch_rs`, usually aliased to `torch` in
  examples and tests.
- Use the repository virtual environment for Python-facing builds and tests.
  Set `PYO3_PYTHON="$PWD/.venv/bin/python"` when invoking Cargo or Maturin with
  `python-bindings`.
- Native [CPU and bounded CUDA support](docs/supported-surface.md) does not imply
  general accelerator/training parity. Skip unavailable hardware cases clearly.
- For device/CUDA/dispatch/transfer/performance changes, test real GPUs when
  available. Default to `CUDA_VISIBLE_DEVICES=0`; record GPU/driver/CUDA/PyTorch
  and build settings. Minimize multi-GPU use. [Paired diagnostics](docs/compile-cuda-matmul.md#explicit-diagnostic-gpu-selection)
  may declare an idle UUID, identical for both builds. Idle snapshots do not reserve
  access; never interrupt another user's jobs.
- Keep build and test artifacts inside the worktree. Do not depend on local
  user configuration, parent checkouts, or globally installed packages.

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

### CUDA compiler-failure tests

On a CUDA host, the compiler-failure tests require `cc` on `PATH` and an
existing, compatible NVRTC shared-library file. Set `TORCH_RS_NVRTC` to that
real compiler library; alternatively, `TEST_PROGRAM_IDENTITY_REAL_NVRTC`
selects it only for the failure tests. The tests build a local injected shim
that forwards to this real library; do not point either setting at the shim.
Both supported suite entry points inherit these settings:

```bash
export TORCH_RS_NVRTC=/absolute/path/to/libnvrtc.so.13
test -f "$TORCH_RS_NVRTC"
command -v cc
CUDA_VISIBLE_DEVICES=0 ./scripts/test-python.sh
# After the implementation is committed:
CUDA_VISIBLE_DEVICES=0 ./scripts/test-python-exact-head.sh
```

Choose the installed NVRTC compatible with the GPU, driver and generated PTX;
the path above is illustrative. Record the loaded NVRTC/runtime versions when
reporting GPU checks: `nvcc --version` identifies the separate toolkit compiler,
not the loaded NVRTC. Unavailable CUDA/reference cases keep their explicit
skips; missing compiler prerequisites on a configured CUDA host are errors.

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
`README.md` for entry points, [docs/README.md](docs/README.md) for navigation,
`FEATURES.md` for weighted coverage, `docs/supported-surface.md` for the
exhaustive contract, `ARCHITECTURE.md` for source maps, and `BENCHMARKING.md`
for performance policy.

Burner owns the managed README progress section and the generated evaluation
artifacts under `docs/burner-evaluation-history.json` and
`docs/burner-evaluation-progress.svg`. Do not edit those files or add
repository-side progress generators, validators, tests, or workflows.
