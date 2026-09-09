# Setup Troubleshooting

Use these fixes from the repository root after following the locked setup in
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Ambient Python Missing Pytest

If `pytest` or `python -m pytest` fails before it reaches repository code, the
command is using an ambient interpreter. The checked-in smoke tests use
`unittest`; run them through the repository environment instead:

```bash
. .venv/bin/activate
python -m unittest tests.test_readme_quickstart
```

For the full Python suite, prefer `./scripts/test-python.sh`; it rebuilds and
checks the installed native extension before running tests.

## `PYTHONPATH=python` Finds Python Files But Not the Native Extension

`PYTHONPATH=python` only exposes the pure-Python package files. It does not build
or install `torch_rs.torch_rs`, so imports can fail while loading
`python/torch_rs/__init__.py`.

Build the extension into the active repository environment, then import without
`PYTHONPATH`:

```bash
unset PYTHONPATH
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
.venv/bin/python -c 'import torch_rs; print(torch_rs.__file__)'
```

## Missing Reference PyTorch 2.13

Reference and differential checks expect PyTorch 2.13.0. If `import torch`
fails, or the preflight reports another version, install the locked reference
dependency group:

```bash
uv sync --locked --no-install-project --group reference
.venv/bin/python -c 'import torch; print(torch.__version__)'
```

The version printed before any local suffix should be `2.13.0`.

## Stale Wheel Installs

If tests import an older `torch-rs` wheel, rebuild and reinstall from this
checkout instead of relying on the previous environment state:

```bash
unset PYTHONPATH
./scripts/test-python.sh
```

The script builds one release wheel from the current worktree, force-installs
it into `.venv`, verifies native extension provenance, and then runs the Python
suite. Clearing `PYTHONPATH` prevents source files from shadowing that wheel.

`maturin develop --release --locked` remains valid for editable development:
it resolves `torch_rs` under `python/torch_rs` so Python edits take effect
immediately. The wheel-only `.github/scripts/verify_native_extension.py` check
intentionally rejects that layout; it requires both `torch_rs` and the native
`torch_rs.torch_rs` extension to resolve inside `.venv`. Use the workflow above
when recovering a stale wheel or preparing an install for wheel verification.
