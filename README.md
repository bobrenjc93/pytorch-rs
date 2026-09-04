# pytorch-rs

`pytorch-rs` is a native Rust tensor and deep-learning engine exposed through a PyTorch-compatible Python API. It pursues PyTorch semantics, broad feature coverage, and competitive performance. It is an early experimental implementation, not yet a PyTorch replacement.

The project is improved through [Burner](https://github.com/bobrenjc93/burner): each increment is developed in an isolated branch, independently reviewed, and measured against the same base revision before it can merge.

## Quickstart

From a source checkout, install the locked Python and Rust dependency graphs.
This requires [uv](https://docs.astral.sh/uv/) and
[rustup](https://rustup.rs/); the repository pins Python dependencies in
`uv.lock`, Cargo dependencies in `Cargo.lock`, and Rust 1.92.0 in
`rust-toolchain.toml`.

```bash
uv venv --clear --python 3.12
uv sync --locked --no-install-project --group dev
VIRTUAL_ENV="$PWD/.venv" PYO3_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/maturin develop --release --locked
```

The distribution is named `torch-rs`; import it as `torch_rs`, conventionally
aliased to `torch`.

### First success

```python
import torch_rs as torch

x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
bias = torch.ones([2, 2])
result = torch.relu(x + bias)
delta = torch.sub(input=bias, other=x)
ratio = x.div(bias)

assert result.tolist() == [[0.0, 3.0], [4.0, 0.0]]
assert delta.tolist() == [[2.0, -1.0], [-2.0, 5.0]]
assert ratio.tolist() == [[-1.0, 2.0], [3.0, -4.0]]
```

The same assertion-only smoke check is available as
[examples/first_success.py](examples/first_success.py).

## Scope

| Surface | Supported today | Unsupported boundary |
| --- | --- | --- |
| Eager CPU tensors | CPU `float32` tensors; core construction and layout/view operations; selected math and neural-network functions; limited first-order autograd. | Additional tensor dtypes and non-CPU tensor execution. |
| CPU-build device probes | `torch.cuda.device_count() == 0`; `torch.cuda.is_available() is False`; `torch.cuda.is_initialized() is False`; `torch.set_default_device(...)` is a CPU-equivalent no-op for `None` or `"cpu"`. | Device selection, mutable default-device routing, CUDA tensors, streams, events, synchronization, allocator APIs, and runtime initialization. |
| CPU-build backend probes | `torch.backends.cuda` preference flags such as `enable_flash_sdp(...)`, `enable_cudnn_sdp(...)`, and `sdp_kernel(...)` as a context manager/decorator. | No `torch.nn.functional.scaled_dot_product_attention`, actual attention-kernel dispatch, CUDA tensors, or CUDA `torch.compile` execution. |
| `torch.compile` eager subset | PyTorch 2.13-shaped argument binding; `disable=True` pass-through; backend default/name resolution through the `torch.compiler` registry; native `backend="eager", fullgraph=True` execution for straight-line one- or two-input CPU `float32` Tensor functions composed from Tensor `neg`, `abs`, `relu`, `detach`, and binary `add`, including Tensor broadcasting, storage-aliasing detach graphlets with `requires_grad=False` outputs, and no-grad inference graphlets, with guarded shape, stride, and `requires_grad` recompilation, bounded per-wrapper input-metadata graph caching, exact non-negative integer `recompile_limit` values, and `torch.compiler.reset()` cache clearing. | No active `__torch_function__` modes, `isolate_recompiles=True`, eager fallback, installed-PyTorch forwarding, callable backend invocation, inductor/CUDA compilation, or broader graph capture/execution. |
| Larger PyTorch stacks | Metadata and helper shims only where listed in the supported surface. | Full module, optimizer, model-serialization, compiler execution, and distributed stacks remain unsupported. |

See [docs/README.md](docs/README.md) for a compact documentation index, the
[exhaustive supported surface](docs/supported-surface.md) for exact API and
limitation details, [FEATURES.md](FEATURES.md) for the weighted coverage
contract, [ARCHITECTURE.md](ARCHITECTURE.md) for a source-oriented contributor
map, and [BENCHMARKING.md](BENCHMARKING.md) for performance policy.

## Evaluation

Correctness gates performance, and missing or unsupported behavior stays in the
denominator. See [BENCHMARKING.md](BENCHMARKING.md) for benchmark methodology
and anti-gaming rules, and [FEATURES.md](FEATURES.md) for weighted feature
coverage.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup preflight, focused test
selection, draft-PR workflow, and documentation ownership.

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo test --doc
uv venv --clear --python 3.12
uv sync --locked --no-install-project
PYO3_PYTHON="$PWD/.venv/bin/python" cargo clippy --all-targets --features python-bindings -- -D warnings
PYO3_PYTHON="$PWD/.venv/bin/python" cargo test --all-targets --features python-bindings
./scripts/test-python.sh
```

The developer Python test command builds a release wheel from the current
worktree, force-installs it into `.venv`, verifies the installed native
extension's provenance, and then runs the full unittest suite. If the suite
fails, it reports the resolved interpreter, package, and extension paths.

To validate the Python package from one freshly built, exact-HEAD release wheel,
run:

```bash
./scripts/test-python-exact-head.sh
```

This exports the exact `HEAD` commit to a temporary directory under `target/`,
creates a Python 3.12 environment there, and installs the locked development and
reference dependencies. It clears inherited environment, import, and Python
optimization markers; builds with the locked Maturin and Cargo dependencies;
force-installs the new wheel; and verifies its native-extension provenance
before checking for PyTorch 2.13.0 and running the full unittest suite. Dirty
worktree files are therefore excluded. To keep every artifact inside the
worktree without inheriting Cargo settings, the command uses a fresh Cargo home
and rejects `.cargo/config` files above the archived checkout. It also ignores
external uv configuration and explicitly installs both locked dependency
groups. Git and tar settings are cleared, and every extracted file is checked
against `HEAD` before testing. The committed Rust channel is explicitly selected
and verified, while ambient Cargo, PyO3, and Python runtime settings, including
warning policy, are cleared. The command rejects a symlinked `target/` before
creating artifacts and uses its verified physical path throughout. It preserves
`CUDA_VISIBLE_DEVICES`, so the existing hardware-aware tests use available CUDA
hardware and skip their CUDA cases when PyTorch reports none.

The checked-in tests are only the public floor. Burner also uses independent generated workloads and side-by-side `torch_rs`/`torch` differential runs.

## License

MIT

<!-- burner-progress:start -->
## Burner evaluation progress

![Burner evaluation progress](docs/burner-evaluation-progress.svg)

Burner updates this graph atomically after each successful merge. It validates a complete finite 0–100 score map for every enabled evaluation, then upserts the canonical baseline-commit or `pr:<number>` key; retrying a merge replaces the existing point instead of duplicating it. Missing or malformed scores abort artifact generation before any file is written. The [raw versioned history](docs/burner-evaluation-history.json) records this merge-coupled policy.
<!-- burner-progress:end -->
