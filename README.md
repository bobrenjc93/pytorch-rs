# pytorch-rs

`pytorch-rs` is an experimental native Rust tensor engine with a
PyTorch-compatible Python API. It supports a CPU `float32` subset and narrow
NVIDIA CUDA paths. It is not a general PyTorch replacement.

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

| Surface | Supported today | Limits |
| --- | --- | --- |
| CPU tensors | Native `float32` construction, views, selected math and NN functions, limited first-order autograd. | No additional tensor dtypes or full training stack. |
| NVIDIA CUDA | Native storage, synchronous CPU transfers, and same-shape contiguous `float32` addition. | Direct factories: 1-D zeros only. No general CUDA math or accelerator training. |
| `torch.compile` | Bounded eager CPU capture and [CUDA addition capture](docs/compile-cuda-add.md), without fusion. | No full Inductor compiler, general graph capture, or eager fallback. |
| Compatibility helpers | Selected device/backend probes, state, data, and JIT helpers. | No full module, `DataLoader`, optimizer, model-serialization, or distributed stacks. |

The [exhaustive supported surface](docs/supported-surface.md) owns exact method
variants and exclusions, including [compiler options and cache behavior](docs/supported-surface.md#jit-and-compiler).
The private H100 compile benchmark path is limited to its fixed workload.

`torch.cuda.is_available()` and `torch.cuda.device_count()` report runtime GPU
visibility; backend build flags do not. See [CUDA setup and tests](docs/troubleshooting.md#optional-native-cuda-runtime)
for runtime requirements. Run hardware checks on an available GPU with
`CUDA_VISIBLE_DEVICES=0`; hardware-only cases skip when unavailable.

## Evaluation

Correctness gates performance, and missing or unsupported behavior stays in the
denominator. See [BENCHMARKING.md](BENCHMARKING.md) for benchmark methodology
and anti-gaming rules, [FEATURES.md](FEATURES.md) for weighted feature coverage,
and the [hardware heterogeneity evaluator](docs/hardware-heterogeneity-evaluator.md)
for accelerator-family coverage.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup preflight, focused test
selection, draft-PR workflow, and documentation ownership. Start with the
README smoke check after installation:

```bash
.venv/bin/python -m unittest tests.test_readme_quickstart
```

For the full Python suite, `./scripts/test-python.sh` builds and installs a
release wheel from the current worktree and verifies extension provenance.
`./scripts/test-python-exact-head.sh` validates a fresh wheel from committed
`HEAD`, excluding local edits; see [validation details](docs/troubleshooting.md#exact-head-validation).
Both use available CUDA hardware and skip hardware-only cases when unavailable.

Browse [docs/README.md](docs/README.md) for focused guides and
[ARCHITECTURE.md](ARCHITECTURE.md) for the source map. Public tests are the floor;
Burner also runs independent generated workloads and differential checks.

## License

MIT

<!-- burner-progress:start -->
## Burner evaluation progress

![Burner evaluation progress](docs/burner-evaluation-progress.svg)

Burner updates this graph atomically after each successful merge. It validates a complete finite 0–100 score map for every enabled evaluation, then upserts the canonical baseline-commit or `pr:<number>` key; retrying a merge replaces the existing point instead of duplicating it. Missing or malformed scores abort artifact generation before any file is written. The [raw versioned history](docs/burner-evaluation-history.json) records this merge-coupled policy.
<!-- burner-progress:end -->
