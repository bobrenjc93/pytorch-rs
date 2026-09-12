# Native compiled CUDA row sums

`torch_rs.compile(..., backend="eager", fullgraph=True)` captures row sums of
contiguous, rank-two CUDA float32 tensors without gradients. It runs the native
row-reduction kernel, not the original Python function or installed PyTorch.
This is bounded inference support, not a general Inductor backend or a
performance-parity claim.

## Example

After the [locked release setup](../CONTRIBUTING.md#locked-setup), run this with
`CUDA_VISIBLE_DEVICES=0` on a machine with a supported NVIDIA GPU:

```python
import torch_rs as torch


def row_sums(x):
    return torch.sum(x, -1)


compiled = torch.compile(row_sums, backend="eager", fullgraph=True, dynamic=False)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to("cuda:0")
y = compiled(x)
assert tuple(y.shape) == (2,)
assert y.cpu().tolist() == [6.0, 15.0]

# The same compiled wrapper executes new values, not cached outputs.
changed = torch.tensor([[3.0, 2.0, 1.0], [-4.0, -5.0, -6.0]]).to("cuda:0")
assert compiled(changed).cpu().tolist() == [6.0, -15.0]
```

## Supported boundary

The method forms `x.sum(1)`, `x.sum(1, True)`, `x.sum(1, keepdim=True)` and
`x.sum(dim=-1, keepdim=False)` are supported. The dimension must be an exact
constant integer `1` or `-1`; `keepdim` must be an exact constant boolean and
defaults to `False`. Constant locals and guarded module globals follow the
existing compiler rules. Keyword lowering is supported on Python 3.10–3.14.

Top-level `torch_rs.sum(x, 1)` / `torch_rs.sum(x, -1)` and imported genuine
aliases accept exactly two positional arguments, with `keepdim=False`.
They share the method reduction planner and executor. Recognition uses the
immutable native function owner retained during package initialization; it does
not trust names or writable exports. Only used module fields are guarded:
changing or deleting an unrelated `sum` leaves existing graphs and their
recompile budget intact. Retained genuine aliases survive export replacement;
changed used bindings are checked on cold calls and cache hits without invoking
replacement callables. Restored bindings can reuse the original specialization.

Inputs must be exact native CUDA tensors, contiguous float32, rank two and
without gradients. Contiguous offset views and singleton layouts are accepted.
For shape `(M, N)`, the output has shape `(M,)` or `(M, 1)`, fresh contiguous
storage at offset zero, and the input's CUDA device. Zero rows and zero-width
rows follow the [native eager reduction semantics](cuda-sum-rows.md).

Reductions compose with the existing supported CUDA graph operations. The
whole graph is validated before execution, and live method and input-metadata
guards apply to both cold calls and cache hits. See the
[supported-surface contract](supported-surface.md) for the full compiler grammar
and shape/stride specialization rules.

Full reductions, dim `0`/`-2`, dimension sequences, other ranks, noncontiguous
inputs, other dtypes and autograd remain unsupported in this capture path.
So do `dtype`, `out`, argument unpacking, nonconstant options and invalid option
types. Top-level keywords and a third positional `keepdim` argument remain
unsupported; existing Tensor method keyword and keepdim forms are unchanged.
CPU compilation is unchanged; no eager fallback is provided for rejected CUDA
graphs.

## Validation and performance evidence

The [validation history](compile-cuda-sum-rows-validation.md) preserves the
clean implementation-commit capture, earlier development attempts, commands
and raw evidence. It includes held-out shapes, changed-input execution without
re-lowering, unsupported boundaries, two-GPU restoration and Python 3.10–3.14
checks. Reproduce the focused checks with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v \
  tests.test_compile_sum_lowering tests.test_compile_cuda_sum_rows \
  tests.test_compile_cuda_module_sum
```

The [top-level development evidence](diagnostics/compile-cuda-module-sum/README.md)
retains the unchanged 144-cell pre-edit probe and subsequent source-bound checks.
The separate [clean-commit capture](diagnostics/compile-cuda-module-sum/postcommit-e46a496/README.md)
records the fresh release build and required checks at `e46a496`.
Dyadic inputs establish bounded correctness, not general exact summation or a
speed improvement. The existing special-value and precision policy is unchanged.

The unchanged hardware evaluator still observes the old unary entry point,
so compiled row sums receive zero credit despite working natively. The
[separate observer campaign](https://github.com/bobrenjc93/pytorch-rs/pull/1970)
requires human review before adoption; the six-case denominator is unchanged.
See the [measurement boundary](compile-cuda-sum-rows-validation.md#current-scoring-boundary)
for this distinction and the earlier captures.

The separate
four-workload private CUDA timing suite is unchanged and does not measure the
new generic row-sum graph. Its saved timing summaries do not contain individual
latency samples; see the [measurement limitations](compile-cuda-sum-rows-validation.md#clean-implementation-commit-capture).
