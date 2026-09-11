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
    return x.sum(dim=-1, keepdim=True)


compiled = torch.compile(row_sums, backend="eager", fullgraph=True, dynamic=False)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to("cuda:0")
y = compiled(x)
assert tuple(y.shape) == (2, 1)
assert y.cpu().tolist() == [[6.0], [15.0]]

# The same compiled wrapper executes new values, not cached outputs.
changed = torch.tensor([[3.0, 2.0, 1.0], [-4.0, -5.0, -6.0]]).to("cuda:0")
assert compiled(changed).cpu().tolist() == [[6.0], [-15.0]]
```

## Supported boundary

The method forms `x.sum(1)`, `x.sum(1, True)`, `x.sum(1, keepdim=True)` and
`x.sum(dim=-1, keepdim=False)` are supported. The dimension must be an exact
constant integer `1` or `-1`; `keepdim` must be an exact constant boolean and
defaults to `False`. Constant locals and guarded module globals follow the
existing compiler rules. Keyword lowering is supported on Python 3.10–3.14.

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
So do `dtype`, `out`, argument unpacking, top-level `torch.sum`, nonconstant
options and invalid option types. CPU compilation is unchanged; no eager
fallback is provided for rejected CUDA graphs.

## Validation and performance evidence

The [validation history](compile-cuda-sum-rows-validation.md) preserves the
clean implementation-commit capture, earlier development attempts, commands
and raw evidence. It includes held-out shapes, changed-input execution without
re-lowering, unsupported boundaries, two-GPU restoration and Python 3.10–3.14
checks. Reproduce the focused checks with:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m unittest -v \
  tests.test_compile_sum_lowering tests.test_compile_cuda_sum_rows
```

The fixed six-case hardware check is correctness evidence. The separate
four-workload private CUDA timing suite is unchanged and does not measure the
new generic row-sum graph. Its saved timing summaries do not contain individual
latency samples; see the [measurement limitations](compile-cuda-sum-rows-validation.md#clean-implementation-commit-capture).
