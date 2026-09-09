# Rank-2 mean diagnostics

Large CPU rank-2 single-dimension `mean` and `sum` use the explicit
`torch.set_num_threads(n)` worker budget, which defaults to one. Independent
outputs are partitioned between the caller and at most `n-1` background workers; no output's partial sums are combined
across workers. Wider column tiles amortize strided reads while retaining the
existing accumulation order. Small reductions and reductions with only one
output remain serial. This does not enable CUDA reductions or an inter-op
executor.

The [diagnostic runner](../scripts/diagnose_rank2_mean.py) compares public method
and function calls at matched one-, four-, and eight-worker budgets. It includes
both axes, contiguous and transposed layouts, irregular and generated shapes,
isolated calls and 32-call batches. Both implementations receive the same seeded
inputs, warmups, reversed execution orders, sample counts, and output checks.
Each report retains raw samples, medians, dispersion, capped and uncapped
ratios, CPU affinity, build durations/cache state, and source/native hashes.
These are feature diagnostics, not evaluation scores.

The default comparison keeps both runtimes in one process. For investigating
worker scheduling or interference, `--implementation native` and
`--implementation torch` run one engine with the other engine's thread setting
at one. Both retain the same inputs, call boundaries, warmups and checks. They
record the actual active/inactive thread settings and emit no parity aggregate
for an incomplete comparison. Retain the default results alongside these
supplemental reports; do not substitute a more favorable process configuration
for the default comparison.

Use the same worktree-local caches and release build helper as the
[CUDA-add diagnostic](cuda-add-diagnostics.md). For retained evidence, run from
the clean committed implementation and runner revision:

```bash
.venv/bin/python scripts/build_cuda_add_diagnostic.py --name mean-candidate --revision HEAD
.venv/bin/python scripts/diagnose_rank2_mean.py \
  --build-record target/cuda-add-diagnostic/mean-candidate/build-record.json \
  --output docs/benchmark-data/rank2-mean-candidate.json
```

Without `--revision`, the helper records a hashed uncommitted source overlay.
Such scratch reports must stay under `target/`; they do not replace clean
committed-source evidence. Burner owns the implementation commit and the later
artifact-only evidence commit. No current candidate measurement is claimed by
the older retained CUDA allocation-repair report.

Correctness checks cover cancellation-sensitive column tails, both axes,
transposes, offsets, gradients, configuration errors, and pool replacement
while native workers are active:

```bash
.venv/bin/python -m unittest tests.test_parallel_reductions \
  tests.test_rank2_mean_reference tests.test_rank2_sum_numerics \
  tests.test_get_num_threads tests.test_get_num_threads_reference \
  tests.test_get_num_interop_threads tests.test_get_num_interop_threads_reference
cargo test --locked parallel::tests
```
