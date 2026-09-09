# Text collation diagnostics

`default_collate` dispatches on its first leaf. An exact Python `str` or `bytes`
returns the original batch, including heterogeneous tails, as PyTorch 2.13 does.
Direct list/tuple batches preserve identity, order, and every input object;
nested dict fields become lists and sequence/namedtuple fields become tuples.
Tails remain opaque metadata, with no execution or conversion support implied.
Numeric-, tensor-, and unsupported non-text-led validation and `default_convert`
retain their existing boundaries.

The [benchmark](../scripts/benchmark_text_collation.py) times equivalent public
calls on precreated inputs. It covers str/bytes, list/tuple batches, direct
homogeneous/mixed tails, and dict/list/namedtuple nesting at sizes 1, 8, 64,
1024, and 8192. Every case uses equal warmup and iteration counts, 12 paired
samples, balanced randomized execution order, and one pinned CPU. Output
identity, container, order, and input preservation checks run outside timing.
The report retains raw nanoseconds per call, medians, median absolute deviations
(MAD), and geometric means of reference/candidate median ratios.

Reproduce after the local source-copy build described in the
[CUDA validation guide](cuda-neg-validation.md):

```bash
PYTHONPATH=python .venv/bin/python scripts/benchmark_text_collation.py \
  --cpu 24 --seed 20260909 --samples 12 \
  --output docs/diagnostics/text-collation.json
```

Choose an allowed CPU from `os.sched_getaffinity(0)` when CPU 24 is unavailable.
Results describe this host and interpreter only, with no timing gate or claim
about general PyTorch performance. The unchanged Burner evaluators own scores.

## Recorded local sample

The [raw report](diagnostics/text-collation.json) measures clean commit
`e3a3ba3bd11f90378000b780b29389e5b1195c59`, with the same production fingerprint
and extension hash as the [CUDA build receipt](diagnostics/composite-cuda-neg/build-record.json).
Its git status is empty and its production diff is empty. The original capture
output under `target/final-validation/` was copied here unchanged after both
measurements; the source and benchmark harness remain unchanged. No historical
workload was rerun or replaced.

Captured `2026-09-09T19:48:54.318392+00:00` to `2026-09-09T19:49:11.044221+00:00` on CPU 24,
CPython 3.12.14, PyTorch 2.13.0+cu130, AMD EPYC 9654.

| Case group | Geometric mean of reference/candidate median |
| --- | ---: |
| dict | 0.428 |
| direct | 1.512 |
| direct_mixed | 1.536 |
| namedtuple | 1.063 |
| sequence | 1.798 |
| all | 1.137 |

Ratios above one mean the candidate median was lower in this sample; below one
mean the reference median was lower. Results vary by container and nesting.
The text capture finished before the full validation suite started. This is a
shared host; CPU affinity does not guarantee exclusive CPU ownership. Raw
samples and MAD expose variability.
These observations support no universal timing or score claim. Inputs, balanced
order, sample counts, summaries, source/harness fingerprints, and extension hash
were independently verified after capture.
