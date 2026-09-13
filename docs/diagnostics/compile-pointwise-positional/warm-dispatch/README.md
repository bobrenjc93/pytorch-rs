# Warm dispatch repair evidence

This non-scoring diagnostic examines Python dispatch overhead within the bounded
[pointwise contract](../../../compile-pointwise-jit.md#cache-behavior). It does
not establish general Inductor parity or replace the fixed-corpus evaluation.

The operator reported the actual `885264b` CUDA result as
**11.360180035407296 (stored 11.4), versus authoritative main 12**, with all eight
supported CUDA cells numerically correct. The preserved operator design also
records CUDA 12 from its same-source coverage run; this is the operator's recorded
context, not a new measurement here. That variation and the negative result remain part of the record;
CPU gains cannot explain the whole shortfall or substitute for qualification.

## Original observation

The following records are byte-identical copies from the read-only operator notes
at `/tmp/burner-approved-default-compile.UA2M6I`, retaining their original source,
interpreter and execution identities. Their historical paths are not current
candidate build paths. The [manifest](original/manifest.json) pins every copy.

- [Design](original/scalar-dispatch-profile-design.md) and
  [independent pre-execution review](original/scalar-dispatch-profile-review.md).
- [Original script](original/profile-scalar-dispatch.py),
  [complete result](original/scalar-dispatch-profile-result.json), and
  [execution receipt](original/scalar-dispatch-profile-execution.json).

That once-run mocked comparison measured `a281503f` against `885264b`. All seven
histories preserved warmed cache cardinalities, made the expected mock boundary
calls, and executed no original body, analysis, lowering or native compilation.
The median CPU microseconds per call were respectively: literal 6.65→14.58,
two-tensor arithmetic 7.03→23.90, broadcast 7.00→22.94, alias 6.90→17.28,
shape revisit 7.10→23.89, promoted capture 8.32→15.89, and eight Boolean entries
8.09→40.26. These include harness and mock costs, not native validation,
public-entrypoint behavior, reset lifecycle or GPU latency.

## Corrected-source diagnostic

The repair precomputes immutable source-position projections and IR hashes within
the existing owners. Each cache map skips a newest-entry mutation only when both
its key and value identity match; logical selection and structural equality remain
unchanged.

[Version 2](profile-dispatch-v2.py) compares immutable `885264b` Git objects with
actual working-tree source bytes. It retains the seven original programs and
histories, 128 profiled calls, four unprofiled 1,024-call batches per source and
balanced before/corrected, corrected/before, corrected/before, before/corrected
orders. It checks all three cache-map cardinalities, exact mock metadata,
validation and launch counts, and zero warm body/analysis/lowering/compilation.
It retains the original positive `Graph.__hash__` instrumentation sanity check.
Production functions are not rewritten; only relative imports are redirected to
the explicitly synthetic boundary, and the actual cache class is constructed.
The registry/global-reset lifecycle is deliberately outside this diagnostic.

Use an isolated, worktree-local Python 3.12 interpreter, after the implementation
and regression checks are ready:

```bash
.venv/bin/python -I -S -B docs/diagnostics/compile-pointwise-positional/warm-dispatch/profile-dispatch-v2.py \
  --output target/warm-dispatch/cpu-v2.json
```

The output records timestamps, command, executable and script hashes, source
hashes rechecked after measurement, HEAD and dirty status. It refuses to overwrite
an existing result and writes partial observations plus the traceback on failure.
Preserve each execution, including failures. Dirty-source results are development
diagnostics and do not satisfy clean-commit hardware captures.

Two development executions are preserved separately. The [first result](cpu-v2-result.json),
[receipt](cpu-v2-execution.json) and [source snapshot](cpu-v2-source-snapshot.json.gz)
precede the final LRU value-identity correction. The [final-source result](cpu-v2-final-result.json),
[receipt](cpu-v2-final-execution.json) and [source snapshot](cpu-v2-final-source-snapshot.json.gz)
measure the corrected bytes. Both passed all seven histories and retained every
batch; the additional execution follows a source correction, not selection of a
faster repeat. Median mocked CPU microseconds per call for the final-source run:

| History | Unchanged 885 | Corrected |
| --- | ---: | ---: |
| literal unary | 14.428 | 12.872 |
| tensor arithmetic | 24.011 | 19.501 |
| broadcast add | 23.174 | 19.099 |
| repeated alias | 17.093 | 14.442 |
| shape revisit | 23.789 | 19.921 |
| promoted capture | 16.006 | 13.453 |
| eight boolean entries | 40.265 | 17.570 |

These mocked CPU reductions neither prove the cause of the CUDA regression nor
qualify the corrected implementation.

The [GPU consumer](gpu-dispatch.py) provides clean-build recording, one public
native/reference leg, and verification of all 16 ordered legs. It has only had
syntax and CLI smoke checks in this revision; no hardware timings were run.
Reproduction commands are part of the plan below. Each leg retains all cold,
warmup and sample observations, with numerical comparison deferred until the
separate-process pair is available. The verifier compares every output; an
unverified leg is not a correctness result. Verification records every input
report identity first, checks all 56 paired histories and every recorded output,
and retains all comparison failures before failing the run. Tensor values use
JSON-safe hexadecimal float strings, including signed zeros and nonfinite values;
comparison preserves the existing JIT absolute-plus-relative tolerance, equal-NaN
policy and signed-zero checks. Hardware-free
synthetic regressions cover success and multiple failures across separate pairs.
No-replay evidence comes from the separate unchanged CPU diagnostic and compiler
regressions; timing traversals are not instrumented with a profiler.

The [balanced hardware plan](measurement-plan.md) fixes the clean-wheel comparison
before any new hardware timing. Corrected clean-commit timings and refreshed
current-candidate captures await Burner's commit/evidence phase; the preserved
`779e512c` measurements remain attributed to that earlier source.

## Development validation

The [validation archive](validation.json.gz) records dirty sources based on
`885264b`, the release wheel/source/import hashes, commands and logs. All 166
pointwise tests ran in 19 disjoint H100 processes; four device-reservation skips
passed separately on GPUs 0 and 1. Hardware-free validation passed 38 tests and
skipped 52 GPU cases; all 12 documentation smoke tests passed. The 17 shared-cache
tests and three new comparison-verifier tests pass, including ordered cache
contents, structural sharing/hash collisions, failed publication, code-object
replacement, multiple retained numerical failures and IEEE serialization.
Formatting and independent design/compatibility reviews also passed.

The first GPU sweep was stopped after discovering an inherited external
`CUDA_CACHE_PATH`; its completed logs and interruption record are retained.
The successful rerun explicitly set CUDA, Torch, Inductor, Triton and temporary
cache paths inside the worktree. No external cache cleanup was performed. An
initial shared-cache test failure against a stale installed wheel is preserved
as a labeled console-derived summary, followed by the passing current-wheel run.
These development checks do not supply clean-commit performance evidence.
