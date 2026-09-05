# torch.compile Coverage Evaluator

The deterministic compile-coverage evaluator runs the versioned v9
reference-eligible corpus against stock PyTorch 2.13 and the current
`torch_rs` wheel. It fails closed on malformed corpus metadata,
reference-import or compile failures, candidate import, compile, runtime, output
or observable-semantic mismatches, opted-in backward-through-sum leaf-gradient
mismatches, eager fallback, installed-PyTorch forwarding, unsupported candidate
cases, skipped eligible cases, and invalid guard coverage. The evaluator pins
the v9 category weights, case order, case callables, helper source hashes,
input factories, input payload hashes, no-grad inference flags, and guard-step
definitions before scoring.

Run the full Burner evaluator from the repository root:

```bash
bash scripts/evaluate_torch_compile_coverage.sh
```

The shell wrapper keeps setup inside this worktree: it creates or reuses the
dedicated environment at `target/torch-compile-coverage/venv`, installs the
locked development and reference dependency groups, builds the current wheel
with maturin, installs it into that environment, verifies extension provenance,
and then executes the evaluator. Its setup, install, verification, and evaluator
steps are serialized by an automatically released `flock` on the evaluator
directory so concurrent or retried invocations cannot race on the shared
environment. Setup and progress diagnostics go to stderr; Burner
EvaluationOutput JSON goes to stdout.

For a faster local screen, run the public strict subset:

```bash
bash scripts/evaluate_torch_compile_coverage.sh --subset public
```

The public subset uses the same corpus metadata, reference eligibility checks,
candidate execution checks, category weights, and scoring formula as the full
gate, but omits held-out cases. The full gate includes the held-out
recompilation-guard, training-autograd, no-grad inference, decomposition, and
mutation-aliasing cases and validates that the current v9 guard scenarios are
present before scoring.

After the wheel and dependencies are already installed, the Python entry point
can be run directly:

```bash
target/torch-compile-coverage/venv/bin/python scripts/evaluate_torch_compile_coverage.py --subset full
```

The reported score is derived only from executed cases as the documented
weighted passed/eligible percentage; it is not a fixed target value.
