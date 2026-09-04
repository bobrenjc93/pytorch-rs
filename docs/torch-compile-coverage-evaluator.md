# torch.compile Coverage Evaluator

The deterministic compile-coverage evaluator runs the versioned v4
reference-eligible corpus against stock PyTorch 2.13 and the current
`torch_rs` wheel. It fails closed on malformed corpus metadata,
reference-import or compile failures, candidate import, compile, runtime, output
or observable-semantic mismatches, eager fallback, installed-PyTorch forwarding,
unsupported candidate cases, skipped eligible cases, and invalid guard coverage.

Run the full Burner evaluator from the repository root:

```bash
bash scripts/evaluate_torch_compile_coverage.sh
```

The shell wrapper keeps setup inside this worktree: it creates or reuses
`.venv`, installs the locked development and reference dependency groups, builds
the current wheel with maturin, installs it into `.venv`, verifies extension
provenance, and then executes the evaluator. Setup and progress diagnostics go
to stderr; Burner EvaluationOutput JSON goes to stdout.

For a faster local screen, run the public strict subset:

```bash
bash scripts/evaluate_torch_compile_coverage.sh --subset public
```

The public subset uses the same corpus metadata, reference eligibility checks,
candidate execution checks, category weights, and scoring formula as the full
gate, but omits held-out cases. The full gate includes the held-out
recompilation-guard cases and validates that the current v4 guard scenarios are
present before scoring.

After the wheel and dependencies are already installed, the Python entry point
can be run directly:

```bash
.venv/bin/python scripts/evaluate_torch_compile_coverage.py --subset full
```

The reported score is derived only from executed cases as the documented
weighted passed/eligible percentage; it is not a fixed target value.
