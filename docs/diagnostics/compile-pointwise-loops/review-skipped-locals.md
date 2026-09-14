# Skipped local reads remain unexecuted

The reviewer reproduced a zero-trip frame error: `y = y + x` inside a skipped
loop raised an unbound-local error even when the remaining program assigned
`y = x` and returned `-y`. The admission-only frame was applying executed-local
lookup rules to instructions that Python never executes.

The existing isolated admission frame now uses a temporary pointwise-data
placeholder for an unbound read, including inside a helper called only by that
frame. The placeholder and every operation using it are discarded with the
scratch IR. They cannot add runtime inputs or bind a local in the executing
frame. Existing operator, helper, binding, data and shared-budget checks remain
active. Executed unbound reads still reject, including after a zero-trip loop,
inside a one-trip loop, and inside an executed helper. No backend, native domain,
cache owner, arithmetic semantics or fallback changed.

The new tests compare the resulting graph with the remaining straight-line
program, reject skipped reductions/unknown calls/helper-local loops/scalar-only
arithmetic, and exercise the exact reviewer example against ordinary default
`torch.compile` on H100. Additional hardware checks cover fresh values and
shapes, input immutability, warm helper-binding rejection and recovery without
lowering/disassembly, reset, and changing the loop from zero to one trip.

Development checks on source based on `a0367d7cf62297803cc07514e5cbc147ec5c5099`:

- The new CPU reproductions failed before the repair (four unbound-local errors);
  that original log is preserved.
- All **245 pointwise tests passed** on the reserved H100s 0 and 1, using local
  CPython 3.12.12 and the freshly packaged wheel.
- Portable loop/helper/signature/diagnostic checks passed on local CPython
  3.10.19, 3.11.15, 3.13.13 and 3.14.5: **66 tests each, ten hardware skips**.
- Checkout Python sources, wheel Python/native bytes and imported package/native
  bytes were verified. The new frontend SHA256 is
  `a70b2059fc15c98878f3ac53ee09b53e5f427be5670bc1e7f9c7babfad59e40e`.

[The evidence archive](review-skipped-locals.json.xz) preserves exact log bytes,
the original failure, source/wheel/import identities and GPU inventories. These
are dirty-source development checks, not timing evidence, a new score, or
clean-commit qualification. No frozen evaluation was rerun. The earlier
`bdcfb051` measurements predate this implementation repair and remain unchanged;
Burner must commit this repair before the required clean evidence refresh.
The previously documented late-raw retention gap remains unchanged.

With the existing canonical `gpu`/`cpu-heavy` reservation and worktree-local
settings from `target/operator-recovery/env.sh`, the principal checks were:

```bash
maturin build --release --locked --offline --out target/skipped-local-fix/wheels
uv --no-config pip install --python .venv/bin/python --force-reinstall --no-deps \
  target/skipped-local-fix/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -B -m unittest discover \
  -s tests -p 'test_compile_pointwise_*.py' -v
```

The wheel reused the existing worktree-local release build of unchanged Rust
sources; no fresh native build timing is claimed. Portable runs extracted this
same wheel and copied the existing `typing_extensions` module locally. All new
files, fixtures, logs, dependencies and writable caches stayed in this worktree.
