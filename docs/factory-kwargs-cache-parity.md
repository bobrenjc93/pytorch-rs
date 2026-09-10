# Factory-kwargs parity across Python caches

On 2026-09-09, unchanged main `28fb6b923843989f9536608a75f3d240ce4c8a7e`
reproduced three order-only assertions in `test_nn_factory_kwargs_reference`.
This is an assertion-contract issue: PyTorch itself changes direct-key order
when its code constants are reconstructed. Production `nn.factory_kwargs`,
CUDA code, validation scripts, and evaluation definitions are unchanged.

The baseline used CPython **3.14.5** (`main, May 10 2026, 19:28:16`, Clang
22.1.3), installed in this worktree's canonical `.venv`, and locked PyTorch
**2.13.0+cu130**. The base interpreter was
`/home/bobren/.local/share/uv/python/cpython-3.14.5-linux-x86_64-gnu/bin/python3.14`.
The reference came from `.venv/lib/python3.14/site-packages/torch/nn/__init__.py`.
Both `inspect.getsource(torch.nn.factory_kwargs)` and the corresponding
`torch_rs.nn.factory_kwargs` source have SHA256
`0bab29e323aee9bf962e7cb108d9555495d6698800752eeb4c84db47ae7ff409`.

Before changing assertions, all 14 existing factory tests ran in separate
processes at **every hash seed 0–11**, using an isolated `PYTHONPYCACHEPREFIX`
under `target/`. Only the two `nn/__init__.py` caches were removed between
states. Cold imports wrote fresh caches; warm imports reused them. A second
matrix also exercised independently cold/warm implementation and reference
caches. No installed reference files were edited.

| Cache state at process start | Unchanged-main result |
| --- | --- |
| Both cold | Seed 10: three order-only failures; other 11 seeds passed |
| Both warm | All 12 seeds passed |
| Reference warm, implementation cold | All 12 seeds passed |
| Implementation warm, reference cold | Seed 10: three order-only failures; other 11 seeds passed |

Thus seed 10 alone does **not** force the historical failure. A passing warm
rerun is expected and does not demonstrate a fix. The existing tests also
reload the modules, so cache state can change during a process's test run.
The full baseline `scripts/test-python.sh` run, with the hash seed unset,
ran 5,341 tests: two factory order-only failures and 14 existing skips;
native-extension provenance passed.

## Reference-only reproduction

This probe imports no `torch_rs` and does not alter installed bytecode caches:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=10 .venv/bin/python - <<'PY'
import inspect
import marshal
import torch.nn as ref

source = inspect.getsource(ref.factory_kwargs)
code = compile(source, "<reference-source-recompiled>", "exec")
values = {"device": object(), "dtype": object(), "memory_format": object()}
for state, variant in (("source", code),
                       ("bytecode", marshal.loads(marshal.dumps(code)))):
    namespace = {}
    exec(variant, namespace)
    result = namespace["factory_kwargs"](values)
    assert result == values
    assert all(result[key] is value for key, value in values.items())
    print(state, list(result))
PY
```

On the pinned 3.14.5 interpreter, source yields
`[memory_format, device, dtype]`; bytecode yields
`[memory_format, dtype, device]`. `marshal` is the code-object serialization
used by `.pyc` files. The frozenset constant's iteration layout can change
on reconstruction even within the same process/hash seed. All 12 seeds
preserved mapping values and identities; seed 10 exposed the order difference.
The installed reference's order depends on its existing cache state.

A reference-only tracking mapping also demonstrated the corresponding
contains/getitem block permutation. For a lone duplicate `device`, source
visited `memory_format, device` before raising; reconstructed bytecode visited
`memory_format, dtype, device`. Both raised exactly
`TypeError('device specified twice, in **kwargs and in factory_kwargs')`
without fetching a value. These incidental permutations are the only ordering
constraints relaxed by the differential checks.

## Regression coverage and validation

```bash
.venv/bin/python -m unittest tests.test_nn_factory_kwargs_cache -v
./scripts/test-python.sh
```

The cache regression runs all seeds 0–11 in fresh interpreters and checks all
nine combinations of installed, source-compiled, and bytecode-reconstructed
implementation/reference functions. It failed with the old assertions at seed
10 and passed after the assertion repair. It does not require a particular
layout difference on other interpreter versions and never selects a passing
seed. The normal installed-module metadata, pickle, and reload tests remain.

Behavioral checks retain exact dictionary values, value identity, fresh output,
input preservation, nested mapping/pair insertion order, per-key
contains-before-getitem access, duplicate detection before fetching, and exact
exception types/messages/arguments. Nested keys must remain the output prefix;
only the appended direct keys and independent membership probes may permute.

After the repair, all 48 process-start cache/seed combinations passed on
3.14.5. The 15 factory tests, including the nine-variant regression at each
seed, passed on both 3.14.5 and system CPython **3.12.13** (`/usr/bin/python3.12`,
`main, Jul 20 2026, 00:00:00`, GCC 11.5.0). The latter also reproduced
reference-only source/bytecode order differences at seeds 6 and 10, with the
same source digest and unchanged values.
Its installed reference is
`.venv/lib64/python3.12/site-packages/torch/nn/__init__.py`.

Full runs use `scripts/test-python.sh`, each interpreter installed at `.venv`,
locked dev/reference groups, and **no PYTHONHASHSEED override**. Build:
Rust 1.92.0, release/abi3-py310, `extension-module`, thin LTO. GPU tests see only
device 0: NVIDIA H100, driver 580.82.07; PyTorch reports CUDA 13.0. Loaded
runtime/compiler libraries are the worktree environment's `libcudart.so.13`
and `libnvrtc.so.13`; system `nvcc` is 12.6.85. Caches and build artifacts stay
under `target/`. No new hardware skips or benchmark results are introduced.

| Canonical full run | Result |
| --- | --- |
| Python 3.14.5 | 5,342 tests; passed, 14 existing skips |
| Python 3.12.13 | 5,342 tests; 3 unrelated failures, 14 existing skips; all factory tests passed |

The 3.12 failures are the two noncanonical boolean-buffer cases in
`test_tensor_buffer_reference` (also documented in the
[earlier main-baseline diagnosis](diagnostics/composite-integration/README.md#independently-reproduced-baseline-failures))
and `test_generated_validator_smoke_artifact_validates` in
`test_top_level_stack_benchmark_artifact`. The latter compares unnormalized
`lib64` package paths with resolved `lib` paths on this system interpreter.
All three reproduced in an isolated nine-test run without the factory tests;
their tests, production code, and validator scripts are identical to starting
main. They were not changed or suppressed. Native-extension provenance passed
on both interpreters: this is separate from the retired review environment's
failure to use the worktree `.venv`.
