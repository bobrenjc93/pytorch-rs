# Review fix: ignored helper arguments on warm hits

The reviewer reproduced a real admission gap in `a8211b07`: after capturing
`helper(x, captured)` with `captured=0.5`, an ignored capture could become a
function, native operator, module or out-of-range integer and still launch a
cached graph. Fresh wrappers rejected those same arguments. The archived
reproduction records ten invalid launches across global and closure bindings.

The frontend now records which root sources crossed a data boundary during
lowering in the existing specialization. Cache hits validate those current
values through the same exact-type/scalar-range check before retrieving a
lowering or launching an executor. These checks do not realize sources, create
scalar-value guards or promotion history, parse helpers, or add a cache owner.
The native IR and numerical admission domains are unchanged.

The new shared hardware-free/H100 regression checks global rebinding and closure
rebinding forwarded through an identity helper and local assignment. It tests
functions, modules, native operators and both signs of out-of-range integers;
fresh and warm rejection; untouched logical/lowering/executor caches and LRU
order; and valid ignored replacements including bools, integer range endpoints,
NaN, infinity and signed zero. Existing positional/tensor replacement, runtime
scalar, reset/GC, native input validation and original-IR tests also pass.

## Revision validation

These are **uncommitted development checks**, based on evidence commit
`98d19f0` with the source hashes in [review-warm-admission.json.xz](review-warm-admission.json.xz).
A new locked, offline release wheel was built in `target/helper-review/build`,
installed into a separate worktree-local environment, and byte-compared against
all checkout Python sources and imported package/extension files. Dependencies
were copied from the existing local environment; earlier measured installations
and raw reports were preserved.

| Check | Result |
| --- | --- |
| Regression against the pre-fix installed frontend | 10 reproduced invalid warm launches; fresh calls rejected |
| Focused helper suite, GPUs hidden | 30 tests, 5 hardware skips, no failures |
| Focused helper suite, physical H100 GPU 0 | 30 tests, 1 explicit two-device reservation skip, no failures |
| Full compiler sweep, GPUs hidden | 928 tests, 355 skips, no failures |
| Wheel/import/source byte verification | Passed |
| Diff whitespace check | Passed |

The canonical `gpu` and `cpu-heavy` resources remained owned by `idea_1f7d364a`.
Only GPU 0 was selected for this fix's hardware checks:
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, H100, driver 580.82.07.
The archive retains GPU inventories, interpreter/wheel/import hashes, generated
CUDA/PTX and runtime identities. Python was 3.12.14, reference PyTorch
`2.13.0+cu130`, loaded CUDA runtime and native NVRTC were 13.0, and Rust was
1.92.0. System nvcc was 12.6.85; it was not the JIT compiler. The ordinary H100
helper checks include unchanged default-Inductor comparisons and body-execution
policing. This revision validation makes no new performance claim.

From the checkout, with local environment/cache settings in the archived
`target/helper-review/env.sh` and the canonical resources held:

```bash
# Build with CARGO_TARGET_DIR=target/helper-review/build and the local
# PYO3_PYTHON/VIRTUAL_ENV pointing at target/helper-review/venv.
python -m maturin build --release --locked --offline --out target/helper-review/wheels
uv --no-config pip install --python target/helper-review/venv/bin/python \
  --force-reinstall --no-deps \
  target/helper-review/wheels/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl
CUDA_VISIBLE_DEVICES='' python -m unittest -v tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=0 python -m unittest -v tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_compile*.py'
```

The archive is a JSON `files` mapping of relative paths to `text`,
`original_sha256` and `uncompressed_sha256`; gzip PTX is decoded losslessly.
It preserves the original failure log, subsequent successful logs, command
metadata, measured source snapshots and exact identities.

## Clean-candidate refresh

The required clean candidate fixed gate and two-H100 helper diagnostic were
refreshed at [76ea91a7](postcommit-76ea91a7/README.md), using the unchanged tools
and full matrix. The earlier [post-commit captures](postcommit-a8211b07/README.md)
still measure `a8211b07` and have not been relabeled or edited. The clean-main
capture remains applicable to unchanged `fe753725` and was revalidated. These
development records do not replace the clean captures, independent review or
exact-head qualification. The operator observer's raw-report retention handoff
remains unchanged.
