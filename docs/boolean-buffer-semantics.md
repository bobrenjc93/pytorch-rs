# Boolean memoryview conversion

This report records source PR #1930 at
`ed078e1a2dec6ad7c01f6c74c36cce52165d6fc3`, based on
`03075466b077c69ed1f7a4f047ebb33004d3306b`. Its test counts and timings are
historical source-PR observations, not measurements of the combined checkout.
The [original evidence bundle](diagnostics/boolean-buffer/source-pr-1930/README.md)
preserves the driver, raw measurements, setup/build/test logs, and post-commit
verification. See [combined validation](boolean-stack-integration.md) separately.

`torch.tensor(view, dtype=torch.float32)` must follow the running interpreter's
memoryview item conversion for `?` and `@?`. Noncanonical boolean storage does
not have a portable byte-level interpretation, even within one Python minor
version. The release wheel built from base `03075466b077c69ed1f7a4f047ebb33004d3306b`
reproduced both existing `test_numeric_buffers_match_pytorch_2_13` failures on
the host GCC build on 2026-09-09:

| Interpreter | `list(memoryview(bytes([0, 1, 2, 3, 254, 255])).cast('?'))`, as integers | Before repair |
| --- | --- | --- |
| Host CPython 3.12.13, GCC 11.5.0 | `[0, 1, 1, 1, 1, 1]` | `?` and `@?` differential failures |
| Managed CPython 3.12.12, Clang 21.1.4 | `[0, 1, 0, 1, 0, 1]` | Pass |
| Managed CPython 3.14.5, Clang 22.1.3 | `[0, 1, 1, 1, 1, 1]` | Pass |

The repair directly converts canonical bytes `0` and `1`. Every other byte is
converted by indexing the original memoryview at its logical index. It neither
guesses from the Python version/compiler nor forwards to installed PyTorch.
The existing byte copy preserves logical order for sliced and reversed views;
it never rewrites the exporter. Float16 conversion and numeric-format/error
dispatch retain their existing paths.

## Regression checks

The same release ABI3 wheel passed 155 tests on each of the three builds:

```sh
python -m unittest \
  tests.test_tensor_buffer tests.test_tensor_buffer_reference \
  tests.test_tensor_data tests.test_tensor_data_reference \
  tests.test_as_tensor tests.test_as_tensor_reference \
  tests.test_scalar_tensor tests.test_scalar_tensor_reference tests.test_python_api
```

All environments used locked dependencies, including PyTorch 2.13.0+cu130 and
NumPy 2.5.1, and passed the native-extension provenance check. New cases cover
all 256 storage bytes together and individually, both native boolean formats,
read-only and writable sources, positive/negative strides, empty views, source
preservation and independent tensor storage, and reference exception parity.
Existing differential inputs and assertions remain intact. The two unit-test
boolean expectations now use the interpreter's memoryview values on the same
original bytes instead of a minor-version heuristic.

`cargo fmt --check`, `cargo clippy --locked --all-targets --features
python-bindings -- -D warnings`, and `cargo test --locked --features
python-bindings --lib` passed (173 Rust tests).

## Conversion cost

These are CPU microbenchmarks of this conversion path, not a broad performance
parity score. The host was an AMD EPYC 9654 (two 96-core sockets), Linux
6.13.2-0_fbk12_0_g0b66b3635210. Measurements pinned the process to CPU 0, used
one PyTorch/OpenMP/MKL/Rayon thread and hid CUDA devices. Both wheels used
rustc 1.92.0, Maturin 1.14.1, `--release --locked`, thin LTO, one codegen unit,
and `extension-module`/ABI3 Python 3.10 support. Initial dependency download and
installation preceded all timings; fresh baseline and incremental repaired
wheel builds took 46.50 s and 38.63 s respectively, outside the timed region.
`uv` reported dependency preparation/install times of 12.90/1.28 s for the
host, cached preparation/1.26 s for managed 3.12, and 6.76/2.82 s for managed
3.14. Native SO SHA256 values were
`2666e69aefda6dea8024972d5ef444d94162a83398cc3c30c1f0ad8c678dc8bf`
(before) and
`326644903ebfc0408dfe6ab55a1468d52c5b2aaf0649322ad5f61603844818aa`
(after), identical across all three interpreter environments.

The matrix used lengths 16, 4,096, and 1,048,576 with contiguous, reversed
(`[::-1]`), and strided (`[1::2]`) views. NumPy `default_rng(682019)` generated
canonical bytes in `[0, 2)` and dense bytes in `[0, 256)` at each length; sparse
inputs copied the canonical bytes and set every 257th byte to 254. Sources
were created before timing, via `memoryview(storage.tobytes()).cast('?')`.

Each implementation received three warmups and five samples in each of two
opposite implementation orders. Samples batched 100, 20, or one conversion for
the three respective lengths. The measured call was
`module.tensor(source, dtype=module.float32, device='cpu')` plus
`tuple(output.shape)`. Outputs are eager; after each timed batch the last output
was materialized with `np.asarray`, checksummed with a float64 sum, and checked
against the interpreter's values. NumPy export/checksum, interpreter startup,
imports, installs, and builds were outside the timed region for both sides.
Before/after runs used the same driver and dependency environments. Baseline
incorrect cells were recorded as incorrect and earn no performance credit.

The tradeoff is bounded by the input: for `n` elements with `k` noncanonical
bytes, conversion uses `k` Python item conversions, an `n`-byte temporary copy,
and `4n` bytes of output, all O(n). There are no Python item conversions when
`k = 0`. Compared with the old pre-3.14 path, dense noncanonical inputs trade
native byte decoding for interpreter calls; compared with the old 3.14 path,
canonical inputs avoid those calls and add the temporary byte copy.

Contiguous conversion timings in microseconds: median [minimum–maximum] of
ten samples. `*` marks incorrect baseline results, included only as a cost
diagnostic. Reference timings are from the repaired-wheel runs.

| Python build | Elements / storage | Before native | After native | PyTorch 2.13 |
| --- | --- | ---: | ---: | ---: |
| GCC 3.12 | 16 / canonical | 1.71 [1.66–2.18] | 1.59 [1.56–1.85] | 3.46 [3.35–3.85] |
| GCC 3.12 | 4,096 / canonical | 11.20 [11.09–12.61] | 7.63 [7.38–9.17] | 262.65 [258.07–294.82] |
| GCC 3.12 | 1,048,576 / canonical | 2,557.42 [2,461.86–3,912.08] | 1,520.06 [1,493.14–3,108.04] | 71,024.77 [69,894.43–72,941.11] |
| GCC 3.12 | 1,048,576 / sparse | 2,543.15 [2,508.83–2,637.21]* | 1,751.02 [1,695.66–1,884.76] | 70,686.79 [70,280.77–82,344.26] |
| GCC 3.12 | 1,048,576 / dense | 2,472.14 [2,452.53–2,674.17]* | 39,327.83 [37,725.24–43,019.79] | 65,488.35 [65,196.48–66,440.37] |
| Clang 3.12 | 16 / canonical | 1.54 [1.45–2.30] | 1.41 [1.38–1.66] | 3.01 [2.95–3.36] |
| Clang 3.12 | 4,096 / canonical | 11.54 [10.77–13.42] | 7.33 [7.20–8.91] | 231.83 [227.87–232.99] |
| Clang 3.12 | 1,048,576 / canonical | 2,468.28 [2,434.28–4,045.86] | 2,165.70 [1,521.66–3,399.36] | 54,816.05 [53,261.07–58,673.89] |
| Clang 3.12 | 1,048,576 / sparse | 2,476.76 [2,441.33–2,500.20] | 1,689.35 [1,628.98–1,733.61] | 55,639.62 [53,308.64–56,999.59] |
| Clang 3.12 | 1,048,576 / dense | 2,485.80 [2,455.65–2,517.57] | 33,831.01 [33,753.98–34,414.23] | 55,717.56 [53,673.19–63,747.93] |
| Clang 3.14 | 16 / canonical | 1.75 [1.72–1.91] | 1.41 [1.39–1.76] | 3.00 [2.94–3.24] |
| Clang 3.14 | 4,096 / canonical | 133.00 [131.54–136.74] | 7.31 [7.22–11.18] | 225.84 [223.71–235.63] |
| Clang 3.14 | 1,048,576 / canonical | 32,281.68 [31,700.15–33,250.92] | 1,550.04 [1,518.70–1,608.41] | 55,746.19 [54,821.78–57,735.90] |
| Clang 3.14 | 1,048,576 / sparse | 32,131.41 [32,043.92–33,067.79] | 1,712.86 [1,680.26–1,966.32] | 57,021.26 [56,034.78–58,070.25] |
| Clang 3.14 | 1,048,576 / dense | 32,324.65 [31,798.39–36,547.12] | 32,346.58 [32,060.48–33,246.07] | 56,936.37 [55,500.82–58,742.39] |

All 81 repaired workload cells (27 per interpreter) matched the interpreter
and PyTorch. Across all three layouts and lengths, the geometric mean canonical
speedup over the baseline was 1.19x on GCC 3.12, 1.16x on Clang 3.12, and 3.41x
on Clang 3.14. The slowest canonical ratio was 0.99x on Clang 3.14; tiny changes
and the wide large-allocation sample ranges should not be overinterpreted.

Dense noncanonical Clang 3.12 buffers are a real regression: at one million
contiguous elements the correct old conversion took 2.49 ms and the repair took
33.83 ms (13.61x). Reversed/strided dense buffers were 4.11x/3.95x slower. On
Clang 3.14, dense contiguous conversion was essentially unchanged, while
reversed/strided dense cases cost 25%/23% more, consistent with the added byte
copy. Sparse large contiguous inputs improved on all three builds. Every
repaired cell remained faster than the stock reference in this run; the
uncapped geometric mean reference/native ratio over all 27 equally weighted
cells was 4.33x, 3.64x, and 3.72x respectively. These fixed-seed diagnostics do
not establish general performance parity or change any evaluation denominator.
