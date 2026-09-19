# Default-compile identity contract check

Untimed developer checks of source `f67248cd3fa4d4a061bb9ce0857374f0828b880f`
on GPU0, 2026-09-18. The runtime is unchanged; this repair documents observable
limits beside the [public alias contract](../compile-pointwise-jit.md#observable-differences-from-upstream-default-compilation).
It supplies no new performance or qualification claim.

Three small programs ran three calls each in separate native-default,
PyTorch-default and PyTorch-eager processes. Ordered alias writes preserved
values/storage on all three paths, but upstream default compilation returned
distinct wrappers for repeated receiver references. Returned constant tuples
also had different source-object identity under upstream default compilation.
Native matched independent eager identity semantics. These two cases differed
on all three calls; the strict comparisons remain failures of equivalence.

A simpler warm `+0.0, -0.0, +0.0` scalar history agreed bitwise across the three
paths. It does not resolve the [historical exceptional-value warm failure or
8/16 strict identity failures](default-alias-mutation-20260918.md).
No historical capture was replaced or resampled, and no numerical tolerance changed.

Four focused GPU regressions passed (receiver/storage identity, current unary
scalar rejection and cache recovery, constant identity, and native-only execution).
Six portable admission tests and twelve documentation/quickstart tests passed.
The native probe blocks PyTorch imports and Python entrypoint replay. No native
execution defect was established; changing receiver identity to match the observed
reference would violate the required native contract. Historical archive size is
unchanged: the repair preserves those records rather than removing audit evidence.

The owned environment used Python 3.12.13, PyTorch 2.13.0+cu130, Rust 1.92.0
and Maturin 1.15.0. GPU0 was H100 `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`,
driver 580.82.07; pre/post snapshots showed 4 MiB and 0% utilization. Snapshots
are not exclusivity evidence. Native reported NVRTC 13.0; its post-call map
snapshot retained CUDA runtime 13 but not NVRTC. The reference-process NVRTC
hash is not attributed to the native process. Inventory nvcc was 12.8 V12.8.93.

Raw commands/exits, complete observations, per-process providers, source export,
release wheel/RECORD/import verification and SHA256/size manifest are retained in
`target/compile-quality-repair/compile-quality-repair-author.tar.gz`.
Archive: 4,570,510 bytes, 49 verified payloads; SHA256
`cec33e31b3daa662f906e0d8102d47d9e73b311d079216d3e553e13238a0906f`.
`MANIFEST.json` records every member size/hash (SHA256
`846591486a1fa05e4adcadf8c123670b8a2fb6cb51adeb8d4507234a499019f9`).
Burner/Main must retain the raw archive before cleanup.
These are scoped developer checks, not an independent review or canonical score.
