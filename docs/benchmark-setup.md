# Setup costs for the September 9 composite reports

The refreshed [stack](top-level-stack-release-timings.md),
[subtract](top-level-subtract-release-timings.md),
[CPU compile](torch-compile-cpu-release-timings.md), and
[H100 compile](torch-compile-cuda-h100-release-timings.md) reports share source
commit `e2f40ff16f8aba5216bc51699b56571e7e82a3e4` and one release-wheel setup.
Their original setup wall times are unavailable: the capture directory recorded
in their JSON (`composite_b4e9c495/target/`) no longer exists, and no setup logs
were retained in the tracked tree. Workload durations and CUDA first-call times
cannot substitute for dependency installation or release-build times.

The [setup record](benchmark-setup/2026-09-09-e2f40ff/setup.json) contains a
**later same-code setup rerun**, captured September 9, 2026, 17:29–17:30 UTC.
These are not timings from the original benchmark capture. No workloads were
rerun; all existing raw JSON, result tables, samples, seeds, denominators, and
aggregates are preserved. The four reports share these costs; do not sum them
four times.

This disclosure follows the [historical-report policy](../BENCHMARKING.md#historical-release-timing-reports).
The source revision is the one measured by the four workload reports. Later
changes, including Rayon dependencies and additional native CUDA execution, are
outside this snapshot. The recorded `capture_checkout_commit` identifies the
checkout that exported the historical source, not the source built by the rerun.
These timings provide no setup-cost or performance claim for the current HEAD.

| Setup stage | Wall seconds | Initial state | Exit status |
| --- | ---: | --- | ---: |
| Create virtualenv | 0.056061 | Absent virtualenv, installed system Python | 0 |
| Install Python dependencies (`uv sync --locked`, dev + reference) | 14.627186 | Empty UV cache, new virtualenv; includes network downloads | 0 |
| Fetch Rust dependencies (`cargo fetch --locked`) | 0.698135 | Empty Cargo registry; includes network downloads | 0 |
| Build release wheel (`maturin build --release --locked --offline`) | 45.108150 | No build outputs; fetched dependency sources cached | 0 |
| Install project wheel (`uv pip install --force-reinstall --no-deps`) | 0.133986 | Dependency-populated virtualenv, local wheel | 0 |

The build includes Rust dependency compilation and wheel packaging. Rust fetching
is separated so the offline build duration excludes network installation. The
rerun started with empty worktree-local caches; the original report describes a
copied Cargo registry, so these cache conditions differ. OS page cache and remote
caches were uncontrolled. Provisioning the preinstalled Python, Rust, uv, system
compiler, driver, and CUDA toolkit is excluded, not reported as zero.

The rerun built a `git archive` of the historical commit, not the newer candidate
checkout. All 831 archived regular files were verified unchanged after setup.
The record identifies the source commit/tree, source archive and lockfile SHA-256
hashes, capture checkout, complete subprocess environment, working directory,
commands, UTC timestamps, monotonic wall durations, statuses, and log hashes.
The [logs](benchmark-setup/2026-09-09-e2f40ff/logs/) retain subprocess output,
including native-extension verification, wheel/runtime provenance, and versions.
Absolute paths describe this capture; recreating it requires a fresh directory
and substituting local paths in the recorded environment and commands. Source
export/extraction and inventory/verification are outside the setup stages above;
the record separately times inventory and verification commands.

The host was Linux x86_64, AMD EPYC 9654, with Rust/Cargo 1.92.0, Python 3.12.13,
uv 0.12.11, maturin 1.14.1, NumPy 2.5.1, and PyTorch 2.13.0+cu130. Release settings
were thin LTO, one codegen unit, `extension-module`, and stripping. GPU metadata
was checked with `CUDA_VISIBLE_DEVICES=0`: NVIDIA H100, driver 580.82.07, CUDA
runtime 13.0, and nvcc 12.6.85. This metadata probe did not repeat CUDA compilation
or the historical workload's runtime preparation/first-call measurements.

Validate disclosure with the standard library, without building or using a GPU:

```bash
python3 scripts/validate_benchmark_setup.py
python3 -m unittest tests.test_benchmark_setup_evidence
```

Schema v1 is enforced by that validator: all setup stages require positive finite
wall durations, UTC timestamps, successful exit status, commands, cache disclosure,
and retained logs. It checks source attribution, log hashes, all four report
links, and hashes of the unchanged workload JSON and result sections. Those
hashes identify the historical snapshots to which the setup costs apply; they do
not certify the recorded worktree paths as current or enforce current-HEAD
provenance. A future current-revision measurement needs its own setup and
workload capture. This validator only checks historical setup disclosure, with
no workload execution or scoring changes.
