# Clean structured-output evidence at 36954660

Measured implementation: `3695466080106e03c4965c0b8801290cf5adc39b`, based on
`main` at `30a3b504ef4d43bf2958998cc39545996cc09970`. These fresh captures qualify
the committed implementation; the earlier uncommitted development captures remain
separate and unchanged. The complete candidate diff introduced no measured reports
to refresh. This record supplies the task-required clean-commit captures.

## Results

The committed tests ran unchanged against a newly built release wheel:

| Check | Passed | Explicit skips |
| --- | ---: | ---: |
| H100 structured outputs, literal loops and shape branches | 70 | 2 two-device tests |
| Dedicated two-physical-device checks | 8 | 0 |
| Native release pointwise and allocation/launch/completion ownership | 39 | 0 |
| Native release Python-conversion failure ownership | 1 | 0 |
| Portable pointwise tests, each CPython version | 143 | 148 hardware tests |

Portable versions were 3.10.19, 3.11.15, 3.12.12, 3.13.13 and 3.14.5. Hardware
skips are not CUDA passes. The conversion test passed while emitting CPython's
`Could not find platform dependent libraries <exec_prefix>` message; its complete
log is preserved.

The [raw archive](raw-captures.tar.gz) includes 66 materialized native/reference
comparisons and their dispatched CUDA source/PTX, plus the child-process output
checking no PyTorch import or original/helper body replay. Tests cover signed and
shared products, competing products, constants, ordering, zeros/nonfinite values,
freshness, aliases, metadata, helpers/loops/branches and the maximum output/scalar
ABI. Every captured PTX has one kernel entry. Existing tolerances and ordinary
`torch_rs.compile(fn)` / stock `torch.compile(fn)` defaults were unchanged.

## Source and runtime provenance

[Measurements](measurements.json.gz) record the clean commit, before/after status,
tracked source hashes, timestamped commands, environment, interpreter/import paths,
wheel members, GPU snapshots, runtime libraries and test outcomes. All tracked
files remained byte-identical through capture. Every interpreter's installed
Python package and extension matched the wheel; no source overlay was used.

- Wheel SHA256: `be0853f9789befd43bfbf23751e287b634452c670c0bc824b2500163397cd258`.
- Locked offline release build: Rust 1.92.0, Maturin 1.15.0, fresh Cargo target,
  thin LTO and one codegen unit. Compiler caches started absent.
- NVIDIA H100, driver 580.82.07; ordinary runs used `CUDA_VISIBLE_DEVICES=0`,
  and dedicated device tests used only `0,1`. Full UUIDs are retained.
- Stock PyTorch 2.13.0+cu130; native NVRTC 13.0 and loaded CUDA runtime 13000.
  PATH `nvcc` is 12.6.85; NVRTC compiles the pointwise kernels.

Use the release build and native-extension verification procedure in
[CONTRIBUTING.md](../../../../CONTRIBUTING.md), then run the unchanged tests:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_RS_STRUCTURED_EVIDENCE="$PWD/target/default-compile-eval/new-capture/raw" \
  .venv/bin/python -m unittest -v tests.test_compile_pointwise_structured_outputs \
  tests.test_compile_pointwise_loops tests.test_compile_pointwise_shape_branches
```

The archived `run_capture.py` records exact build, install, two-device, native and
portable commands. It orchestrates existing repository tooling; it does not change
the workload matrix or assertions. Use a new report directory for any rerun.

## Retention and limits

The [manifest](raw-retention-manifest.json) verifies all 239 archived raw files and
identifies the retained wheel, committed source archive and full reference-cache
archive under `target/default-compile-eval/structured-outputs-postcommit-36954660/`.
All archive members were verified byte-for-byte. The original 798 development
evidence files were also verified unchanged, including failures and nested-report
archives. No disposable-worktree cleanup occurred. The canonical report roots are
covered by Burner's outer archival observer; this record does not claim an
external archive path.

No official coverage/performance evaluation or unrelated full suite was rerun,
and no score is claimed. Earlier broader Rust NaN-bit failures and their baseline
reproductions remain development records. This step changes only new evidence,
its documentation and the compiler-guide link. Historical reports, implementation,
tests, dependencies, harnesses, evaluators and managed progress artifacts remain
unchanged. Independent review and normal merge gates are still required.
