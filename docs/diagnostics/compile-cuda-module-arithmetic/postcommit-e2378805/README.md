# Clean-commit CUDA module arithmetic capture

Non-scoring validation of implementation commit
`e2378805eee2d1092d21c0678f503d888851e47c`, based on
`64dd296ee65bff37d5d2868506de2e61aa5effb5`. Every measured command verified
that commit and empty Git status before and after execution. The release wheel
was freshly built from its clean source export. These reports and this guide
were added only after measurements completed.

The [development capture](../README.md), including the original 32-case gap,
failed attempts and final development checks, remains byte-for-byte unchanged.
This capture completes its deferred clean-commit step. It does not replace
independent review, evaluation or delivery gates, or adopt PR #1970/#1971.

| Check | Result |
| --- | --- |
| Complete compiler sweep: every `test_compile*.py` and `test_top_level_compile.py`, 59 modules | 664 run, 649 passed, 15 expected device-mask skips |
| Focused CUDA module arithmetic and eager CUDA add/neg reference tests | 32 run, 27 passed, 5 two-device skips |
| CUDA-hidden frontend, CPU compiler and eager add/neg reference tests | 116 run, 108 passed, 8 hardware skips |
| New and existing restoration checks on GPUs 0–1 | 2 passed |
| Rust default all-targets | 395 passed, 0 ignored |
| Python-binding native graph/planner checks, GPU 0 | 15 passed |
| CUDA-hidden native graph/planner checks | 15 passed; hardware sections return early |
| Formatting, default and Python-binding Clippy | Passed |
| README/docs smoke and compiled guide example | 12 tests passed; example passed |
| Wheel/source/native verification, interpreter inventory and seeded input hashes | Passed before/after checks |

Counts are test methods, with overlapping suites; they should not be summed as
unique cases. The focused tests include all four supported policies and the 32
seeded Inductor spelling cells. The complete sweep retains the existing corpus,
unsupported outcomes and assertions. CPU module/direct-import call capture
remains rejected; this change does not extend CPU graph support.

The [command receipts](commands.json) retain exact argv, timestamps, elapsed
seconds, exit codes, test summaries and skip reasons, selected environment,
clean Git checks, GPU snapshots and raw-log hashes.
[Publication verification](verification.json) separately records the final
artifact-only diff, link checks and repeated docs smoke; it does not label that
report-writing stage a clean checkout. [Capture metadata](capture.json)
links the unchanged source/test input manifest and freshly regenerated input
hashes. [Build provenance](build-record.json.gz) is the complete, losslessly
compressed output of the repository build tool, including its full source
manifest. [Before](logs/provenance-before.log) and
[after](logs/provenance-after.log) verification bind the installed Python files
and native extension to that wheel, source export and worktree.

## Environment and scope

The existing worktree-local canonical `.venv` and verified CPython 3.12.14
installation were reused. The complete interpreter file/mode/internal-symlink
inventory was rechecked before and after; no interpreter or environment was
copied in this step. Locked dev/reference dependencies were synced, then a new
release wheel was built with Rust 1.92.0 in an empty Cargo target. Worktree-local
Cargo/uv download caches and CUDA/Inductor/Triton caches were warm. The original
interpreter setup remains documented in [historical provenance](../interpreter.json).

Native and reference execution loaded the local CUDA runtime 13.0 (runtime
version 13000); reference PyTorch was `2.13.0+cu130`. Installed nvcc was 12.6.85
and was not invoked by the native build, which uses driver JIT of embedded PTX.
The driver was 580.82.07. GPU 0 was NVIDIA H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`; restoration additionally used H100
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296` as GPU 1.

Resource use was `gpu` and `cpu-heavy`. Ordinary commands used
`CUDA_VISIBLE_DEVICES=0`, restoration used `0,1`, and portability checks used
an empty mask. Snapshots are observations, not reservations. All build outputs,
environments, caches and temporary files stayed within the worktree.

Cold/cache-hit tests describe compiled wrapper behavior. Warm persistent caches
and functional test elapsed times do not establish cold compilation latency,
speed, fusion, training, general Inductor support or hardware parity. The known
Inductor signed-zero behavior and native/eager bit checks remain documented in
the unchanged development report.

## Reproduction

Use a clean checkout at the measured commit and the locked contributor setup
with the verified local Python 3.12.14 distribution. Before adding reports to the
checkout, copy the parent `env.sh.txt`, `compiler_sweep.py.txt` and
`inductor_inputs.py.txt` into `target/postcommit-e2378805/`, dropping `.txt`.
Copy this directory's `record.py.txt`, `provenance.py.txt` and `checks.sh.txt`
there in the same way. These are capture recipes, not scoring infrastructure.
The build name below must be absent; the build tool creates a fresh export and
release target. Paths derive from the current worktree.

```sh
. target/postcommit-e2378805/env.sh
export GIT_OPTIONAL_LOCKS=0
python3 target/postcommit-e2378805/record.py release-build python3 scripts/build_cuda_add_diagnostic.py --name module-postcommit-e2378805 --revision HEAD
bash target/postcommit-e2378805/checks.sh
```

`checks.sh.txt` records every validation command. `publish.py.txt` records the
lossless packaging and verification recipe; it also consumes the recorded
`candidate-audit.json`. Existing baseline, development and unrelated evidence
are preserved. No implementation, dependency, test, harness, evaluator or
managed Burner progress artifact changed in this evidence step.
