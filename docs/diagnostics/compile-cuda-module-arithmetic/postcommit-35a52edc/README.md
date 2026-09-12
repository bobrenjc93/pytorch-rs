# Clean-commit capture after owner-initialization review

Non-scoring validation of implementation commit
`35a52edcd4af20ce655f6bffa1779aa7bb1b0781`, based on
`64dd296ee65bff37d5d2868506de2e61aa5effb5`. Every measured command verified the
commit and empty Git status before and after execution. The release wheel was
freshly built from that clean commit's source export. Evidence publication and
its documentation link were added only after measurements finished.

This capture completes the deferred clean-commit step for the
[owner-initialization review fix](../owner-startup-review/README.md). It retains
the genuine immutable callable owner during eager package initialization,
before the compiler frontend is imported lazily. The fresh-process tests cover
counterfeit, hostile and deleted exported owners, exact rejection errors,
canonical aliases, cold/warm execution and cache recovery under all four policies.
The reviewer's original counterfeit probe rejects with
`CompileTraceUnsupportedError`; genuine negation remains correct with a hostile
exported owner on both cold and warm calls, without invoking its callback.

The [initial clean capture](../postcommit-e2378805/README.md),
[original baseline/development evidence](../README.md) and review-development
failures remain byte-for-byte unchanged. They retain their original source
identities. This capture does not replace independent review, evaluation or
delivery gates, and does not adopt PR #1970/#1971.

| Check | Result |
| --- | --- |
| Complete compiler sweep: every `test_compile*.py` and `test_top_level_compile.py`, 59 modules | 666 run, 651 passed, 15 expected device-mask skips |
| Focused CUDA module arithmetic and eager add/neg reference tests | 34 run, 29 passed, 5 two-device skips |
| CUDA-hidden frontend, CPU compiler and eager reference tests | 118 run, 109 passed, 9 hardware skips |
| New and existing restoration checks on GPUs 0–1 | 2 passed |
| Rust default all-targets | 395 passed, 0 ignored |
| Python-binding native graph/planner checks, GPU 0 | 15 passed |
| CUDA-hidden native graph/planner checks | 15 passed; hardware sections return early |
| Formatting, default and Python-binding Clippy | Passed |
| README/docs smoke and compiled guide example | 12 tests passed; example passed |
| Source/wheel/native verification, interpreter inventory, seeded input hashes and original review probes | Passed |

Counts are test methods and overlap between suites. The focused suite includes
the six startup-substitution subprocess cases, four policies and all 32 seeded
Inductor spelling cells. Their input hashes are unchanged from the committed
review-development capture. No test, scoring corpus, evaluator, dependency or
supported behavior changed in this evidence step. CPU global call capture
remains unsupported; these spellings do not expand CPU graph support.

[Command receipts](commands.json) record exact argv, timestamps, durations,
statuses, test counts/skips, selected environment, clean Git checks, GPU
snapshots and raw-log hashes. [Capture metadata](capture.json) reuses the
unchanged input-manifest chain. The complete [build record](build-record.json.gz)
is losslessly compressed and retains its full source manifest.
[Before](logs/provenance-before.log) and [after](logs/provenance-after.log)
verification bind installed Python/native files to the wheel and source export.
[Publication verification](verification.json) separately records the
artifact-only dirty state, final hashes and documentation checks.

## Environment and reproduction

The worktree-local canonical `.venv` and verified CPython 3.12.14 interpreter
were reused. The complete interpreter file/mode/internal-symlink inventory was
rechecked before and after; no interpreter, environment or native package was
copied. Locked dev/reference dependencies were synced and a fresh release wheel
was built with Rust 1.92.0 in an empty Cargo target. Worktree-local Cargo/uv
download caches and CUDA/Inductor/Triton caches were warm.

Native and reference loaded the local CUDA 13.0 runtime (version 13000);
PyTorch was `2.13.0+cu130`, driver 580.82.07. Installed nvcc was 12.6.85 and was
not invoked by the native build, which uses driver JIT of embedded PTX.
Resource use was `gpu` and `cpu-heavy`. Ordinary runs used H100 GPU 0,
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, with `CUDA_VISIBLE_DEVICES=0`.
Only restoration additionally used GPU 1,
`GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`. Portability checks used an empty mask.
Snapshots are observations, not reservations. All temporary files and caches
stayed inside the worktree. These functional checks and their elapsed times do
not establish performance, fusion, general Inductor, training or hardware parity.

At the measured clean commit, use the locked contributor setup and verified
local Python distribution. Copy the parent `env.sh.txt`, `compiler_sweep.py.txt`
and `inductor_inputs.py.txt` into `target/postcommit-35a52edc/`, dropping `.txt`.
Copy this directory's `record.py.txt`, `provenance.py.txt` and `checks.sh.txt` there
in the same way. The build name below must be absent so the repository tool
creates a fresh source export and Cargo target. Paths derive from the worktree.

```sh
. target/postcommit-35a52edc/env.sh
export GIT_OPTIONAL_LOCKS=0
python3 target/postcommit-35a52edc/record.py release-build python3 scripts/build_cuda_add_diagnostic.py --name module-postcommit-35a52edc --revision HEAD
bash target/postcommit-35a52edc/checks.sh
```

The scripts are capture recipes, not scoring infrastructure. `checks.sh.txt`
records every validation command, including the unchanged committed reviewer
probe. `publish.py.txt` and `verify_publication.py.txt` record packaging and
publication verification, using the initial `candidate-audit.json`. Prior
measurements are preserved; managed Burner progress artifacts are untouched.
