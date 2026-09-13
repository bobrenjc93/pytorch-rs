# Clean post-commit evidence

Candidate `779e512cf87f8774c103dedf255eb76ad52f0508` and actual main
`a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4` were measured from clean checkouts with
the unchanged `public-default-compile-v2` gate. This refresh follows the NaN-guard
and diagnostic-consumer repairs. It does not replace Burner's independent review
or exact-head qualification.

| Build | Measured commit | Weighted coverage | CUDA performance | Common-success reference/native latency ratio |
| --- | --- | ---: | ---: | ---: |
| [main](main.json.gz) | `a281503f3391` | 8% (6/112) | 12% (6/56) | 1.820680653 |
| [candidate](candidate.json.gz) | `779e512cf87f` | 9% (8/112) | 12% (8/56) | 1.266226496 |

Both reports are valid, non-diagnostic measurements with empty source/build
working-tree status. They retain all 112 coverage cells and 56 CUDA cells, both
variants, changed-value checks, unsupported and failed outcomes, both CUDA
implementation orders, five warmups, 17 synchronized samples and cold/steady
timings. No workload, tolerance, weight, reference configuration or denominator
changed. No reference or infrastructure failure was discarded.

The candidate's two additional passing cells are CUDA `scalar_guard` variants.
The other program in that category remains unsupported, leaving its CUDA
performance contribution zero. Common-success sets differ between builds; the
reported ratios are not a paired speedup claim. Fixed-corpus scores do not
establish general Inductor parity.

## Source and build identities

The candidate runs from this worktree. Main runs from the independent detached
repository at `target/postcommit-main/`, initialized and fetched from the local
`main` ref without changing a parent or sibling checkout. Its interpreter,
environment, dependencies, builds and caches also live beneath that directory.
Worktree-local Python, uv and Cargo caches were copied into it before setup.
Build/dependency caches may be warm; each gate builds and installs a new locked
release wheel. Each worker has initially empty Inductor/Triton caches. Candidate
and main GPU measurements ran sequentially on the same H100.

Both use Python 3.12.12, PyTorch 2.13.0+cu130, Rust 1.92.0, GPU 0
(`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`) and driver 580.82.07. Burner held the
`gpu` and `cpu-heavy` resources. Generated kernels use NVRTC 13.0 and CUDA runtime
13000; installed nvcc 12.6 is recorded separately. Reports preserve setup times,
source/build/wheel hashes, runtime paths, GPU inventory and before/after snapshots.

## Dispatched modules and regressions

All three committed capture tools ran with the candidate gate's exact installed
wheel and clean sources. Each capture identifies the executable selected by the
successful call:

- Positional [CUDA](../kernel.cu), [PTX](../kernel.ptx.gz), [source manifest](../source-manifest.json.gz) and [provenance](../provenance.json).
- Bounded-broadcast [CUDA](bounded-codegen/kernel.cu), [PTX](bounded-codegen/kernel.ptx.gz) and [provenance](bounded-codegen/provenance.json), including the return to an earlier shape after another module.
- Pointwise-JIT [CUDA](jit-codegen/kernel.cu), [PTX](jit-codegen/kernel.ptx.gz) and [provenance](jit-codegen/provenance.json), recording each observed dispatch.

[Commands and outcomes](capture-checks.json) also record 16 passing focused tests:
the five unchanged shape-history methods (including the former 18 failing
subtests), eight persistent-specialization methods (including twelve signed/payload
NaNs in both orders for parameter/global/closure sources), the CPU NaN guard/IR
regression and both diagnostic-consumer tests. Persistent histories keep both
ordinary default wrappers alive; assertions and limits are unchanged. Unrelated
full validation suites were not rerun.

The [verification receipt](verification.json) checks source, wheel, installed
native extension, worker output/log hashes, matrix completeness and all three
captured modules. [Logs](logs.json.gz) are preserved losslessly; report files retain
raw latency samples and raw tensor-observation hashes. Full raw observation files
remain at the worktree-local paths in the receipt.

Earlier measured reports and failures remain unchanged. The former root-level
`7a74596b` capture was copied byte-for-byte into [its original record](../postcommit-7a74596b/codegen/provenance.json)
before refreshing the root capture. The [operator repair archive](../operator-review-fix/README.md)
retains the original ninth-NaN rejection, consumer failures and development checks
with their original identities. These earlier records supply no current performance credit.

## Reproduce

Start at the measured clean commit, keeping environments and caches inside its
checkout. Run the repository wrapper with no diagnostic or backend overrides:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/postcommit-779e512c/REPORT.json
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python \
  docs/diagnostics/compile-pointwise-positional/capture.py \
  "$PWD/target/new-positional-capture" "$PWD/target/default-compile-eval/wheels.DIRECTORY/torch_rs-WHEEL.whl"
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python \
  docs/diagnostics/compile-pointwise-broadcast/capture_bounded.py target/new-bounded-capture
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python \
  docs/diagnostics/compile-pointwise-jit/capture.py target/new-jit-capture
```

Run the same gate from clean main for the comparison. Exact regression commands,
cache paths, timestamps and return codes are preserved in the command receipt.
Burner owns artifact commits, independent review, final qualification and delivery.
