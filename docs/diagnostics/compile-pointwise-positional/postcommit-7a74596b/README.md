# Clean post-commit evidence

The candidate at `7a74596b5fa7fc17843eda8d90c885dc8cd72f12` and actual main at
`a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4` were measured from clean checkouts with
the unchanged `public-default-compile-v2` gate. The focused H100 rerun passes all
12 persistent-specialization tests, including the 18 formerly failing signed-zero
subtests with their original assertions. These measurements do not replace
Burner's independent review or exact-head qualification.

| Build | Measured commit | Weighted coverage | CUDA performance | Common-success reference/native latency ratio |
| --- | --- | ---: | ---: | ---: |
| [main](main.json.gz) | `a281503f3391` | 8% (6/112) | 12% (6/56) | 1.850404484 |
| [candidate](candidate.json.gz) | `7a74596b5fa7` | 9% (8/112) | 12% (8/56) | 1.237361082 |

Both reports have `valid: true`, `diagnostic: false` and empty recorded working-tree
changes for source and build. They retain all 112 coverage cells, all 56 CUDA
cells, two variants per program, changed-value checks, unsupported outcomes,
both CUDA implementation orders, five warmups, 17 synchronized samples, and
cold/steady timings. Default public compile calls and all corpus tolerances and
weights are unchanged. No reference or infrastructure failure was discarded.

The two additional passing candidate cells are the CUDA `scalar_guard` variants.
The other program in that category remains unsupported, so its CUDA performance
contribution remains zero. Common-success sets differ between builds; the ratios
above are not a paired speedup claim. Fixed-corpus scores do not establish general
Inductor parity.

## Source, build and device provenance

Actual main was measured again in the clean detached checkout under
`target/postcommit-main/` inside this worktree. Its interpreter, virtual environment,
dependencies, build outputs and caches are also rooted there. The checkout was
previously fetched from the local `main` ref into an independent local Git repository;
no parent or sibling checkout was modified. Dependency/build caches may be warm;
each gate invocation builds and installs a new locked release wheel and every
worker receives initially empty Inductor/Triton cache directories. Candidate and
main GPU measurements ran sequentially on the same physical H100.

Both builds used Python 3.12.12, locked PyTorch 2.13.0+cu130, Rust 1.92.0 and H100
GPU 0 (`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`), driver 580.82.07. The candidate
capture used NVRTC 13.0 and CUDA runtime 13000. Installed nvcc 12.6 was recorded
but did not compile the generated pointwise kernel. Burner held the `gpu` and
`cpu-heavy` resources. Exact hashes, setup durations, timestamps, runtime paths,
GPU inventories and before/after snapshots are retained in the reports and the
[module provenance](../provenance.json).

## Capture and verification

The committed [capture tool](../capture.py) was run using the exact wheel from the
candidate gate. The refreshed [CUDA](../kernel.cu), [PTX](../kernel.ptx.gz),
[source manifest](../source-manifest.json.gz) and [provenance](../provenance.json)
identify the actually dispatched module of the independent positional-scalar
example. Source/native wheel bytes and all 128 source/test manifest entries were
verified. The final call selects the earlier runtime specialization, distinct
from both the first static module and the last Boolean specialization.

[Capture commands and outcomes](capture-checks.json) record the successful capture
and the `ScalarShapeSpecializationHardware` and `PersistentSpecializationHardware`
rerun against the clean candidate wheel. All 12 tests pass without reference-only
resets or changed assertions. The [logs](logs.json.gz) retain the full outputs.
No implementation, test, dependency, harness or evaluator changed in this evidence
step. Unrelated full validation suites were not rerun.

The [verification receipt](verification.json) records source/build/wheel checks,
worker hashes, complete cell counts, both orders and all samples. The reports
preserve raw latency samples and worker output hashes. Full raw tensor-observation
files remain at the worktree-local run paths listed in that receipt; worker logs
are additionally checked in losslessly as `logs.json.gz`.

The [earlier clean measurements](../postcommit-4ed105f5/README.md), including their
18 failures, remain unchanged. Their former root-level module capture was copied
byte-for-byte to [its own archive](../postcommit-4ed105f5/codegen/provenance.json)
before refreshing the current capture. Historical main reports, reference probes
and development validation archives are preserved with their original identities;
none supplies current-candidate performance credit.

## Reproduce

Start from the measured clean revision with all environments and caches under its
checkout. Use the repository wrapper, with no diagnostic or backend override:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/postcommit-7a74596b/REPORT.json
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python \
  docs/diagnostics/compile-pointwise-positional/capture.py \
  "$PWD/target/postcommit-7a74596b/codegen" "$PWD/target/default-compile-eval/wheels.DIRECTORY/torch_rs-WHEEL.whl"
CUDA_VISIBLE_DEVICES=0 target/default-compile-eval/venv/bin/python -m unittest -v \
  tests.test_compile_pointwise_runtime_scalars.ScalarShapeSpecializationHardware \
  tests.test_compile_pointwise_runtime_scalars.PersistentSpecializationHardware
```

Use the gate's newly built wheel for the candidate capture. Run the gate from the
independent clean main checkout for the comparison. Exact executed commands,
local cache settings and return codes are in the retained records. Burner owns
independent review, artifact commits, exact-head qualification, publication and merge.
