# Clean post-commit helper evidence

These captures measured clean implementation commit
`a8211b07d03f19ed73fa7f581272ef28e23eeb06` and clean main commit
`fe7537251a53b54e1064309d28db2631605fb720` on September 13, 2026.
Both builds and all imported packages were inside the admitted
`agent_19fa0fee` worktree. Main was checked out detached in
`target/postcommit-a821/main` inside that worktree. All measurements finished
before adding these evidence files. The original [development records](../README.md)
and their failures are preserved unchanged.

## Results

| Check | Candidate | Clean main |
| --- | --- | --- |
| Fixed public-default-compile-v2 coverage | 12.5%; 12/112 cells passed | 11.0%; 10/112 cells passed |
| Fixed CUDA performance score | 20.0%; 12/56 cells passed | 20.0%; 10/56 cells passed |
| Common-success geometric reference/native latency ratio | 1.272547 | 1.283077 |
| Frozen gate validity | Valid | Valid |
| Helper diagnostic, two H100s and both framework orders | 8 successful legs; 64 paired histories passed | Not run; new helper domain is unsupported |
| Focused helper admission/cache/native tests | 28 passed | Not applicable |
| Physical GPU 1 hardware repeat | 3 passed; 1 explicit two-device reservation skip | Not applicable |
| Rust pointwise regressions | 34 passed | Not repeated |

The only fixed-corpus pass/fail changes are `python_helper` CUDA variants 0 and 1.
Their measured reference/native ratios are 1.164409 and 0.991313, respectively.
The custom-functions category still contains unsupported cells, so its geometric
performance contribution remains zero. The complete 112-cell coverage and
56-cell CUDA denominators, 14 category weights, unsupported results and slow
samples remain in the reports. These are fixed-corpus percentages, not coverage
of all Python programs or general Inductor equivalence. The common-success
ratios describe different success sets and are not a paired speedup claim.

The separate helper diagnostic covers multiple/composed helpers, identity and
runtime-scalar forwarding, scalar-literal returns, and the unchanged frozen
helper factory. Each fresh process keeps its wrappers through four
shape/value/alias states. All 64 paired histories passed fixed `rtol=1e-5`,
`atol=1e-6`, including explicit NaN/infinity/signed-zero checks. Individual median
reference/native ratios range from **0.588314 to 1.576784**; values below one
are slower native results. All samples are retained. Native cold timings include
body-execution policing and are separately reported; steady timings are
unprofiled. The hardware tests cover the original native IR rejection boundary
and native input validation in addition to successful helper programs.

## Reproduction and identities

Burner's canonical `gpu` and `cpu-heavy` locks belonged to `idea_1f7d364a`
through these runs. GPUs 0 and 1 were explicitly selected for helper work; only
GPU 0 was visible for each frozen gate. The reservation record and command
metadata retain full inventories before and after measurements:

- GPU 0: `GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`
- GPU 1: `GPU-11979b85-93e3-21d3-e68f-df37b8a4c296`

Both are H100s on driver 580.82.07. Python was 3.12.14 and reference PyTorch
`2.13.0+cu130`. Both frameworks synchronized with worktree-local CUDA runtime
13.0 (`cudaRuntimeGetVersion=13000`). The native helper kernels used NVRTC 13.0,
`compute_90`, `--fmad=true`, `--ftz=false`, `--prec-div=true`, and
`--prec-sqrt=true`. System nvcc 12.6.85 was recorded but did not generate those
JIT kernels. Rust was 1.92.0. Native processes rejected importing PyTorch.

The unchanged repository wrapper built and installed a new locked release wheel
for each clean tree, verified its native extension, and ran ordinary
`torch_rs.compile(fn)` against ordinary `torch.compile(fn)`/Inductor. Each gate
used one host thread, separate processes and fresh framework caches, five
warmups, 17 synchronized samples, both CUDA framework orders, and the frozen
matrix and tolerances. No compiler limit or reference configuration was changed.
Dependencies and the Python distribution were copied from the existing
worktree-local installation into separate local environments; locked offline
sync verified them. Dependency caches were reused. The release build logs and
setup timestamps are retained; no historical setup duration is reused.

| Identity | Candidate | Main |
| --- | --- | --- |
| Frontend SHA256 | `1c4405368457998bee39d940e521e4fd283c4dcec44a39c241635f35a7d2fc95` | `e76b55a179e6adf4109d32664ab697c3f1e446fc1321aed5e4efc472547bd25c` |
| Wheel SHA256 | `1edea1769752bb9e41571d9d364eb06ab57a2eb2cdb259a3ed588e2e9165b5e2` | `d73898519d8ba89b8001ca28e365bb52445a486e7a009957ec24a1cb0d5e7c76` |
| Imported extension SHA256 | `0fdb7ccb205ece5477f928b4cab896041066c2ff5a26b1914eeb4d5ece9ea697` | `f241c080b85b9dee2ed7cced8160b2d694ea4cfc4a0d7fa4e19c1cf6a0efc409` |

With canonical resources held, run from each clean checkout using its local
interpreter/dependencies and writable caches. The archived `env.sh` and
`record-command.py` contain the exact environment and command recorder. The
measurement commands were:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/postcommit-a821/candidate-fixed.json
# Same unchanged command in the nested clean main checkout, output main-fixed.json.
CUDA_VISIBLE_DEVICES=0,1 python docs/diagnostics/compile-pointwise-helpers/compare.py \
  --output target/postcommit-a821/helpers \
  --wheel target/default-compile-eval/wheels.RnUQc4/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl \
  --devices 0,1
CUDA_VISIBLE_DEVICES=0,1 python -m unittest -v tests.test_compile_pointwise_helpers
CUDA_VISIBLE_DEVICES=1 python -m unittest -v tests.test_compile_pointwise_helpers.HelperHardware
CUDA_VISIBLE_DEVICES=0,1 cargo test --locked --offline --lib pointwise
```

The wheel directory above is the actual recorded build; a new invocation of the
wrapper creates a new directory. Use that newly built wheel when reproducing.
An initial main launcher used an incorrect relative path to the command recorder
and exited 2 before any build or measurement. Its error and the corrected
launch are both retained. Each actual gate ran once; neither had a reference or
infrastructure failure. There was no selection among repeated measurements.

## Retained artifacts and remaining handoff

- [candidate-fixed.json](candidate-fixed.json) and [main-fixed.json](main-fixed.json)
  are byte-for-byte copies of the generated frozen reports, including every
  cell, score, failure, timing sample, setup identity and worker artifact hash.
- [captures.json.xz](captures.json.xz) contains all helper reports, generated
  CUDA/PTX, per-leg logs, source manifest, build/test/command logs, environment,
  resource check and the evidence audit procedure. Its `files` mapping stores
  original relative paths, decoded `text`, `original_sha256` and
  `uncompressed_sha256`. Gzip inputs are decoded losslessly for compression.
- [audit.json](audit.json) records the clean status, verified source/import/wheel
  identities, unchanged frozen evaluator/corpus/definitions, exact pass/fail
  delta, helper kernel hashes and unchanged historical archives.
- [raw-retention-manifest.json](raw-retention-manifest.json) inventories all
  12 original frozen worker reports and their 12 logs, totaling 294,240,375 bytes.
  Every file remains at its recorded worktree-local path and its hash was
  verified. Detailed helper reports are included in the checked-in archive;
  large frozen worker reports remain in their original run directories.

**External retention receipt remains unverified.** The existing separately
reviewed campaign-scoped pre-cleanup observer must preserve the frozen workers'
full reports before ordinary worktree removal. This worker has not changed or
replaced that observer and does not claim its export succeeded. The manifest
provides the exact files and hashes for that handoff. These captures do not
replace Burner's independent review, exact-head qualification or merge gates.

To read the helper summary without extracting files:

```python
import json, lzma
from pathlib import Path
archive = json.loads(lzma.decompress(Path(
    "docs/diagnostics/compile-pointwise-helpers/postcommit-a8211b07/captures.json.xz"
).read_bytes()))
summary = json.loads(archive["files"][
    "target/postcommit-a821/helpers/summary.json.gz"]["text"])
assert summary["passed"]
```
