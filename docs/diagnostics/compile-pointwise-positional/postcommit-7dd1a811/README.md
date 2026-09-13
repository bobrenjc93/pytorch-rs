# Clean synchronized warm-dispatch evidence

This record measures candidate `7dd1a811bfbc8936a3f9bf25fade6f71c9a25af3`, actual
main `a281503f3391bbd98a0ab6de2ff8f0c0a55a12d4`, and unchanged comparison source
`885264b5319d76165e6be8d6df405450b3830c47` from clean checkouts. It completes the
predeclared v2 H100 comparison after the synchronization repair and refreshes the
fixed gates and dispatched modules. This bounded evidence does not establish
general Inductor parity or replace Burner's independent qualification.

## Fixed measurements

| Build | Commit | Weighted coverage | CUDA performance | Common-success reference/native ratio |
| --- | --- | ---: | ---: | ---: |
| [main](main.json.gz) | `a281503f3391` | 8% (6/112) | 12% (6/56) | 1.61949957 |
| [candidate](candidate.json.gz) | `7dd1a811bfbc` | 9% (8/112) | 12% (8/56) | 1.16903351 |

The [candidate](candidate.json.gz) and [main](main.json.gz) reports are valid,
non-diagnostic runs of the unchanged `public-default-compile-v2` gate. Both retain
all 112 coverage cells and 56 CUDA cells, unsupported outcomes, both variants,
changed-value checks, both CUDA implementation orders, five warmups, 17 samples,
synchronization, cold/steady timings and setup provenance. Their common-success
sets differ; the last column is not a paired speedup claim. No evaluator, corpus,
tolerance, denominator, reference compiler configuration or timeout changed.

## Balanced H100 comparison

The committed [plan](../warm-dispatch/measurement-plan.md) ran once in its full
16-leg order on physical H100 GPU 0
(`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`). Build B is clean
[885264b](comparison/build-B.json); build C is clean
[7dd1a811](comparison/build-C.json). Every leg used a fresh process and separate,
empty CUDA, Inductor and Triton caches. Both frameworks used the same locked CUDA
runtime's `cudaSetDevice(0)` and `cudaDeviceSynchronize`, with matching runtime
path/hash/version/API records.

All 16 legs passed. The [verifier](comparison/verification.json) checked all 56
paired histories and 3,264 outputs, including metadata, input invariance and
fresh-output assertions performed by each leg. All seven histories retained two
setup traversals, five warmups and 17 samples. The ordinary default wrappers
stayed alive for each full history; native processes imported no PyTorch. Every
leg is preserved beside the verifier as `leg-00.json.gz` through `leg-15.json.gz`.

| History | B native µs, rounds 1–4 | C native µs, rounds 1–4 | B/C, all four rounds |
| --- | --- | --- | ---: |
| literal_unary | 32.309, 35.363, 33.481, 32.189 | 30.356, 30.506, 30.185, 30.476 | 1.0965 |
| tensor_arithmetic | 40.962, 44.638, 62.324, 42.714 | 36.255, 37.858, 39.560, 39.100 | 1.2305 |
| broadcast_add | 41.382, 41.933, 44.978, 41.293 | 35.784, 37.957, 63.025, 37.827 | 0.9988 |
| repeated_alias | 35.654, 38.058, 39.320, 35.914 | 33.050, 34.302, 33.350, 34.662 | 1.0996 |
| shape_revisit | 39.865, 42.849, 44.477, 40.871 | 35.063, 38.167, 37.747, 37.782 | 1.1294 |
| promoted_capture | 28.483, 30.773, 30.072, 29.685 | 26.106, 26.367, 26.457, 27.345 | 1.1196 |
| eight_boolean_entries | 51.956, 52.646, 51.969, 52.076 | 29.188, 32.745, 33.831, 31.212 | 1.6457 |

Each list contains native median microseconds per call for rounds 1–4 in their
predeclared order. The ratio is the geometric mean of all four B/C median ratios;
values above one favor the corrected source. The [complete summary](comparison/summary.json)
retains all native and reference medians for every round, and the leg reports
retain every cold, warmup and sample duration and output. No round or slow sample
was removed or repeated to obtain a favorable result. Broadcast is slightly
slower across the four rounds (B/C 0.9988), including the slower third corrected
round. These synchronized public call timings do not isolate Python dispatch
overhead or explain the entire prior CUDA score shortfall. They award no fixed-corpus credit.

The [original negative result and CPU profiles](../warm-dispatch/README.md) and
[failed v1 comparison](../postcommit-ab6a3acd/README.md#balanced-comparison-blocker)
remain byte-identical measured records. No v1 reference leg was reused. Their
failures and the earlier same-source timing variation remain part of the record.

## Dispatched modules and checks

The three existing capture tools used the candidate gate's exact installed wheel
and selected the executor actually dispatched by their non-corpus program:

- Positional [CUDA](positional-codegen/kernel.cu), [PTX](positional-codegen/kernel.ptx.gz), [provenance](positional-codegen/provenance.json) and [source manifest](positional-codegen/source-manifest.json.gz).
- Bounded broadcast [CUDA](bounded-codegen/kernel.cu), [PTX](bounded-codegen/kernel.ptx.gz) and [provenance](bounded-codegen/provenance.json).
- Pointwise JIT [CUDA](jit-codegen/kernel.cu), [PTX](jit-codegen/kernel.ptx.gz) and [provenance](jit-codegen/provenance.json).

All 37 focused regressions passed: persistent scalar and shape histories,
signed zeros/NaNs, cache order/reset/failure and structural sharing, no warm body
or eager replay, capture selection, runtime synchronization errors and diagnostic
verification. Existing assertions and limits were unchanged. The full compiler
and Rust suites were not repeated during this evidence-only step.
The [documentation checks](artifact-checks.json) record 12 passing smoke tests,
affected/inbound links, manifest relationships and unchanged historical artifacts.

The [verification receipt](verification.json) checks clean source/build identities,
wheel and imported package bytes, full matrices, sample counts, raw worker hashes,
selected CUDA/PTX hashes and the entire balanced comparison. [Lossless logs and
command receipts](logs.json.gz) preserve builds, runs and focused checks. Raw
worker observation files remain at the verified worktree-local paths recorded in
the receipt. Historical captures and failure records are unchanged.

## Reproduction and provenance

Candidate sources are this worktree; independently initialized detached `885`
and main checkouts are under `target/postcommit-7dd1a811/before` and
`target/postcommit-7dd1a811/main`. Every interpreter, dependency, wheel, build,
temporary file and writable cache remains inside this worktree. Dependency/build
caches may be warm; each build produced a fresh locked release wheel, and the
fixed evaluator created fresh worker compiler caches. GPU measurements ran
sequentially on the same H100, with one host thread.

Reports preserve Python 3.12.12, PyTorch 2.13.0+cu130, Rust 1.92.0, driver
580.82.07, selected CUDA runtime/NVRTC versions, installed nvcc 12.6 and GPU
before/after snapshots. Commands, timestamps, script/source/build/wheel hashes,
import paths and setup times are recorded separately for every build and leg.

At each measured clean revision, run the unchanged gate with an output beneath
that checkout:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_torch_compile_default.sh \
  --metric both --output target/new-clean-report.json
```

Use the three existing capture scripts with the newly installed candidate wheel;
exact invocations are archived. Follow the linked plan for fresh builds and all
16 v2 legs, using new output directories. Burner owns artifact commits, independent
review, exact-head qualification and delivery.
