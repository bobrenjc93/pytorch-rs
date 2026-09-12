# Default compiler baseline: 0 coverage, 0 CUDA performance

Measured on 2026-09-12 from clean commit
`5688dab40b3dcb1e1ae0fe788f7123f2288b46d2`, using `public-default-compile-v2`.
The Rust/Python implementation and dependency locks are unchanged from main at
`a8ea1225406d3aaa0412d74091ecd8aef97ea1a7`. This is a measurement correction, not
an implementation regression. Adoption into the main campaign awaits review.

| Gate | Definition | Passing candidate cells | Score |
| --- | --- | ---: | ---: |
| Default compile coverage | `eval_a61c0e71` v4 | 0 / 112 | 0 / 100 |
| Default compile CUDA performance | `eval_6f98c42d` v3 | 0 / 56 | 0 / 100 |

All 28 reference programs passed default PyTorch Inductor on CPU and CUDA,
including the reversed-order CUDA runs. There are no common-success CUDA
workloads, so a relative latency ratio is **undefined**, not 1× or parity.
Ordinary native default compilation raises `NotImplementedError`; additional
programs expose missing module/custom-autograd APIs and CUDA gradients.

## Evidence

- [Complete scored report](baseline-2026-09-12.json): both command results,
  every denominator cell, reference samples/counters, failure reasons,
  source/wheel/library hashes, setup costs, and hardware metadata.
- [Contract and reproduction](../../torch-compile-default-evaluator.md):
  unchanged public compile calls, corpus, weights, and symmetric tolerances.
- [Initial v1 capture](../default-compile-v1/README.md): retained unchanged;
  superseded after review found its asymmetric bfloat16 tolerance.

Both commands ran sequentially through Burner's command evaluator at
`ea8866960b1553f7390d66b929a8969836751d85`, from 15:48:00 to 15:53:24 UTC. No
scheduler, feature admission, PR merge, or canonical history update ran.

Hardware was a real NVIDIA H100 (compute capability 9.0), GPU UUID
`8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07. The host has eight
H100s; both frameworks used the same single visible device. Python was 3.12.14,
reference PyTorch 2.13.0+cu130, with one host compute thread and CUDA 13 runtime
synchronization. Release-build/dependency caches were warm; each reference
worker started with empty Inductor/Triton disk caches.

The archive was audited against 123 source-file hashes and 20 raw-output/log
hashes. All 140 reference program executions passed across the five reference
workers; all 140 candidate program executions failed across the matching
workers. Repeated runs do not enlarge the fixed 112/56 scoring denominators.
143 focused tests and Ruff checks passed; Burner's 224 tests and builds passed.

Full-value observations are retained locally under
`target/default-compile-eval/run-20260912T154801Z-956cf1b0/` and
`target/default-compile-eval/run-20260912T155123Z-6f8434b1/`. Their paths and
hashes are in the committed report; the large arrays are not checked into Git.
The report includes all scoring inputs and timing samples, with full-value
correctness checks completed before acceptance.

After reviewing/adopting the definitions, use Burner's normal baseline workflow
at the actual adopted main revision. Do not relabel this branch's capture, carry
old 100 scores forward, or treat the score correction as an implementation gain.
