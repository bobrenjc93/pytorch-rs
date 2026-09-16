# Clean post-commit attempt — 2026-09-16

The required public comparison is **incomplete**. Candidate C was clean commit
`47b97f203b35f936dc9fc0d38ef1255a68764f93`; baseline B was clean accepted main
`60202557b4f110d07777f585e804ab5f55e1ff7b`. Both source checkouts, environments,
release wheels and caches were rooted inside the current worktree. The committed
consumer and frozen protocol were unchanged. This report records no candidate
speedup, reference parity, score or qualification verdict.

The first reference leg failed before any matrix call:

```text
ValueError: Actual loaded CUDA runtime differs from the single pinned barrier/runtime
```

The process mapped two distinct `libcudart.so.13` files: the explicitly pinned
barrier library in the candidate environment and PyTorch's library in the
separate baseline environment. Both reported CUDA runtime 13.0 and SHA256
`96c42e418cec19054186b9429c321603cc190bf26a18104e19408117a2a817b0`.
Equal library bytes do not satisfy the frozen requirement for one loaded runtime.
Read-only inspection of the installed reference's `torch/__init__.py` found
`_load_global_deps()` calling `_preload_cuda_deps()`, which loads package-local
CUDA libraries by absolute path. The failed record preserves both actual paths
and mappings. No runtime check or dependency bytes were changed to bypass this.

| Phase | Outcome |
| --- | --- |
| Hardware-free controls | 33 passed before freeze |
| Clean consumer freeze | Passed; consumer SHA256 `1861fca73674d8c89f6285b07883ab9a83e6719c1b6a0c12cd2be1ae7c4c8b42` |
| B/C release builds and wheel/source/install checks | Passed |
| Candidate H100 preflight | Passed all fixed families, reuse/eviction and ownership/reset controls |
| Leg 0: B-native | Captured all 12 cells, 204 samples, 60 warmups, 12 first calls and 12 churn calls |
| Leg 1: B-reference | Failed runtime provenance before timing |
| Legs 2–7 | Not started after prerequisite failure |
| Offline acceptance | Incomplete sequence rejected: all eight ordered legs are mandatory |

The candidate preflight selected direct code for affine, trig, shared arithmetic,
broadcast and structured results. The unchanged over-cap fixture selected VM
with 325 instructions at both sizes. Receipt calls were untimed companions;
release-unobservable compiler/build/upload counters remain explicitly marked
unobservable. No candidate timing or reference timing was captured. The baseline
capture retains every sample and its median, quartiles, MAD, minimum and maximum;
it supplies no standalone before/after conclusion.

Builds used the consumer's release/locked Maturin command, Rust 1.92.0 and Python
3.12.12. The selected runtime and NVRTC were worktree-local CUDA 13.0 libraries;
system nvcc 12.6 was recorded but did not compile native JIT kernels. Reference
PyTorch was `2.13.0+cu130`. GPU0 was H100 UUID
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07, with exactly one
visible device. The attempt used fresh process/cache directories and the same
runtime barrier pin. No other GPU was used or job interrupted. The post-failure
snapshot reported no compute process. Native NVRTC preload remains outside
cold-call timing as declared by the consumer.

The [manifest](manifest.json) inventories 16 immutable records in
[attempt-1.tar.gz](attempt-1.tar.gz): controls, freeze, build and installation
records, candidate preflight, both ordered leg records, logs, and the final
integrity/device snapshot. All archive members were hash-verified. Before writing
this evidence, both checkouts were rechecked against their recorded clean source
inventories; baseline sample statistics and finite values were checked without
removing samples. Original development and historical evidence remains unchanged.

The remaining prerequisite is to reconcile the separate B/reference environment
with the frozen single-loaded-runtime contract through independent review.
No hardware leg was retried and no missing leg was fabricated. A complete new
authorized attempt, unchanged full qualification and canonical independent review
remain outstanding; this evidence-only step does not approve the candidate.
