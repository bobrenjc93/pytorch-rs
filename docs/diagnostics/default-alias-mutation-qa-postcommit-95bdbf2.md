# Alias-mutation QA: committed-source validation

Measured commit `95bdbf21ebb0bde911d6b49942b96bb3d63dbaa0` on 2026-09-18.
This focused correctness capture supplements the unchanged
[dirty author evidence](default-alias-mutation-qa-20260918.md); it is not a
performance measurement, canonical evaluation, or review approval. The
[compiler contract](../compile-pointwise-jit.md#alias-only-views-and-scalar-mutation)
remains authoritative.

A clean Git export supplied every build and test input, verified against Git
blob IDs and SHA256s. A fresh, offline, locked release build used an explicitly
selected worktree-owned interpreter and Cargo output. The wheel's 66 RECORD
entries and Python source bytes were verified before selecting its extracted
package through `PYTHONPATH`; no installation was performed. Native unit tests
used a separate debug test build of the identical committed Rust sources.

| Check | Result |
| --- | --- |
| Public default alias-mutation and input-transpose regression modules, GPU0 | 63 passed |
| Same modules, CUDA hidden | 63 run, OK; 34 hardware skips |
| Native CUDA graph bridge tests, including late valid packing and hostile callbacks | 14 passed; 294 unrelated tests filtered out |
| Actual Python 3.10 keyword execution, native-only | 9 valid calls and 5 pre-write rejections passed |
| Release source/RECORD/import and native-only runtime/provider verification | Passed |

The untimed public-default tuple characterization again records equal values,
within-call sharing and fresh dynamic containers. Native preserves original
constant-pool identity across warm calls/reset/code replacement. PyTorch
2.13 preserves warm tuple reuse but not original source-object identity, and
changes tuple identity after reset. Earlier strict identity failures and the
warm signed-zero reference divergence remain limitations, not waived parity.
No timing campaign or unrelated full-suite rerun was performed.

Two setup attempts are preserved: the initial local `python` launcher selected
its external base interpreter and import verification failed for missing
`typing_extensions`; a new release build with explicit `python-owned` passed.
No shared environment was installed into or repaired. The first native filter
selected zero tests; the corrected existing module filter ran all 14 bridge tests.
Neither unsuccessful attempt is counted as validation.

The provider report initially collided with its command-receipt filename. The
command receipt is preserved; only the untimed provider check was repeated with
a distinct label. Its worktree-dirty flag reflects this new untracked evidence
page; the exported build inputs and tracked implementation remain byte-identical
to the measured commit. The first package receipt is retained separately.


Python was 3.12.14+meta and 3.10.20; reference PyTorch was 2.13.0+cu130.
Selected runtime/NVRTC were CUDA 13.0, Rust 1.92.0; inventory nvcc was
12.8 V12.8.93, not the pointwise compiler. Physical GPU0 was H100
`GPU-8f8e55a5-a9eb-eb79-bc43-807a19bcb1c1`, driver 580.82.07, with
`CUDA_VISIBLE_DEVICES=0`. Pre/post snapshots show 4 MiB and 0% utilization;
snapshots do not establish exclusivity. These Linux checks do not establish
macOS, Python 3.14 or multi-device results.

## Retained receipts

All new raw files are under `target/alias-mutation-qa/postcommit-95bdbf2/`.
The archive contains exact commands/statuses/logs, source export and manifest,
both attempted release wheels, import/provider identities and characterization
receipts. `MANIFEST.json` records every member's size and SHA256; all 57 payloads
were reread and verified. Historical archives are neither changed nor nested.
Main must copy and verify the raw archive before removing this worktree.

| Artifact | SHA256 |
| --- | --- |
| `alias-mutation-qa-postcommit-95bdbf2.tar.gz` (5,997,188 bytes) | `f83bd50b3f331f98e5d8fb58fc5a28d4f5e5a2272ac34dd95619e05a74de54b9` |
| Archive `MANIFEST.json` | `2baf6a647d542a4bb190f4b1eaf4934501a14f7b538d02f684c217948d32d6f5` |
| Verified owned-interpreter release wheel | `282abd448c19f3a51ba05eaf7a4be297a1a2b1908eea0f3362a6c0674d2b852b` |
| Loaded `torch_rs.abi3.so` | `e6fe9b6f3b4eb232dc61b5c515a5c956828bb78723673972596f24b8b171e1df` |
