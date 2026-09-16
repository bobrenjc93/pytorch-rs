# Compact evidence archive audit

Passed integrity checks for all three compact archives. No measurement,
framework, GPU, build or test execution was performed; this is read-only
verification of existing records. Only this local report is retained.

All 4,442 regular members have unique names, exactly match their manifest set,
and match recorded byte counts/SHA256 values. Each also matches its original
recorded worktree-local file. The private and public xz files decompress to
exactly the same tar byte streams as their original, hash-verified gzip packs;
no member content or embedded provenance was rewritten by recompression.

| Archive | Members | Uncompressed member bytes | Archive SHA256 |
| --- | ---: | ---: | --- |
| attempt-002 | 2245 | 24253432 | `fcb9037e2fba58306f1dc32157098f28047c53365a672fbf982af8186b03c00a` |
| public-attempt-002 | 2008 | 607600604 | `e028b6a98ef6ddd51379d4a90bfe58e562663142d2371a91637f002cf0fc6ea8` |
| supporting-records | 189 | 101582937 | `0f68e2c5c54684b3b62bbfc9a43ec906703f635ab1ea2cd4ee0d881829deba52` |

Private source hash remains
f5f99ceb9ea941fbcc7c46a840374f4482c4fa94ccc8ad9033c98ada32f32e4f;
public source hash remains
2a2c379799f24fd1c7866e4b4ec6dc47293a9459ff43c3eb3940cb6cd5d7812a.
Archived comparison records reproduce the previously independently audited
335 private and 33 eager / 236 compiled / 33 no-NVRTC public comparisons.
The README's zero finite-bit and signed-zero differences and 1,924 NaN-word
differences per compiled dataset match those audits. NaN payload equivalence
is explicitly not claimed.

All 11 local Markdown link targets in README.md resolve. Result/test-count
claims match the retained comparison records and logs: 257 Rust library tests,
15 new Python tests, 48 focused regressions with one skip, 17 generator tests.
The supporting record preserves full fixed-corpus report/log/identity evidence
without pretending it contains the bulky fixed worker raw tensors; their
continued worktree-local location and exclusion are disclosed. Failed capture
attempts remain distinct from numerical acceptance.

Two minor prose corrections were reported to the author: say the timing
protocol uses five warmups and retains 17 samples (it does not retain warmup
samples), and limit the NVRTC13 environment wording to compiled captures,
because ordinary eager with absent NVRTC maps no such library. Neither issue
changes the archived measurements or their integrity.

The README accurately labels all captures as dirty staged development evidence,
keeps the fixed diagnostic valid=false, and explicitly requires clean-commit
refreshes and unchanged official scoring/full qualification after Burner commits.
This audit does not promote them to final scores, certify performance gain,
cover a future source revision, or approve delivery.

Assembled 2026-09-16T07:49:21.671880+00:00.
README SHA256 inspected: `372e12c53d25eee75a1d5e6598c50098839f9d1cb09029828bb6235d9b3a503c`.
