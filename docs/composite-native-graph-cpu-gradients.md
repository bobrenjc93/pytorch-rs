# Native CUDA graph / CPU division and mean L1 integration

The combined implementation is committed as `3c320cb2e36ae1f165437e095ecf69cbcd1f59f4`. It retains the
single native CUDA execution bridge from PR #1959, first-order CPU scalar
division from #1960 and default/explicit mean L1 backward from #1961, with
unsupported forms and eager semantics unchanged. The integration commit fixes
the exact float32 Clippy assertion and adds combined division/L1 differentials.

[Clean committed evidence](diagnostics/compile-cuda-graph/postcommit-3c320cb/README.md)
now supplies the previously deferred empty-target build, installed source-matched
wheel verification, actual command receipts, source/test/harness/runtime hashes,
all four declared diagnostic runs and complete raw data. Both Clippy modes,
67 focused CPU tests, H100 CUDA regressions, seven separate two-GPU checks,
the 38-case compiler corpus and fixed CUDA math passed. All four fixed scoring
shapes were eligible. See the linked reports for exact results and accounting.

An unrelated workload occupied GPU 0 during timing. Every case and raw call
is retained, but these measurements establish neither isolated performance
non-regression nor repeatable acceleration. Native capture remains
`backend="eager"`; no broad Inductor, training or hardware parity is claimed.

[Historical source evidence](diagnostics/compile-cuda-graph/postcommit-786c1b2/README.md)
and original `fbb0aa0` baseline captures remain byte-identical. The earlier
precommit checks and setup limitations are recorded in this document at commit
`3c320cb2`; they are not the clean-commit evidence. In particular, the earlier
uv-created home-directory symlink remains untouched by this evidence step.

Independent exact-head review, all ten current-definition non-regressing gates,
and exact-head CI remain Burner-owned merge prerequisites. This publication
does not grant approval or change source associations, admission settings or
managed progress artifacts. Controlled performance validation still requires
appropriate GPU scheduling.
