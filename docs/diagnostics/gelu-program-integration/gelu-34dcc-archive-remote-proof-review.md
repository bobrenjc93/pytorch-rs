# Archive preservation and existing remote proof

Conclusion: **the saved canonical remote-head observation plus the verified local Git object chain is sufficient for the proposal's narrow pushed-reachability checkpoint at the recorded time.** Another commit-lookup API call or `ls-remote` is not intrinsically required. This is not a remote byte-download test, a guarantee of future availability, or approval to change the running owner's checkout.

## What was verified

The canonical v2 `before-observations.json` capture is stamped `2026-09-17T01:21:35.292Z`, SHA-256 `d8fef479de17265172f17a5f84eede7e6aafb99c971fc96021a4377e0cfa3b42`. Its PR2004 probe succeeded with no recorded capture failures: OPEN/DRAFT, branch `burner/integrate-native-gelu-into-once-compiled-c-40b456`, remote head `34dcc47454b3faaa27ef46a5189b34e81d3802f2`, base `60202557b4f110d07777f585e804ab5f55e1ff7b`. Both named CI checks were successful; CI is supporting context, not the archive proof.

I read the actual capture source. It obtains this value through synchronous `gh pr view 2004 --repo bobrenjc93/pytorch-rs`, with a thrown command failure producing `ok:false`; it does not synthesize remote head from a local ref. The retained canonical leaf identifies that same repository and branch. The observation is therefore materially stronger than `refs/remotes/origin/...` alone. I made no new remote request.

The local inventory is independently consistent with immutable `34dcc`, not merely internally consistent prose. At `2026-09-17T01:30:10.293Z`, a read-only verifier completed:

- All six original manifest hashes matched their `34dcc` Git blobs and current checkout copies.
- Their ordered part inventories exactly partitioned the declared 28 unique files, with no omissions or extras.
- Every original path resolved to its declared regular Git blob. Streaming each blob independently reproduced its Git SHA-1, size and SHA-256; streaming its checkout copy reproduced the same size/SHA-256. All six ordered concatenation hashes matched the original manifests.
- Total: **28 blobs, 1,612,770,062 bytes**. These are actual large Git blobs, not LFS pointer files.

| Original archive | Parts/files | Bytes |
| --- | ---: | ---: |
| `development.tar.gz` | 7 | 416,003,036 |
| `postcommit-5bf3e9b0.tar.gz` | 10 | 648,030,978 |
| `feature-gate-repair.tar.gz` | 2 | 107,468,295 |
| `admission-repair-development.tar.gz` | 5 | 289,273,973 |
| `postcommit-728d147.tar.gz` | 3 | 137,748,914 |
| `ci-gate-repair.tar.gz` | 1 | 14,244,866 |

The commit object hashes to `34dcc`; its tree is `a23f2091b4b18658a28e6e9a7cb9269499509111`. I also recomputed the intervening tree object identities through `docs/diagnostics/gelu-program-integration`. History is non-shallow, and `34dcc` was still the worktree HEAD and thus its own inclusive ancestor. The scoped archive/manifest diff was empty. At the `01:35:04Z` recheck, both new preservation documents remained untracked with unchanged reviewed hashes; no archive removal was observed.

## Interpretation and remaining obligations

Under ordinary Git content-addressing and accepted-ref connectivity semantics, GitHub reporting this exact pushed PR head identifies the same commit/tree and therefore the 28 contained blobs. It is an operational reachability proof based on the trusted remote head observation, not an independent acquisition of remote payload bytes. Reusing this existing canonical observation is the smallest sound treatment; a new service, alternate network route, or invented retention mechanism is unnecessary.

The author's blocked commit lookup and CONNECT-403 attempt remain real failures. Its decision to retain copies without access to a fresh authoritative observation is sound. This review supplements that limited observation; it does not rewrite those failures or claim the worker successfully contacted either denied endpoint.

The proposal and v2 guidance still require more than this one checkpoint:

- A saved observation proves its recorded time, not the current remote ref indefinitely. Any later contradictory observation or ancestry change must be reconciled; do not relabel the snapshot as a new live check.
- `34dcc` is not already retained through accepted base `602`. Later acceptance must preserve the archive-containing ancestor. Burner's actual leaf merge uses `--merge --match-head-commit` (`git.ts:399–402`), and settlement proves source-to-landing/target ancestry (`git.ts:693–709`). These are intended future safeguards, not evidence that a merge has happened. A hash/URL alone and unverified PR-ref retention are insufficient substitutes.
- The required retrieval and exact-byte verification after any checkout-copy removal remains outstanding. My reads verified local retrieval before removal only. Existing full-history/no-Git instructions must remain honest; mandatory offline fixtures and original manifests remain outside the removal scope.
- Neither this snapshot nor local hashing proves raw HTTP access, successful shallow fetch/download from this environment, host retention forever, or offline bytes in an sdist. Both blocked routes stay blocked for this review. No current network workaround was attempted or recommended.

Thus there is no additional missing *remote-head* proof in the combined local evidence, but there is also no completed migration, future retrieval guarantee, or new public-operation admission. Main retains the next ownership/admission decision; the live author was not steered.

## Bindings and verification limits

Reviewed SHA-256 values:

```text
06c64af258cd4eca888f0ad486d43eb438db2acad56181b45fd7f50f923bf9ee  archive-preservation-observation.json
2680b0263a524837f5bd7cd1d596cd16236109c5e11ef27e0eb7cee20dbcc217  archive-preservation.md
2d647f46025c25fa8724e73bebfdc6bb64db5335cba44017a178c8ec7b656970  gelu-34dcc-next-repair-proposal.md
3b875a173bbb8634d633c0de9c4d7df0c3892d5c766d0f630a6087cf74ea1956  gelu-34dcc-next-repair-notes-v2.md
3165299dd7475d8302ef560dab0e6d1cdf03569abc9859fe7a7cac07fb6c7db6  run-gelu-34dcc-guided-repair-v2.mjs
d0eb091f4e00c03367d70e527c92adc8f6d0cf0a136b062665225db8c49667f5  Burner src/lib/git.ts
6061be123581af38e2c256af353ce590a7b6e27ad0868d4bd472d79443a200e4  Burner src/lib/orchestrator.ts
```

The hashed inventory binds all 28 per-file identities and all six original manifest digests without duplicating that schema here. The existing tensor-madd README and historical-blob factual inventory were read as context, not substituted for the checks above. Moduler design/review guidance favored reuse of the authoritative observation over another verification path.

Executed diagnostics: bounded `sed`/`rg`/`wc`/`sha256sum` reads; local Git `show`, `cat-file`, `rev-parse`, `ls-tree`, `merge-base`, scoped `status` and `diff`; inline Node standard-library JSON projections and streaming hash comparisons. Git reads disabled lazy fetching/optional locks and replacement objects. The blob verifier exited 0 (tool session 36705, completion chunk `83c727`); it did not decompress, extract, execute or inspect archive members. One instruction-path read and one search for a nonexistent `github.ts` failed before their correct paths were read; neither affected the audit. No archived script, controller/product import, framework, workload, GPU, test/evaluation, private transcript, target/venv scan, or network request was executed. Only this report was written. Future evolving source or preservation artifacts are not reviewed by this version-bound conclusion.
