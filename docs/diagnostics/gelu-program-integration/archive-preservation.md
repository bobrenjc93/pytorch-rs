# Historical archive preservation status

The current warm-call repair starts at `34dcc47454b3faaa27ef46a5189b34e81d3802f2`,
after rejection `leaf-evaluation_0f7369bf`. Its later public guidance permits
removing six historical compressed archive groups from the checkout only after
local preservation and pushed-remote reachability proofs succeed. See the
[current repair scope](../pointwise-admission-repair/warm-repair-scope.md).

[The local observation](archive-preservation-observation.json) verifies all six
original manifests, **28 files totaling 1,612,770,062 bytes**, concatenated archive
hashes, exact Git blob identities, sizes and SHA256 hashes. Every original blob
was streamed from the named commit and checked against the checkout bytes.
That commit is present in full local history and is the local remote-tracking
branch head. These are preservation checks, not new workload measurements.

**All checkout copies remain.** Fresh GitHub API access was denied by the
destination filter; ordinary `git ls-remote origin` also failed with HTTP 403.
A local remote-tracking ref does not establish current pushed-remote
reachability. Raw outputs and completed command receipts are retained with the
current repair evidence under `target/warm-repair/archive-proof`; the repair
handoff must archive those receipts. No archive, manifest, historical result,
or mandatory offline fixture was removed or rewritten.

A canonical owner must supply a verifiable current remote observation before
checkout removal can proceed. Any later merge must retain the archive-containing
ancestor. The existing [Git-blob retrieval pattern](../compile-pointwise-tensor-madd/README.md#historical-wheels-stay-in-git-history)
remains the model: explicit fetch for missing objects, pinned immutable URLs for
no-Git readers, size/hash verification, and no automatic test fetches or execution
of historical binaries. Moving copies into history would reduce checkout size;
it would not reduce full-clone history or provide offline bytes in source archives.
