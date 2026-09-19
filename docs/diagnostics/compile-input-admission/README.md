# Historical admission evidence

The [experiment index](../../compile-input-admission-warm.md) separates historical
captures from the current author diagnostic. This directory's former 55 files
and the original warm-call note are preserved byte-for-byte in
[historical-evidence.tar.gz](historical-evidence.tar.gz), indexed by the
[member/hash manifest](historical-evidence-manifest.json). Original failures,
measurements, commands and provenance are unchanged; archived worktree paths are
historical identities, not current locations.

From the repository root, extract into a fresh local directory:

```sh
mkdir -p target/admission-history
sha256sum docs/diagnostics/compile-input-admission/historical-evidence.tar.gz
tar -xzf docs/diagnostics/compile-input-admission/historical-evidence.tar.gz -C target/admission-history
```

Compare the archive hash with `archive_sha256` in the manifest. Each member has
its original repository path, SHA-256, byte count and mode. The original entry
points are `docs/diagnostics/compile-input-admission/README.md`, its
`postcommit-32105a22/README.md`, and `docs/compile-input-admission-warm.md` under
the extraction directory; their relative evidence links resolve there.
