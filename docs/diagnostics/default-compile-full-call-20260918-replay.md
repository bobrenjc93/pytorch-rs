# Replaying the retained full-call experiment

This corrects the packaging described in the [original note](default-compile-full-call-20260918.md),
which remains unchanged. Its compact archive omitted 12 NPZ payloads from its
90-entry manifest. The exact original [raw archive](default-compile-full-call-20260918-raw.tar.xz)
is included in this author output for Burner to commit: **20,204,660 bytes**, SHA256
`82ac63e93cee8d76d2f78226f9324ec92e032becf154bfac84861cd42c2ea265`.
All 90 manifest payloads verify locally across 91 tar members, including hard links.
Durable repository replay requires Burner to include this raw archive in its commit.
No timing, array, failure, source identity or original receipt was rewritten.

The [single comparator](../../scripts/diagnose_compile_full_call.py) now loads
each NPZ beside its JSON report, using the recorded basename. The original
absolute path remains capture provenance; replay never falls back to it.
Each NPZ archive hash and every contained array's content hash are checked
before the existing output, metadata, alias and signed-zero comparisons.

From the repository root, with a worktree-local Python environment containing
NumPy, verify and replay offline (no CUDA, native extension or PyTorch needed):

```bash
echo '82ac63e93cee8d76d2f78226f9324ec92e032becf154bfac84861cd42c2ea265  docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz' | sha256sum -c -
mkdir -p target/full-call-relocated
tar -xJf docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz -C target/full-call-relocated
for build in accepted incoming measured; do
  .venv/bin/python -B -I scripts/diagnose_compile_full_call.py --compare \
    "target/full-call-relocated/full-call/$build-reference-second.json" \
    "target/full-call-relocated/full-call/$build-native-first.json"
  .venv/bin/python -B -I scripts/diagnose_compile_full_call.py --compare \
    "target/full-call-relocated/full-call/$build-reference-first.json" \
    "target/full-call-relocated/full-call/$build-native-second.json"
done
.venv/bin/python -B -m unittest tests.test_diagnose_compile_full_call -v
```

[Offline results](default-compile-full-call-20260918-replay.json): both accepted-main
orders reproduce 58 checked cells and 18 explicit rejections; incoming and the
discarded experiment each reproduce 76 checked cells in both orders, with no
comparison errors. Ten portable tests pass, including manifest verification,
historical replay, relocation after deleting the original location, missing or
corrupted arrays, and refusal to substitute a still-existing original archive.

These are replays of historical developer observations, not new performance
measurements or qualification. The archived measured script and this later
comparator have distinct identities. Separately timestamped NVRTC receipts
remain separate from the original timed captures. The required future clean
implementation-commit capture remains outstanding pending a justified executable
improvement and separate continuation admission.
