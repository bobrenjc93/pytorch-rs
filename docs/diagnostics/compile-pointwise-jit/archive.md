# Pointwise compiler evidence archive

The [current evidence index](README.md) owns the latest results and reproduction
commands. Baseline and current reports remain directly accessible there.
Superseded captures and development records are consolidated below, with their
original paths, bytes, failures and measured source identities intact.

## Collections

| Collection | Original files | Stored from commit | Manifest |
| --- | ---: | --- | --- |
| [Early captures](history.tar.gz) | 56 | `ed6ad9ea` | [Paths and hashes](history-manifest.json) |
| [Later captures and repair history](history-later/history.tar.gz) | 56 | `91fd9524` | [Paths and hashes](history-later/history-manifest.json) |

“Stored from” identifies the checkout whose files were archived, not the source
measured by each report. The manifests pin each member's size and SHA-256 and
the container's hash. All members were compared byte-for-byte with their source
commit before the duplicate checkout copies were removed. The early archive
and manifest themselves are unchanged.

## Verify and read

From the repository root, verify **both** collections using only the Python
standard library. Missing collections, corrupt data, duplicate paths and unsafe
archive members fail verification. Nothing is extracted or executed.

```bash
python3 docs/diagnostics/compile-pointwise-jit/verify_archive.py
```

Use a manifest's original relative path to read a member. For example, inspect a
repair narrative or decode an earlier complete coverage report:

```bash
tar -xOf docs/diagnostics/compile-pointwise-jit/history-later/history.tar.gz \
  review-product-priority.md
tar -xOf docs/diagnostics/compile-pointwise-jit/history.tar.gz \
  postcommit-5fc75c/candidate-coverage.json.gz | gzip -dc
```

Archived Markdown retains its original relative links as historical provenance;
use the manifests to locate those members. Git history also retains every
original checkout path.

## Clean-capture map

Each prefix contains coverage and CUDA-performance reports, a receipt, logs,
generated CUDA/PTX, provenance and a source manifest. “Root” means no prefix
inside that archive.

| Measured source | Collection | Prefix |
| --- | --- | --- |
| `5b93c983` | Later | Root |
| `60abd863` | Early | `postcommit-60abd/` |
| `53c10058` | Early | `postcommit-53c100/` |
| `dbd1a0f6` | Early | `postcommit-dbd1a0/` |
| `0c836a49` | Early | `postcommit-0c836a/` |
| `40a57b36` | Early | `postcommit-40a57b/` |
| `f745c45c` | Early | `postcommit-f745c4/` |
| `5fc75c41` | Early | `postcommit-5fc75c/` |
| `42959e14` | Later | `postcommit-42959e/` |
| `74602c07` | Later | `postcommit-74602c/` |
| `08e37fe5` | Later | `postcommit-08e37f/` |

These captures predate the current libdevice-order repair. Their timings and
scores belong only to their recorded revisions and builds; none is credited as
a measurement of the current candidate.

## Development records

The later collection also preserves the initial diagnostic, validation
inventory, logs and IEEE probes, plus the ten earlier `review-*.md` narratives
and their matching JSON bundles. They retain original/intermediate failures,
dirty-source checks and repair histories; development diagnostics are not scores.

The [latest repair narrative](review-libdevice-order.md),
[repair bundle](review-libdevice-order.json.gz) and
[independent operator observations](operator-libdevice-compositions.json.gz)
remain directly accessible alongside the current capture.

Temporary wheel, interpreter and raw-observation paths recorded inside old
reports may disappear during worktree cleanup. The checked-in artifacts are the
durable record. Archive verification proves preservation; it does not rerun
historical tests or replace independent review and merge qualification.
