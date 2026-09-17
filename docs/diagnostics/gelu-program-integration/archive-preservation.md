# Historical archives in Git history

The six historical archive groups now live in ancestor
`34dcc47454b3faaa27ef46a5189b34e81d3802f2`: development, postcommit-5bf3e9b0,
feature-gate-repair, admission-repair-development, postcommit-728d147 and
ci-gate-repair. Only their **28 checkout copies/parts (1,612,770,062 bytes)** were
removed. All six original member manifests, failures, weaker measurements,
source/build identities and small readers/scripts/licenses remain unchanged.
The three mandatory offline history archives and PR1924 inventories are untouched.

[The original inventory](archive-preservation-observation.json) binds every path,
Git blob, byte count, SHA256 and immutable same-repository URL. Its earlier
API/destination-filter and CONNECT-403 failures remain true. The subsequently
supplied [Main remote observations](gelu-archive-remote-head-facts.json) and
[independent review](gelu-34dcc-archive-remote-proof-review.md) establish recorded
pushed-head reachability at 01:21:35.292Z (34dcc) and 02:48:30.609Z (f83),
2026-09-17. They do not claim a fresh remote payload download or future availability.

[The relocation receipt](archive-relocation-observation.json) records verification
of all original manifests, the complete regular-blob inventory, sizes, hashes,
ordered concatenations and current ancestry before removal. Afterwards every
original blob was retrieved again from local Git and all sizes/SHA256 and six
concatenation hashes reverified, without executing historical code or binaries.
[Raw relocation and validation receipts](method-guard-repair-manifest.json) are
retained with the current repair. This follows the existing [tensor-madd provenance approach](../compile-pointwise-tensor-madd/README.md#historical-wheels-stay-in-git-history).
Later acceptance **must retain 34dcc ancestry**. This reduces current checkout
size, not full-clone Git history. Source packages omitting these historical blobs
do not offer offline raw-byte access.

## Retrieve without executing historical artifacts

From a full-history repository root, choose one of the six manifest names below.
The snippet explicitly streams its original parts into one worktree-local gzip,
verifies each part's size/SHA256 and the complete concatenation, and never extracts
or executes a member. To retrieve all six, repeat with each original manifest.

```bash
mkdir -p target/historical-evidence
python - <<'PY'
import hashlib, json, os, pathlib, subprocess
root = pathlib.Path.cwd()
d = root / 'docs/diagnostics/gelu-program-integration'
# Other names: postcommit-5bf3e9b0, feature-gate-repair,
# admission-repair-development, postcommit-728d147, ci-gate-repair.
m = json.loads((d / 'development-manifest.json').read_text())
records = {r['path']: r for r in json.loads(
    (d / 'archive-preservation-observation.json').read_text())['files']}
parts = m.get('parts', [dict(path=m['archive'], size=m['size'], sha256=m['sha256'])])
env = dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_NO_REPLACE_OBJECTS='1', GIT_OPTIONAL_LOCKS='0')
whole = hashlib.sha256()
with (root / 'target/historical-evidence' / m['archive']).open('xb') as output:
    for part in parts:
        record = records[(d / part['path']).relative_to(root).as_posix()]
        process = subprocess.Popen(['git', 'cat-file', 'blob', record['gitBlob']],
                                   stdout=subprocess.PIPE, env=env)
        digest, size = hashlib.sha256(), 0
        while chunk := process.stdout.read(1024 * 1024):
            output.write(chunk); digest.update(chunk); whole.update(chunk); size += len(chunk)
        assert process.wait() == 0
        assert size == part['size'] and digest.hexdigest() == part['sha256']
    assert output.tell() == m['size'] and whole.hexdigest() == m['sha256']
print('Verified', m['archive'], m['size'], m['sha256'])
PY
```

For shallow clones or missing objects, retrieve explicitly before the snippet;
tests never fetch. Fetching one commit does not establish full history. Use
`git fetch --unshallow origin` separately if full-history ancestry checks are needed.

```bash
git fetch --no-tags origin 34dcc47454b3faaa27ef46a5189b34e81d3802f2
```

Without Git, use each inventory entry's immutable GitHub URL and **Raw/Download
raw file**, saving parts in the original manifest order. Check every downloaded
part's size/SHA256, concatenate in that order, and check the manifest's total
size/SHA256 before inspecting. Alternatively, create a full clone and use the
snippet. Network retrieval is explicit and was not tested by this relocation;
it may be unavailable. Never install historical wheels or run archived scripts.
Offline provenance checks still run; optional Git-byte checks explicitly skip
when Git/history is unavailable.
