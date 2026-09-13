# Tensor-leaf multiply-add evidence

This collection documents the original tensor-input-only `a*b+c` / `c+a*b`
broadcast exception. The [compiler contract](../../compile-pointwise-jit.md)
owns the supported boundary. This repair changes packaging only; it does not
change compilation, capture behavior, numerical expectations or evaluation rules.

## Evidence and outcomes

| Record | Identity and outcome |
| --- | --- |
| Development | Uncommitted work based on main `77aa16fc2d0cbd258f0fa309140dc75708cf2ee3`; original tests, exploratory failures and interrupted runs are archived unchanged. |
| Clean post-commit measurements | Implementation `bf9085787aca29e523a7a38773f20f965167e734` versus main `77aa16fc2d0cbd258f0fa309140dc75708cf2ee3`; see the [measurement index](postcommit-bf908578/README.md), unchanged [candidate report](postcommit-bf908578/fixed/candidate/run-20260913T173018Z-d4056067/report.json) and [main report](postcommit-bf908578/fixed/main/run-20260913T173516Z-6151e3a3/report.json). |
| Canonical evaluation | Packaging/evaluated source `48fb3588c692db5b2a4f69033b5bf8a2bebb2019`: Repository polish fell from 83 to 82 in **all three** actual samples: `evalrun_a9316d55`, `evalrun_7f0fe688`, `evalrun_53ed1eb0`. The [sealed producer result](operator-result.json) and [original audit with complete samples](operator-audit.json) are preserved byte-for-byte. Helper PID 1742628 exited 2, unqualified, without submission to the full gate. This was a polish regression, not a fabricated kernel execution failure. |
| Later operator GPU QA | A separate normal gate; neither historical measurements nor this packaging repair establish its completion or authorize merge. |

Normal independent review, exact-head evaluation, median-of-three confirmation,
full merge qualification and operator GPU QA remain separate. No sample is
replaced or waived here. Historical captures are not measurements of this
packaging repair and do not establish broad default-Inductor equivalence.

## One archive, complete original inventory

[history.tar.gz](history.tar.gz) contains **all 134 original non-wheel files**
from Git tree `48fb3588c692db5b2a4f69033b5bf8a2bebb2019`, under their exact original
relative names. [history-manifest.json](history-manifest.json) uses the existing
archive schema and records every size/hash and the full source commit.
The byte-identical [operator inventory](original-inventory.json), SHA256
`08da41c49c979077c67d2305341e149f41bccb28d984e4a045b714584b96905a`, partitions the
original 137 files into those 134 members and exactly three historical wheels.

Members include `README.md`, `sha256.json`, `development/validation.json`,
`development/exploratory-status.json`, `development/compiler-module-logs.tar.gz`,
`development/verified-build/source.tar.gz`, `postcommit-bf908578/README.md`,
`postcommit-bf908578/sha256.json`, `postcommit-bf908578/audit.json`, and every
original timing, CUDA/PTX archive, source snapshot, duplicate, log and failure.
Old self-manifests describe only that original inventory; they are archive
members, not manifests of today's checkout. [capture.py](capture.py) and both
fixed summary reports remain byte-identical at their original paths.

Verify from the repository root, offline, without extracting or executing members:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
p = Path('docs/diagnostics')
spec = importlib.util.spec_from_file_location('archive', p / 'compile-pointwise-jit/verify_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
manifest = archive.verify(p / 'compile-pointwise-tensor-madd')
print(manifest['source_commit'], len(manifest['files']))
PY
python -m unittest discover -s tests -p test_compile_evidence_archive.py -v
```

Safe inspection: list `manifest['files']` in the snippet above, or read an exact
ordinary member in memory with `tarfile.open(..., 'r:gz').extractfile(name).read()`
after verification. Do not unpack or execute archived scripts. The verifier's
existing CLI still verifies only its original two pointwise-JIT collections.

## Historical wheels stay in Git history

[wheel-provenance.json](wheel-provenance.json) binds each of the three original
paths to its byte count, SHA256, Git blob ID, full source commit and immutable
same-repository GitHub URL. They are historical build artifacts, **not binaries
to install**. Only their current checkout copies were removed; no wheel is
hidden inside this archive or a renamed payload, and no Git history/blob was
deleted. This reduces checkout clutter, not historical clone size. The later
normal merge must retain ancestor `48fb3588c692db5b2a4f69033b5bf8a2bebb2019`.

[Packaging provenance](packaging-provenance.json) records the full-history,
ancestor and original-blob checks performed before removal. The canonical retry
owner had freshly verified the pushed PR head; its pinned helper/receipt and the
local remote-tracking head were checked here. Live worker GitHub reads were
blocked by the destination filter; that limitation is explicit in the receipt.
The [packaging check log](packaging-checks.txt) retains 13 passing archive tests,
12 passing README smoke tests and the initial local-environment setup failure.

For an existing clone missing that commit, explicitly fetch it yourself (tests
never fetch). A shallow clone can retrieve the objects without claiming full
history; `git fetch --unshallow origin` is needed to run full-history checks.

```bash
git fetch --no-tags origin 48fb3588c692db5b2a4f69033b5bf8a2bebb2019
# Example: stream the original diagnostic wheel into a worktree-local file.
mkdir -p target/historical-evidence
git cat-file blob 753b55dc40aebce21ea422b3c086d725eb9d7b1f > target/historical-evidence/tensor-madd.whl
```

Check the result against its size and SHA256 in `wheel-provenance.json` before
inspection. Source-archive readers without `.git` can use each immutable GitHub
URL's **Raw/Download raw file** action, or create a separate clone and perform
the explicit retrieval above. Offline archive/schema/summary tests still run;
unavailable historical Git-byte checks explicitly skip, never silently pass.

Reproduction builds from pinned source with worktree-local tools/caches and
source/installed-wheel identity checks; see the preserved original reproduction
instructions in archive members `README.md` and `postcommit-bf908578/README.md`.
Do not install historical wheels or recreate wheel/build-output clutter here.
The normal Burner post-commit evidence step remains intact: this packaging-only
repair needs identity checks, not new GPU measurements. Genuinely required new
evidence must retain truthful attribution without rewriting these records.

The detailed frozen-worker reports are **separate** operator-retained evidence,
not contents of this inventory archive. Their original external archive paths,
manifest hashes and complete audit remain in [operator-audit.json](operator-audit.json)
and [operator-result.json](operator-result.json); the original staging references
also survive in member `postcommit-bf908578/raw-retention.json`. No external
archive, observer, audit or expectation catalog was changed.
