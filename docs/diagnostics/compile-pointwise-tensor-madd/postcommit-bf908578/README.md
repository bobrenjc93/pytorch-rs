# Historical clean post-commit captures

These measurements belong to implementation
`bf9085787aca29e523a7a38773f20f965167e734` and actual main
`77aa16fc2d0cbd258f0fa309140dc75708cf2ee3`. Both checkouts were clean when measured.
The later packaging/evaluated source is
`48fb3588c692db5b2a4f69033b5bf8a2bebb2019`; neither packaging revision is a new
measurement of the implementation.

| Original fixed capture | Cells passed | Weighted coverage | CUDA cells passed | Weighted capped CUDA performance |
| --- | ---: | ---: | ---: | ---: |
| [Candidate, bf908578](fixed/candidate/run-20260913T173018Z-d4056067/report.json) | 10/112 | 11% | 10/56 | 20% |
| [Main, 77aa16fc](fixed/main/run-20260913T173516Z-6151e3a3/report.json) | 8/112 | 9% | 8/56 | 12% |

Both reports are byte-identical to the original Git tree and archive members
at these same relative paths prefixed with `postcommit-bf908578/`. All fixed
cells and unsupported outcomes remain: 112 coverage / 56 CUDA cells, both CUDA
implementation orders, five warmups, 17 samples, one host thread and real runtime
synchronization. Ordinary default Inductor settings, tolerances and limits were
unchanged. These finite-corpus results do not imply broad Inductor equivalence.

## Separate evidence stages

- **Development:** dirty/uncommitted runs and their failures are under archive
  prefix `development/`; they do not satisfy clean-commit capture requirements.
- **Clean captures:** both six-program comparison orders passed 42 states per
  leg, 966 calls each, with exact finite/zero-sign comparisons and no NaN-payload
  equality requirement. Ten focused tests passed on reserved H100 GPUs 0/1,
  including real cached-kernel execution. Historical NVRTC was 13.0, runtime
  13000, host nvcc 12.6, PyTorch 2.13.0+cu130 and driver 580.82.07.
- **Canonical evaluation:** all three actual Repository polish samples were 82
  versus baseline 83 at `48fb3588`: `evalrun_a9316d55`, `evalrun_7f0fe688` and
  `evalrun_53ed1eb0`. The [sealed result](../operator-result.json) and
  [complete samples/audit](../operator-audit.json) retain the original unqualified
  outcome and helper PID 1742628's exit 2. Packaging does not resample or waive
  normal median-of-three confirmation.
- **Later operator GPU QA:** separate from these historical captures and from
  independent review, exact-head evaluation and the full merge gate; no later
  qualification or merge is claimed here.

## Exact archive locations

Use [the archive verification and retrieval instructions](../README.md) and
[history-manifest.json](../history-manifest.json). All names below are relative
to the root of [history.tar.gz](../history.tar.gz):

| Evidence | Member paths |
| --- | --- |
| Four full diagnostic reports | `postcommit-bf908578/paired-0-native.json.gz`, `paired-1-reference.json.gz`, `paired-2-reference.json.gz`, `paired-3-native.json.gz` (all with that same prefix) |
| Actual dispatched CUDA/PTX | `postcommit-bf908578/paired-0-native.tar.gz`, `postcommit-bf908578/paired-3-native.tar.gz` |
| Exact comparisons | `postcommit-bf908578/comparison-0-1.json`, `postcommit-bf908578/comparison-2-3.json` |
| Source/build identity | `postcommit-bf908578/build/build.json`, `postcommit-bf908578/build/source.tar.gz` |
| Commands, audit and failures | `postcommit-bf908578/commands.json`, `postcommit-bf908578/audit.json`, `postcommit-bf908578/inspection-notes.txt`, and every original member under `postcommit-bf908578/logs/` |
| Original index/self-manifest | `postcommit-bf908578/README.md`, `postcommit-bf908578/sha256.json` (historical only) |
| Frozen-worker staging references | `postcommit-bf908578/raw-retention.json` |

The original build helper's legacy `kind` text is retained verbatim alongside
its actual commit, empty status and verified identities. No provenance was
relabelled. This archive covers all 134 original checked-in non-wheel files;
it does **not** contain the separately retained detailed frozen-worker reports.
The immutable [operator audit](../operator-audit.json) identifies those external
raw archives and their hashes. Source/build/audit duplicates and every original
failure remain available inside the single history archive.

The three historical wheel objects remain in the pushed ancestor, with exact
bindings and immutable URLs in [wheel-provenance.json](../wheel-provenance.json).
They are not installable release recommendations. Reproduction builds pinned
source and verifies installed-wheel identity. Shallow/source-archive retrieval
and explicit unavailable-history skips are documented in the parent index.
Do not recreate platform wheels or redundant build-output collections during
normal post-commit evidence handling of this packaging-only repair.
