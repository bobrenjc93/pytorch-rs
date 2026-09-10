# Observer campaign evidence

See the [campaign guide](../../cuda-compilation-observer-campaign.md) and
[manifest](../../../scripts/campaigns/cuda-compilation-observer-v1.json).
The successful paired development capture is in `development/`. Its baseline
has unchanged main production plus uncommitted campaign files; its historical
candidate checkout is clean. It is not evidence from a final evaluator-only
commit and does not provide implementation credit or human approval.

The required clean-commit capture is now in
[`postcommit-10b4cc76/`](postcommit-10b4cc76/campaign.json). It freshly rebuilds
and compares evaluator-only commit `10b4cc76f1bf211714e7588b2b92790895a1d6a6`
against historical candidate `b3659e76011239388da710d4de2f30c52018016f` using
the pinned evaluator and seeds on GPU0. Both checkouts were clean before and
after execution. Baseline **5/6** and historical candidate **6/6** were verified
from raw values; 17 rejection controls passed per interpreter. The new capture
includes complete worker results, fresh build/runtime identity, snapshots,
commands and hashes. Its `checks/` directory records the preflight, raw audit
and portable publication validation.

Earlier development attempts are retained separately with all command output
and file hashes:

- `attempts/01-uuid-representation/`: environment setup stopped before the native
  build or workload evaluation. PyTorch's UUID omitted nvidia-smi's `GPU-`
  prefix. The corrected runner compares the exact UUID payload. The evaluator
  did not change.
- `attempts/02-control-working-directory/`: the baseline completed with 5/6;
  the following rejection-control suite had 16 passes and one timeout error.
  The runner launched the controller-owned CLI escape test from a nested
  checkout, so `../escape.json` was inside the controller's repository and
  started an unintended negative-control evaluation instead of rejecting its
  path. The subprocess was terminated by the test timeout; there is no complete
  result from that control. Its failure traceback is retained. No candidate
  comparison ran. The runner now executes those unchanged controls from their
  own repository directory, with each revision's local interpreter. A new
  paired run rebuilt both revisions after this correction.

Neither failed attempt is relabelled as a clean or completed campaign. All
artifacts, including the unintended control's temporary working directory,
stayed within this task's worktree. The third attempt uses the same two seeds
and exactly the same pinned observer, common helper and matrix.

The post-commit capture satisfies the deferred rebuild/comparison requirement;
existing attempts and development reports remain intact. Independent review,
human campaign approval and normal gates are still required. No independent
campaign review or human approval occurred during implementation or this
evidence refresh, and no implementation or timing credit is claimed.
