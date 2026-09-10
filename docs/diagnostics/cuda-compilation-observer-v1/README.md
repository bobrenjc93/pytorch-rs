# Observer campaign evidence

See the [campaign guide](../../cuda-compilation-observer-campaign.md) and
[manifest](../../../scripts/campaigns/cuda-compilation-observer-v1.json).
The successful paired development capture is in `development/`. Its baseline
has unchanged main production plus uncommitted campaign files; its historical
candidate checkout is clean. It is not evidence from a final evaluator-only
commit and does not provide implementation credit or human approval.

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

A future Burner refresh must publish a new directory from the final campaign
commit and run portable validation before review and normal gates. Existing
attempts and development reports must remain intact. No independent campaign
review or human campaign approval occurred in this implementation task.
