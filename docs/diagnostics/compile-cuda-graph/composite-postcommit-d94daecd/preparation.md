The enclosing worktree was clean at d94daecdd9c086ac23afe690a52df11cbaca4f8a.
The seed declaration was written before any new timing measurements. Existing
worktree-local CPython 3.12.14 and dependency caches came from integration setup.
No historical timing report or prior native build was reused.

Preparatory checkout/cache commands (performed before the timestamped setup and
build receipts; these notes do not assign invented timestamps to those commands):

```bash
mkdir -p target/composite-postcommit-d94daecd/baseline
GIT_CONFIG_NOSYSTEM=1 git init --quiet target/composite-postcommit-d94daecd/baseline
GIT_CONFIG_NOSYSTEM=1 git -C target/composite-postcommit-d94daecd/baseline -c gc.auto=0 fetch --quiet --no-tags "$PWD" b7936239ceb7d7713a15ecd8a2c82c4bddd6d214
GIT_CONFIG_NOSYSTEM=1 git -C target/composite-postcommit-d94daecd/baseline checkout --quiet --detach FETCH_HEAD
mkdir -p target/composite-postcommit-d94daecd/baseline/target/local
cp -a --reflink=auto target/integration/uv-python target/composite-postcommit-d94daecd/baseline/target/local/uv-python
cp -al target/integration/uv-cache target/composite-postcommit-d94daecd/baseline/target/local/uv-cache
cp -al target/integration/cargo target/composite-postcommit-d94daecd/baseline/target/local/cargo
```

This makes a detached nested repository; it creates no delivery branch, commit
or remote publication. All copied files, local cache hardlinks and Python links
are inside the composite worktree. The baseline `uv sync --locked` uses copy
installation mode; its exact command, environment and timestamps are in
baseline/setup.receipt.json. Both fresh builds and all measurement commands have
separate timestamped receipts. Baseline Python, installed reference and runtime
content identities are compared with the candidate after installation.
