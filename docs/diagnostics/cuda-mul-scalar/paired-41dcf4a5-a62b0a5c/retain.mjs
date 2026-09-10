import assert from 'node:assert/strict';
import { constants } from 'node:fs';
import { copyFile, mkdir, readFile, readdir, realpath, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { runCommand } from '/data/users/bobren/a/burner/dist/lib/process.js';

const root = dirname(fileURLToPath(import.meta.url));
const repair = join(root, 'polish-repair');
const destination = join(repair, 'docs/diagnostics/cuda-mul-scalar/paired-41dcf4a5-a62b0a5c');
const sha = async (path) => createHash('sha256').update(await readFile(path)).digest('hex');
const summary = JSON.parse(await readFile(join(root, 'summary.json'), 'utf8'));
assert.equal(summary.comparedRows, 152);
assert.equal(summary.allOutputsAndMetadataIdentical, true);
assert.deepEqual(summary.revisions, {
  baseline: '41dcf4a5a015337a61f4507940cbeb20fd4ff006',
  candidate: 'a62b0a5c2709898ebcb1b906b37ffcb7c1a2c1d9',
});
const status = await runCommand('git', ['status', '--porcelain'], { cwd: repair });
assert.equal(status.exitCode, 0);
assert.equal(status.stdout.trim(), '', 'Do not overwrite a changed repair worktree');
await mkdir(destination);
const hashes = {};
async function retain(source, relative) {
  const target = join(destination, relative);
  await mkdir(dirname(target), { recursive: true });
  await copyFile(source, target, constants.COPYFILE_EXCL);
  hashes[relative] = await sha(source);
  assert.equal(await sha(target), hashes[relative], 'Raw evidence must remain byte-identical');
}
async function reports(source, relative) {
  for (const entry of (await readdir(source, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
    assert.ok(entry.isFile() && /\.(json|log)$/.test(entry.name), `Unexpected report input: ${entry.name}`);
    await retain(join(source, entry.name), join(relative, entry.name));
  }
}
for (const name of ['plan.json', 'selected-revisions.json', 'results.json', 'summary.json', 'run.mjs',
  'summarize.mjs', 'retain.mjs', 'setup-rejection.json', 'setup-rejection-old-uv.json']) await retain(join(root, name), name);
const postRun = {};
for (const variant of ['baseline', 'candidate']) {
  await reports(join(root, variant, 'target/root-paired/reports'), join(variant, 'commands'));
  for (const name of ['build-record.json', 'build.log', 'dependency_setup.log', 'origin.patch']) {
    await retain(join(root, variant, 'target/cuda-add-diagnostic/paired', name), join(variant, 'build', name));
  }
  const build = JSON.parse(await readFile(join(root, variant, 'target/cuda-add-diagnostic/paired/build-record.json'), 'utf8'));
  const python = await realpath(join(root, variant, '.venv/bin/python'));
  const pythonSha256 = await sha(python);
  assert.equal(pythonSha256, 'f7c6210eb40fadcd3c2889dddd24a15fc2c9f926aec5a03bf9da66e12d581526');
  assert.equal(await sha(build.installed_native), build.installed_native_sha256);
  postRun[variant] = { python, pythonSha256, native: build.installed_native,
    nativeSha256: build.installed_native_sha256, measuredCodeCommit: build.measured_code_commit };
}
for (const attempt of ['rejected-system-python', 'rejected-old-uv']) {
  await reports(join(root, attempt, 'baseline/target/root-paired/reports'), join('setup-attempts', attempt, 'commands'));
  await retain(join(root, attempt, 'selected-revisions.json'), join('setup-attempts', attempt, 'selected-revisions.json'));
}
const postPath = join(destination, 'post-run-binary-checks.json');
await writeFile(postPath, JSON.stringify({ checkedAt: new Date().toISOString(),
  purpose: 'Post-run verification, not a fabricated earlier capture timestamp', binaries: postRun }, null, 2) + '\n');
hashes['post-run-binary-checks.json'] = await sha(postPath);
await writeFile(join(destination, 'artifact-sha256.json'), JSON.stringify(hashes, null, 2) + '\n');
console.log(JSON.stringify({ destination, retainedFiles: Object.keys(hashes).length,
  byteIdenticalCopies: Object.keys(hashes).length - 1, noBinariesWheelsEnvironmentsOrCaches: true }));
