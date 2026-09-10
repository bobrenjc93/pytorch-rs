import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { LockManager } from '/data/users/bobren/a/burner/dist/lib/locks.js';
import { runCommand } from '/data/users/bobren/a/burner/dist/lib/process.js';

const root = dirname(fileURLToPath(import.meta.url));
const repo = '/data/users/bobren/a/pytorch-rs-burner';
const plan = JSON.parse(await readFile(join(root, 'plan.json'), 'utf8'));
const candidate = process.argv[2];
assert.match(candidate ?? '', /^[0-9a-f]{40}$/, 'Pass the exact published candidate commit');
const revisions = { baseline: plan.baseline, candidate };
const controller = new AbortController();
const stop = () => controller.abort();
process.on('SIGINT', stop);
process.on('SIGTERM', stop);
const sha = async (path) => createHash('sha256').update(await readFile(path)).digest('hex');
const json = async (path, data) => writeFile(path, JSON.stringify(data, null, 2) + '\n');
const exists = async (path) => Boolean(await stat(path).catch((error) => {
  if (error.code === 'ENOENT') return undefined;
  throw error;
}));

async function idle() {
  const response = await fetch('http://127.0.0.1:4321/api/dashboard', { signal: AbortSignal.timeout(15000) });
  assert.equal(response.ok, true);
  const { state, runtime } = await response.json();
  assert.equal(state.orchestrator.enabled, false, 'Dispatch must remain paused');
  for (const key of ['runningAgents', 'runningEvaluations', 'runningComposites']) assert.equal(runtime[key], 0, key);
  const agent = state.agentRuns.find((item) => item.id === plan.candidate_agent);
  assert.equal(agent?.status, 'completed');
  assert.equal(agent.reviewApproved, true);
  assert.ok(agent.prNumber, 'Candidate must be published');
  assert.equal(agent.baseCommit, plan.baseline);
  return agent;
}

async function command(executable, args, cwd, env, label, timeoutMs = 1800000) {
  const reports = join(cwd, 'target/root-paired/reports');
  await mkdir(reports, { recursive: true });
  const startedAt = new Date().toISOString();
  const started = performance.now();
  console.log(`Starting ${label} in ${cwd}`);
  const result = await runCommand(executable, args, { cwd, env, timeoutMs, signal: controller.signal });
  const log = join(reports, `${label}.log`);
  await writeFile(log, result.stdout + '\n' + result.stderr);
  const receipt = { executable, args, cwd, startedAt, completedAt: new Date().toISOString(),
    seconds: (performance.now() - started) / 1000, exitCode: result.exitCode, log, logSha256: await sha(log) };
  await json(join(reports, `${label}.receipt.json`), receipt);
  console.log(`Finished ${label}: exit ${result.exitCode}, ${receipt.seconds.toFixed(2)} seconds`);
  assert.equal(result.exitCode, 0, `See ${log}; no automatic retry`);
  return result.stdout.trim();
}

async function git(args, cwd) {
  const result = await runCommand('git', args, { cwd, signal: controller.signal, timeoutMs: 120000 });
  assert.equal(result.exitCode, 0, result.stderr);
  return result.stdout.trim();
}

async function snapshot(cwd) {
  assert.equal(await git(['status', '--porcelain'], cwd), '', 'Source must remain clean');
  return { commit: await git(['rev-parse', 'HEAD'], cwd), tree: await git(['rev-parse', 'HEAD^{tree}'], cwd),
    diagnosticSha256: await sha(join(cwd, 'scripts/diagnose_cuda_add.py')),
    builderSha256: await sha(join(cwd, 'scripts/build_cuda_add_diagnostic.py')),
    lockSha256: await sha(join(cwd, 'uv.lock')) };
}

async function environment(cwd) {
  const env = {};
  for (const key of Object.keys(process.env)) {
    if (/^(PYO3_|PYTHON|CARGO_|UV_|TORCH_RS_)/.test(key) || ['VIRTUAL_ENV', 'CONDA_PREFIX', 'RUSTFLAGS', 'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER'].includes(key)) env[key] = undefined;
  }
  for (const [key, directory] of Object.entries({ UV_CACHE_DIR: 'uv-cache', UV_PYTHON_INSTALL_DIR: 'uv-python',
    CARGO_HOME: 'cargo-home', CARGO_TARGET_DIR: 'preflight-cargo', TMPDIR: 'tmp', XDG_CACHE_HOME: 'xdg-cache',
    CUDA_CACHE_PATH: 'cuda-cache', TORCHINDUCTOR_CACHE_DIR: 'inductor-cache', TRITON_CACHE_DIR: 'triton-cache',
    TORCH_HOME: 'torch-cache', TORCH_EXTENSIONS_DIR: 'torch-extensions' })) {
    env[key] = join(cwd, 'target', directory);
    await mkdir(env[key], { recursive: true });
  }
  for (const tool of ['rustc', 'rustdoc']) {
    const resolved = await runCommand('rustup', ['which', '--toolchain', '1.92.0', tool], { cwd, signal: controller.signal });
    assert.equal(resolved.exitCode, 0, resolved.stderr);
    env[tool.toUpperCase()] = resolved.stdout.trim();
  }
  return { ...env, UV_PROJECT_ENVIRONMENT: join(cwd, '.venv'), UV_PYTHON: plan.python, UV_MANAGED_PYTHON: '1',
    PATH: join(root, 'uv-tool/bin') + ':' + process.env.PATH,
    VIRTUAL_ENV: join(cwd, '.venv'), PYO3_PYTHON: join(cwd, '.venv/bin/python'),
    PYTHONDONTWRITEBYTECODE: '1', PYTHONHASHSEED: '0', CUDA_VISIBLE_DEVICES: String(plan.gpu),
    CARGO_BUILD_JOBS: '4', RUSTUP_TOOLCHAIN: '1.92.0', HTTPS_PROXY: 'http://fwdproxy:8080',
    CARGO_HTTP_PROXY: 'http://fwdproxy:8080' };
}

await idle();
const locks = new LockManager(join(repo, '.burner/locks'));
const lease = await locks.tryAcquireAll(['cpu-heavy', 'gpu'], 'root-scalar-public-add-pair');
assert.ok(lease, 'CPU/GPU resource leases must be idle');
try {
  const agent = await idle();
  assert.equal(await git(['rev-parse', agent.branch], repo), candidate);
  assert.equal(await exists(join(root, 'results.json')), false, 'Never overwrite an existing series');
  for (const name of plan.order) assert.equal(await exists(join(root, name)), false, `Never overwrite ${name}`);
  await json(join(root, 'selected-revisions.json'), { revisions, prNumber: agent.prNumber, selectedAt: new Date().toISOString() });
  const setups = {};
  for (const name of plan.order) {
    const cwd = join(root, name);
    await git(['clone', '--quiet', '--shared', '--no-hardlinks', '--no-checkout', repo, cwd], root);
    await git(['checkout', '--quiet', '--detach', revisions[name]], cwd);
    const before = await snapshot(cwd);
    assert.equal(before.commit, revisions[name]);
    const env = await environment(cwd);
    await command(env.RUSTC, ['--version', '--verbose'], cwd, env, 'rustc-version');
    await command('uv', ['--version'], cwd, env, 'uv-version');
    await command('uv', ['sync', '--locked', '--no-install-project', '--group', 'dev', '--group', 'reference', '--managed-python', '--python', plan.python], cwd, env, 'dependencies');
    const python = join(cwd, '.venv/bin/python');
    assert.equal(await sha(await realpath(python)), plan.python_binary_sha256, 'Interpreter binary must match author setup');
    await command('cargo', ['fetch', '--locked'], cwd, env, 'prime-cargo-download-cache');
    await command(python, ['-B', 'scripts/build_cuda_add_diagnostic.py', '--name', 'paired', '--revision', 'HEAD'], cwd, env, 'build-clean-export');
    await command(python, ['-B', '.github/scripts/verify_native_extension.py'], cwd, env, 'verify-provenance');
    const versions = JSON.parse(await command(python, ['-B', '-c', 'import sys,json,torch,numpy;print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,numpy=numpy.__version__)))'], cwd, env, 'versions'));
    const buildPath = join(cwd, 'target/cuda-add-diagnostic/paired/build-record.json');
    const build = JSON.parse(await readFile(buildPath, 'utf8'));
    assert.equal(build.measured_code_commit, revisions[name]);
    assert.equal(build.source_matches_commit, true);
    assert.equal(await sha(build.installed_native), build.installed_native_sha256);
    assert.deepEqual(await snapshot(cwd), before);
    setups[name] = { cwd, env, source: before, versions, buildPath, build };
  }
  for (const key of ['diagnosticSha256', 'builderSha256', 'lockSha256']) assert.equal(setups.baseline.source[key], setups.candidate.source[key], key);
  assert.deepEqual(setups.baseline.versions, setups.candidate.versions);
  const records = [];
  await json(join(root, 'results.json'), records);
  for (const name of plan.order) {
    await idle();
    const { cwd, env, source, buildPath, build } = setups[name];
    assert.deepEqual(await snapshot(cwd), source);
    assert.equal(await sha(build.installed_native), build.installed_native_sha256);
    const report = join(cwd, 'target/root-paired/reports/public-add.json');
    await command('taskset', ['-c', String(plan.cpu), join(cwd, '.venv/bin/python'), '-B', 'scripts/diagnose_cuda_add.py',
      '--build-record', buildPath, '--output', report, '--seed', String(plan.seed), '--samples', String(plan.samples_per_order),
      '--warmups', String(plan.warmups_per_order)], cwd, env, 'public-add-diagnostic');
    assert.deepEqual(await snapshot(cwd), source);
    const data = JSON.parse(await readFile(report, 'utf8'));
    assert.equal(data.cases.reduce((sum, item) => sum + item.rows.length, 0), plan.expected_rows_per_variant);
    records.push({ variant: name, source, versions: setups[name].versions, report, reportSha256: await sha(report),
      cases: data.cases.map((item) => ({ cache: item.cache_state, rows: item.rows.length, cappedParity: item.diagnostic_capped_geometric_parity_percent })) });
    await json(join(root, 'results.json'), records);
    console.log(JSON.stringify(records.at(-1)));
  }
} finally {
  await lease.release();
  process.off('SIGINT', stop);
  process.off('SIGTERM', stop);
}
