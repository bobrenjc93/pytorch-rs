import { spawn } from 'node:child_process';
import { LockManager } from '/data/users/bobren/a/burner/dist/lib/locks.js';

const locks = new LockManager('/data/users/bobren/a/pytorch-rs-burner/.burner/locks');
const lease = await locks.tryAcquireAll(['cpu-heavy', 'gpu'], 'root-paired-performance-1924');
if (!lease) throw new Error('Paired validation requires idle CPU/GPU resource locks');
let child;
const stop = (signal) => { if (child && child.exitCode === null) child.kill(signal); };
const interrupt = () => stop('SIGINT');
const terminate = () => stop('SIGTERM');
process.on('SIGINT', interrupt);
process.on('SIGTERM', terminate);
try {
  child = spawn('/data/users/bobren/a/pytorch-rs-burner/.venv/bin/python',
    ['-B', '/tmp/pytorch-rs-1924-paired.9gl9RC/validate.py', 'measure'],
    { cwd: '/tmp/pytorch-rs-1924-paired.9gl9RC', stdio: 'inherit' });
  process.exitCode = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code) => resolve(code ?? 1));
  });
} finally {
  process.off('SIGINT', interrupt);
  process.off('SIGTERM', terminate);
  await lease.release();
}
