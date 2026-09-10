import { readFileSync, writeFileSync } from 'node:fs';
const root = '/tmp/pytorch-rs-1924-paired.9gl9RC';
const rows = JSON.parse(readFileSync(`${root}/results.json`, 'utf8'));
const expected = ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate'];
if (JSON.stringify(rows.map(r => r.variant)) !== JSON.stringify(expected)) throw Error('Incomplete or changed run order');
if (rows.some(r => r.receipt.exit_code !== 0 || r.aggregates.workload_count !== 4 ||
    r.aggregates.per_workload.some(w => w.candidate_status !== 'ok' || !w.eligible_cuda_compile_evidence))) throw Error('Invalid workload result');
const median = values => [...values].sort((a, b) => a - b)[1];
const workloads = rows[0].aggregates.per_workload.map(({ name }) => {
  const values = variant => rows.filter(r => r.variant === variant).map(r =>
    r.aggregates.per_workload.find(w => w.name === name).torch_rs_steady_median_us);
  const baseline = median(values('baseline'));
  const candidate = median(values('candidate'));
  return { name, baseline_native_us: baseline, candidate_native_us: candidate,
    candidate_over_baseline: candidate / baseline, percent_change: 100 * (candidate / baseline - 1) };
});
const summary = { order: expected, scores: rows.map(r => r.aggregates.coverage_adjusted_overall_percent),
  runs: 6, reference_eligible_native_passes: 24, workloads,
  scope: 'Matched Python 3.12.12, CPU 24, H100 GPU 0; four fixed forward-only workloads. No benchmark changes or retries. Does not identify the cause of earlier Python 3.12.14+meta unpinned timing variation or establish general compiler parity.' };
writeFileSync(`${root}/summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
