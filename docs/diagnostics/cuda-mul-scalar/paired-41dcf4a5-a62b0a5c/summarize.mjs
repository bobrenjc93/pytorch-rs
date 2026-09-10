import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(import.meta.url));
const records = JSON.parse(await readFile(join(root, 'results.json'), 'utf8'));
assert.deepEqual(records.map((item) => item.variant), ['baseline', 'candidate']);
const reports = {};
for (const record of records) {
  const contents = await readFile(record.report);
  assert.equal(createHash('sha256').update(contents).digest('hex'), record.reportSha256);
  reports[record.variant] = JSON.parse(contents);
}
const identity = (cache, row) => JSON.stringify([cache, row.shape, row.layout, row.form, row.calls_per_sample, row.chained]);
const flatten = (report) => new Map(report.cases.flatMap((condition) => condition.rows.map((row) =>
  [identity(condition.cache_state, row), { cache: condition.cache_state, ...row }])));
const baseline = flatten(reports.baseline);
const candidate = flatten(reports.candidate);
assert.equal(baseline.size, 152);
assert.equal(candidate.size, 152);
assert.deepEqual([...baseline.keys()].sort(), [...candidate.keys()].sort());
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
};
const geomean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
const rows = [];
for (const [key, before] of baseline) {
  const after = candidate.get(key);
  for (const row of [before, after]) {
    for (const variant of ['native', 'torch']) {
      assert.equal(row.checks[variant].bitwise_equal, true);
      assert.equal(row.samples_ns_per_call[variant].length, 18);
      assert.ok(row.samples_ns_per_call[variant].every((value) => Number.isFinite(value) && value > 0));
      assert.ok(Math.abs(median(row.samples_ns_per_call[variant]) - row.median_ns[variant]) < 1e-8);
    }
    assert.equal(row.checks.native.sha256, row.checks.torch.sha256);
  }
  assert.deepEqual(after.checks.native, before.checks.native, 'Cross-revision outputs and metadata must be identical');
  const nativeRatio = after.median_ns.native / before.median_ns.native;
  const referenceRatio = after.median_ns.torch / before.median_ns.torch;
  rows.push({ cache: before.cache, shape: before.shape, layout: before.layout, form: before.form,
    callsPerSample: before.calls_per_sample, chained: before.chained,
    baselineMedianNs: before.median_ns, candidateMedianNs: after.median_ns,
    candidateToBaselineNativeLatency: nativeRatio,
    candidateToBaselineReferenceLatency: referenceRatio,
    relativeNativeLatencyChangeVsReference: nativeRatio / referenceRatio });
}
const groups = new Map();
for (const row of rows) {
  const key = JSON.stringify([row.cache, row.callsPerSample, row.chained]);
  const group = groups.get(key) ?? [];
  group.push(row);
  groups.set(key, group);
}
const grouped = [...groups.values()].map((group) => ({ cache: group[0].cache,
  callsPerSample: group[0].callsPerSample, chained: group[0].chained, rows: group.length,
  candidateToBaselineNativeLatency: geomean(group.map((row) => row.candidateToBaselineNativeLatency)),
  candidateToBaselineReferenceLatency: geomean(group.map((row) => row.candidateToBaselineReferenceLatency)),
  relativeNativeLatencyChangeVsReference: geomean(group.map((row) => row.relativeNativeLatencyChangeVsReference)),
  nativeLatencyRatioRange: [Math.min(...group.map((row) => row.candidateToBaselineNativeLatency)),
    Math.max(...group.map((row) => row.candidateToBaselineNativeLatency))] }));
assert.equal(grouped.length, 8);
assert.ok(grouped.every((group) => group.rows === 19));
const summary = { purpose: 'Matched diagnostic only; does not alter Burner scores',
  limitation: 'One fixed baseline-then-candidate series, with separate clean/saturated processes and 18 samples per implementation per row; not proof of universal parity or a causal explanation of historical timing changes.',
  interpretation: 'Latency ratios above 1 mean the candidate is slower. Reference-normalized ratios account descriptively for the same-run PyTorch timing change; no statistical significance is inferred.',
  revisions: Object.fromEntries(records.map((record) => [record.variant, record.source.commit])),
  versions: records.map((record) => ({ variant: record.variant, ...record.versions })),
  comparedRows: rows.length, allOutputsAndMetadataIdentical: true, grouped, rows };
await writeFile(join(root, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify({ comparedRows: summary.comparedRows, allOutputsAndMetadataIdentical: true, grouped }, null, 2));
