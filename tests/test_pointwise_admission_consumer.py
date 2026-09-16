"""Offline corruption controls for the bounded admission diagnostic verifier."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / 'docs/diagnostics/pointwise-admission-repair/consumer.py'
SPEC = importlib.util.spec_from_file_location('admission_consumer', PATH)
CONSUMER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONSUMER)


def fixture():
    """Small synthetic records isolate verifier policy, never measurement evidence."""
    payload = {'values': ['0x0.0p+0']}
    cells = []
    for name, sizes, _ in CONSUMER.CASES:
        for size in sizes:
            cells.append({'family': name, 'size': size, 'passed': True, 'warmups': list(range(1, 6)),
                          'inputs': [payload] * 23, 'first': payload, 'firstNs': 100,
                          'outputs': [payload] * 17, 'samples': [10] * 17,
                          'statistics': CONSUMER.common.summary([10] * 17),
                          'ownership': {'retainedOutputs': 17,
                                        'computedPointers': list(range(100, 134 if name == 'nested' else 117)),
                                        'inputPointers': list(range(1, 18 if name == 'unaryflat' else 35))}})
    reports = []
    for index, label in enumerate(CONSUMER.ORDER):
        build = 'B' if label == 'B' else 'C'
        reports.append({'passed': True, 'index': index, 'label': label, 'bindings': CONSUMER.bindings(),
                        'gpu': CONSUMER.common.GPU, 'gpuBefore': 'fixture', 'gpuAfter': 'fixture',
                        'interpreter': str(CONSUMER.ROOT / build / 'python'), 'interpreterSha256': build,
                        'source': {'root': str(CONSUMER.ROOT / build), 'commit': CONSUMER.BASE,
                                   'status': '' if build == 'B' else ' M src/python_pointwise.rs',
                                   'files': {'python/torch_rs/_compile_pointwise.py': build, 'src/python_pointwise.rs': build}},
                        'installed': {'fixtureBuild': build},
                        'environment': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                        'synchronization': {'api': 'cudaDeviceSynchronize', 'logicalDevice': 0,
                                            'path': 'fixture-runtime', 'sha256': 'fixture-sha', 'version': 13000},
                        'compiler': {'path': 'fixture-compiler', 'sha256': 'fixture-sha', 'version': [13, 0]},
                        'frameworkVersion': '2.13.0+cu130', 'referenceUuid': CONSUMER.common.GPU,
                        'referenceExtension': {'fixtureReference': True},
                        'started': str(index * 2), 'finished': str(index * 2 + 1),
                        'cells': copy.deepcopy(cells),
                        'churn': [{'family': name, 'size': sizes[0], 'phase': 25, 'input': payload,
                                   'output': payload, 'elapsedNs': 50} for name, sizes, _ in CONSUMER.CASES]})
    # ISO timestamps in real receipts sort chronologically as strings.
    for index, report in enumerate(reports):
        report.update(started=f'2026-09-16T00:00:{index * 2:02d}+00:00',
                      finished=f'2026-09-16T00:00:{index * 2 + 1:02d}+00:00')
    return reports


class AdmissionVerifier(unittest.TestCase):
    def verify(self, reports):
        # Native library validation has its own existing pure controls. These
        # fixtures exercise the new retention/thread/barrier contract alone.
        with patch.object(CONSUMER.common, 'validate_libraries'):
            return CONSUMER.verify_records(reports)

    def test_complete_fixture(self):
        result = self.verify(fixture())
        self.assertEqual(len(result['ratios']), 16)
        self.assertFalse(result['qualificationOrScoreProduced'])

    def test_ownership_corruption_rejected(self):
        corruptions = (
            lambda owner: owner.clear(),
            lambda owner: owner.update(retainedOutputs=16),
            lambda owner: owner.update(retainedOutputs=True),
            lambda owner: owner['computedPointers'].pop(),
            lambda owner: owner['computedPointers'].__setitem__(0, owner['computedPointers'][1]),
            lambda owner: owner['computedPointers'].__setitem__(0, owner['inputPointers'][0]),
            lambda owner: owner['computedPointers'].__setitem__(0, True),
            lambda owner: owner['inputPointers'].reverse(),
            lambda owner: owner['inputPointers'].pop(),
        )
        for mutate in corruptions:
            with self.subTest(mutate=mutate):
                reports = fixture()
                mutate(reports[2]['cells'][0]['ownership'])
                with self.assertRaises((ValueError, KeyError)):
                    self.verify(reports)

    def test_one_host_thread_required(self):
        for key, value in (('OMP_NUM_THREADS', '2'), ('MKL_NUM_THREADS', '8'), ('OMP_NUM_THREADS', None)):
            reports = fixture()
            reports[3]['environment'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'one host thread'):
                self.verify(reports)

    def test_completion_barrier_and_logical_device_required(self):
        for key, value in (('api', 'cudaStreamSynchronize'), ('logicalDevice', 1)):
            reports = fixture()
            reports[4]['synchronization'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'completion barrier'):
                self.verify(reports)

    def test_missing_leg_changed_identity_and_wrong_values_rejected(self):
        reports = fixture()
        with self.assertRaisesRegex(ValueError, 'six ordered'):
            self.verify(reports[:-1])
        reports[3]['installed'] = {'fixtureBuild': 'foreign'}
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.verify(reports)
        reports = fixture()
        reports[2]['cells'][0]['first'] = {'values': ['0x1.0p+0']}
        with self.assertRaisesRegex(ValueError, 'Native bits differ'):
            self.verify(reports)

    def test_failed_reordered_or_rebound_legs_rejected(self):
        for key, value in (('passed', False), ('index', 4), ('label', 'R'), ('bindings', {})):
            reports = fixture()
            reports[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.verify(reports)

    def test_incomplete_cells_or_churn_rejected(self):
        for field in ('cells', 'churn'):
            reports = fixture()
            reports[3][field].pop()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify(reports)
        reports = fixture()
        reports[4]['cells'][0]['outputs'].pop()
        with self.assertRaisesRegex(ValueError, 'Incomplete cell'):
            self.verify(reports)

    def test_overlapping_processes_rejected(self):
        reports = fixture()
        reports[2]['started'] = reports[1]['started']
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            self.verify(reports)


if __name__ == '__main__':
    unittest.main()
