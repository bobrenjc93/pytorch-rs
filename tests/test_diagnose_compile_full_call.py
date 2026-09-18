"""Portable receipt-integrity checks for the non-scoring public-call diagnostic."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('full_call_diagnostic', ROOT / 'scripts/diagnose_compile_full_call.py')
diag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


class FullCallReceiptTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'target'
        folder.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        data = np.array([0., 2.], dtype=np.float32)
        self.key = hashlib.sha256(data.tobytes()).hexdigest()
        arrays = self.folder / 'arrays.npz'
        np.savez_compressed(arrays, **{self.key: data})
        tensor = dict(array=self.key, shape=[2], stride=[1], offset=0,
                      dtype='torch.float32', device='cuda:0', grad=False,
                      wrapper=0, input_aliases=[])
        self.report = dict(samples=17, warmups=5, calls_per_sample=1,
                           arrays=str(arrays), arrays_sha256=diag.sha(arrays), cells=[])
        for fn, shapes in diag.PROGRAMS:
            for mode in ('reused', 'fresh'):
                for shape in shapes:
                    self.report['cells'].append(dict(program=fn.__name__, shape=list(shape), mode=mode,
                        status='ok', no_body_replay=True, samples_ns=[123] * 17,
                        outputs=[copy.deepcopy(tensor) for _ in range(18)]))
        self.reference = self.folder / 'reference.json'
        self.reference.write_text(json.dumps(self.report))

    def compare(self, report):
        candidate = self.folder / 'candidate.json'
        candidate.write_text(json.dumps(report))
        with contextlib.redirect_stdout(io.StringIO()):
            return diag.compare(self.reference, candidate)

    def test_complete_equal_receipts(self):
        self.assertFalse(self.compare(self.report))

    def test_missing_samples_outputs_or_replay_check(self):
        for field in ('samples_ns', 'outputs', 'no_body_replay'):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                if field == 'no_body_replay':
                    report['cells'][0][field] = False
                else:
                    report['cells'][0][field].pop()
                with self.assertRaisesRegex(ValueError, 'incomplete successful'):
                    self.compare(report)

    def test_missing_cell(self):
        report = copy.deepcopy(self.report)
        report['cells'].pop()
        with self.assertRaisesRegex(ValueError, 'workload matrix'):
            self.compare(report)

    def test_signed_zero_mismatch_is_not_tolerance_equivalence(self):
        report = copy.deepcopy(self.report)
        data = np.array([-0., 2.], dtype=np.float32)
        key = hashlib.sha256(data.tobytes()).hexdigest()
        arrays = self.folder / 'negative-zero.npz'
        np.savez_compressed(arrays, **{key: data})
        report.update(arrays=str(arrays), arrays_sha256=diag.sha(arrays))
        for cell in report['cells']:
            for output in cell['outputs']:
                output['array'] = key
        self.assertTrue(self.compare(report))

    def test_archive_and_content_hashes(self):
        report = copy.deepcopy(self.report)
        report['arrays_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            self.compare(report)
        corrupt = self.folder / 'corrupt.npz'
        np.savez_compressed(corrupt, **{self.key: np.array([7., 8.], dtype=np.float32)})
        report.update(arrays=str(corrupt), arrays_sha256=diag.sha(corrupt))
        with self.assertRaisesRegex(ValueError, 'content hash'):
            self.compare(report)

    def test_metadata_alias_and_runtime_failure_are_not_passes(self):
        for field, value in (('stride', [2]), ('input_aliases', [0]), ('wrapper', 1)):
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report['cells'][0]['outputs'][0][field] = value
                self.assertTrue(self.compare(report))
        report = copy.deepcopy(self.report)
        report['cells'][0].update(status='error', error='launch failed')
        self.assertTrue(self.compare(report))
        report['cells'][0].update(status='unsupported', error='native rejects view')
        self.assertFalse(self.compare(report))  # Retained, explicitly unsupported; never a success.
        report['cells'][0]['status'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'unknown cell status'):
            self.compare(report)


if __name__ == '__main__':
    unittest.main()
