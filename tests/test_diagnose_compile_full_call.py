"""Portable receipt-integrity checks for the non-scoring public-call diagnostic."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
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

    def test_relocated_receipts_after_original_directory_is_removed(self):
        original = self.folder / 'capture'
        relocated = self.folder / 'relocated'
        original.mkdir()
        shutil.move(self.folder / 'arrays.npz', original / 'arrays.npz')
        report = copy.deepcopy(self.report)
        report['arrays'] = str(original / 'arrays.npz')
        for name in ('reference.json', 'candidate.json'):
            (original / name).write_text(json.dumps(report))
        shutil.copytree(original, relocated)
        shutil.rmtree(original)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(diag.compare(relocated / 'reference.json', relocated / 'candidate.json'))

    def test_missing_local_arrays_do_not_fall_back_to_original(self):
        relocated = self.folder / 'relocated'
        relocated.mkdir()
        for name in ('reference.json', 'candidate.json'):
            (relocated / name).write_text(json.dumps(self.report))
        self.assertTrue(Path(self.report['arrays']).is_file())
        with self.assertRaisesRegex(ValueError, 'missing array archive beside report'):
            diag.compare(relocated / 'reference.json', relocated / 'candidate.json')

    def test_corrupted_relocated_archive_does_not_use_valid_original(self):
        relocated = self.folder / 'relocated'
        relocated.mkdir()
        for name in ('reference.json', 'candidate.json'):
            (relocated / name).write_text(json.dumps(self.report))
        (relocated / 'arrays.npz').write_bytes(b'corrupted NPZ')
        with self.assertRaisesRegex(ValueError, 'array archive hash mismatch'):
            diag.compare(relocated / 'reference.json', relocated / 'candidate.json')

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


class ProfiledBundleTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'target'
        folder.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.base = ROOT / 'docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz'
        committed = ROOT / 'docs/diagnostics/profiled-full-call-20260918.tar.xz'
        self.assertEqual(committed.stat().st_size, 203344)
        self.assertEqual(diag.sha(committed), 'de35b43f2fc8fa5e31f9917953126f0244dcd1efd687c253351e47ba67f0370d')
        with tarfile.open(committed) as archive:
            helper = archive.extractfile('profiled-call/bundle.py').read()
        self.assertEqual(hashlib.sha256(helper).hexdigest(),
                         'a4f6f8e9ca0d64df3d07e337da709ab22aeb09a3eecb892d3145f25b30acbe1d')
        original_helper = self.folder / 'bundle.py'
        original_helper.write_bytes(helper)
        capture = self.folder / 'capture'
        capture.mkdir()
        with tarfile.open(self.base) as archive:
            manifest = json.load(archive.extractfile('full-call/manifest.json'))
            member = next(name for name in manifest if name.endswith('.npz'))
            (capture / 'mapped.npz').write_bytes(archive.extractfile('full-call/' + member).read())
        np.savez_compressed(capture / 'embedded.npz', novel=np.array([123.25], dtype=np.float32))
        for name in ('mapped', 'embedded'):
            arrays = capture / (name + '.npz')
            receipt = dict(arrays=str(arrays), arrays_sha256=diag.sha(arrays))
            (capture / (name + '.json')).write_text(json.dumps(receipt, indent=3) + '\n\n')
        self.expected = {path.name: path.read_bytes() for path in capture.iterdir()}
        packed = self.folder / 'packed.tar.xz'
        result = self.cli(original_helper, 'pack', capture, self.base, packed)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.relocated = self.folder / 'elsewhere' / 'profiled-call'
        self.relocated.mkdir(parents=True)
        with tarfile.open(packed) as archive:
            self.members = {Path(member.name).name for member in archive.getmembers()}
            for member in archive.getmembers():
                (self.relocated / Path(member.name).name).write_bytes(archive.extractfile(member).read())
        shutil.rmtree(capture)  # Only synthetic test data; recorded absolute paths now fail.
        original_helper.unlink()
        self.assertFalse(capture.exists())
        self.helper = self.relocated / 'bundle.py'
        self.manifest = self.relocated / 'bundle-manifest.json'

    def cli(self, helper, *arguments):
        return subprocess.run([sys.executable, '-B', '-I', str(helper), *map(str, arguments)],
                              cwd=self.folder, capture_output=True, text=True)

    def test_pack_relocate_restore_preserves_mapped_and_embedded_bytes(self):
        metadata = json.loads(self.manifest.read_text())
        self.assertIn('base_member', metadata['arrays']['mapped.npz'])
        self.assertNotIn('mapped.npz', metadata['payloads'])
        self.assertNotIn('mapped.npz', self.members)
        self.assertNotIn('base_member', metadata['arrays']['embedded.npz'])
        self.assertIn('embedded.npz', metadata['payloads'])
        self.assertIn('embedded.npz', self.members)
        result = self.cli(self.helper, 'restore', self.relocated, self.base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('restored 2 NPZ files', result.stdout)
        for name, data in self.expected.items():
            self.assertEqual((self.relocated / name).read_bytes(), data, name)

    def test_restore_rejects_wrong_base(self):
        wrong_base = self.folder / 'wrong-base.tar.xz'
        wrong_base.write_bytes(b'not the committed base archive')
        result = self.cli(self.helper, 'restore', self.relocated, wrong_base)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('base archive hash/size mismatch', result.stderr)

    def test_restore_rejects_tampered_ordinary_and_mapped_receipts(self):
        original = self.manifest.read_bytes()
        for section, name in (('payloads', 'embedded.json'), ('arrays', 'mapped.npz')):
            with self.subTest(section=section, name=name):
                metadata = json.loads(original)
                metadata[section][name]['sha256'] = '0' * 64
                self.manifest.write_text(json.dumps(metadata))
                result = self.cli(self.helper, 'restore', self.relocated, self.base)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('payload hash/size mismatch', result.stderr)


class HistoricalFullCallReplayTests(unittest.TestCase):
    def test_committed_archive_manifest_and_historical_replay(self):
        archive = ROOT / 'docs/diagnostics/default-compile-full-call-20260918-raw.tar.xz'
        self.assertEqual(archive.stat().st_size, 20204660)
        self.assertEqual(diag.sha(archive), '82ac63e93cee8d76d2f78226f9324ec92e032becf154bfac84861cd42c2ea265')
        with tarfile.open(archive) as bundle:
            manifest = json.load(bundle.extractfile('full-call/manifest.json'))
            self.assertEqual(len(bundle.getmembers()), 91)
            self.assertEqual(len(manifest), 90)
            for name, receipt in manifest.items():
                # extractfile resolves the historical tar's deduplicated hard links.
                data = bundle.extractfile('full-call/' + name).read()
                self.assertEqual(len(data), receipt['bytes'], name)
                self.assertEqual(hashlib.sha256(data).hexdigest(), receipt['sha256'], name)
            folder = ROOT / 'target'
            folder.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                relocated = Path(temporary)
                # Only these two reports and their arrays are needed for a historical
                # pair; no imports, extension, GPU, or recorded absolute paths.
                for name in ('incoming-reference-first.json', 'incoming-native-first.json'):
                    data = bundle.extractfile('full-call/' + name).read()
                    report = json.loads(data)
                    (relocated / name).write_bytes(data)
                    array_name = Path(report['arrays']).name
                    (relocated / array_name).write_bytes(bundle.extractfile('full-call/' + array_name).read())
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertFalse(diag.compare(relocated / 'incoming-reference-first.json',
                                                  relocated / 'incoming-native-first.json'))
                result = json.loads(output.getvalue())
                self.assertEqual(result['compared'], 76)
                self.assertEqual(result['candidate_rejections'], [])
                self.assertEqual(result['errors'], [])


if __name__ == '__main__':
    unittest.main()
