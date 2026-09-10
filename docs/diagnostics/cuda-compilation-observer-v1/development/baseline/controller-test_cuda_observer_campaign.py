"""Portable campaign boundary checks; synthetic checks confer no hardware credit."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('campaign', ROOT / 'scripts/cuda_compilation_campaign.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.evaluator = c.load_manifest()
        work = ROOT / 'target/campaign-tests'
        work.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=work)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pins_reject_every_changed_observer_input(self):
        for path in self.manifest['observer_files']:
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, dest)
        self.assertEqual(c.pins(self.root, self.manifest), self.manifest['observer_files'])
        for path in self.manifest['observer_files']:
            with self.subTest(path=path):
                dest = self.root / path
                original = dest.read_bytes()
                dest.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    c.pins(self.root, self.manifest)
                dest.write_bytes(original)

    def test_production_digest_rejects_byte_changes(self):
        # Test the guard without requiring future main to stay at this campaign's
        # historical baseline or requiring full Git history in portable CI.
        source = self.root / 'src/lib.rs'
        source.parent.mkdir()
        source.write_bytes(b'original production')
        files = {'src/lib.rs': c.sha(source)}
        manifest = {'baseline': {'production': {'files': files, 'sha256':
            hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}}}
        c.production(self.root, manifest, 'baseline')
        source.write_bytes(b'changed production')
        with self.assertRaisesRegex(ValueError, 'production bytes changed'):
            c.production(self.root, manifest, 'baseline')

    def test_scope_excludes_implementation_and_other_evaluators(self):
        self.assertFalse(c.allowed_baseline_path('src/cuda.rs'))
        self.assertFalse(c.allowed_baseline_path('python/torch_rs/_compile_trace.py'))
        self.assertFalse(c.allowed_baseline_path('Cargo.toml'))
        self.assertFalse(c.allowed_baseline_path('scripts/evaluate_cuda_math.py'))
        self.assertFalse(c.allowed_baseline_path('docs/hardware-heterogeneity-matrix-v1.json'))
        self.assertFalse(c.allowed_baseline_path('.burner/evaluations.json'))

    def test_gpu0_is_pinned_not_any_idle_gpu(self):
        row = 'index, uuid, name\n0, ' + self.manifest['device']['uuid'] + ', NVIDIA H100\n'
        c.check_gpu(row, self.manifest)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            c.check_gpu(row.replace(self.manifest['device']['uuid'], 'GPU-other'), self.manifest)
        c.check_torch_uuid(self.manifest['device']['uuid'].removeprefix('GPU-'), self.manifest)
        with self.assertRaisesRegex(ValueError, 'UUID differs'):
            c.check_torch_uuid('11979b85-93e3-21d3-e68f-df37b8a4c296', self.manifest)

    def test_missing_reduction_retains_zero_slot(self):
        spec = importlib.util.spec_from_file_location('observer_tests', ROOT / 'tests/test_cuda_compilation_evaluator.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fixture = module.AccountingTests()
        fixture.setUp()
        for trial in fixture.trials:
            if trial['case_id'] == 'cuda_f32_compile_sum_rows':
                trial['candidate'].update(status='failed', executions=[],
                    error="AttributeError: module 'torch_rs.torch_rs' has no attribute '_compile_trace_reduction'")
        score = fixture.score()
        self.assertEqual(score['denominator'], 6)
        self.assertEqual([row['credit'] for row in score['cases']], [1, 1, 1, 1, 0, 1])

    def test_seed_selection_and_case_set_are_preserved(self):
        self.assertEqual(self.manifest['case_set'], self.evaluator.corpus())
        self.assertEqual(self.manifest['seed_selection']['seeds'],
                         [8454740352132840342, 4006623364652518084])
        self.assertEqual(self.manifest['options'], self.evaluator.OPTIONS)

    def test_published_evidence_and_tamper_rejection(self):
        published = ROOT / 'docs/diagnostics/cuda-compilation-observer-v1/development'
        if not published.exists():
            self.skipTest('development evidence not published yet')
        actual = c.validate(published, self.manifest, self.evaluator)
        self.assertEqual(actual['mode'], 'development')
        self.assertFalse(actual['human_approval'])
        shutil.copytree(published, self.root / 'evidence')
        receipt = self.root / 'evidence/campaign.json'
        data = c.read(receipt)
        data['seeds'][0] += 1
        c.dump(receipt, data)
        with self.assertRaisesRegex(ValueError, 'seed mismatch'):
            c.validate(receipt.parent, self.manifest, self.evaluator)
        shutil.copyfile(published / 'campaign.json', receipt)
        result = receipt.parent / 'baseline/result.json'
        data = c.read(result)
        data['accounting']['passed'] = 6
        c.dump(result, data)
        with self.assertRaisesRegex(ValueError, 'evidence hash mismatch'):
            c.validate(receipt.parent, self.manifest, self.evaluator)
