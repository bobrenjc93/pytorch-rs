"""Hardware-free checks of the non-scoring diagnostic's evidence boundaries."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'docs/diagnostics/compile-pointwise-loops/compare.py'
spec = importlib.util.spec_from_file_location('loop_diagnostic', PATH)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class DiagnosticContract(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'target/loop-diagnostic-tests'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_reference_zero_always_checks_sign_without_requiring_exact_zero(self):
        def compare(a, b):
            diagnostic.compare({'values': [float(a).hex()]}, {'values': [float(b).hex()]})
        for a, b in ((-1e-7, 0.0), (1e-7, -0.0), (-0.0, 0.0), (0.0, -0.0)):
            with self.subTest(a=a, b=b), self.assertRaises(AssertionError): compare(a, b)
        for a, b in ((1e-7, 0.0), (-1e-7, -0.0), (0.0, 0.0), (-0.0, -0.0), (1.000005, 1.0)):
            compare(a, b)
        for a, b in ((2e-6, 0.0), (-2e-6, -0.0), (1.0001, 1.0)):
            with self.subTest(a=a, b=b), self.assertRaises(AssertionError): compare(a, b)

    def capture_run(self, fail=False):
        history = {'warmup_ns': [], 'samples_ns': []}
        dispatches = []
        first = types.SimpleNamespace(source='first source', ptx='first ptx', nvrtc_version=(13, 0), options=[], device=0)
        offending = types.SimpleNamespace(source='offending source', ptx='offending ptx', nvrtc_version=(13, 0), options=[], device=0)
        calls = []
        def compiled():
            calls.append(history['phase'])
            if history['phase'] == 'cold':
                dispatches.append(('cold', first))
                return 'initial'
            if history['phase'] == 'warm_guard':
                dispatches.append(('warm_guard', offending))
                if fail: raise RuntimeError('first execution failure')
                return 'different final output'
            return 'initial'
        def save(result):
            diagnostic.capture_observations(history, result, (), dispatches, self.directory, 'case')
            diagnostic.write(self.directory / 'report.json.gz', history)
        with patch.object(diagnostic, 'encode', side_effect=lambda value: value):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'first execution failure'):
                    diagnostic.measure_history(compiled, (), lambda: None, history, save, lambda *args: None)
            else:
                diagnostic.measure_history(compiled, (), lambda: None, history, save, lambda *args: None)
                with self.assertRaises(AssertionError):
                    assert history['observed_output'] == history['output']
        saved = json.loads(gzip.decompress((self.directory / 'report.json.gz').read_bytes()))
        self.assertEqual(len(calls), 24)  # cold + five warmups + 17 samples + warm probe; no replay
        self.assertEqual(saved['output'], 'initial')
        self.assertEqual(saved['observed_output'], None if fail else 'different final output')
        self.assertEqual(saved['observed_inputs'], [])
        self.assertEqual(saved['phase'], 'warm_guard')
        self.assertEqual(len(saved['observed_kernels']), 2)
        self.assertEqual((self.directory / 'case-observed-1.cu').read_text(), 'offending source')
        self.assertEqual(gzip.decompress((self.directory / 'case-observed-1.ptx.gz').read_bytes()), b'offending ptx')

    def test_mismatch_saved_with_actual_observed_receivers_before_assertion(self):
        self.capture_run()

    def test_execution_failure_saves_available_evidence_without_replay(self):
        self.capture_run(fail=True)

    def test_observation_failure_does_not_discard_other_available_artifacts(self):
        receiver = types.SimpleNamespace(source='available', nvrtc_version=(13, 0), options=[], device=0)
        history = {}
        with patch.object(diagnostic, 'encode', side_effect=ValueError('unreadable result')):
            diagnostic.capture_observations(history, object(), (), [('cold', receiver)], self.directory, 'partial')
        self.assertEqual((self.directory / 'partial-observed-0.cu').read_text(), 'available')
        self.assertEqual(len(history['observation_errors']), 2)

    def test_timeout_stops_all_later_legs_and_preserves_available_report(self):
        output, workspace = self.directory / 'raw', self.directory / 'workspace'
        output.mkdir(); workspace.mkdir()
        args = types.SimpleNamespace(output=output, workspace=workspace, wheel=ROOT/'unused.whl', devices='0,1')
        def timeout(command, **kwargs):
            raw = Path(command[command.index('--durable-raw-output') + 1])
            diagnostic.write(raw/'report.json.gz', {'passed': False, 'partial': 'available'})
            for key in ('TMPDIR', 'CUDA_CACHE_PATH', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'XDG_CACHE_HOME'):
                self.assertTrue(Path(kwargs['env'][key]).is_relative_to(workspace))
                self.assertFalse(Path(kwargs['env'][key]).is_relative_to(output))
            raise subprocess.TimeoutExpired(command, 600)
        with patch.object(diagnostic, 'command', return_value=''), patch.object(diagnostic.subprocess, 'run', side_effect=timeout) as launch:
            self.assertEqual(diagnostic.run(args), 1)
        self.assertEqual(launch.call_count, 1)
        saved = json.loads(gzip.decompress((output/'summary.json.gz').read_bytes()))
        self.assertFalse(saved['passed'])
        self.assertIn('quiescence', saved['stopped'])
        self.assertFalse(saved['legs'][0]['descendants_quiescent'])
        self.assertEqual(saved['comparisons'], [])
        self.assertTrue((output/saved['legs'][0]['directory']/'report.json.gz').exists())

    def test_explicit_durable_destination_is_exclusive_and_workspace_stays_local(self):
        checkout = self.directory/'checkout'
        checkout.mkdir()
        raw = self.directory/'durable-attempt'
        def args():
            return types.SimpleNamespace(output=checkout/'workspace', durable_raw_output=raw,
                                         wheel=checkout/'wheel.whl', devices='0,1', role=None)
        with patch.object(diagnostic, 'ROOT', checkout), patch.object(diagnostic.sys, 'executable', str(checkout/'python')):
            configured = args()
            diagnostic.prepare_paths(configured)
            self.assertEqual(configured.output, raw)
            self.assertEqual(configured.workspace, checkout/'workspace')
            with self.assertRaises(FileExistsError): diagnostic.prepare_paths(args())
            invalid = args(); invalid.output = self.directory/'outside-workspace'
            with self.assertRaises(AssertionError): diagnostic.prepare_paths(invalid)


if __name__ == '__main__': unittest.main()
