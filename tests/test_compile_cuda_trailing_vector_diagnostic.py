"""The new non-scoring diagnostic must retain failed and interrupted evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_cuda_add import torch
if torch is not None:
    from scripts import diagnose_compile_cuda_trailing_vector as diagnostic

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch is not None, "requires optional PyTorch")
class TrailingVectorDiagnosticTests(unittest.TestCase):
    def test_failures_and_interruptions_are_retained(self):
        (ROOT / 'target').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'target') as directory:
            output = Path(directory) / 'report.json'
            argv = ['--case-set', 'trailing_vector_v1', '--build-record', 'unused', '--output', str(output)]
            bad = dict(reference_eligible=False, native_outcome='error', expectation_met=False, credit=0)
            with patch.object(diagnostic, 'environment', return_value={}), \
                 patch.object(diagnostic, 'cases', return_value=[('x + y', (3, 7), 'offset', True)]), \
                 patch.object(diagnostic, 'run_case', side_effect=[bad, KeyboardInterrupt()]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diagnostic.main(argv), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report['cases'], [bad])
            self.assertIn('KeyboardInterrupt', report['setup_or_interruption_error'])
            with patch.object(diagnostic, 'environment', side_effect=RuntimeError('source mismatch')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(diagnostic.main(argv), 1)
            self.assertIn('source mismatch', json.loads(output.read_text())['setup_or_interruption_error'])

    def test_unsupported_and_ineligible_cases_never_earn_credit(self):
        for expression, supported in (('x + y', True), ('x.add(y, alpha=2)', False)):
            with patch.object(diagnostic.common.torch, 'compile', side_effect=RuntimeError('bad reference')), \
                 patch.object(diagnostic.common.native, 'compile', side_effect=NotImplementedError('unsupported')):
                record = diagnostic.run_case(expression, (3, 7), 'offset', supported, True)
            self.assertEqual(record['credit'], 0)
            self.assertFalse(record['expectation_met'])
            self.assertIn('bad reference', record['reference_error'])
            self.assertEqual(record['native_outcome'], 'unsupported')
