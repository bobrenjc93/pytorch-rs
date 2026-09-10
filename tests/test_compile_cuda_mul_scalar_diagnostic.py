"""Failure accounting for the separate, non-scoring scalar diagnostic."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_cuda_add import torch

if torch is not None:
    from scripts import diagnose_compile_cuda_mul_scalar as diagnostic


@unittest.skipUnless(torch is not None, 'requires optional PyTorch reference dependency')
class ScalarDiagnosticAccountingTests(unittest.TestCase):
    def test_failed_reference_wrong_data_and_unsupported_results_are_retained(self):
        outputs = [{'sha256': 'first'}, {'sha256': 'second'}]
        supported = dict(reference_eligible=True, expected_supported=True,
                         native_outcome='pass', reference_outputs=outputs,
                         native_outputs=outputs)
        unsupported = dict(reference_eligible=True, expected_supported=False,
                           native_outcome='unsupported', native_error='unsupported scalar')
        scenarios = (
            (supported, 0), (unsupported, 0),
            ({**supported, 'native_outputs': outputs[:1]}, 1),
            ({**supported, 'native_outputs': [{'sha256': 'wrong'}, outputs[1]]}, 1),
            ({**supported, 'reference_eligible': False}, 1),
            ({**supported, 'native_outcome': 'unsupported'}, 1),
            ({**unsupported, 'reference_eligible': False}, 1),
            ({**unsupported, 'native_outcome': 'pass'}, 1),
            ({**unsupported, 'native_outcome': 'error'}, 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'report.json'
            for record, status in scenarios:
                with self.subTest(record=record), \
                     patch.object(diagnostic.neg_add, 'environment',
                                  return_value={'source_sha256': {}, 'environment': {}}), \
                     patch.object(diagnostic, 'cases', return_value=(
                         [(diagnostic.scale_left, (3,), 1, 'offset', True)], {})), \
                     patch.object(diagnostic.frozen, 'command', return_value='test toolchain'), \
                     patch.object(diagnostic.frozen, 'run_case',
                                  side_effect=lambda *args: dict(record)), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(diagnostic.main([
                        '--case-set', 'mul_neg_add_v1', '--output', str(output)]), status)
                    report = json.loads(output.read_text())
                    self.assertEqual(report['summary']['expectation_failures'], 2 * status)
                    for retained in report['cases']:
                        self.assertEqual({k: v for k, v in retained.items()
                                          if k != 'expectation_met'}, record)

    def test_setup_failure_is_written_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(diagnostic.neg_add, 'environment',
                          side_effect=RuntimeError('source mismatch')), \
             contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory) / 'report.json'
            self.assertEqual(diagnostic.main([
                '--case-set', 'mul_neg_add_v1', '--output', str(output)]), 1)
            report = json.loads(output.read_text())
            self.assertEqual(report['setup_error'], 'RuntimeError: source mismatch')
            self.assertEqual(report['cases'], [])


if __name__ == '__main__':
    unittest.main()
