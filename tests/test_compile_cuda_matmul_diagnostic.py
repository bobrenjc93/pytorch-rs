"""Accounting of the separate diagnostic must retain missing/failed cells."""
import importlib.util
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


class MatmulDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location('matmul_diagnostic', scripts/'diagnose_compile_cuda_matmul.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.diagnostic = module
        finally:
            sys.path.pop(0)

    def test_fixed_denominator_and_failures(self):
        module = self.diagnostic
        cells = module.cells(12)
        self.assertEqual(cells, module.cells(12))
        self.assertEqual(len(cells), 12)
        self.assertNotEqual(cells, module.cells(13))
        self.assertEqual(module.aggregate(cells), 0)
        for cell in cells:
            cell.update(status='passed', capped_parity=.5)
        self.assertAlmostEqual(module.aggregate(cells), .5)
        cells[0].update(status='failed', capped_parity=1.)
        self.assertEqual(module.aggregate(cells), 0)
        cells[0].update(status='passed', capped_parity=.25)
        self.assertAlmostEqual(module.aggregate(cells), (.25*.5**11)**(1/12))

    def test_failed_output_check_retains_prior_and_partial_samples(self):
        entry = {}
        checks = []

        def check(value):
            checks.append(value)
            if len(checks) == 7:
                raise AssertionError('bad output midway through reference block')

        order = ['native', 'pytorch']
        with patch.object(self.diagnostic.time, 'perf_counter_ns',
                          side_effect=itertools.count(step=1000).__next__):
            with self.assertRaisesRegex(AssertionError, 'bad output'):
                self.diagnostic.sample_order(
                    {'native': lambda: 'native', 'pytorch': lambda: 'pytorch'},
                    {key: () for key in order}, order, lambda: None, check, entry)
        native, reference = (entry['samples'][key] for key in order)
        self.assertEqual(native['samples_us'], [1.])
        self.assertEqual(native['calls_us'], [1.] * 5)
        self.assertEqual(native['checked_calls'], 5)
        self.assertEqual(reference['samples_us'], [])
        self.assertEqual(reference['calls_us'], [1., 1.])
        self.assertEqual(reference['checked_calls'], 1)
