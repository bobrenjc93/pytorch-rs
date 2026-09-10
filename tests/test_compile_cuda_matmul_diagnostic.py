"""Accounting of the separate diagnostic must retain missing/failed cells."""
import importlib.util
from pathlib import Path
import sys
import unittest


class MatmulDiagnosticTests(unittest.TestCase):
    def test_fixed_denominator_and_failures(self):
        scripts = Path(__file__).resolve().parents[1] / 'scripts'
        sys.path.insert(0, str(scripts))
        try:
            spec = importlib.util.spec_from_file_location('matmul_diagnostic', scripts/'diagnose_compile_cuda_matmul.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
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
