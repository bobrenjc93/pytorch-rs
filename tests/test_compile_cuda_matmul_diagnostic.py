"""Accounting of the separate diagnostic must retain missing/failed cells."""
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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

    def test_measurement_definition_matches_merged_main_golden(self):
        # Byte/structural goldens from 46db0021e8db4b563327ac4b8290eb7eab4318f4.
        # These freeze the public diagnostic, not a scoring corpus. Updating
        # routing/provenance must never require updating these goldens.
        source = Path(self.diagnostic.__file__).read_text()
        tree = ast.parse(source)
        goldens = {
            'cells': 'd09a212930253200735d8a6b42ab26a4bbddfbe13b0c0f18ee056d01117e2c05',
            'aggregate': '69af3e19d8d402ea87dbe4df6ea592e0b83e08c8ce1dfc26d16e53d1bd381723',
            'program': '774d37539f64fd4eb448defd3a9d5bf07a8bcaed726afc755472287e3f279fa9',
            'sample_order': 'b45ddf43bca727a3da80f9fb9419e4c94ad75581d2ad2fcfb97db1b6b8e58280',
        }
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        for name, expected in goldens.items():
            with self.subTest(function=name):
                self.assertEqual(digest(ast.get_source_segment(source, functions[name])), expected)
        policy = next(n for n in ast.walk(functions['main']) if isinstance(n, ast.Dict)
                      and any(isinstance(k, ast.Constant) and k.value == 'warmups_per_order'
                              for k in n.keys))
        self.assertEqual(digest(ast.dump(policy)),
                         'd32ff54063387f7338fdcd16673efae86fe94085b1ea4d64608c26c24b259bf3')
        measurement = source[source.index('        rng = np.random.default_rng(args.seed)'):
                             source.index('        libraries = sorted')]
        self.assertEqual(digest(measurement),
                         'b427e1dc79de85516b21033a3842c609ecc4131f39a644abc812b6ad1b4ec34c')
        self.assertIn("report['capped_geometric_parity'] = aggregate(report['cases'])", source)

    def inventory(self):
        return {'devices': [
            {'index': '0', 'uuid': 'GPU-00000000-0000-0000-0000-000000000000',
             'pci.bus_id': '00000000:01:00.0'},
            {'index': '2', 'uuid': 'GPU-22222222-2222-2222-2222-222222222222',
             'pci.bus_id': '00000000:03:00.0'}]}

    def test_environment_retains_compiler_controls_without_secrets(self):
        controls = {'TORCHINDUCTOR_MAX_AUTOTUNE': '1',
                    'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM': '1',
                    'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS': 'ATEN',
                    'TORCHINDUCTOR_FX_GRAPH_CACHE': '0', 'TORCH_COMPILE_DISABLE': '1'}
        with patch.dict(os.environ, {**controls, 'CARGO_REGISTRY_TOKEN': 'secret',
                                     'TORCH_PROVIDER_AUTH': 'secret'}):
            observed = self.diagnostic.environment_provenance()
        for key, value in controls.items():
            self.assertEqual(observed[key], value)
        self.assertNotIn('secret', observed.values())

    def test_default_and_explicit_masks(self):
        inventory = self.inventory()
        for index, declared, mask in ((0, None, '0'), (1, inventory['devices'][1]['uuid'], '2'),
                                      (1, inventory['devices'][1]['uuid'], inventory['devices'][1]['uuid'])):
            with self.subTest(mask=mask):
                selection = {'cuda_visible_devices': mask, 'declared_uuid': declared}
                self.assertEqual(self.diagnostic.select_gpu(selection, inventory), inventory['devices'][index])

    def test_reject_absent_ambiguous_and_unbounded_selection(self):
        inventory = self.inventory()
        gpu = inventory['devices'][1]['uuid']
        for mask, declared in ((None, None), ('', None), ('0,2', None), ('0,', None),
                               ('2', None), (gpu, None), ('0', gpu), ('-1', gpu),
                               (gpu, ''), (gpu, gpu[:12]), (gpu, 'MIG-' + gpu),
                               (gpu + ',0', gpu), (gpu, 'GPU-33333333-3333-3333-3333-333333333333')):
            with self.subTest(mask=mask, declared=declared), self.assertRaises(RuntimeError):
                self.diagnostic.select_gpu({'cuda_visible_devices': mask, 'declared_uuid': declared}, inventory)
        for devices in ([], inventory['devices'] * 2,
                        [inventory['devices'][0], dict(inventory['devices'][1], index='0')],
                        [inventory['devices'][0], dict(inventory['devices'][1],
                                                       **{'pci.bus_id': '0000:01:00.0'})]):
            with self.subTest(devices=devices), self.assertRaises(RuntimeError):
                self.diagnostic.select_gpu({'cuda_visible_devices': '0', 'declared_uuid': None},
                                           {'devices': devices})

    def test_bind_rejects_runtime_order_count_uuid_and_pci_mismatch(self):
        physical = self.inventory()['devices'][1]
        observed = {'visible_count': 1, 'logical_index': 0, 'uuid': physical['uuid'],
                    'pci.bus_id': '0000:03:00.0'}
        self.diagnostic.bind_gpu_identity(observed, physical)
        self.assertEqual(observed['physical_index'], 2)
        for field, value in (('visible_count', 0), ('visible_count', 2), ('logical_index', 2),
                             ('uuid', self.inventory()['devices'][0]['uuid']),
                             ('pci.bus_id', '0000:01:00.0')):
            with self.subTest(field=field, value=value), self.assertRaises(RuntimeError):
                self.diagnostic.bind_gpu_identity(dict(observed, **{field: value}), physical)

    def test_each_runtime_is_independently_bound_and_observations_retained(self):
        module = self.diagnostic
        inventory = self.inventory()
        gpu = inventory['devices'][1]['uuid']
        selection = {'cuda_visible_devices': gpu, 'declared_uuid': gpu}
        good = {'visible_count': 1, 'logical_index': 0, 'uuid': gpu, 'pci.bus_id': '0000:03:00.0'}
        for bad_role in (None, 'driver', 'native_runtime', 'reference_runtime'):
            driver, runtime = copy.deepcopy(good), copy.deepcopy(good)
            reference_uuid = gpu
            wrong = inventory['devices'][0]['uuid']
            if bad_role == 'driver': driver['uuid'] = wrong
            if bad_role == 'native_runtime': runtime['pci.bus_id'] = '0000:01:00.0'
            if bad_role == 'reference_runtime': reference_uuid = wrong
            torch = SimpleNamespace(version=SimpleNamespace(cuda='13.0'), cuda=SimpleNamespace(
                device_count=lambda: 1, current_device=lambda: 0,
                get_device_properties=lambda _: SimpleNamespace(uuid=reference_uuid)))
            native = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))
            observations = {}
            with self.subTest(role=bad_role), patch.dict(os.environ, {
                    'CUDA_VISIBLE_DEVICES': gpu, 'TORCH_RS_CUDART': module.__file__}), \
                    patch.object(module, 'cuda_identity', side_effect=[driver, runtime]) as probe:
                if bad_role:
                    with self.assertRaisesRegex(RuntimeError, 'UUID'):
                        module.verify_gpu(selection, inventory, torch, native, observations)
                else:
                    module.verify_gpu(selection, inventory, torch, native, observations)
                self.assertEqual(probe.call_count, 2)
                self.assertEqual(set(observations), {'driver', 'native_runtime',
                                                     'native_visible_count', 'reference_runtime'})
                self.assertEqual(observations['reference_runtime']['uuid'], reference_uuid)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], gpu)

    def test_native_bus_absence_and_changed_mask_rejected_before_work(self):
        module = self.diagnostic
        inventory = self.inventory()
        gpu = inventory['devices'][1]['uuid']
        selection = {'cuda_visible_devices': gpu, 'declared_uuid': gpu}
        driver = {'visible_count': 1, 'logical_index': 0, 'uuid': gpu, 'pci.bus_id': '0000:03:00.0'}
        runtime = dict(driver, **{'pci.bus_id': '0000:05:00.0'})
        observations = {}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': gpu, 'TORCH_RS_CUDART': module.__file__}), \
                patch.object(module, 'cuda_identity', side_effect=[driver, runtime]):
            with self.assertRaisesRegex(RuntimeError, 'PCI identity absent'):
                module.verify_gpu(selection, inventory, None, None, observations)
        self.assertEqual(observations['native_runtime']['pci.bus_id'], '0000:05:00.0')
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                patch.object(module, 'cuda_identity') as probe:
            with self.assertRaisesRegex(RuntimeError, 'mask changed'):
                module.verify_gpu(selection, inventory, None, None, {})
            probe.assert_not_called()

    def test_rejection_report_retains_inventory_denominator_and_zero(self):
        module = self.diagnostic
        target = module.ROOT / 'target'
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as directory:
            output = Path(directory) / 'rejected.json'
            with patch.object(sys, 'argv', [module.__file__, '--build-record', 'unused.json',
                                           '--output', str(output)]), \
                    patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,2'}), \
                    patch.object(module, 'gpu_inventory', return_value=self.inventory()), \
                    patch.object(module, 'source_provenance') as provenance:
                self.assertEqual(module.main(), 1)
                provenance.assert_not_called()
                report = json.loads(output.read_text())
                self.assertEqual(report['seed'], 798431)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['capped_geometric_parity'], 0)
                self.assertEqual(len(report['cases']), 12)
                self.assertTrue(all(c['capped_parity'] == 0 for c in report['cases']))
                self.assertEqual(report['gpu_inventories']['before'], self.inventory())
                self.assertEqual(report['gpu_inventories']['after'], self.inventory())
                self.assertIn('exactly one device', report['error'])
                with self.assertRaises(FileExistsError):
                    module.main()
