"""Real compiler-owner failures, isolated from non-scoring public timings."""
import ctypes
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import Mock, patch

import torch_rs as native
from tests.test_compile_pointwise_jit import available, cache, program


def real_nvrtc_provider(environment):
    """Mirror native NVRTC loader names/override precedence before interposing.

    Keep the verified library alive; loader-resolved sonames are valid pins.
    An explicit pin never falls through to another provider.
    """
    pin = environment.get('TEST_PROGRAM_IDENTITY_REAL_NVRTC',
                          environment.get('TORCH_RS_NVRTC'))
    paths = [pin] if pin is not None else ['libnvrtc.so.13', 'libnvrtc.so.12', 'libnvrtc.so']
    errors = []
    for path in paths:
        try:
            library = ctypes.CDLL(path)
            for symbol in ('nvrtcVersion', 'nvrtcCreateProgram', 'nvrtcCompileProgram',
                           'nvrtcGetProgramLogSize', 'nvrtcGetProgramLog',
                           'nvrtcGetPTXSize', 'nvrtcGetPTX', 'nvrtcDestroyProgram'):
                getattr(library, symbol)
        except (OSError, AttributeError) as error:
            errors.append(f'{path}: {error}')
            continue
        library.nvrtcVersion.argtypes = [ctypes.POINTER(ctypes.c_int)] * 2
        library.nvrtcVersion.restype = ctypes.c_int
        major, minor = ctypes.c_int(), ctypes.c_int()
        status = library.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        if status != 0:
            raise RuntimeError(f'{path}: nvrtcVersion status {status}')
        return path, library
    raise RuntimeError('cannot load real NVRTC: ' + '; '.join(errors))


class NvrtcProviderSetup(unittest.TestCase):
    def test_default_discovery_and_explicit_loader_names(self):
        library = Mock()
        library.nvrtcVersion.return_value = 0
        for environment, expected in (
                ({}, 'libnvrtc.so.13'),
                ({'TORCH_RS_NVRTC': 'libnvrtc.so.13'}, 'libnvrtc.so.13'),
                ({'TORCH_RS_NVRTC': '/owned/libnvrtc.so.13'}, '/owned/libnvrtc.so.13'),
                ({'TEST_PROGRAM_IDENTITY_REAL_NVRTC': 'libnvrtc.so.13',
                  'TORCH_RS_NVRTC': '/interposer.so'}, 'libnvrtc.so.13')):
            with self.subTest(environment=environment), patch.object(ctypes, 'CDLL', return_value=library) as load:
                self.assertEqual(real_nvrtc_provider(environment), (expected, library))
                load.assert_called_once_with(expected)
        with patch.object(ctypes, 'CDLL', side_effect=[OSError('absent'), library]) as load:
            self.assertEqual(real_nvrtc_provider({}), ('libnvrtc.so.12', library))
            self.assertEqual([call.args[0] for call in load.call_args_list],
                             ['libnvrtc.so.13', 'libnvrtc.so.12'])

    def test_selected_provider_version_failure_is_not_hidden(self):
        library = Mock()
        library.nvrtcVersion.return_value = 7
        with patch.object(ctypes, 'CDLL', return_value=library) as load:
            with self.assertRaisesRegex(RuntimeError, 'nvrtcVersion status 7'):
                real_nvrtc_provider({})
            load.assert_called_once_with('libnvrtc.so.13')

    def test_explicit_invalid_pin_does_not_fall_back(self):
        for key in ('TORCH_RS_NVRTC', 'TEST_PROGRAM_IDENTITY_REAL_NVRTC'):
            with self.subTest(key=key), patch.object(ctypes, 'CDLL', side_effect=OSError('invalid pin')) as load:
                with self.assertRaisesRegex(RuntimeError, 'cannot load real NVRTC.*invalid pin'):
                    real_nvrtc_provider({key: '/invalid/nvrtc.so'})
                load.assert_called_once_with('/invalid/nvrtc.so')
        with self.assertRaisesRegex(RuntimeError, 'cannot load real NVRTC'):
            real_nvrtc_provider({'TORCH_RS_NVRTC': str(Path(__file__) / 'missing-nvrtc.so')})


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ProgramIdentityCompilerFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.real_nvrtc, cls.real_library = real_nvrtc_provider(os.environ)
        destination = root / 'target' / 'program-identity-compiler-failures'
        destination.mkdir(parents=True, exist_ok=True)
        cls.library = destination / 'libprogram_identity_nvrtc.so'
        subprocess.run(['cc', '-std=c11', '-shared', '-fPIC', '-O2', '-Wall', '-Wextra', '-Werror',
                        str(root / 'tests' / 'fixtures' / 'program_identity_nvrtc.c'),
                        '-o', str(cls.library), '-ldl'], check=True,
                       env={**os.environ, 'TMPDIR': str(destination)})
        # Keep the test library loaded so counters survive the native loader's
        # success/error drops. The forwarded real NVRTC owner is process-local.
        cls.shim = ctypes.CDLL(str(cls.library))
        cls.shim.test_program_identity_reset.argtypes = [ctypes.c_int]
        cls.shim.test_program_identity_reset.restype = None
        cls.shim.test_program_identity_count.argtypes = [ctypes.c_int]
        cls.shim.test_program_identity_count.restype = ctypes.c_uint

    def tearDown(self):
        native.compiler.reset()

    def snapshot(self, compiled):
        state = cache(compiled)
        return (tuple((key, id(entry), tuple(entry.lowerings.items()),
                       tuple(entry.observations.items()), entry.numerical_hint)
                      for key, entry in state.graphs.items()),
                tuple(state.executors.items()), tuple(state.prepared.items()), state.prepared_bytes)

    def check_failure(self, phase, message, expected_counts):
        fn = program('def f(x):\n return -x')
        compiled = native.compile(fn)
        x = native.tensor([1., -2., 3.]).to('cuda:0')
        retained = compiled(x)
        prepared = next(reversed(cache(compiled).prepared.values()))[0]
        executor = next(reversed(cache(compiled).executors.values()))
        self.assertEqual(executor.kind, 'direct')
        before = self.snapshot(compiled)
        # Changing the original Graph creates a genuinely new executable. No
        # shape-specific fixture or private compilation entry point is used.
        fn.__code__ = program('def f(x):\n return x+x').__code__
        for cold in (False, True):
            current = native.compile(fn) if cold else compiled
            snapshot = self.snapshot(current)
            self.shim.test_program_identity_reset(phase)
            with self.subTest(cold=cold), patch.dict(os.environ, {
                    'TORCH_RS_NVRTC': str(self.library),
                    'TEST_PROGRAM_IDENTITY_REAL_NVRTC': self.real_nvrtc}):
                with self.assertRaisesRegex(RuntimeError, message):
                    current(x)
            self.assertEqual(self.snapshot(current), snapshot)
            # Exactly one selected direct compile was attempted. A backup VM
            # would add version/create calls and expose its opcode switch.
            self.assertEqual(tuple(self.shim.test_program_identity_count(index)
                                   for index in range(7)), expected_counts)
            self.assertEqual(self.snapshot(compiled), before)
            self.assertTrue(prepared.belongs_to(executor))
            self.assertEqual(prepared.run((x,))[0].cpu().tolist(), [-1., 2., -3.])
            self.assertEqual(retained.cpu().tolist(), [-1., 2., -3.])
        # Restore the original graph; retained ordinary execution stays usable
        # even while compiler discovery is deliberately unavailable.
        fn.__code__ = program('def f(x):\n return -x').__code__
        with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/program-identity-nvrtc'):
            self.assertEqual(compiled(x).cpu().tolist(), [-1., 2., -3.])

    def test_success_compiles_only_one_direct_executable(self):
        self.shim.test_program_identity_reset(0)
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.tensor([1., -2., 3.]).to('cuda:0')
        with patch.dict(os.environ, {
                'TORCH_RS_NVRTC': str(self.library),
                'TEST_PROGRAM_IDENTITY_REAL_NVRTC': self.real_nvrtc}):
            self.assertEqual(compiled(x).cpu().tolist(), [-1., 2., -3.])
            self.assertEqual(compiled(x).cpu().tolist(), [-1., 2., -3.])
        self.assertEqual(tuple(self.shim.test_program_identity_count(i) for i in range(7)),
                         (1, 1, 1, 1, 1, 0, 0))
        self.assertEqual(next(iter(cache(compiled).executors.values())).kind, 'direct')

    def test_version_failure_never_creates_a_program(self):
        self.check_failure(1, r'NVRTC status 7', (1, 0, 0, 0, 0, 0, 0))

    def test_create_failure_has_no_program_to_destroy(self):
        self.check_failure(2, r'NVRTC status 2', (1, 1, 0, 0, 0, 0, 0))

    def test_compile_failure_destroys_the_created_program(self):
        self.check_failure(3, r'NVRTC compilation failed \(6\): program identity injected compile failure',
                           (1, 1, 1, 0, 1, 0, 0))

    def test_invalid_real_compiler_ptx_propagates_module_load_failure(self):
        self.check_failure(4, r'cuModuleLoadData.*CUDA_ERROR_INVALID_(?:PTX|IMAGE)', (1, 1, 1, 1, 1, 0, 0))


if __name__ == '__main__':
    unittest.main()
