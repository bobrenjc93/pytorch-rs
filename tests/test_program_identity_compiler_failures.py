"""Real compiler-owner failures, isolated from non-scoring public timings."""
import ctypes
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import torch_rs as native
from tests.test_compile_pointwise_jit import available, cache, program


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class ProgramIdentityCompilerFailures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.real_nvrtc = os.environ.get('TEST_PROGRAM_IDENTITY_REAL_NVRTC',
                                        os.environ.get('TORCH_RS_NVRTC', ''))
        if not cls.real_nvrtc or not Path(cls.real_nvrtc).is_file():
            raise RuntimeError('pin TEST_PROGRAM_IDENTITY_REAL_NVRTC to the actual NVRTC library')
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
