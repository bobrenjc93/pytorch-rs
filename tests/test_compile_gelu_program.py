"""Combined selected-module failure publication and eager separation controls."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch_rs as native
from tests.test_compile_gelu import no_gelu_replay
from tests.test_compile_pointwise_jit import available, cache, program
from tests import test_program_identity_compiler_failures as failures


@unittest.skipUnless(available(), "requires native CUDA and reference PyTorch CUDA")
class GeluProgram(unittest.TestCase):
    setUpClass = classmethod(failures.ProgramIdentityCompilerFailures.setUpClass.__func__)
    snapshot = failures.ProgramIdentityCompilerFailures.snapshot

    def tearDown(self):
        native.compiler.reset()

    def test_equivalent_gelu_programs_compile_once_across_new_shape_binds(self):
        fn = program("def f(x):\n return fw.nn.functional.gelu(x)")
        compiled = native.compile(fn)
        self.shim.test_program_identity_reset(0)
        retained = []
        with patch.dict(os.environ, {"TORCH_RS_NVRTC": str(self.library),
                                    "TEST_PROGRAM_IDENTITY_REAL_NVRTC": self.real_nvrtc}):
            for size in (3, 5, 13, 3):
                x = native.tensor([.5] * size).to("cuda:0")
                with no_gelu_replay(fn):
                    result = compiled(x)
                retained.append((result, result.cpu().tolist()))
        self.assertEqual(tuple(self.shim.test_program_identity_count(i) for i in range(7)),
                         (1, 1, 1, 1, 1, 0, 0))
        self.assertEqual(len(cache(compiled).executors), 1)
        for result, values in retained:
            self.assertEqual(result.cpu().tolist(), values)

    def test_selected_compile_and_link_failures_publish_nothing(self):
        fn = program("def f(x):\n return fw.nn.functional.gelu(x)")
        x = native.tensor([-.75, -0., .5]).to("cuda:0")
        for phase, message, counts in [
            (1, "NVRTC status 7", (1, 0, 0, 0, 0, 0, 0)),
            (2, "NVRTC status 2", (1, 1, 0, 0, 0, 0, 0)),
            (3, "NVRTC compilation failed", (1, 1, 1, 0, 1, 0, 0)),
            (4, "cuLinkAddData_v2", (1, 1, 1, 1, 1, 0, 0)),
            (5, "cuLinkComplete", (1, 1, 1, 1, 1, 0, 0)),
        ]:
            compiled = native.compile(fn)
            before = self.snapshot(compiled)
            self.shim.test_program_identity_reset(phase)
            with self.subTest(phase=phase), patch.dict(os.environ, {
                    "TORCH_RS_NVRTC": str(self.library),
                    "TEST_PROGRAM_IDENTITY_REAL_NVRTC": self.real_nvrtc}):
                with no_gelu_replay(fn), self.assertRaisesRegex(RuntimeError, message):
                    compiled(x)
            self.assertEqual(before, self.snapshot(compiled))
            self.assertEqual(tuple(self.shim.test_program_identity_count(i) for i in range(7)), counts)
            with no_gelu_replay(fn):
                old = compiled(x)
            executor = next(iter(cache(compiled).executors.values()))
            self.assertEqual(executor.kind, "direct")
            self.assertIn("--relocatable-device-code=true", executor.options)
            snapshot = old.cpu().tolist()
            with patch.dict(os.environ, TORCH_RS_NVRTC="/nonexistent/gelu-nvrtc"):
                with no_gelu_replay(fn):
                    self.assertEqual(compiled(x).cpu().tolist(), snapshot)
            native.compiler.reset()
            self.assertEqual(old.cpu().tolist(), snapshot)

    def test_eager_works_without_nvrtc_and_empty_compiled_gelu_links_vm(self):
        subprocess.run([sys.executable, "-c", """
import torch_rs as native
x = native.tensor([0.]).to('cuda:0')
assert native.nn.functional.gelu(x).cpu().tolist() == [0.]
"""], check=True, env={**os.environ, "TORCH_RS_NVRTC": "/nonexistent/gelu-nvrtc"})
        x = native.tensor([-.75, .5]).to("cuda:0")
        expected = native.nn.functional.gelu(x).cpu().tolist()
        with patch.dict(os.environ, TORCH_RS_NVRTC="/nonexistent/gelu-nvrtc"):
            self.assertEqual(native.nn.functional.gelu(x).cpu().tolist(), expected)
            with self.assertRaisesRegex(RuntimeError, "cannot load NVRTC"):
                native.compile(program("def f(x):\n return fw.nn.functional.gelu(x)"))(x)
        fn = program("def f(x):\n return fw.nn.functional.gelu(x)")
        compiled = native.compile(fn)
        with no_gelu_replay(fn):
            result = compiled(native.tensor([]).to("cuda:0"))
        self.assertEqual(result.shape, (0,))
        executor = next(iter(cache(compiled).executors.values()))
        self.assertEqual(executor.kind, "vm")
        self.assertIn("torch_rs_erf(R(a))", executor.source)
        self.assertIn("--relocatable-device-code=true", executor.options)


if __name__ == "__main__":
    unittest.main()
