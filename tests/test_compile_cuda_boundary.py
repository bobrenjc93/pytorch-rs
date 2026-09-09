"""Ordinary CUDA tensors must not enter the CPU eager compiler's graph cache."""
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_trace, _compiler_state
from tests.test_cuda_add import available


CAPTURE = None


def add_inputs(x, y):
    return x + y


def add_capture(x):
    return x + CAPTURE


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CompileCudaBoundaryTests(unittest.TestCase):
    def compile_with_cache(self, model, fullgraph, dynamic):
        cache = _compiler_state.new_native_eager_compile_cache()
        with patch.object(_compiler_state, "new_native_eager_compile_cache", return_value=cache):
            compiled = native.compile(model, backend="eager", fullgraph=fullgraph,
                                      dynamic=dynamic, recompile_limit=1)
        return compiled, cache

    def test_cuda_inputs_rejected_before_cache_lookup_or_lowering(self):
        cpu = native.tensor([1., 2., 3.])
        cuda = cpu.to("cuda:0")
        for fullgraph in (True, False):
            for dynamic in ((None, False, True) if fullgraph else (None,)):
                for warmed in (False, True):
                    with self.subTest(fullgraph=fullgraph, dynamic=dynamic, warmed=warmed):
                        compiled, cache = self.compile_with_cache(add_inputs, fullgraph, dynamic)
                        if warmed:
                            self.assertEqual(compiled(cpu, cpu).tolist(), [2., 4., 6.])
                        before = dict(cache.graphs)
                        for args in ((cuda, cuda), (cpu, cuda), (cuda, cpu)):
                            with self.assertRaisesRegex(NotImplementedError, "CPU.*CUDA"):
                                compiled(*args)
                            self.assertEqual(cache.graphs, before)
                        self.assertEqual(compiled(cpu, cpu).tolist(), [2., 4., 6.])
                        self.assertEqual(len(cache.graphs), 1)
        # The restriction belongs to tracing; the eager operation remains supported.
        self.assertEqual((cuda + cuda).cpu().tolist(), [2., 4., 6.])

    def test_cuda_capture_rejected_before_caching_fresh_and_cpu_warmed(self):
        cpu = native.tensor([1., 2., 3.])
        cuda = cpu.to("cuda:0")
        for fullgraph in (True, False):
            for warmed in (False, True):
                with self.subTest(fullgraph=fullgraph, warmed=warmed):
                    compiled, cache = self.compile_with_cache(add_capture, fullgraph, None)
                    with patch(__name__ + ".CAPTURE", cpu):
                        if warmed:
                            self.assertEqual(compiled(cpu).tolist(), [2., 4., 6.])
                        before = dict(cache.graphs)
                        with patch(__name__ + ".CAPTURE", cuda):
                            with self.assertRaisesRegex(NotImplementedError, "CPU.*CUDA"):
                                compiled(cpu)
                        self.assertEqual(cache.graphs, before)
                        self.assertEqual(compiled(cpu).tolist(), [2., 4., 6.])
                        self.assertEqual(len(cache.graphs), 1)

    def test_private_trace_entrypoints_reject_cuda(self):
        cpu = native.tensor([1., 2., 3.])
        cuda = cpu.to("cuda:0")
        backend = _compile_trace._native
        calls = [lambda: _compile_trace._metadata_from_native_tensor(cuda)]
        calls.extend(lambda target=target: backend._compile_trace_unary(cuda, target)
                     for target in ("float", "detach", "neg"))
        calls.extend(lambda left=left, right=right: backend._compile_trace_binary(left, right, "add")
                     for left, right in ((cpu, cuda), (cuda, cpu), (cuda, cuda)))
        for index, call in enumerate(calls):
            with self.subTest(entrypoint=index):
                with self.assertRaisesRegex(NotImplementedError, "CPU.*CUDA"):
                    call()


if __name__ == "__main__":
    unittest.main()
