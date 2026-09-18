"""Real native bridge controls for code identity, separate from public timing."""
import os
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import torch_rs as bridge, _compile_pointwise as frontend
from tests.test_compile_pointwise_jit import available, program, lower


@unittest.skipUnless(available(), 'CUDA unavailable')
class NativeProgramIdentity(unittest.TestCase):
    def tearDown(self):
        native.compiler.reset()

    def test_original_graph_admission_precedes_hint_and_order_parsing(self):
        x = native.tensor([1., 2., 3.]).to('cuda:0')
        incompatible = native.tensor([4., 5.]).to('cuda:0')
        graph = lower(program('def f(x,y):\n return x+y'), 2)
        with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/identity-admission-order'):
            for hint, order in ((-1, (0,)), (3, (0, 0))):
                with self.subTest(hint=hint, order=order):
                    with self.assertRaisesRegex(NotImplementedError, 'incompatible.*broadcast'):
                        bridge._pointwise_host_plan((x, incompatible), graph.nodes,
                                                    graph.outputs, hint, order)
            with self.assertRaises(OverflowError):
                bridge._pointwise_host_plan((x, x), graph.nodes, graph.outputs, -1, (0,))
            with self.assertRaises(ValueError):
                bridge._pointwise_host_plan((x, x), graph.nodes, graph.outputs, 3, (0, 0))
            # Valid checked host preparation also succeeds without discovering
            # the compiler; only its explicit compile operation needs NVRTC.
            host = bridge._pointwise_host_plan((x, x), graph.nodes, graph.outputs, 3, (0,))
            with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                host.compile()

    def test_evicted_preparation_rebuild_needs_no_compiler_and_receipt_is_exact(self):
        compiled = frontend.implementation(program('def f(x):\n return -x'), 2)
        outputs = []
        first_executor = None
        for index, size in enumerate((3, 5, 7, 9, 5)):
            x = native.tensor([float(index + 1)] * size).to('cuda:0')
            if index == 0:
                out, prepared = compiled._torch_rs_pointwise_receipt(x)
                first_executor = next(iter(compiled._torch_rs_pointwise_cache.executors.values()))
            else:
                with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/identity-test-nvrtc'):
                    out, prepared = compiled._torch_rs_pointwise_receipt(x)
            self.assertEqual(prepared.kind, 'direct')
            self.assertTrue(prepared.belongs_to(first_executor))
            self.assertEqual(len(compiled._torch_rs_pointwise_cache.executors), 1)
            self.assertEqual(out.cpu().tolist(), [-float(index + 1)] * size)
            outputs.append(out)
        self.assertEqual(outputs[0].cpu().tolist(), [-1.] * 3)
        self.assertEqual(len({out.data_ptr() for out in outputs}), len(outputs))
        with patch.object(bridge, '_pointwise_host_plan', side_effect=AssertionError('warm construction')), \
                patch.object(frontend, '_prepared_entry_bytes', side_effect=AssertionError('warm accounting')):
            compiled(x)
        native.compiler.reset()
        with patch.dict(os.environ, TORCH_RS_NVRTC='/nonexistent/identity-test-nvrtc'):
            with self.assertRaisesRegex(RuntimeError, 'NVRTC'):
                compiled(x)
        self.assertFalse(compiled._torch_rs_pointwise_cache.executors)
        self.assertEqual(prepared.run((x,))[0].cpu().tolist(), [-5.] * 5)

    def test_native_bind_rejects_distinct_graph_address_and_owner(self):
        x = native.tensor([1., 2., 3.]).to('cuda:0')
        y = native.tensor([4.]).to('cuda:0')
        graph = lower(program('def f(x,y):\n return x+y'), 2)
        host = bridge._pointwise_host_plan((x,x), graph.nodes, graph.outputs, 3, (0,))
        other = bridge._pointwise_host_plan((x,y), graph.nodes, graph.outputs, 3, (0,))
        executor = host.compile()
        replacement = host.compile()
        prepared = executor.bind(host)
        self.assertNotEqual(host.executable_identity, other.executable_identity)
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            executor.bind(other)
        self.assertTrue(prepared.belongs_to(executor))
        self.assertFalse(prepared.belongs_to(replacement))
        for name in ('run', 'prepare'):
            self.assertFalse(hasattr(executor, name))
        for owner, name in ((host, 'executable_identity'), (executor, 'kind'), (prepared, 'retained_bytes')):
            with self.assertRaises((AttributeError, TypeError)):
                setattr(owner, name, None)

    def test_actual_programs_separate_and_same_words_share_across_hints(self):
        x = native.tensor([0.125]*13).to('cuda:0')
        graph = lower(program('def f(x):\n p=x*x\n return (-p,p.sin())'))
        hosts = [bridge._pointwise_host_plan((x,), graph.nodes, graph.outputs, hint, order)
                 for hint in (1,2,13,257,65537) for order in ((0,1),(1,0))]
        identities = {host.executable_identity for host in hosts}
        self.assertGreater(len(identities), 1)
        self.assertLess(len(identities), len(hosts))
        executors = {}
        for host in hosts:
            key = host.executable_identity
            if key not in executors:
                executors[key] = host.compile()
            prepared = executors[key].bind(host)
            self.assertEqual(prepared.executable_identity, key)
            prepared.run((x,))
        a, b = list(executors.values())[:2]
        host = next(host for host in hosts if host.executable_identity == a.executable_identity)
        with self.assertRaisesRegex(RuntimeError, 'identity mismatch'):
            b.bind(host)


if __name__ == '__main__':
    unittest.main()
