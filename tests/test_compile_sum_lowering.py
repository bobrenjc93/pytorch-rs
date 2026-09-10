"""Constant reduction bytecode contract, runnable on Python 3.10 through 3.14."""
from dataclasses import replace
import os
import unittest

import torch_rs as native

from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace


def program(expression):
    namespace = {}
    exec('def reduction(x):\n    return ' + expression, namespace)
    return namespace['reduction']


class SumLoweringTests(unittest.TestCase):
    def lower(self, expression, device='cuda:0'):
        metadata = trace.CompileTraceTensorMetadata(
            (5, 19), (19, 1), trace.float32, device, False, 0)
        return bytecode.lower_compile_graph(program(expression), (metadata,))

    def test_constant_forms_have_explicit_shape_changing_ir(self):
        for expression, keepdim in (
            ('x.sum(1)', False), ('x.sum(-1)', False),
            ('x.sum(dim=1)', False), ('x.sum(dim=-1, keepdim=False)', False),
            ('x.sum(1, True)', True), ('x.sum(-1, keepdim=True)', True),
            ('x.sum(keepdim=True, dim=1)', True),
        ):
            with self.subTest(expression=expression):
                graph = self.lower(expression)
                node, = graph.operations
                self.assertEqual((node.op, node.target, node.reduction, node.scalar),
                                 ('call_reduction', 'sum', (1, keepdim), None))
                self.assertEqual(node.metadata.shape, (5, 1) if keepdim else (5,))
                self.assertEqual(node.metadata.stride, (1, 1) if keepdim else (1,))
                self.assertEqual(node.metadata.storage_offset, 0)

    def test_local_and_guarded_global_constant_options(self):
        namespace = {'DIM': 1, 'KEEP': True}
        exec('def local(x):\n    dim = -1\n    keep = True\n    return x.sum(dim, keepdim=keep)\n'
             'def global_options(x):\n    return x.sum(dim=DIM, keepdim=KEEP)', namespace)
        meta = self.lower('x.sum(1)').inputs[0].metadata
        for name in ('local', 'global_options'):
            graph = bytecode.lower_compile_graph(namespace[name], (meta,))
            self.assertEqual(graph.operations[0].reduction, (1, True))
        fn = namespace['global_options']
        original = bytecode.prepare_compile_cache_request(fn, (meta,)).key
        namespace['KEEP'] = False
        changed = bytecode.prepare_compile_cache_request(fn, (meta,)).key
        self.assertNotEqual(original, changed)
        self.assertEqual(bytecode.lower_compile_graph(fn, (meta,)).operations[0].metadata.shape, (5,))
        namespace['DIM'] = 0
        with self.assertRaises(NotImplementedError):
            bytecode.lower_compile_graph(fn, (meta,))

    def test_unsupported_options_and_other_keywords(self):
        for expression in ('x.sum()', 'x.sum(0)', 'x.sum(-2)', 'x.sum(True)',
                           'x.sum(1.0)', 'x.sum(None)', 'x.sum((1,))',
                           'x.sum(1, 1)', 'x.sum(1, None)', 'x.sum(x)',
                           'x.sum(1, dim=1)', 'x.sum(1, False, False)',
                           'x.sum(dim=1, dtype=None)', 'x.sum(dim=1, out=x)',
                           'x.sum(**{"dim": 1})', 'x.add(other=x)',
                           'x.neg(out=x)'):
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                self.lower(expression)
        with self.assertRaisesRegex(NotImplementedError, 'CUDA'):
            self.lower('x.sum(1)', device='cpu')

    def test_keyword_helper_call_is_not_enabled(self):
        namespace = {'__name__': __name__}
        exec('def helper(x):\n    return x.sum(1)\ndef root(x):\n    return helper(x=x)', namespace)
        metadata = trace.CompileTraceTensorMetadata((5, 19), (19, 1), trace.float32, 'cuda:0', False, 0)
        with self.assertRaisesRegex(NotImplementedError, 'keyword'):
            bytecode.lower_compile_graph(namespace['root'], (metadata,))

    def test_reduction_metadata_rejects_layout_dtype_and_autograd(self):
        meta = self.lower('x.sum(1)').inputs[0].metadata
        for invalid in (replace(meta, shape=(95,), stride=(1,)),
                        replace(meta, stride=(1, 5)), replace(meta, requires_grad=True),
                        replace(meta, dtype=object()), replace(meta, device='cpu')):
            with self.subTest(metadata=invalid), self.assertRaises(NotImplementedError):
                trace._reduction_output_metadata(invalid, 1, False)

    @unittest.skipUnless(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and native.cuda.is_available(),
                         'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
    def test_native_keyword_calls_on_running_python_version(self):
        # No reference import is needed for these exactly representable values;
        # the separate held-out differential suite checks random float32 inputs.
        for expression, keep in (('x.sum(dim=1, keepdim=False)', False),
                                 ('-x.sum(-1, keepdim=True)', True)):
            compiled = native.compile(program(expression), backend='eager', fullgraph=True)
            for value in (0.25, -0.5):
                x = native.full((8, 29), value).to('cuda:0')
                result = compiled(x)
                expected = -value * 29 if keep else value * 29
                self.assertEqual(result.cpu().tolist(), [[expected]] * 8 if keep else [expected] * 8)
                self.assertEqual(result.stride(), (1, 1) if keep else (1,))
                self.assertEqual(str(result.device), 'cuda:0')
