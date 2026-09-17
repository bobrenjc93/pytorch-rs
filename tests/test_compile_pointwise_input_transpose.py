"""Input-rooted default transpose recipes: source-bound portable and GPU checks."""
import concurrent.futures
import gc
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, program
from tests.test_compile_pointwise_helpers import no_bodies
from tests import test_compile_pointwise_structured_outputs as structured_tests
from tests.test_cuda_contiguous import read_bits, write_bits


def lower(source, shapes=((2, 3),), **bindings):
    fn = program(source, **bindings)
    p = frontend.analyze(fn, len(shapes))
    _, values = frontend.resolve(fn, p)
    metadata = tuple((shape, (3, 1) if len(shape) == 2 else (1,) if shape else (),
                      False, 'torch.float32', 'cuda:0', 0) for shape in shapes)
    result = frontend.lower(p, values, len(shapes), metadata=metadata)
    frontend.preflight_views(result, values, metadata)
    return result


class Metadata(unittest.TestCase):
    def test_exact_types_without_callbacks_and_private_export(self):
        self.assertNotIn('_compile_trace_cuda_transpose_metadata', native.__all__)
        effects = []
        class Integer(int):
            def __index__(self): effects.append('index'); return 0
        class Sequence(tuple):
            def __iter__(self): effects.append('iter'); return iter(())
        query = bridge._compile_trace_cuda_transpose_metadata
        valid = [(2, 3), (3, 1), 4, (0, 1)]
        for slot, values in ((0, [Sequence((2, 3)), (True, 3), (Integer(2), 3)]),
                             (1, [(1.0, 1), Sequence((3, 1))]),
                             (2, [False, Integer(0), 0.0]),
                             (3, [[0, 1], Sequence((0, 1)), (True, 1), (Integer(0), 1), (0,), (0.0, 1)])):
            for value in values:
                args = valid.copy(); args[slot] = value
                with self.subTest(slot=slot, value=type(value)), self.assertRaises(TypeError): query(*args)
        self.assertEqual(effects, [])

    def test_layout_axes_offsets_and_overflow(self):
        q = bridge._compile_trace_cuda_transpose_metadata
        for shape, stride, axes, want in (((), (), (-1, 0), ((), ())),
                ((3,), (1,), (0, -1), ((3,), (1,))),
                ((2, 3), (3, 1), (-1, 0), ((3, 2), (1, 3))),
                ((0, 3), (3, 1), (0, 1), ((3, 0), (1, 3))),
                ((2, 3), (0, 7), (0, 1), ((3, 2), (7, 0)))):
            out = q(shape, stride, 11, axes)
            self.assertEqual((tuple(out[0]), tuple(out[1]), out[2]), (*want, 11))
        for args in (((2, 3), (1,), 0, (0, 1)), ((2, 3, 4), (12, 4, 1), 0, (0, 1)),
                     ((2, 3), (3, 1), 0, (0, 2)), ((), (), 0, (0, 1)),
                     ((2,), (1,), -1, (0, 0)), ((2,), (-1,), 0, (0, 0)),
                     ((2**63, 3), (3, 1), 0, (0, 1)),
                     ((2,), (2**64-1,), 1, (0, 0)),
                     ((2,), (1,), 0, (0, 2**64))):
            with self.subTest(args=args), self.assertRaises((ValueError, OverflowError, NotImplementedError, IndexError)):
                q(*args)


class Admission(unittest.TestCase):
    def test_identity_provenance_and_numerical_projection(self):
        plain = lower('def f(x):\n a=x.sin()\n b=x+2\n return (b,a)')
        mixed = lower('def f(x):\n v=x.transpose(0,1)\n a=x.sin()\n w=v.transpose(0,1)\n b=x+2\n return (v,b,x,w,a,v,x.transpose(0,1))')
        self.assertEqual(plain.graph, mixed.graph)
        self.assertEqual(plain.result.output_order, mixed.result.output_order)
        self.assertEqual(mixed.views, (('input', frontend.BindingSource('parameter','x',0), (0, 1)), ('view', 0, (0, 1)), ('input', frontend.BindingSource('parameter','x',0), (0, 1))))
        owners = [object() for _ in mixed.views]
        values = {frontend.BindingSource('parameter', 'x', 0): frontend.Value(0)}
        result = mixed.result.reconstruct(('a', 'b'), values, ('x',), (), owners)
        self.assertIs(result[0], result[5]); self.assertIsNot(result[0], result[6])
        self.assertEqual(result[1], 'b'); self.assertEqual(result[4], 'a')
        self.assertIsNone(lower('def f(x):\n return x.transpose(0,1)').graph)

    def test_constants_helpers_selected_leaves(self):
        helper = program('def f(x,a,b):\n return x.transpose(a,b)')
        for axes in ('0,1', 'left,right'):
            lower('def f(x):\n a=helper(x,'+axes+')\n return [a,a.transpose(1,0),x.shape[0]]',
                  helper=helper, left=0, right=1)
        axis = 1
        def closure(x): return x.transpose(0, axis)
        p = frontend.analyze(closure, 1); _, values = frontend.resolve(closure, p)
        self.assertEqual(frontend.lower(p, values, 1).views[0][2], (0, 1))

    def test_rejections_include_unused_computed_sources_and_consumers(self):
        bodies = ['y=-x; v=y.transpose(0,1); return y',
                  'v=(-x).transpose(0,1); v=x; return -x',
                  'v=x.transpose(0,1); return v+1', 'v=x.transpose(0,1); return v.sin()',
                  'v=x.transpose(0,1); return (v,v.shape[0])',
                  'return x.t()', 'return x.reshape(3,2)', 'return x.contiguous()',
                  'return fw.transpose(x,0,1)', 'return x.transpose(dim0=0,dim1=1)',
                  'return x.transpose(False,1)', 'return x.transpose(0.0,1)',
                  'a=-x; return x.transpose(0,1)', 'return (x,1)',
                  'v=x.transpose(0,1); return x', 'return x.transpose(0,9)']
        for body in bodies:
            with self.subTest(body=body), self.assertRaises((NotImplementedError, IndexError)):
                lower('def f(x):\n '+body)
        for body in (' y=-x\n v=y.transpose(0,1)', ' v=x.transpose(0,9)'):
            for source in ('def f(x):\n for i in range(0):\n '+body.replace('\n','\n ')+'\n return -x',
                           'def f(x):\n if x.shape[0] < 4:\n  return -x\n'+body+'\n return -x'):
                with self.subTest(source=source), self.assertRaises((NotImplementedError, IndexError)):
                    lower(source)


class Transactions(unittest.TestCase):
    snapshot = structured_tests.StructuredCache.snapshot

    def setUp(self):
        structured_tests.StructuredCache.setUp(self)
        self.aliases = self.stack.enter_context(patch.object(bridge, '_compile_trace_cuda_graph',
            side_effect=lambda inputs, nodes: tuple(object() for _ in nodes)))

    def test_pure_no_numerical_cache_and_late_preflight(self):
        for body in ('v=x.transpose(0,1); return [v,v,x]',
                     'y=-x; v=x.transpose(0,1); return (y,v)'):
            fn = program('def f(x):\n '+body)
            compiled = native.compile(fn)
            x = native.ones(2,3)
            actual = compiled(x)
            before = self.snapshot(compiled)
            for seam in ((bridge, '_compile_trace_cuda_graph'), (frontend.ResultSpec, 'reconstruct')):
                with patch.object(*seam, side_effect=MemoryError('injected')):
                    with self.assertRaises(MemoryError): compiled(x)
                    self.assertEqual(self.snapshot(compiled), before)
                    cold = native.compile(fn)
                    with self.assertRaises(MemoryError): cold(x)
                    self.assertEqual(self.snapshot(cold), ([], [], [], 0))
            compiled(x)
            if body.startswith('v='):
                self.assertIs(actual[0], actual[1])
                self.assertFalse(cache(compiled).executors)
                self.assertFalse(cache(compiled).prepared)
                self.assertEqual(cache(compiled).prepared_bytes, 0)
        self.codegen.reset_mock(); self.aliases.reset_mock()
        for tail in ('v=x.transpose(0,9)', 'v=x.transpose(0,1); w=v.transpose(0,9)'):
            compiled = native.compile(program('def f(x):\n y=-x\n '+tail+'\n return y'))
            with self.assertRaises(IndexError): compiled(native.ones(2,3))
            self.assertEqual(self.snapshot(compiled), ([], [], [], 0))
        self.codegen.assert_not_called(); self.aliases.assert_not_called()

    def test_conditional_binding_guard_and_inactive_helper_warm_reuse(self):
        helper = program('def f(x):\n return x.transpose(0,1)')
        fn = program('def f(x):\n if x.shape[0] < 4:\n  return -x\n return helper(x)', helper=helper)
        compiled = native.compile(fn)
        compiled(native.ones(2,3))
        helper.__code__ = program('def f(x):\n return (-x).transpose(0,1)').__code__
        compiled(native.ones(2,3))  # Inactive helper body is not rescanned on a hit.
        with self.assertRaises(NotImplementedError): compiled(native.ones(5,3))
        with patch.object(native.Tensor, 'transpose', lambda *a: None):
            with self.assertRaisesRegex(NotImplementedError, 'patched Tensor'): compiled(native.ones(2,3))
            native.compile(program('def f(x):\n return -x'))(native.ones(2,3))

    def test_inactive_warm_preflight_current_sources_without_helper_rescan(self):
        helper=program('def f(y):\n return y.transpose(0,1)')
        fn=program('def f(x,y):\n if x.shape[0]<4:\n  return -x\n return helper(y)',helper=helper)
        compiled=native.compile(fn); x=native.ones(2,3)
        compiled(x,native.ones(2,3)); self.aliases.assert_not_called()
        before=self.snapshot(compiled); launches=len(self.launches)
        helper.__code__=program('def f(y):\n return (-y).transpose(0,1)').__code__
        with self.assertRaises(NotImplementedError): compiled(x,native.ones(2,3,4))
        self.assertEqual(len(self.launches),launches)
        self.assertEqual(self.snapshot(compiled),before)
        compiled(x,native.ones(4,7))  # Geometry changes; inactive helper stays frozen.
        self.aliases.assert_not_called()
        fn=program('def f(x,y):\n for i in range(0):\n  v=y.transpose(0,1)\n return -x')
        compiled=native.compile(fn); compiled(x,native.ones(2,3))
        with self.assertRaises(NotImplementedError): compiled(x,native.ones(2,3,4))
        self.aliases.assert_not_called()

    def test_older_entry_and_new_abi_failures_preserve_independent_lrus(self):
        fn=program('def f(unused,x):\n return (x.transpose(0,1),-x)')
        original=fn.__code__; compiled=native.compile(fn); x=native.ones(2,3)
        compiled(False,x); compiled(native.ones(2,3),x)
        fn.__code__=program('def f(unused,x):\n return (x.transpose(0,1),x.sin())').__code__
        compiled(False,x)
        fn.__code__=original
        before=self.snapshot(compiled)
        for owner,name in ((bridge,'_compile_trace_cuda_graph'),(frontend.ResultSpec,'reconstruct')):
            with patch.object(owner,name,side_effect=MemoryError('injected')):
                for unused in (False,native.ones(2,3),x):
                    with self.assertRaises(MemoryError): compiled(unused,x)
                    self.assertEqual(self.snapshot(compiled),before)
                with self.assertRaises(MemoryError): compiled(False,native.ones(5,3))
                self.assertEqual(self.snapshot(compiled),before)
        compiled(False,x)
        self.assertNotEqual(self.snapshot(compiled),before)

    def test_runtime_axes_and_new_abi_require_admission(self):
        compiled = native.compile(program('def f(x,axis):\n return x.transpose(axis,1)'))
        for axis in (0,0.0,False):
            with self.assertRaises(NotImplementedError): compiled(native.ones(2,3),axis)
        helper=program('def f(x):\n return x.transpose(0,1)')
        fn=program('def f(unused,x):\n if x.shape[0] < 4:\n  return -x\n return helper(x)',helper=helper)
        compiled=native.compile(fn); compiled(False,native.ones(2,3))
        helper.__code__=program('def f(x):\n return (-x).transpose(0,1)').__code__
        compiled(False,native.ones(2,3))
        with self.assertRaises(NotImplementedError): compiled(native.ones(2,3),native.ones(2,3))
        self.assertEqual(len(cache(compiled).executors),1)

    def test_current_abi_rebinding_and_view_only_observed_input(self):
        fn = program('def f(unused,x):\n return [x.transpose(0,1),-x]')
        compiled = native.compile(fn)
        for unused in (False, native.ones(2,3), True): compiled(unused, native.ones(2,3))
        self.assertEqual(len(cache(compiled).graphs), 1)
        self.assertEqual(len(next(iter(cache(compiled).graphs.values())).lowerings), 2)
        self.assertEqual(self.hints, [6,6,6])
        fn = program('def f(x,y):\n return [-x,y.transpose(0,1)]')
        compiled = native.compile(fn); compiled(native.ones(2,3), native.ones(1,7))
        self.assertEqual(self.hints[-1], 6)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class Hardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def tearDown(self):
        native.compiler.reset()
        self.torch.compiler.reset()

    def upload(self, shape, shift=0, offset=False, framework=native):
        import math
        count = math.prod(shape)
        x = framework.tensor([float(i+shift) * .25 - 2 for i in range(count+int(offset))],
                             dtype=framework.float32).to('cuda:0')
        if offset: x = x[1:]
        return x.reshape(shape)

    def compare(self, actual, expected):
        if isinstance(expected, self.torch.Tensor):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            import numpy as np
            np.testing.assert_allclose(actual.cpu().tolist(), expected.cpu().tolist(), rtol=1e-5, atol=1e-6, equal_nan=True)
        elif isinstance(expected, (tuple,list)):
            self.assertIs(type(actual),type(expected))
            self.assertEqual(len(actual),len(expected))
            for a,b in zip(actual, expected): self.compare(a,b)
        elif type(expected) is dict:
            self.assertIs(type(actual),dict)
            self.assertEqual(list(actual),list(expected))
            for key in expected: self.compare(actual[key],expected[key])
        else: self.assertEqual(actual, expected)

    def test_default_pairs_shape_history_alias_identity_offset_and_lifetime(self):
        histories = (((),), ((1,), (7,), (0,), (7,)),
                     ((2,3), (4,5), (2,3)), ((0,3), (1,3), (0,3)))
        for shapes in histories:
            native.compiler.reset()
            self.torch.compiler.reset()
            axes = '(0,-1)'
            source = 'def f(x):\n v=x.transpose'+axes+'\n w=v.transpose'+axes+'\n return (v,v,x.transpose'+axes+',w,x)'
            fn = program(source); compiled = native.compile(fn)
            reference = self.torch.compile(program(source, self.torch))
            for shape in shapes:
                for shift in (0,7):
                    x = self.upload(shape,shift,shift == 0); r = self.upload(shape,shift,shift == 0, self.torch)
                    with no_bodies(fn): actual = compiled(x)
                    expected = reference(r); self.compare(actual,expected)
                    self.assertIs(actual[0],actual[1]); self.assertIsNot(actual[0],actual[2])
                    self.assertIs(actual[-1],x); self.assertIsNot(actual[3],x)
                    self.assertEqual(actual[0].data_ptr(),x.data_ptr())
                    import numpy as np
                    bits = np.full(x.numel(), 3.5, dtype=np.float32).view(np.uint32)
                    write_bits(x, bits); write_bits(r, bits); self.compare(actual,expected)
                    del x,r; gc.collect(); self.compare(actual,expected)
            self.assertFalse(cache(compiled).executors); self.assertFalse(cache(compiled).prepared)

    def test_raw_bits_equal_axes_unused_views_and_deleted_sources(self):
        import numpy as np
        bits = np.array([0,0x80000000,0x7f800001,0xff800001,0x7fc12345,
                         0xffc54321,0x7f800000,0xff800000,1,0x80000001,
                         0x3f800000,0xbf800000],dtype=np.uint32)
        source = 'def f(x):\n unused=x.transpose(0,1)\n a=x.transpose(0,0)\n b=x.transpose(0,1)\n return (a,b,b.transpose(0,1),a)'
        fn=program(source); compiled=native.compile(fn)
        reference=self.torch.compile(program(source,self.torch))
        x=self.upload((3,4)); r=self.upload((3,4),framework=self.torch)
        write_bits(x,bits); write_bits(r,bits)
        with no_bodies(fn): actual=compiled(x)
        expected=reference(r)
        for a,b in zip(actual,expected):
            np.testing.assert_array_equal(read_bits(a.contiguous()),read_bits(b.contiguous()))
        self.assertIs(actual[0],actual[3]); self.assertIsNot(actual[0],x)
        write_bits(actual[2],bits[::-1].copy()); write_bits(expected[2],bits[::-1].copy())
        pointer=x.data_ptr(); del x,r; gc.collect()
        for a,b in zip(actual,expected):
            self.assertEqual(a.data_ptr(),pointer)
            np.testing.assert_array_equal(read_bits(a.contiguous()),read_bits(b.contiguous()))

    def test_mixed_trig_fma_output_order_and_selected_input_tree(self):
        sources = [
            'def f(x,y):\n v=y.transpose(0,1)\n p=x*y\n return (v,p+x,v,x)',
            'def f(x,y):\n v=y.transpose(0,1)\n p=x.sin()\n q=x.cos()\n return (v,q,p,v,x)',
            'def f(x,y):\n v=y.transpose(0,1)\n return (v,x*0.125+1.5,v,x)',
        ]
        for source in sources:
            fn = program(source); compiled = native.compile(fn)
            reference = self.torch.compile(program(source,self.torch))
            for shape in ((2,3),(4,5),(2,3)):
                for shift in (0,11):
                    args = [self.upload(shape,shift),self.upload(shape,shift+1)]
                    refs = [self.upload(shape,shift,framework=self.torch),self.upload(shape,shift+1,framework=self.torch)]
                    with no_bodies(fn): actual = compiled(*args)
                    self.compare(actual,reference(*refs))
                    for entry in cache(compiled).graphs.values():
                        for lowering in entry.lowerings.values():
                            self.assertNotIn('transpose',[node[0] for node in lowering.graph.nodes])
                            self.assertEqual(lowering.result.output_order, tuple(reversed(range(len(lowering.graph.outputs)))) if 'q,p' in source else (0,))
        fn = program('def f(tree):\n x=tree["x"][0]\n y=tree["y"]\n return (-x,y.transpose(0,1))')
        compiled = native.compile(fn)
        x,y = self.upload((2,3)),self.upload((1,7))
        result = compiled({'x':[x], 'y':y})
        self.assertEqual(tuple(result[1].shape),(7,1))
        self.assertEqual(result[1].data_ptr(),y.data_ptr())

    def test_sensitive_fma_and_captured_scalar_histories(self):
        xv=[3e38,-3e38,2e38,-2e38,1.0000001192092896,-0.,0.,1e-45,-1e-45]
        yv=[-1.5,-1.5,-1.5,-1.5,-0.9999999403953552,0.,-0.,-1.,-1.]
        for expression in ('x*y+x','x+x*y'):
            source='def f(x,y):\n v=y.transpose(0,1)\n return (v,'+expression+',v)'
            fn=program(source); compiled=native.compile(fn)
            reference=self.torch.compile(program(source,self.torch))
            for shape in ((1,9),(3,3)):
                args=[native.tensor(v).reshape(shape).to('cuda:0') for v in (xv,yv)]
                refs=[self.torch.tensor(v,device='cuda:0').reshape(shape) for v in (xv,yv)]
                with no_bodies(fn): actual=compiled(*args)
                self.compare(actual,reference(*refs))
        source='def f(x):\n v=x.transpose(0,1)\n return (v,x*scale,v)'
        fn=program(source,scale=.125); ref_fn=program(source,self.torch,scale=.125)
        compiled=native.compile(fn); reference=self.torch.compile(ref_fn)
        for scale in (.125,1.137,-.375,.125):
            fn.__globals__['scale']=scale; ref_fn.__globals__['scale']=scale
            x=self.upload((3,7)); r=self.upload((3,7),framework=self.torch)
            with no_bodies(fn): actual=compiled(x)
            self.compare(actual,reference(r))
        self.assertTrue(any(any(node[0]=='scalar' for node in lowering.graph.nodes)
                            for entry in cache(compiled).graphs.values() for lowering in entry.lowerings.values()))

    def test_helpers_loops_and_nested_view_results(self):
        helper_source='def f(a,axis):\n v=a.transpose(0,axis)\n return [v,v]'
        helper=program(helper_source); ref_helper=program(helper_source,self.torch)
        source='def f(tree):\n x=tree["x"]\n shared=helper(x,1)\n v=shared[0]\n for i in range(2):\n  v=v.transpose(0,1)\n return {"same":shared,"again":shared,"chain":v,"shape":x.shape[0]}'
        fn=program(source,helper=helper); compiled=native.compile(fn)
        reference=self.torch.compile(program(source,self.torch,helper=ref_helper))
        for shape in ((2,3),(5,7),(2,3)):
            x=self.upload(shape); r=self.upload(shape,framework=self.torch)
            with no_bodies(fn,helper): actual=compiled({'x':x})
            self.compare(actual,reference({'x':r}))
            self.assertIs(actual['same'],actual['again'])
            self.assertIs(actual['same'][0],actual['same'][1])
            self.assertIsNot(actual['chain'],actual['same'][0])
            self.assertEqual(actual['chain'].data_ptr(),x.data_ptr())

    def test_axes_rebinding_and_inactive_geometry_on_real_inputs(self):
        source='def f(x):\n return x.transpose(axis,1)'
        fn=program(source,axis=0); ref_fn=program(source,self.torch,axis=0)
        compiled=native.compile(fn); reference=self.torch.compile(ref_fn)
        x=self.upload((2,3)); r=self.upload((2,3),framework=self.torch)
        for axis in (0,1,-2,0):
            fn.__globals__['axis']=axis; ref_fn.__globals__['axis']=axis
            self.compare(compiled(x),reference(r))
        helper=program('def f(y):\n return y.transpose(0,1)')
        source='def f(x,y):\n if x.shape[0]<4:\n  return -x\n return helper(y)'
        compiled=native.compile(program(source,helper=helper))
        with patch.object(bridge,'_compile_trace_cuda_graph',side_effect=AssertionError('inactive alias executed')):
            compiled(x,self.upload((2,3)))
            helper.__code__=program('def f(y):\n return (-y).transpose(0,1)').__code__
            with self.assertRaises(NotImplementedError): compiled(x,self.upload((2,3,4)))
            self.compare(compiled(x,self.upload((4,7))),-r)

    def test_deferred_view_continuation_across_both_arms(self):
        source='def f(x):\n if x.shape[0] < 4:\n  v=x.transpose(0,1)\n else:\n  return x.transpose(0,1)\n return v.transpose(0,1)'
        fn=program(source); compiled=native.compile(fn)
        reference=self.torch.compile(program(source,self.torch))
        for shape in ((2,3),(5,3),(2,3),(7,3)):
            x=self.upload(shape); r=self.upload(shape,framework=self.torch)
            with no_bodies(fn): actual=compiled(x)
            self.compare(actual,reference(r))
            self.assertEqual(actual.data_ptr(),x.data_ptr())

    def test_rebinding_cross_thread_reset_and_invalid_inputs(self):
        compiled = native.compile(program('def f(unused,x):\n return x.transpose(0,1)'))
        x = self.upload((2,3))
        for unused in (False,self.upload((1,7)),True):
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                result = pool.submit(compiled,unused,x).result()
            self.assertEqual(result.data_ptr(),x.data_ptr())
        for bad in (native.ones(2,3), self.upload((2,3)).transpose(0,1),
                    self.torch.ones(2,3,dtype=self.torch.float64,device='cuda:0'),
                    native.ones(2,3,requires_grad=True)):
            with self.assertRaises((NotImplementedError, ValueError)): compiled(bad,x)
        native.compiler.reset(); self.assertFalse(cache(compiled).graphs)
        self.assertEqual(compiled(False,x).data_ptr(),x.data_ptr())

    def test_native_only_without_reference_import(self):
        code = '''
import sys
class Block:
 def find_spec(self, name, *args):
  if name == 'torch' or name.startswith('torch.'):
   raise AssertionError('reference import')
sys.meta_path.insert(0, Block())
import torch_rs as t
def f(x):
 v=x.transpose(0,1)
 return (v,v,x.transpose(0,1),-x)
x=t.ones(2,3).to('cuda:0')
compiled=t.compile(f)
y=compiled(x)
assert y[0] is y[1] and y[0] is not y[2]
assert y[0].data_ptr()==x.data_ptr()
assert 'torch' not in sys.modules
print('native-only transpose passed')
print('nvrtc_version',next(iter(compiled._torch_rs_pointwise_cache.executors.values())).nvrtc_version)
from pathlib import Path
print(sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if 'libcudart.' in line or 'libnvrtc.' in line or 'libcuda.' in line}))
'''
        result = subprocess.run([sys.executable,'-c',code],text=True,capture_output=True)
        directory = os.environ.get('BURNER_EVALUATION_ARTIFACT_DIR')
        if directory: Path(directory,'transpose-native-only.log').write_text(result.stdout+result.stderr)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
