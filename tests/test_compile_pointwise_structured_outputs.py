"""Bounded structured native results; portable mocks make no hardware claims."""
from collections import OrderedDict
from contextlib import ExitStack
import dataclasses
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_helpers import no_bodies
from tests import test_compile_pointwise_jit as jit_tests
from tests.test_compile_pointwise_jit import available, cache, program, two_device_reservation


def kernel(compiled):
    # Successful execution moves its executable to the end, including warm
    # revisits and scalar promotions. Capture the dispatched kernel's evidence.
    return next(reversed(cache(compiled).executors.values()))


def shape_lower(fn, shapes=((3,),), arguments=None):
    if arguments is None:
        arguments = tuple(frontend.Value(i) for i in range(len(shapes)))
    parsed = frontend.analyze(fn, len(arguments))
    _, values = frontend.resolve(fn, parsed, arguments)
    metadata = []
    for shape in shapes:
        strides, stride = [], 1
        for size in reversed(shape):
            strides.append(stride)
            stride *= max(size, 1)
        metadata.append((shape, tuple(reversed(strides)), False, 'torch.float32', 'cuda:0'))
    return frontend.lower(parsed, values, len(shapes), metadata=tuple(metadata))


class StructuredAdmission(unittest.TestCase):
    def test_original_ssa_slots_and_topology_are_separate(self):
        left = shape_lower(program('def f(x):\n a=-x\n b=-x\n return {"a":a,"b":[b,a,x]}'))
        right = shape_lower(program('def f(x):\n a=-x\n b=-x\n return (b,a)'))
        self.assertEqual(left.graph, right.graph)
        self.assertEqual(len(left.graph.outputs), 2)
        self.assertEqual(left.graph.outputs, tuple(sorted(left.graph.outputs)))
        self.assertNotEqual(left.result, right.result)
        self.assertEqual(left.result.output_order, (0, 1))
        self.assertEqual(right.result.output_order, (1, 0))
        self.assertEqual(len(shape_lower(program('def f(x):\n a=-x\n return [a,a]')).graph.outputs), 1)

    def test_all_small_constructors_literal_metadata_and_duplicates(self):
        fn = program('def f(x):\n a=-x\n return {"same":x,"z":[a,None,True,7,0.25,"text"],"same":a,"empty":{}}')
        with no_bodies(fn):
            result = shape_lower(fn)
        self.assertEqual(len(result.graph.outputs), 1)
        for spelling in ('(-x,)', '[-x]', '{"v":-x}', '(-x, [], {})'):
            with self.subTest(spelling=spelling):
                shape_lower(program('def f(x):\n return '+spelling))

    def test_noninteger_result_metadata_does_not_require_predicate_provenance(self):
        fn = program('def f(x):\n b=True\n n=-b\n c=0.25\n return (-x,b,n,-c,False,None,"metadata")')
        with no_bodies(fn):
            lowering = shape_lower(fn)
            output = object()
            result = lowering.result.reconstruct((output,), {}, (), ())
        self.assertIs(result[0], output)
        self.assertIs(result[1], True)
        self.assertIs(type(result[2]), int)
        self.assertEqual(result[2], -1)
        self.assertIs(type(result[3]), float)
        self.assertEqual(result[3:], (-0.25, False, None, 'metadata'))

    def test_constant_key_pool_rejected_before_disassembly_without_callbacks(self):
        effects = []
        class Key(str):
            def __repr__(self):
                effects.append('repr')
                return 'key'
            def __hash__(self):
                effects.append('hash')
                return 1
        class Keys(tuple):
            def __iter__(self):
                effects.append('iter')
                return super().__iter__()
            def __repr__(self):
                effects.append('repr tuple')
                return 'keys'
        for invalid in ((Key('k'),), Keys(('k',)), (1,), (('k',),), ('k', object())):
            fn = program('def f(x):\n return -x')
            fn.__code__ = fn.__code__.replace(co_consts=fn.__code__.co_consts+(invalid,))
            with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('disassembled unsafe constants')):
                with self.assertRaises(NotImplementedError):
                    frontend.analyze(fn, 1)
            self.assertEqual(effects, [])

    def test_metadata_provenance_and_excluded_container_language(self):
        for expression in ('(x, None)', '(1, "x")', '(x+1, x.shape)',
                           '(x+1, (x+1).shape[0])', '(x+1, x.shape[0]+1)',
                           '(x+1, [x][0])', '(x+1, tuple([x]))',
                           '{1:x+1}', '{key:x+1}', '(x+1, scale)'):
            fn = program('def f(x, scale):\n return '+expression, key='v')
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                shape_lower(fn, arguments=(frontend.Value(0), 0.5))
        for value in (1.5, True, 7, None, 'captured'):
            fn = program('def f(x):\n return (-x, captured)', captured=value)
            with self.subTest(captured=value), self.assertRaises(NotImplementedError):
                shape_lower(fn)
        helper = program('def f(a):\n return a')
        fn = program('def f(x):\n y=helper(x)\n return (-x, y.shape[0])', helper=helper)
        shape_lower(fn)
        for expression in ('helper(x).shape[0]', 'helper(x.shape[0])'):
            fn = program('def f(x):\n if ' + expression + ' < 4:\n  return -x\n return x+1', helper=helper)
            with self.assertRaises(NotImplementedError):
                shape_lower(fn)
        for source in ('def f(x):\n pair=(-x,x)\n a,b=pair\n return a',
                       'def f(x):\n a=[-x]\n a.append(x)\n return a',
                       'def f(x):\n return [-x,*[x]]'):
            with self.assertRaises(NotImplementedError):
                shape_lower(program(source))

    def test_output_count_depth_and_expanded_loop_edge_budgets(self):
        for count, accepted in ((64, True), (65, False)):
            assignments = ''.join(f' a{i}=-x\n' for i in range(count))
            # Pairwise nesting avoids CPython's unrelated large-literal opcodes.
            def balanced(names):
                if len(names) == 1: return names[0]
                middle = len(names)//2
                return '('+balanced(names[:middle])+','+balanced(names[middle:])+')'
            expression = balanced([f'a{i}' for i in range(count)])
            fn = program('def f(x):\n'+assignments+' return '+expression)
            if accepted:
                self.assertEqual(len(shape_lower(fn).graph.outputs), count)
            else:
                with self.assertRaises(NotImplementedError): shape_lower(fn)
        maximum = program('def f(x):\n a=-x\n for i in range(64):\n  a=[a]\n return a')
        shape_lower(maximum)
        for count in (65, 200):
            fn = program('def f(x):\n a=-x\n'+' a=[a]\n'*count+' return a')
            with self.assertRaises(NotImplementedError): shape_lower(fn)
        # Each assignment allocates an independent 16-edge container, including
        # overwritten containers in a normalized root loop.
        fn = program('def f(x):\n a=-x\n for i in range(260):\n  result=['+','.join(['a']*16)+']\n return result')
        with self.assertRaises(NotImplementedError): shape_lower(fn)

    def test_private_output_roots_and_original_dead_nodes_validate(self):
        nodes = (("input",0,0,0),("neg",0,0,0),("relu",0,0,0))
        for roots in ((), (0,), (3,), (1,1), (2,1), (True,), [1], (1,)*65):
            with self.subTest(roots=roots), self.assertRaises((TypeError,ValueError,OverflowError)):
                bridge._pointwise_source(nodes, roots, 1)
        with self.assertRaises(ValueError):
            bridge._pointwise_source(nodes+(("add",99,0,0),), (1,2), 1)
        source = bridge._pointwise_source(nodes, (1,2), 1)
        self.assertEqual(source.count('__global__'),1)

    def test_helpers_loops_branches_and_immutable_branch_snapshots(self):
        helper = program('def f(a):\n b=-a\n return {"out":[b,a],"literal":None}')
        fn = program('def f(x):\n for i in range(2):\n  result=helper(x)\n return (result,-x,x.shape[-1])', helper=helper)
        with no_bodies(fn, helper): shape_lower(fn)
        fn = program('def f(x):\n a=[-x]\n if x.shape[0]<4:\n  result=(a,a)\n else:\n  a=[x.relu()]\n  result={"other":a}\n return result')
        a = shape_lower(fn, ((3,),))
        b = shape_lower(fn, ((8,),))
        self.assertEqual(a, shape_lower(fn, ((3,),)))
        self.assertNotEqual(a.result, b.result)
        for bad in ('x.sum()', '{1:-x}', '(-x, x.shape)', '(-x, captured)'):
            fn = program('def f(x):\n if x.shape[0]<4:\n  return [-x]\n return '+bad, captured=0.5)
            with self.subTest(inactive=bad), self.assertRaises(NotImplementedError):
                shape_lower(fn, ((3,),))

    def test_shape_axis_metadata_inside_expanded_loops(self):
        for trips in (0, 1, 3):
            fn = program('def f(x):\n result=[-x]\n for i in range(' + str(trips)
                         + '):\n  result={"computed":x+1,"axis":x.shape[-1]}\n return result')
            result = shape_lower(fn)
            self.assertEqual(len(result.graph.outputs), 1)
            for invalid in ('[x][0]', '(x+1).shape[0]', 'x.shape[i]'):
                bad = program('def f(x):\n for i in range(' + str(trips)
                              + '):\n  result=[-x,' + invalid + ']\n return -x')
                with self.subTest(trips=trips, invalid=invalid), self.assertRaises(NotImplementedError):
                    shape_lower(bad)

    def test_helpers_cannot_query_computed_or_local_shapes(self):
        identity = program('def f(a):\n return a')
        query = program('def f(a):\n return a.shape[0]')
        for expression, helper in (('helper(x+1).shape[0]', identity), ('helper(x)', query)):
            fn = program('def f(x):\n return (-x,' + expression + ')', helper=helper)
            with self.assertRaises(NotImplementedError):
                shape_lower(fn)


class StructuredCache(unittest.TestCase):
    """Exercise the real frontend and cache transaction with a mocked launch."""
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.addCleanup(native.compiler.reset)
        original = bridge._compile_trace_tensor_metadata
        def metadata(tensor):
            result = list(original(tensor))
            result[4] = 'cuda:0'
            return tuple(result)
        self.stack.enter_context(patch.object(bridge, '_compile_trace_tensor_metadata', metadata))
        self.stack.enter_context(patch.object(bridge, '_pointwise_validate_inputs', lambda tensors: None))
        self.launches = []
        self.hints = []
        self.orders = []
        def compile_(tensors, nodes, outputs):
            def run(tensors, scalars, numerical_hint, output_order):
                self.hints.append(numerical_hint)
                self.orders.append(output_order)
                result = tuple(object() for _ in outputs)
                self.launches.append(result)
                return result
            return types.SimpleNamespace(run=run)
        self.codegen = self.stack.enter_context(patch.object(bridge, '_pointwise_compile', side_effect=compile_))

    def snapshot(self, compiled):
        state = cache(compiled)
        return ([(key, id(entry), tuple(entry.lowerings.items()), dict(entry.observations), entry.numerical_hint)
                 for key, entry in state.graphs.items()], list(state.executors.items()))

    def test_numerical_hint_follows_successful_specialization_history(self):
        fn = program('def f(x):\n p=x*x\n return (-p,p.sin())')
        compiled = native.compile(fn)
        compiled(native.ones(1))
        before = self.snapshot(compiled)
        with patch.object(frontend.ResultSpec, 'reconstruct', side_effect=MemoryError('result')):
            with self.assertRaises(MemoryError):
                compiled(native.ones(2))
        self.assertEqual(self.snapshot(compiled), before)
        for size in (3, 2, 257):
            compiled(native.ones(size))
        self.assertEqual(self.hints, [1, 2, 3, 3, 3])
        self.assertEqual([entry.numerical_hint for entry in cache(compiled).graphs.values()], [1, 3])
        self.assertEqual(len(cache(compiled).executors), 1)
        native.compiler.reset()
        compiled(native.ones(2))
        self.assertEqual(self.hints[-1], 2)
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_numerical_hint_uses_live_source_after_tensor_abi_rebinding(self):
        fn = program('def f(unused,x):\n return -x')
        compiled = native.compile(fn)
        compiled(False, native.ones(13))
        compiled(native.ones(1), native.ones(13))
        compiled(True, native.ones(13))
        self.assertEqual(self.hints, [13, 13, 13])
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_identity_freshness_current_metadata_and_no_input_retention(self):
        fn = program('def f(x):\n a=-x\n equal=-x\n shared=[a,x,x.shape[-1]]\n return {"a":shared,"b":shared,"equal":[equal],"meta":[None,True,7,0.25,"s",a]}')
        compiled = native.compile(fn)
        held = []
        for size in (3,5,7,3):
            x = native.ones(size)
            with no_bodies(fn): actual = compiled(x)
            self.assertIs(actual['a'], actual['b'])
            self.assertIs(actual['a'][1], x)
            self.assertEqual(actual['a'][2], size)
            self.assertIsNot(actual['a'][0], actual['equal'][0])
            self.assertEqual(actual['meta'][:-1], [None,True,7,0.25,'s'])
            for earlier in held:
                self.assertIsNot(earlier, actual)
                self.assertIsNot(earlier['a'], actual['a'])
                self.assertIsNot(earlier['a'][0], actual['a'][0])
            held.append(actual)
        del actual, earlier
        held.clear()
        self.codegen.reset_mock()  # The spy intentionally records its input arguments.
        gc.collect()
        # CPython 3.14 may borrow LOAD_FAST operands, so getrefcount results
        # depend on bytecode position. Inspect the bounded cache ownership graph
        # directly instead of asserting interpreter-specific reference counts.
        pending = [cache(compiled).graphs, cache(compiled).executors]
        seen = set()
        while pending:
            owner = pending.pop()
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            self.assertIsNot(type(owner), native.Tensor)
            self.assertLess(len(seen), 10000)
            if type(owner) in (dict, OrderedDict, types.MappingProxyType):
                pending.extend(owner.keys())
                pending.extend(owner.values())
            elif type(owner) in (tuple, list):
                pending.extend(owner)
            elif dataclasses.is_dataclass(owner) or type(owner) is types.SimpleNamespace:
                pending.extend(vars(owner).values())
            elif type(owner) is types.FunctionType and owner.__closure__:
                pending.extend(cell.cell_contents for cell in owner.__closure__)


    def test_dict_order_duplicates_literal_types_and_equal_containers(self):
        fn=program('def f(x):\n a=-x\n first=[a]\n equal=[a]\n return {"dup":x,"middle":first,"dup":a,"equal":equal,"empty":{},"literals":[a,None,True,-7,-0.0,"metadata"]}')
        actual=native.compile(fn)(native.ones(3))
        self.assertEqual(list(actual),['dup','middle','equal','empty','literals'])
        self.assertIs(actual['dup'],actual['middle'][0])
        self.assertIs(actual['middle'][0],actual['equal'][0])
        self.assertIsNot(actual['middle'],actual['equal'])
        self.assertEqual(actual['empty'],{})
        metadata=actual['literals'][1:]
        self.assertEqual([type(v) for v in metadata],[type(None),bool,int,float,str])
        self.assertEqual(metadata,[None,True,-7,-0.0,'metadata'])
        self.assertEqual(math.copysign(1,metadata[3]),-1)

    def test_helper_identity_and_deep_shared_loop_dag(self):
        helper=program('def f(a):\n return a')
        fn=program('def f(x):\n a=[-x]\n for i in range(60):\n  a=[a,a]\n return (a,helper(a))',helper=helper)
        compiled=native.compile(fn)
        with no_bodies(fn,helper): actual=compiled(native.ones(3))
        self.assertIs(actual[0],actual[1])
        current=actual[0]
        for _ in range(60):
            self.assertIs(current[0],current[1])
            current=current[0]
        self.assertEqual(len(current),1)
        # The reconstruction graph is linear despite 2**60 references.
        entry=next(iter(cache(compiled).graphs.values()))
        lowering=next(iter(entry.lowerings.values()))
        self.assertLess(len(lowering.result.entries),70)

    def test_warm_capture_admission_and_reconstruction_retry(self):
        fn=program('def f(x):\n a=x*scale\n return [a,x,x.shape[0]]',scale=0.25)
        compiled=native.compile(fn)
        x=native.ones(3)
        compiled(x)
        before=self.snapshot(compiled)
        fn.__globals__['scale']=object()
        with self.assertRaises(NotImplementedError): compiled(x)
        self.assertEqual(self.snapshot(compiled),before)
        fn.__globals__['scale']=0.25
        with patch.object(frontend, 'lower', side_effect=AssertionError('warm reparse')):
            self.assertIs(compiled(x)[1],x)

    def test_same_graph_different_topology_shares_executor(self):
        fn = program('def f(x):\n a=-x\n return (a,x)')
        compiled = native.compile(fn)
        x = native.ones(3)
        first = compiled(x)
        fn.__code__ = program('def f(x):\n a=-x\n return {"changed":[a,x,a]}').__code__
        second = compiled(x)
        self.assertIs(type(first), tuple)
        self.assertIs(type(second), dict)
        self.assertIs(second['changed'][0], second['changed'][2])
        self.assertIs(second['changed'][1], x)
        self.assertEqual(self.codegen.call_count, 1)
        self.assertEqual(len(cache(compiled).executors), 1)
        self.assertEqual(len(cache(compiled).graphs), 2)

    def test_reconstruction_and_launch_failure_preserve_all_cache_orders(self):
        fn = program('def f(x):\n return [-x,x.shape[0]]')
        compiled = native.compile(fn)
        x = native.ones(3)
        original = fn.__code__
        compiled(x)
        fn.__code__ = program('def f(x):\n return [x.relu(),x.shape[0]]').__code__
        compiled(x)
        fn.__code__ = original
        before = self.snapshot(compiled)
        with patch.object(frontend.ResultSpec, 'reconstruct', side_effect=MemoryError('reconstruction')):
            with self.assertRaisesRegex(MemoryError, 'reconstruction'): compiled(x)
            self.assertEqual(self.snapshot(compiled), before)
            cold = native.compile(fn)
            with self.assertRaises(MemoryError): cold(x)
            self.assertEqual(self.snapshot(cold), ([], []))
        executor = next(iter(cache(compiled).executors.values()))
        with patch.object(executor, 'run', side_effect=RuntimeError('launch')):
            with self.assertRaisesRegex(RuntimeError, 'launch'): compiled(x)
        self.assertEqual(self.snapshot(compiled), before)
        self.assertEqual(compiled(x)[1], 3)
        native.compiler.reset()
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class StructuredHardware(unittest.TestCase):
    setUpClass = classmethod(jit_tests.Hardware.setUpClass.__func__)
    tearDown = jit_tests.Hardware.tearDown
    upload = jit_tests.Hardware.upload
    compare = jit_tests.Hardware.compare
    without_replay = jit_tests.Hardware.without_replay

    def compare_tree(self, actual, expected):
        if isinstance(expected, self.torch.Tensor):
            self.compare(actual, expected)
        elif type(expected) in (tuple,list):
            self.assertIs(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for a,e in zip(actual,expected): self.compare_tree(a,e)
        elif type(expected) is dict:
            self.assertIs(type(actual), dict)
            self.assertEqual(list(actual),list(expected))
            for key in expected: self.compare_tree(actual[key],expected[key])
        else:
            self.assertIs(type(actual),type(expected))
            self.assertEqual(actual,expected)

    def retain(self, label, compiled, actual, expected):
        directory = os.environ.get('TORCH_RS_STRUCTURED_EVIDENCE')
        if not directory: return
        root = Path(__file__).resolve().parents[1]
        directory = Path(directory).resolve()
        if root not in directory.parents:
            raise AssertionError('structured evidence must remain inside the current worktree')
        directory.mkdir(parents=True, exist_ok=True)
        def encode(value):
            if type(value) is native.Tensor or isinstance(value,self.torch.Tensor):
                return {'shape':list(value.shape),'values':value.cpu().tolist()}
            if type(value) is dict: return {k:encode(v) for k,v in value.items()}
            if type(value) in (tuple,list): return [encode(v) for v in value]
            return value
        # Unique filenames preserve earlier observations and failures on retries.
        import uuid
        prefix = directory / (label+'-'+uuid.uuid4().hex)
        prefix.with_suffix('.json').write_text(json.dumps({'actual':encode(actual),'expected':encode(expected)},allow_nan=True))
        generated = kernel(compiled)
        prefix.with_suffix('.cu').write_text(generated.source)
        prefix.with_suffix('.ptx').write_text(generated.ptx)
        entry = next(reversed(cache(compiled).graphs.values()))
        lowering = next(reversed(entry.lowerings.values()))
        pending = [(actual, lowering.result.root)]
        scalar_output = False
        while pending:
            value, index = pending.pop()
            kind, payload = lowering.result.entries[index]
            if kind == 'output':
                scalar_output = tuple(value.shape) == ()
                break
            if kind in ('tuple', 'list'):
                pending.extend(zip(value, payload))
            elif kind == 'dict':
                pending.extend((value[key], child) for key, child in payload)
        prefix.with_suffix('.plan').write_text(generated.plan(
            entry.numerical_hint, lowering.result.output_order,
            scalar_output=scalar_output))

    def test_joint_numerics_shared_signed_competing_products_and_order(self):
        bodies = (
            'p=x*y\n return (p,p+x)',
            'p=x*y\n return (p+x,p)',
            'p=x*y\n return (p,-p+x,p-x)',
            'p=x*y\n return (p,p+x,p+y)',
            'p=x*y\n q=y*y\n return {"p":p,"q":q,"out":p-q}',
            'p=x*1.0000001192092896\n return (p,p-y)',
            'p=x*1.137\n q=y*-3.713\n return (q,p,p+q)',
            'p=x*y\n return (p,-p,-p+p)',
            'a=x+y\n b=x+y\n return [b,a,a]',
            'p=x*y\n q=p*-1.0\n return (p,q,q+x)',
            'p=x*y\n q=p*-2.0\n return (q+x,p,q)',
            'p=x*y\n q=p*-1.0\n return (p,q,q+x,q-y)',
            'p=x*y\n q=-2.0*p\n return (q,q+x,p,q-y)',
            'p=x*0.0\n q=p+1.137\n return (p,q,q*y)',
            'p=x*-0.0\n q=p-0.375\n return (q*y,p,q,q+x)',
            'p=x*0\n q=p+1.137\n return (p,q,q*y)',
            'p=x*False\n q=p+1.137\n return (q*y,p,q,q+x)',
        )
        histories = (
            ([1e10,16777216.,1.0000001192092896,1e-38,0.,-0.,2e38,-2e38],
             [1.0000001192092896,-16777215.,-1.,-1e-38,-0.,0.,-2e38,2e38]),
            ([float('inf'),-float('inf'),float('nan'),0.,-0.,1.,-1.,1e-38],
             [0.,float('inf'),1.,-0.,0.,-1.,1.,1e-38]),
        )
        for index, body in enumerate(bodies):
            source='def f(x,y):\n '+body
            fn, ref_fn=program(source), program(source,self.torch)
            compiled, reference=native.compile(fn), self.torch.compile(ref_fn)
            for history,(left,right) in enumerate(histories):
                args=[self.upload(v,(8,)) for v in (left,right)]
                refs=[self.upload(v,(8,),self.torch) for v in (left,right)]
                with self.subTest(body=body,history=history):
                    expected=reference(*refs)
                    actual=self.without_replay(fn,compiled,args)
                    self.retain(f'numerics-{index}-{history}',compiled,actual,expected)
                    self.compare_tree(actual,expected)
                    self.assertEqual(kernel(compiled).ptx.count('.visible .entry'),1)

    def test_generated_nested_trees_and_keys(self):
        rng = random.Random(762931)
        for case in range(8):
            operations = [rng.choice(('x+y','x-y','x*y','-x','y.relu()')) for _ in range(4)]
            statements = ''.join(f' a{i}={expression}\n' for i,expression in enumerate(operations))
            def tree(depth):
                if depth == 0:
                    return rng.choice(('a0','a1','a2','a3','x','y','x.shape[-1]','None','True','7','"leaf"'))
                left,right=tree(depth-1),tree(depth-1)
                kind=rng.randrange(3)
                if kind == 0: return '(a0,'+left+','+right+')'
                if kind == 1: return '[a1,'+left+','+right+']'
                key='key_'+str(rng.randrange(100000))
                return '{'+repr(key)+':'+left+',"other":'+right+',"tensor":a2}'
            # Include a dynamic value at every constructor so constant folding
            # cannot turn the test into an excluded constant-container opcode.
            topology=tree(3)
            source='def f(x,y):\n'+statements+' return (a0,a1,a2,a3,'+topology+')'
            fn,ref_fn=program(source),program(source,self.torch)
            compiled,reference=native.compile(fn),self.torch.compile(ref_fn)
            left=[rng.uniform(-3,3) for _ in range(17)]
            right=[rng.uniform(-3,3) for _ in range(17)]
            args=[self.upload(v,(17,)) for v in (left,right)]
            refs=[self.upload(v,(17,),self.torch) for v in (left,right)]
            with self.subTest(case=case,source=source):
                expected=reference(*refs)
                actual=self.without_replay(fn,compiled,args)
                self.retain(f'generated-{case}',compiled,actual,expected)
                self.compare_tree(actual,expected)

    def test_returned_competing_products_select_contraction_by_live_uses(self):
        # A stored product is observable, but can still fuse if its competitor
        # is also shared or there is no competing product. Keep singleton and
        # reordered returns in the same differential matrix.
        bodies = (
            'p=(-x)*y\n q=x*y\n r=p+q',
            'p=x*(-y)\n q=x*y\n r=p+q',
            'p=(-x)*(-y)\n q=x*y\n r=p-q',
            'p=x*y\n q=x.sin()*y.cos()\n r=p+q',
            'p=x*y\n q=x.sin()*y.cos()\n r=q+p',
            'p=x.sin()*y.cos()\n q=x*y\n r=p-q',
            'p=x*y\n q=x*y\n r=p+x',
            'p=x*y\n q=p*-1.0\n r=q+x*y',
            'p=x*y\n q=p-0.0\n r=q+x.sin()*y.cos()',
            'p=x*y\n q=p*-1.0\n r=q+x.sin()*y.cos()',
            'p=x*y\n q=p*-2.0\n r=q+x.sin()*y.cos()',
            'p=x*y\n q=x.sin()*y.cos()\n r=(p+q)+p',
            'p=x*y\n q=(-x)*y\n r=(p+q)+p',
        )
        returns = ('r', '(p,r)', '(r,p)', '(q,r)', '(p,q,r)', '(r,q,p)', '(p,p,r)')
        left = [1e10, 16777216., 1.0000001192092896, 1e-38, 0., -0.,
                2e38, -2e38, 1., -1., float('inf'), -float('inf'), float('nan')]
        right = [1.0000001192092896, -16777215., -1., -1e-38, -0., 0.,
                 -2e38, 2e38, -1., 1., 0., float('inf'), 1.]
        rng = random.Random(293)
        left += [rng.uniform(-3, 3) for _ in range(257-len(left))]
        right += [rng.uniform(-3, 3) for _ in range(257-len(right))]
        args = [self.upload(values, (257,)) for values in (left, right)]
        refs = [self.upload(values, (257,), self.torch) for values in (left, right)]
        for case, body in enumerate(bodies):
            for order, result in enumerate(returns):
                source = 'def f(x,y):\n ' + body + '\n return ' + result
                fn, ref_fn = program(source), program(source, self.torch)
                compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
                with self.subTest(body=body, result=result):
                    expected = reference(*refs)
                    actual = self.without_replay(fn, compiled, args)
                    self.retain(f'competing-uses-{case}-{order}', compiled, actual, expected)
                    self.compare_tree(actual, expected)
                    self.assertEqual(kernel(compiled).ptx.count('.visible .entry'), 1)

    def test_duplicate_computed_outputs_do_not_bias_competing_products(self):
        bodies = (
            ('p=x*y\n p2=x*y\n q=(-x)*y\n q2=q', True, False),
            ('p=x*y\n p2=p\n q=(-x)*y\n q2=(-x)*y', False, True),
            ('p=x*y\n p2=x*y\n q=(-x)*y\n q2=(-x)*y', True, True),
            ('p=x*y\n p2=(x*1.0)*y\n q=(-x)*y\n q2=q', True, False),
            ('p=x*1.25\n p2=x*1.25\n q=(-x)*1.25\n q2=q', True, False),
        )
        returns = ('(p,p2,q,q2,r,p)', '(r,q2,q,p2,p,p)',
                   '{"left":[p,p2,p],"right":[q,q2],"value":r}')
        histories = ((1e10, 1.0000001192092896), (2e38, -2e38),
                     (1e-38, -1e-38), (1e-38, 1e-38), (0., -0.),
                     (-0., 1.), (float('inf'), 0.), (float('nan'), 1.))
        for case, (body, separate_p, separate_q) in enumerate(bodies):
            for order, result in enumerate(returns):
                source = 'def f(x,y):\n ' + body + '\n r=p+q\n return ' + result
                for size in (1, 13, 257):
                    fn, ref_fn = program(source), program(source, self.torch)
                    compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
                    prior = None
                    for history, values in enumerate(histories):
                        args = [self.upload([v]*size, (size,)) for v in values]
                        refs = [self.upload([v]*size, (size,), self.torch) for v in values]
                        for repeat in range(2):
                            with self.subTest(case=case, order=order, size=size,
                                              history=history, repeat=repeat):
                                expected = reference(*refs)
                                actual = self.without_replay(fn, compiled, args)
                                self.retain(f'duplicate-products-{case}-{order}-{size}-{history}-{repeat}',
                                            compiled, actual, expected)
                                self.compare_tree(actual, expected)
                                if order == 0:
                                    p, p2, q, q2, _, repeated = actual
                                elif order == 1:
                                    _, q2, q, p2, p, repeated = actual
                                else:
                                    p, p2, repeated = actual['left']
                                    q, q2 = actual['right']
                                self.assertIs(p, repeated)
                                for first, second, separate in ((p,p2,separate_p),(q,q2,separate_q)):
                                    if separate:
                                        self.assertIsNot(first, second)
                                        self.assertNotEqual(first.data_ptr(), second.data_ptr())
                                    else:
                                        self.assertIs(first, second)
                                if prior is not None:
                                    old_actual, old_expected, old_p = prior
                                    self.compare_tree(old_actual, old_expected)
                                    self.assertIsNot(p, old_p)
                                    self.assertNotEqual(p.data_ptr(), old_p.data_ptr())
                                prior = actual, expected, p

    def test_shared_sibling_products_and_observable_uses(self):
        bodies = (
            'p=x*y\n q=(-x)*y\n r=(p+q)+(p-q)',
            'p=x*y\n q=(-x)*y\n r=(p-q)+(p+q)',
            'p=x*y\n q=x.sin()*y.cos()\n r=(p+q)+(p-q)',
            'p=x*y\n q=(-x)*y\n r=((p+q)+(p-q))+q',
            'p=x*y\n q=(-x)*y\n r=((p-0.0)+q)+((p-0.0)-q)',
        )
        returns = ('r', '(p,r)', '(r,p)', '(q,r)', '(p,q,r)')
        histories = ((2e38, -2e38), (1e-38, -1e-38), (1e-38, 1e-38),
                     (1e10, 1.0000001192092896))
        for case, body in enumerate(bodies):
            for order, result in enumerate(returns):
                source = 'def f(x,y):\n ' + body + '\n return ' + result
                for size in (1, 13, 257):
                    fn, ref_fn = program(source), program(source, self.torch)
                    compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
                    for history, (left, right) in enumerate(histories):
                        args = [self.upload([v]*size, (size,)) for v in (left, right)]
                        refs = [self.upload([v]*size, (size,), self.torch) for v in (left, right)]
                        for repeat in range(2):
                            with self.subTest(case=case, result=result, size=size,
                                              history=history, repeat=repeat):
                                expected = reference(*refs)
                                actual = self.without_replay(fn, compiled, args)
                                self.retain(f'siblings-{case}-{order}-{size}-{history}-{repeat}',
                                            compiled, actual, expected)
                                self.compare_tree(actual, expected)
                                self.assertEqual(kernel(compiled).ptx.count('.visible .entry'), 1)

    def test_nonlinear_regions(self):
        bodies = (
            'p=x*y\n q=-p\n r=p.sin()',
            'p=x*x\n q=-p\n r=p.sin()',
            'p=x*y\n q=-p\n r=p.cos()',
            'p=x*y\n q=-p\n r=(p+p).sin()',
            'p=x*y\n q=-p\n r=x.sin()',
            'p=x*y\n q=p+x\n r=p.sin()',
            'p=x*y\n q=-p\n r=p.sin()+q',
        )
        histories = ((1e-38, 1e-38), (1e-38, -1e-38),
                     (2e38, -2e38), (1e10, 1.0000001192092896))
        for case, body in enumerate(bodies):
            for order, result in enumerate(('(q,r)', '(r,q)', '(p,q,r)', 'r')):
                source = 'def f(x,y):\n ' + body + '\n return ' + result
                self.torch.compiler.reset()
                fn = program(source)
                compiled = native.compile(fn)
                reference = self.torch.compile(program(source, self.torch))
                for size in (1, 2, 3, 13, 257):
                    for history, (left, right) in enumerate(histories):
                        args = [self.upload([v]*size, (size,)) for v in (left, right)]
                        refs = [self.upload([v]*size, (size,), self.torch) for v in (left, right)]
                        for repeat in range(2):
                            with self.subTest(case=case, result=result, size=size,
                                              history=history, repeat=repeat):
                                expected = reference(*refs)
                                actual = self.without_replay(fn, compiled, args)
                                self.retain(f'regions-{case}-{order}-{size}-{history}-{repeat}',
                                            compiled, actual, expected)
                                self.compare_tree(actual, expected)
                                self.assertEqual(kernel(compiled).ptx.count('.visible .entry'), 1)
                    self.assertEqual(len(cache(compiled).executors), 1)

    def test_nonlinear_single_input_cold_and_persistent_histories(self):
        for order, result in enumerate(('(q,r)', '(r,q)')):
            source = 'def f(x):\n p=x*x\n q=-p\n r=p.sin()\n return ' + result
            # Singleton sequences are fresh-compile controls. Longer sequences
            # keep BOTH ordinary wrappers alive, including the promoted hint.
            for sequence in ((1,), (13,), (257,), (1,2,3,13,257), (13,1,2,3,257)):
                self.torch.compiler.reset()
                fn = program(source)
                compiled = native.compile(fn)
                reference = self.torch.compile(program(source, self.torch))
                for size in sequence:
                    args = (self.upload([1e-38]*size, (size,)),)
                    refs = (self.upload([1e-38]*size, (size,), self.torch),)
                    for repeat in range(2):
                        with self.subTest(order=order, sequence=sequence, size=size, repeat=repeat):
                            actual = self.without_replay(fn, compiled, args)
                            expected = reference(*refs)
                            self.retain(f'history-{order}-{sequence}-{size}-{repeat}',
                                        compiled, actual, expected)
                            self.compare_tree(actual, expected)
                            self.assertEqual(len(cache(compiled).executors), 1)
                            self.assertEqual(kernel(compiled).ptx.count('.visible .entry'), 1)

    def test_private_numerical_hint_admission_never_calls_integer_hooks(self):
        x = self.upload([1e-38]*13, (13,))
        compiled = native.compile(program('def f(x):\n p=x*x\n return (-p,p.sin())'))
        compiled(x)
        generated = kernel(compiled)
        effects = []
        class Integer(int):
            def __index__(self):
                effects.append('index')
                return 1
            def __int__(self):
                effects.append('int')
                return 1
        for invalid in (True, Integer(1), 1.0, -1, 2**64):
            with self.subTest(invalid=type(invalid)), self.assertRaises((TypeError, OverflowError)):
                generated.run((x,), (), invalid)
        self.assertEqual(effects, [])

    def test_private_output_order_admission_preserves_executor(self):
        fn = program('def f(x):\n p=x*x\n return (-p,p.sin())')
        x = self.upload([1e-38]*13, (13,))
        compiled = native.compile(fn)
        expected = compiled(x)
        generated = kernel(compiled)
        effects = []
        class Integer(int):
            def __index__(self):
                effects.append('index')
                return 0
        class Tuple(tuple):
            def __iter__(self):
                effects.append('iter')
                return super().__iter__()
        for invalid in ([0, 1], Tuple((0, 1)), (), (0,), (0, 0),
                        (0, 2), (-1, 0), (True, 1), (Integer(0), 1)):
            with self.subTest(invalid=type(invalid)), self.assertRaises((TypeError, ValueError, OverflowError)):
                generated.run((x,), (), 13, invalid)
        SpoofBoolean = type('bool', (), {
            '__module__': 'numpy',
            '__bool__': lambda self: effects.append('bool') or True,
        })
        with self.assertRaises(TypeError):
            generated.plan(13, (0, 1), scalar_output=SpoofBoolean())
        self.assertEqual(effects, [])
        actual = compiled(x)
        self.assertIs(kernel(compiled), generated)
        for before, after in zip(expected, actual):
            self.assertEqual(before.cpu().tolist(), after.cpu().tolist())

    def test_nonlinear_large_and_zero_read_producers(self):
        # Returned runtime producers connect regions even beyond the small-graph
        # certificate. Zero-read tensors, including scalar-bound expressions,
        # are cheap and must not acquire a tensor-buffer realization boundary.
        bodies = (
            'p=x*y\n q=-p\n r=p.sin()' + '\n r=r.sin()'*8,
            'p=x*0\n q=-p\n r=p.sin()',
            'p=x*0+scale\n q=-p\n r=p.sin()',
        )
        for case, body in enumerate(bodies):
            source = 'def f(x,y,scale):\n ' + body + '\n return (p,q,r)'
            fn = program(source)
            compiled = native.compile(fn)
            reference = self.torch.compile(program(source, self.torch))
            for size in (1, 13, 257):
                for step, (left, right, scale) in enumerate((
                        (1e-38, 1e-38, 0.0), (2e38, -2e38, -0.0),
                        (1.25, -2.5, 0.125))):
                    args = [self.upload([v]*size, (size,)) for v in (left,right)]
                    refs = [self.upload([v]*size, (size,), self.torch) for v in (left,right)]
                    with self.subTest(case=case, size=size, step=step):
                        actual = self.without_replay(fn, compiled, (*args,scale))
                        expected = reference(*refs,scale)
                        self.retain(f'region-controls-{case}-{size}-{step}', compiled, actual, expected)
                        self.compare_tree(actual, expected)
                        self.assertEqual(kernel(compiled).ptx.count('.visible .entry'), 1)

    def test_nonlinear_large_unreturned_producer(self):
        # Returning p too
        # would force vertical fusion and conceal the finite rounding failure.
        source = ('def f(x,y):\n p=x*y\n q=p-x\n'
                  ' r=p.sin().sin().sin().sin().sin().sin().sin()\n return (q,r)')
        for size in (1, 13, 257):
            self.torch.compiler.reset()
            fn = program(source)
            compiled = native.compile(fn)
            reference = self.torch.compile(program(source, self.torch))
            args = tuple(self.upload([v]*size, (size,)) for v in (1e10, 1.0000001192092896))
            refs = tuple(self.upload([v]*size, (size,), self.torch)
                         for v in (1e10, 1.0000001192092896))
            with self.subTest(size=size):
                actual = self.without_replay(fn, compiled, args)
                expected = reference(*refs)
                self.retain(f'large-unreturned-{size}', compiled, actual, expected)
                self.compare_tree(actual, expected)

    def test_large_realization_return_order_reuses_native_executable(self):
        self.check_large_realization_order(duplicate_product=False)

    def test_large_realization_duplicate_product_preserves_return_order(self):
        self.check_large_realization_order(duplicate_product=True)

    def test_returned_product_independent_duplicate_preserves_realization(self):
        self.check_returned_product_realization(reuse_product=False)

    def test_returned_product_reuse_crosses_a_realized_boundary(self):
        self.check_returned_product_realization(reuse_product=True)

    def check_returned_product_realization(self, reuse_product):
        # Independent tensor products retain separate producer identities;
        # spelling x*y twice does not make q a consumer of returned p. The
        # genuinely reused case must instead read p's rounded import when the
        # long chain separates their regions. Check the selected plan below.
        case = 'reuse' if reuse_product else 'independent'
        lines = ['def f(x,y):', ' p=x*y',
                 ' q=p-1e10' if reuse_product else ' q=x*y-1e10', ' r=p']
        for _ in range(64):
            lines.extend([' r=r.sin()'] * 30)
            lines.append(' r=r.sin()+r.cos()')
        lines.append(' r=r.sin()')
        sources = ['\n'.join(lines + [' return '+order])
                   for order in ('(p,r,q)', '(q,r,p)')]
        functions = [program(source) for source in sources]
        lowerings = [shape_lower(fn, ((2,), (2,))) for fn in functions]
        self.assertEqual(lowerings[0].graph, lowerings[1].graph)
        graph = lowerings[0].graph
        self.assertEqual(len(graph.outputs), 3)
        # This fixture creates p, q, then r; original SSA root order is stable.
        product_id, q_id, r_id = graph.outputs
        self.assertEqual(lowerings[0].result.output_order, (0, 2, 1))
        self.assertEqual(lowerings[1].result.output_order, (1, 2, 0))
        original_product = graph.nodes[product_id]
        self.assertEqual(original_product[0], 'mul')
        self.assertEqual(graph.nodes[q_id][0], 'sub')
        self.assertEqual(graph.nodes[r_id][0], 'sin')
        q_product_id = graph.nodes[q_id][1]
        self.assertEqual(graph.nodes[q_product_id], original_product)
        self.assertEqual(graph.nodes.count(original_product), 1 if reuse_product else 2)
        if reuse_product:
            self.assertEqual(q_product_id, product_id)
        else:
            self.assertNotEqual(q_product_id, product_id)
        q_slot = 1
        fn = functions[0]
        compiled = native.compile(fn)
        generated = None
        prior = []
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))
        try:
            for order, source in enumerate(sources):
                self.torch.compiler.reset()
                fn.__code__ = program(source).__code__
                reference = self.torch.compile(program(source, self.torch))
                for history, y in enumerate((1.0000001192092896, 1.000000238418579)):
                    args = tuple(self.upload([v]*2, (2,)) for v in (1e10, y))
                    refs = tuple(self.upload([v]*2, (2,), self.torch) for v in (1e10, y))
                    for repeat in range(2):
                        with self.subTest(case=case, order=order, history=history, repeat=repeat):
                            actual = self.without_replay(fn, compiled, args)
                            expected = reference(*refs)
                            self.retain(f'returned-product-{case}-{order}-{history}-{repeat}',
                                        compiled, actual, expected)
                            self.compare_tree(actual, expected)
                            current = kernel(compiled)
                            if generated is None:
                                generated = current
                            self.assertIs(current, generated)
                            self.assertEqual(len(cache(compiled).executors), 1)
                            self.assertEqual(current.ptx.count('.visible .entry'), 1)
                            if order == history == repeat == 0:
                                # Diagnostic reconstruction for these exact
                                # arguments, not a captured GPU instruction stream.
                                listing = current.plan(
                                    2, lowerings[order].result.output_order,
                                    scalar_output=False)
                                regions = listing.split('// numerical region ')[1:]
                                q_regions = [(i, region) for i, region in enumerate(regions)
                                             if f'out{q_slot}[i] = ' in region]
                                self.assertEqual(len(q_regions), 1)
                                q_index, q_region = q_regions[0]
                                imports = [line for line in q_region.splitlines()
                                           if line.startswith('const float ')
                                           and line.endswith(f' = export{product_id};')]
                                q_store = next(line for line in q_region.splitlines()
                                               if line.startswith(f'out{q_slot}[i] = '))
                                q_value = q_store.partition(' = ')[2].removesuffix(';')
                                if reuse_product:
                                    producers = [i for i, region in enumerate(regions)
                                                 if any(line.startswith(f'export{product_id} = ')
                                                        for line in region.splitlines())]
                                    self.assertEqual(len(producers), 1)
                                    self.assertLess(producers[0], q_index)
                                    self.assertEqual(len(imports), 1)
                                    imported = imports[0].split()[2]
                                    self.assertIn(
                                        f'const float {q_value} = __fsub_rn({imported}, ', q_region)
                                else:
                                    self.assertEqual(imports, [])
                                    self.assertIn(f'const float {q_value} = fmaf(', q_region)
                            for old_actual, old_expected in prior:
                                self.compare_tree(old_actual, old_expected)
                                for old in old_actual:
                                    for new in actual:
                                        self.assertIsNot(new, old)
                            prior.append((actual, expected))
        finally:
            sys.setrecursionlimit(old_limit)

    def check_large_realization_order(self, duplicate_product):
        # Cross the ordinary reference's realized-unit fusion limit. Only the
        # return expression changes; arithmetic SSA and native code stay shared.
        lines = ['def f(x,y):', ' p=x*y',
                 ' q=x*y-x' if duplicate_product else ' q=p-x', ' r=p']
        for _ in range(64):
            lines.extend([' r=r.sin()'] * 30)
            lines.append(' r=r.sin()+r.cos()')
        lines.append(' r=r.sin()')
        sources = ['\n'.join(lines + [' return '+order])
                   for order in ('(q,r)', '(r,q)')]
        functions = [program(source) for source in sources]
        self.assertEqual(shape_lower(functions[0], ((2,), (2,))).graph,
                         shape_lower(functions[1], ((2,), (2,))).graph)
        fn = functions[0]
        compiled = native.compile(fn)
        generated = None
        prior = []
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, 20000))
        try:
            for order, source in enumerate(sources):
                self.torch.compiler.reset()
                fn.__code__ = program(source).__code__
                reference = self.torch.compile(program(source, self.torch))
                for repeat, values in enumerate(((1e10, 1.0000001192092896),
                                                  (2e38, 2.0), (1e-38, 1e-38))):
                    args = tuple(self.upload([v]*2, (2,)) for v in values)
                    refs = tuple(self.upload([v]*2, (2,), self.torch) for v in values)
                    with self.subTest(order=order, repeat=repeat):
                        actual = self.without_replay(fn, compiled, args)
                        expected = reference(*refs)
                        self.retain(f'realization-order-{duplicate_product}-{order}-{repeat}', compiled, actual, expected)
                        self.compare_tree(actual, expected)
                        current = kernel(compiled)
                        if generated is None:
                            generated = current
                        self.assertIs(current, generated)
                        self.assertEqual(len(cache(compiled).executors), 1)
                        self.assertEqual(current.ptx.count('.visible .entry'), 1)
                        for old_actual, old_expected in prior:
                            self.compare_tree(old_actual, old_expected)
                            for old in old_actual:
                                for new in actual:
                                    self.assertIsNot(new, old)
                        prior.append((actual, expected))
        finally:
            sys.setrecursionlimit(old_limit)

    def test_nonlinear_rank_and_nested_observable_order(self):
        for operation, values in (('p-x', (1e10, 1.0000001192092896)),
                                  ('-p', (1e-38, 1e-38))):
            body = ('def f(x,y):\n p=x*y\n q='+operation+'\n'
                    ' r=p.sin().sin().sin().sin().sin().sin().sin()\n')
            returns = (' shared=[q,r,q]\n return {"first":shared,"again":shared,"r":r}',
                       ' shared=[r,q,r]\n return {"first":shared,"again":shared,"q":q}')
            for shape in ((), (1,), (13,), (257,)):
                compiled = None
                generated = None
                for order, result in enumerate(returns):
                    self.torch.compiler.reset()
                    source = body+result
                    if compiled is None:
                        fn = program(source)
                        compiled = native.compile(fn)
                    else:
                        fn.__code__ = program(source).__code__
                    reference = self.torch.compile(program(source, self.torch))
                    size = math.prod(shape)
                    args = tuple(self.upload([v]*size, shape) for v in values)
                    refs = tuple(self.upload([v]*size, shape, self.torch) for v in values)
                    for repeat in range(2):
                        with self.subTest(operation=operation, shape=shape, order=order, repeat=repeat):
                            actual = self.without_replay(fn, compiled, args)
                            expected = reference(*refs)
                            self.retain(f'nested-order-{operation}-{shape}-{order}-{repeat}', compiled, actual, expected)
                            self.compare_tree(actual, expected)
                            self.assertIs(actual['first'], actual['again'])
                            self.assertIs(actual['first'][0], actual['first'][2])
                            if generated is None:
                                generated = kernel(compiled)
                            self.assertIs(kernel(compiled), generated)
                            self.assertEqual(len(cache(compiled).executors), 1)

    def test_maximum_output_and_runtime_scalar_abi(self):
        def balanced(names):
            if len(names) == 1:
                return names[0]
            middle = len(names) // 2
            return '(' + balanced(names[:middle]) + ',' + balanced(names[middle:]) + ')'

        names = [f'a{i}' for i in range(64)]
        source = ('def f(x,' + ','.join(f's{i}' for i in range(64)) + '):\n'
                  + ''.join(f' a{i}=x*s{i}\n' for i in range(64))
                  + ' return ' + balanced(names))
        fn, ref_fn = program(source), program(source, self.torch)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        x = self.upload([0., -0., 1.25, -3.5, 9.], (5,))
        rx = self.upload([0., -0., 1.25, -3.5, 9.], (5,), self.torch)
        for step in (0, 1, 2):
            scalars = tuple((i + step + 1) / 17. for i in range(64))
            actual = self.without_replay(fn, compiled, (x, *scalars))
            expected = reference(rx, *scalars)
            self.retain(f'max-abi-{step}', compiled, actual, expected)
            self.compare_tree(actual, expected)
        generated = kernel(compiled)
        self.assertIn('float* out63', generated.source)
        self.assertIn('float s63', generated.source)
        self.assertEqual(generated.ptx.count('.visible .entry'), 1)

    def test_retained_outputs_shapes_metadata_aliases_and_distinct_allocations(self):
        source='def f(x):\n a=x*1.25\n b=x*1.25\n shared=[a,x,x.shape[-1]]\n return {"one":shared,"again":shared,"equal":b,"literal":(None,False,9,"held",a)}'
        fn,ref_fn=program(source),program(source,self.torch)
        compiled,reference=native.compile(fn),self.torch.compile(ref_fn)
        held=[]
        for size,shift in ((7,0),(7,4),(257,-2),(1,3),(0,0),(7,9)):
            values=[(i+shift)*0.125 for i in range(size)]
            x,rx=self.upload(values,(size,)),self.upload(values,(size,),self.torch)
            actual=self.without_replay(fn,compiled,(x,))
            expected=reference(rx)
            self.retain(f'fresh-{size}-{shift}',compiled,actual,expected)
            self.compare_tree(actual,expected)
            self.assertIs(actual['one'],actual['again'])
            self.assertIs(actual['one'][1],x)
            self.assertIsNot(actual['one'][0],actual['equal'])
            if size:
                self.assertNotEqual(actual['one'][0].data_ptr(),actual['equal'].data_ptr())
            for previous,snapshot in held:
                self.assertEqual(previous['one'][0].cpu().tolist(),snapshot)
                self.assertIsNot(previous['one'],actual['one'])
                self.assertIsNot(previous['one'][0],actual['one'][0])
            held.append((actual,actual['one'][0].cpu().tolist()))

    def test_helper_loop_branch_integration_and_scalar_and_offset_storage(self):
        helper_source='def f(a):\n return {"v":[-a,a],"tag":"helper"}'
        source='def f(x):\n for i in range(2):\n  nested=helper(x)\n if x.shape[-1]<4:\n  return (nested,x*2,x.shape[-1])\n return {"nested":nested,"v":x+2,"axis":x.shape[-1]}'
        helper,ref_helper=program(helper_source),program(helper_source,self.torch)
        fn,ref_fn=program(source,helper=helper),program(source,self.torch,helper=ref_helper)
        compiled,reference=native.compile(fn),self.torch.compile(ref_fn)
        for size in (3,5,9,3,0):
            values=[i*0.25 for i in range(size+1)]
            x=self.upload(values,(size+1,))[1:]
            rx=self.upload(values,(size+1,),self.torch)[1:]
            with no_bodies(fn,helper): actual=compiled(x)
            expected=reference(rx)
            self.retain(f'composition-{size}',compiled,actual,expected)
            self.compare_tree(actual,expected)
        fn,ref_fn=program('def f(x):\n return (x+1,-x,x)'),program('def f(x):\n return (x+1,-x,x)',self.torch)
        self.compare_tree(native.compile(fn)(self.upload([2.],())),self.torch.compile(ref_fn)(self.upload([2.],(),self.torch)))

        source = ('def f(x):\n for i in range(2):\n'
                  '  result={"values":[x+1,-x],"axis":x.shape[0]}\n return result')
        fn, ref_fn = program(source), program(source, self.torch)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        for size in (3, 5, 9):
            values = [i * 0.125 for i in range(size)]
            actual = self.without_replay(fn, compiled, (self.upload(values, (size,)),))
            expected = reference(self.upload(values, (size,), self.torch))
            self.retain(f'loop-metadata-{size}', compiled, actual, expected)
            self.compare_tree(actual, expected)

        helper_source = 'def f(a):\n return a'
        source = ('def f(x):\n alias=helper(x)\n'
                  ' return (-x,alias,alias.shape[0],helper(x.shape[0]))')
        helper, ref_helper = program(helper_source), program(helper_source, self.torch)
        fn, ref_fn = program(source, helper=helper), program(source, self.torch, helper=ref_helper)
        compiled, reference = native.compile(fn), self.torch.compile(ref_fn)
        for size in (3, 5, 9, 3):
            values = [i * 0.125 for i in range(size)]
            x = self.upload(values, (size,))
            with no_bodies(fn, helper):
                actual = compiled(x)
            expected = reference(self.upload(values, (size,), self.torch))
            self.retain(f'helper-metadata-{size}', compiled, actual, expected)
            self.compare_tree(actual, expected)
            self.assertIs(actual[1], x)

    @unittest.skipUnless(two_device_reservation(), 'requires explicit two-device reservation')
    def test_two_device_restoration_and_wrong_device_failure(self):
        torch=self.torch
        if torch.cuda.device_count()<2: self.skipTest('requires two CUDA devices')
        source='def f(x):\n a=x*0.5\n return {"a":a,"b":[-a,x]}'
        fn,ref_fn=program(source),program(source,torch)
        compiled,reference=native.compile(fn),torch.compile(ref_fn)
        inputs=[]
        for target in (0,1):
            x=native.tensor([1.,-2.]).to(f'cuda:{target}')
            rx=torch.tensor([1.,-2.],device=f'cuda:{target}')
            inputs.append(x)
            with torch.cuda.device(1-target):
                actual=compiled(x)
                self.assertEqual(torch.cuda.current_device(),1-target)
                self.compare_tree(actual,reference(rx))
                self.assertIs(actual['b'][1],x)
        selected=next(executor for executor in cache(compiled).executors.values() if executor.device==0)
        with torch.cuda.device(1):
            with self.assertRaises(RuntimeError): selected.run((inputs[1],))
            self.assertEqual(torch.cuda.current_device(),1)
            native.compiler.reset()
            gc.collect()
            self.assertEqual(torch.cuda.current_device(),1)

    def test_structured_native_path_without_installed_pytorch(self):
        source = r"""
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('installed PyTorch forwarding')
sys.meta_path.insert(0, Block())
import torch_rs as m
def helper(a):
    return [-a,a]
def f(x):
    pair=helper(x)
    return {'pair':pair,'again':pair,'axis':x.shape[0]}
compiled=m.compile(f)
codes=(f.__code__,helper.__code__)
def forbid(frame,event,arg):
    if event=='call' and any(frame.f_code is code for code in codes):
        raise AssertionError('Python body replay')
sys.setprofile(forbid)
for values in ([1.,-2.],[3.,4.,5.],[]):
    x=m.tensor(values).to('cuda:0')
    result=compiled(x)
    assert result['pair'] is result['again']
    assert result['pair'][1] is x
    assert result['axis']==len(values)
    assert result['pair'][0].cpu().tolist()==[-v for v in values]
sys.setprofile(None)
assert 'torch' not in sys.modules
print('structured native calls passed without PyTorch import or body replay')
"""
        completed=subprocess.run([sys.executable,'-I','-B','-c',source],capture_output=True,timeout=90)
        # Retain raw child output, including failure output, when evidence is enabled.
        directory=os.environ.get('TORCH_RS_STRUCTURED_EVIDENCE')
        if directory:
            import uuid
            path=Path(directory).resolve()
            self.assertIn(Path(__file__).resolve().parents[1],path.parents)
            path.mkdir(parents=True,exist_ok=True)
            label='no-pytorch-'+uuid.uuid4().hex
            (path/(label+'.stdout')).write_bytes(completed.stdout)
            (path/(label+'.stderr')).write_bytes(completed.stderr)
        self.assertEqual(completed.returncode,0,(completed.stdout+completed.stderr).decode(errors='replace'))

    def test_union_dependencies_broadcast_and_differently_shaped_input_alias(self):
        cases=(
            ('return (-x,-y,x,y)',(2,3),(2,3)),
            ('return (x+y,x-y,x,y)',(2,1),(1,3)),
            ('return (-x,x+x,y)',(2,1),(1,3)),
        )
        for body,left_shape,right_shape in cases:
            source='def f(x,y):\n '+body
            fn,ref_fn=program(source),program(source,self.torch)
            compiled,reference=native.compile(fn),self.torch.compile(ref_fn)
            left=[i*0.5-1 for i in range(math.prod(left_shape))]
            right=[i*0.25+2 for i in range(math.prod(right_shape))]
            args=[self.upload(left,left_shape),self.upload(right,right_shape)]
            refs=[self.upload(left,left_shape,self.torch),self.upload(right,right_shape,self.torch)]
            with self.subTest(body=body):
                actual=self.without_replay(fn,compiled,args)
                expected=reference(*refs)
                self.retain('union-dependencies',compiled,actual,expected)
                self.compare_tree(actual,expected)
                self.assertIs(actual[-1],args[1])

    def test_unequal_computed_shapes_reject_before_publication(self):
        # Each root is individually admitted; equal numel cannot hide unequal shape.
        fn=program('def f(x,y):\n return (-x,-y)')
        compiled=native.compile(fn)
        with self.assertRaisesRegex(NotImplementedError, 'computed outputs require the same actual shape'):
            compiled(self.upload([1.,2.],(1,2)),self.upload([3.,4.],(2,1)))
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)
        # Original numerical limits apply even when one root alone is shallow.
        fn=program('def f(x,y):\n return (x+y,x*y+x*2)')
        compiled=native.compile(fn)
        with self.assertRaisesRegex(NotImplementedError, 'unequal input shapes require at most one arithmetic stage'):
            compiled(self.upload([1.,2.],(2,1)),self.upload([3.,4.],(1,2)))
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)


if __name__ == '__main__':
    unittest.main()
