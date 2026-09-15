"""Input trees: admission is complete; source observation is lazy."""
import unittest
import types
import dataclasses
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend
from tests.test_compile_pointwise_jit import program, cache
from tests.test_compile_pointwise_helpers import no_bodies
from tests import test_compile_pointwise_structured_outputs as output_tests
from tests import test_compile_pointwise_jit as jit_tests


class InputContracts(unittest.TestCase):
    setUp = output_tests.StructuredCache.setUp
    snapshot = output_tests.StructuredCache.snapshot
    def test_flat_large_signature_and_nested_runtime_scalar_bound(self):
        names = [f'a{i}' for i in range(4097)]
        fn = program('def f(x,' + ','.join(names) + '):\n return -x')
        native.compile(fn)(native.ones(1), *([False] * len(names)))
        with self.assertRaisesRegex(NotImplementedError, '4096'):
            frontend.bind_arguments((native.ones(1), *([False] * 4095), []))
        for count in (64, 65):
            fn = program('def f(p):\n return p["x"]+' +
                         '+'.join(f'p["scalars"][{i}]' for i in range(count)))
            compiled = native.compile(fn)
            compiled({'x': native.ones(1), 'scalars': [1.25] * count})
            before = self.snapshot(compiled)
            if count == 65:
                with self.assertRaisesRegex(NotImplementedError, '64 runtime scalar'):
                    compiled({'x': native.ones(1), 'scalars': [2.5] * count})
                self.assertEqual(self.snapshot(compiled), before)
            else:
                compiled({'x': native.ones(1), 'scalars': [2.5] * count})

    def test_nested_selection_unpack_and_current_owner(self):
        helper = program('def f(p):\n a,b=p\n return {"chosen":[a,b]}')
        fn = program('def f(p):\n q=helper(p["pair"])\n x,g=q["chosen"]\n return (x*g,x)', helper=helper)
        compiled = native.compile(fn)
        for gain in (0.5, 1.5):
            x = native.ones(3)
            with no_bodies(fn, helper):
                actual = compiled({'pair': [x, gain], 'ignored': [False]})
            self.assertIs(actual[1], x)
        self.assertEqual(len(cache(compiled).graphs), 2)


class InputAdmission(unittest.TestCase):
    def test_flat_arity_and_structured_edges(self):
        x = native.ones(1)
        tensors, _ = frontend.bind_arguments((x,) + (False,) * 4097)
        self.assertEqual(tensors, (x,))
        frontend.bind_arguments((x, [], {}, ()))
        frontend.bind_arguments(([x] + [False] * 4094,))
        with self.assertRaisesRegex(NotImplementedError, '4096'):
            frontend.bind_arguments(([x] + [False] * 4095,))

    def test_repeated_tensor_occurrences_and_container_tree_boundary(self):
        x = native.ones(1)
        tensors, _ = frontend.bind_arguments(({'a': x, 'b': [x]},))
        self.assertEqual(len(tensors), 2)
        self.assertIs(tensors[0], tensors[1])
        for child in ([], (), {}):
            with self.assertRaisesRegex(NotImplementedError, 'repeated|cycle'):
                frontend.bind_arguments((x, [child, child]))
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(NotImplementedError):
            frontend.bind_arguments((x, cycle))
        with self.assertRaises(NotImplementedError):
            frontend.bind_arguments(([x, x, x],))

    def test_depth_limit_and_callback_free_complete_admission(self):
        x = native.ones(1)
        tree = x
        for _ in range(64):
            tree = [tree]
        frontend.bind_arguments((tree,))
        with self.assertRaisesRegex(NotImplementedError, '64'):
            frontend.bind_arguments(([tree],))
        effects = []
        class Meta(type):
            def __eq__(self, other):
                effects.append('type equality')
                return False
        class Bad(metaclass=Meta):
            def __repr__(self):
                effects.append('repr')
                return 'bad'
            def __hash__(self):
                effects.append('hash')
                return hash('live')
            def __eq__(self, other):
                effects.append('equality')
                return False
        class List(list):
            def __iter__(self):
                effects.append('iter')
                return super().__iter__()
        bad_key = {Bad(): x}
        effects.clear()
        for invalid in (bad_key, List([x]), {'live': x, 'unused': Bad()},
                        {'live': x, 'unused': None}, {'live': x, 'unused': 1}):
            with self.assertRaises(NotImplementedError):
                frontend.bind_arguments((invalid,))
            self.assertEqual(effects, [])


class InputLanguage(unittest.TestCase):
    def lower(self, source, tree, **captures):
        fn = program(source, **captures)
        tensors, parameters = frontend.bind_arguments((tree,))
        parsed = frontend.analyze(fn, 1)
        _, values = frontend.resolve(fn, parsed, parameters)
        metadata = (((3,), (1,), False, 'torch.float32', 'cuda:0'),) * len(tensors)
        with no_bodies(fn, *(v for v in captures.values() if type(v) is types.FunctionType)):
            return frontend.lower(parsed, values, len(tensors), metadata=metadata)

    def check_local_tuple_assignment(self, names, expression):
        tree = [native.ones(3), 2.0, 3.0]
        assignment = ','.join(names) + '=(' + ','.join(f'p[{i}]' for i in range(len(names))) + ')'
        explicit = '\n '.join(f'{name}=p[{i}]' for i, name in enumerate(names))
        expected = self.lower(f'def f(p):\n {explicit}\n return {expression}', tree)
        source = f'def f(p):\n {assignment}\n return {expression}'
        helper = program(source)
        cases = ((source, {}),
                 ('def f(p):\n return helper(p)', {'helper': helper}),
                 (f'def f(p):\n for i in range(2):\n  {assignment}\n return {expression}', {}))
        for source, captures in cases:
            with self.subTest(source=source):
                self.assertEqual(self.lower(source, tree, **captures), expected)
        # Even zero-trip bodies must validate rotations without touching the iterator.
        self.assertEqual(self.lower(f'def f(p):\n for i in range(0):\n  {assignment}\n return -p[0]', tree),
                         self.lower('def f(p):\n return -p[0]', tree))

    def test_two_item_local_tuple_assignment(self):
        self.check_local_tuple_assignment(('a', 'b'), 'a-b')

    def test_three_item_local_tuple_assignment(self):
        self.check_local_tuple_assignment(('a', 'b', 'c'), '(a-b)*c')

    def test_local_selection_and_unpacking_moved_from_negative_contracts(self):
        x = native.ones(3)
        for source in ('def f(x):\n return (x+1,[x][0])',
                       'def f(x):\n pair=(-x,x)\n a,b=pair\n return a',
                       'def f(x):\n for i in range(2):\n  a,b=[x,-x]\n return (a+1,b)',
                       'def f(x):\n for i in range(0):\n  a=[x][0]\n return -x'):
            self.lower(source, x)

    def test_unsupported_selectors_unpacking_and_input_returns(self):
        x = native.ones(3)
        for source in ('def f(p):\n return (-p[0],p)',
                       'def f(p):\n return (-p[0],{"input":p})',
                       'def f(p):\n a,*b=p\n return -a',
                       'def f(p):\n a,b,c=p\n return -a',
                       'def f(p):\n return -p[9]',
                       'def f(p):\n return -p[True]',
                       'def f(p):\n return -p[0:1]',
                       'def f(p):\n return -p[p[1]]',
                       'def f(p):\n p[0]=p[0]+1\n return -p[0]',
                       'def f(p):\n return -p[index]',
                       'def f(p):\n if p[1]:\n  return -p[0]\n return p[0]+1'):
            with self.subTest(source=source), self.assertRaises(NotImplementedError):
                self.lower(source, [x, 0.0], index=0)
        with self.assertRaises(NotImplementedError):
            self.lower('def f(p):\n a,b=p\n return -a', {'a': x, 'b': False})
        with self.assertRaises(NotImplementedError):
            self.lower('def f(p):\n return -p["missing"]', {'x': x})

    def test_shape_provenance_survives_selection_but_not_helpers(self):
        x = native.ones(3)
        self.lower('def f(p):\n x=p[0]\n if x.shape[0]<4:\n  return -x\n return x+1', [x])
        helper = program('def f(p):\n return p')
        self.lower('def f(p):\n x,g=helper(p)\n return x*g', [x, 0.5], helper=helper)
        with self.assertRaises(NotImplementedError):
            self.lower('def f(p):\n x=helper(p)[0]\n if x.shape[0]<4:\n  return -x\n return x+1', [x], helper=helper)
        # Helper-local literals are selectors, not new predicate authority.
        helper = program('def f(p):\n return p[0]')
        self.lower('def f(p):\n return -helper(p)', [x], helper=helper)


class InputTransactions(InputContracts):
    test_nested_selection_unpack_and_current_owner = None
    test_flat_large_signature_and_nested_runtime_scalar_bound = None

    def test_logical_miss_and_hit_new_abi_failures_are_atomic(self):
        for kind in ('compile', 'prepare', 'run', 'reconstruct'):
            for miss in (False, True):
                with self.subTest(stage=kind, logical_miss=miss):
                    compiled = native.compile(program('def f(p):\n return (p["x"]*p["gain"],p["x"])'))
                    compiled({'x': native.ones(3), 'gain': -0.0})
                    compiled({'x': native.ones(5), 'gain': True})
                    before = self.snapshot(compiled)
                    current = {'unused': native.ones(3), 'gain': 0.5 if miss else 0.0,
                               'x': native.ones(3)}
                    original = self.codegen.side_effect
                    def failing(*args):
                        if kind == 'compile':
                            raise RuntimeError('injected')
                        executor = original(*args)
                        def fail(*args):
                            raise RuntimeError('injected')
                        if kind in ('prepare', 'run'):
                            setattr(executor, kind, fail)
                        return executor
                    with patch.object(frontend.ResultSpec, 'reconstruct',
                                      side_effect=RuntimeError('injected')) if kind == 'reconstruct' else patch.object(frontend._native, '_pointwise_compile', failing):
                        with self.assertRaisesRegex(RuntimeError, 'injected'):
                            compiled(current)
                    self.assertEqual(self.snapshot(compiled), before)
                    result = compiled(current)
                    self.assertIs(result[1], current['x'])
                    self.assertEqual(len(cache(compiled).graphs), 3 if miss else 2)

    def test_warm_invalid_trees_and_missing_paths_do_not_publish(self):
        compiled = native.compile(program('def f(p):\n return -p["x"][0]'))
        x = native.ones(3)
        compiled({'x': [x]})
        before = self.snapshot(compiled)
        cycle = []
        cycle.append(cycle)
        for tree in ({'other': x}, {'x': x}, {'x': []}, {'x': [x], 'bad': object()},
                     {'x': [x], 'bad': cycle}, {'x': [x], 'bad': [x, x]}):
            with self.assertRaises(NotImplementedError):
                compiled(tree)
            self.assertEqual(self.snapshot(compiled), before)
        compiled({'x': [x]})
        self.assertEqual(len(cache(compiled).graphs), 1)

    def test_lazy_observation_helpers_inactive_branch_and_zero_trip(self):
        helper = program('def f(p,x):\n return -x')
        fn = program('def f(p):\n unused=p["ignored"]\n return helper(unused,p["x"])', helper=helper)
        compiled = native.compile(fn)
        for ignored in (0.0, 16777217.0, [False], {'new': True}):
            compiled({'x': native.ones(3), 'ignored': ignored})
        self.assertEqual(len(cache(compiled).graphs), 1)
        entry = next(iter(cache(compiled).graphs.values()))
        self.assertNotIn(frontend.BindingSource('parameter', 'p', 0, ('ignored',)), entry.observed)
        for source in (
                'def f(p):\n x=p["x"]\n if x.shape[0]<4:\n  return -x\n return x*p["other"]',
                'def f(p):\n x=p["x"]\n for i in range(0):\n  unused=x*p["other"]\n return -x'):
            compiled = native.compile(program(source))
            compiled({'x': native.ones(3), 'other': 0.0})
            compiled({'x': native.ones(3), 'other': 1.5})
            self.assertEqual(len(cache(compiled).graphs), 2 if "range(0)" in source else 1)

    def test_caches_do_not_retain_input_owners_descriptors_or_unused_keys(self):
        compiled = native.compile(program('def f(p):\n return (p["x"]*p["gain"],p["x"])'))
        inputs = []
        keys = []
        for index in range(5):
            key = 'unique_unused_key_' + str(index)
            x = native.ones(3)
            tree = {key: [native.ones(3)], 'x': x, 'gain': -0.0 if index == 0 else 0.0}
            compiled(tree)
            inputs.extend((tree, tree[key], x, tree[key][0]))
            keys.append(key)
        forbidden = {id(value) for value in inputs + keys}
        self.codegen.reset_mock()
        pending = [cache(compiled).graphs, cache(compiled).executors, cache(compiled).prepared]
        seen = set()
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            self.assertNotIn(id(value), forbidden)
            self.assertIsNot(type(value), frontend.InputTree)
            self.assertIsNot(type(value), native.Tensor)
            if type(value) is dict:
                pending.extend(value.keys())
                pending.extend(value.values())
            elif type(value) in (list, tuple):
                pending.extend(value)
            elif dataclasses.is_dataclass(value) or type(value) is types.SimpleNamespace:
                pending.extend(vars(value).values())
            elif type(value) is types.FunctionType and value.__closure__:
                pending.extend(cell.cell_contents for cell in value.__closure__)
        native.compiler.reset()
        self.assertFalse(cache(compiled).graphs)
        self.assertFalse(cache(compiled).executors)
        self.assertFalse(cache(compiled).prepared)


@unittest.skipUnless(jit_tests.available(), 'requires native CUDA and reference PyTorch CUDA')
class InputHardware(unittest.TestCase):
    setUpClass = classmethod(jit_tests.Hardware.setUpClass.__func__)
    tearDown = jit_tests.Hardware.tearDown
    upload = jit_tests.Hardware.upload
    compare_tree = output_tests.StructuredHardware.compare_tree

    def compare(self, actual, expected):
        jit_tests.Hardware.compare(self, actual, expected, exact=True)
        torch = self.torch
        host = torch.tensor(actual.cpu().tolist(), dtype=torch.float32).reshape(expected.shape)
        ref = expected.cpu()
        mask = ~ref.isnan()
        self.assertTrue(torch.equal(host.view(torch.int32)[mask], ref.view(torch.int32)[mask]))

    def pair(self, source, helper_source=None):
        helpers, refs = {}, {}
        if helper_source:
            helpers['helper'] = program(helper_source)
            refs['helper'] = program(helper_source, self.torch)
        fn = program(source, **helpers)
        ref = program(source, self.torch, **refs)
        return fn, native.compile(fn), self.torch.compile(ref), tuple(helpers.values())

    def check(self, pair, args, refs):
        fn, compiled, reference, helpers = pair
        with no_bodies(fn, *helpers):
            actual = compiled(*args)
        expected = reference(*refs)
        self.compare_tree(actual, expected)
        return actual

    def tensors(self, step, shape=(5,), offset=False):
        import math
        size = math.prod(shape)
        data = [(-1 if (i + step) % 2 else 1) * (i + 1 + step) * 0.25 for i in range(size)]
        if size >= 2:
            data[-2:] = [0.0, -0.0]
        if offset:
            data.insert(0, 99.0)
            return tuple(self.upload(data, (size + 1,), module)[1:].reshape(shape)
                         for module in (native, self.torch))
        return tuple(self.upload(data, shape, module) for module in (native, self.torch))

    def test_dict_signed_zero_abi_churn_and_unequal_current_replacements(self):
        pair = self.pair('def f(p):\n x=p["x"]\n return (x*p["gain"],x)')
        p, rp = {}, {}
        held = []
        # Keep the wrapper and mutate/replace caller dictionaries through history.
        for step, layout in enumerate(('xg', 'uxg', 'gxu', 'gx', 'ugx', 'xg')):
            x, rx = self.tensors(step)
            u, ru = self.tensors(step + 21)
            data = {'x': x, 'g': -0.0 if step == 0 else 0.0, 'u': u}
            rdata = {'x': rx, 'g': data['g'], 'u': ru}
            p.clear(); rp.clear()
            for name in layout:
                key = {'g': 'gain', 'u': 'unused'}.get(name, name)
                p[key], rp[key] = data[name], rdata[name]
            actual = self.check(pair, (p,), (rp,))
            self.assertIs(actual[1], x)
            self.assertTrue(self.torch.equal(self.torch.signbit(self.torch.tensor(actual[0].cpu().tolist())),
                                             self.torch.signbit((rx * -0.0).cpu())))
            held.append((actual[0], actual[0].cpu().tolist()))
            if step == 2:
                p, rp = dict(p), dict(rp)
        self.assertEqual(len(cache(pair[1]).graphs), 1)
        for result, snapshot in held:
            self.assertEqual(result.cpu().tolist(), snapshot)

    def test_positive_source_across_list_tuple_and_length_precision_history(self):
        pair = self.pair('def f(x,p):\n return ((x*0)+p[0])-16777216.0')
        history = ([16777216.0], (16777217.0,), [16777218.0, False],
                   (16777219.0, True), [16777220.0], [16777221.0])
        for step, p in enumerate(history):
            x, rx = self.tensors(step)
            self.check(pair, (x, p), (rx, p))

    def test_negative_selection_normalizes_and_changes_source_with_length(self):
        pair = self.pair('def f(x,p):\n return (x*p[-1],x*p[1])')
        for step, p in enumerate(([False, -0.0], (False, 0.0),
                                  [False, 16777216.0, 16777217.0],
                                  [False, 16777218.0, 16777219.0], [True, 0.5])):
            x, rx = self.tensors(step)
            self.check(pair, (x, p), (rx, p))

    def test_noncommutative_reversed_realization_roles_and_runtime_abi_reorder(self):
        pair = self.pair('def f(p):\n b=p["b"][0]\n a=p["a"][0]\n return b-a')
        for step, (tensor_key, scalar) in enumerate((('a', 16777216.0), ('b', 16777217.0),
                                                    ('b', 16777218.0), ('a', 16777219.0),
                                                    ('a', 16777220.0))):
            x, rx = self.tensors(step)
            scalar_key = 'b' if tensor_key == 'a' else 'a'
            p, rp = {scalar_key: [scalar], tensor_key: [x]}, {scalar_key: [scalar], tensor_key: [rx]}
            self.check(pair, (p,), (rp,))
        pair = self.pair('def f(p):\n return p["x"]*p["a"]-p["b"]')
        for step, order in enumerate(('xab', 'bax', 'ubxa', 'axbu', 'bxa')):
            x, rx = self.tensors(step)
            u, ru = self.tensors(step + 10)
            d = {'x': x, 'a': 1.25 + step, 'b': 0.125 + step, 'u': u}
            rd = dict(d, x=rx, u=ru)
            self.check(pair, ({k: d[k] for k in order},), ({k: rd[k] for k in order},))

    def test_tensor_aliases_distinct_equal_views_offsets_ranks_and_shapes(self):
        pair = self.pair('def f(p):\n a,b=p\n return (b-a,a,b)')
        for step, mode in enumerate(('same', 'equal', 'view', 'unequal', 'same')):
            x, rx = self.tensors(step, offset=True)
            if mode == 'same':
                y, ry = x, rx
            elif mode == 'equal':
                y, ry = self.tensors(step)
            elif mode == 'view':
                y, ry = x[:], rx[:]
            else:
                y, ry = self.tensors(step + 10)
            actual = self.check(pair, ([x, y],), ([rx, ry],))
            self.assertIs(actual[1], x)
            self.assertIs(actual[2], y)
            self.assertEqual(actual[1] is actual[2], mode == 'same')
        pair = self.pair('def f(p):\n x=p["x"]\n return (-x,x,x.shape[-1])')
        for step, shape in enumerate(((3,), (5,), (1, 5), (0,), (1,), (3,))):
            x, rx = self.tensors(step, shape, offset=True)
            actual = self.check(pair, ({'x': x},), ({'x': rx},))
            self.assertIs(actual[1], x)

    def test_helper_local_unpack_loop_shape_branch_and_failed_recovery(self):
        pair = self.pair('def f(p):\n x=p["x"]\n for i in range(2):\n  a,b=helper([x,p["gain"]])\n if x.shape[0]<4:\n  return (a*b,x)\n return (a-b,x)',
                         'def f(p):\n q={"items":p}\n a,b=q["items"]\n return [a,b]')
        for step, size in enumerate((3, 5, 7, 3)):
            x, rx = self.tensors(step, (size,))
            actual = self.check(pair, ({'x': x, 'gain': 0.5 + step},), ({'x': rx, 'gain': 0.5 + step},))
            self.assertIs(actual[1], x)
            before = tuple(cache(pair[1]).graphs.items())
            with self.assertRaises(NotImplementedError):
                pair[1]({'x': x, 'gain': 0.5, 'unused': None})
            self.assertEqual(tuple(cache(pair[1]).graphs.items()), before)

    def test_bool_float_nonfinite_history(self):
        pair = self.pair('def f(p):\n return p["x"]*p["gain"]')
        for step, gain in enumerate((False, True, -0.0, float('nan'), float('inf'), 0.5, 1.5)):
            x, rx = self.tensors(step)
            self.check(pair, ({'x': x, 'gain': gain},), ({'x': rx, 'gain': gain},))

    def test_helper_input_descriptor_round_trip(self):
        pair = self.pair('def f(p):\n x,g=helper(p["pair"])\n return (x*g,x)',
                         'def f(p):\n return p')
        for step, sequence in enumerate((list, tuple, list)):
            x, rx = self.tensors(step)
            actual = self.check(pair, ({'pair': sequence((x, 0.5 + step))},),
                                ({'pair': sequence((rx, 0.5 + step))},))
            self.assertIs(actual[1], x)

    def test_local_tuple_assignments(self):
        for assignment, expression in (('a,b=(p[0],p[1])', 'a-b'),
                                       ('a,b,c=(p[0],p[1],p[2])', '(a-b)*c')):
            helper = f'def f(p):\n {assignment}\n return {expression}'
            for source, helper_source in ((helper, None),
                                          ('def f(p):\n return helper(p)', helper),
                                          (f'def f(p):\n for i in range(2):\n  {assignment}\n return {expression}', None)):
                with self.subTest(source=source, helper=helper_source):
                    pair = self.pair(source, helper_source)
                    for step in range(3):
                        x, rx = self.tensors(step)
                        self.check(pair, ([x, 2.0 + step, 3.0 + step],),
                                   ([rx, 2.0 + step, 3.0 + step],))

    def test_aliased_tensor_to_scalar_keeps_realization_history(self):
        pair = self.pair('def f(p):\n b=p["b"]\n a=p["a"]\n return ((b*0)+a)-16777216.0')
        x, rx = self.tensors(0)
        for a, ra in ((x, rx), (16777217.0, 16777217.0),
                      (16777218.0, 16777218.0), (x[:], rx[:])):
            self.check(pair, ({'a': a, 'b': x},), ({'a': ra, 'b': rx},))
