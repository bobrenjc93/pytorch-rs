"""Bounded public CUDA scalar add_: binding, shared storage and completion."""
import ctypes
import gc
import inspect
from pathlib import Path
import subprocess
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch_rs as native
from tests.test_cuda_add import available, runtime, torch, upload
from tests.test_cuda_contiguous import metadata, read_bits, write_bits
from tests.test_compile_pointwise_jit import cache, program


class AddInplaceBindingTests(unittest.TestCase):
    def test_real_descriptor_and_modern_only_binding(self):
        x = native.tensor([1.0])
        descriptor = inspect.getattr_static(native.Tensor, 'add_')
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(x.add_), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, 'add_')
        self.assertNotIn('add_', native.Tensor.__dict__)
        with self.assertRaises(TypeError):
            descriptor(1, 2)
        for call in (lambda: x.add_(), lambda: x.add_(1, 2),
                     lambda: x.add_(1, other=2), lambda: x.add_(x2=1),
                     lambda: x.add_(1, x2=2), lambda: x.add_(1, out=x),
                     lambda: x.add_(1, alpha=1, device='cuda:0')):
            with self.subTest(call=call), self.assertRaises(TypeError):
                call()
        self.assertEqual(x.tolist(), [1.0])

    def test_cpu_grad_saved_alias_and_meta_rejection(self):
        for requires_grad in (False, True):
            x = native.tensor([1.0, -2.0], requires_grad=requires_grad)
            saved = x * x
            for view in (x, x.detach(), x[:1]):
                for call in (lambda: view.add_(2), lambda: view.add_(other=2, alpha=1.0)):
                    with self.assertRaises(NotImplementedError):
                        call()
            self.assertEqual(x.tolist(), [1.0, -2.0])
            self.assertEqual(saved.tolist(), [1.0, 4.0])
            if requires_grad:
                saved.sum().backward()
                self.assertEqual(x.grad.tolist(), [2.0, -4.0])
        # Meta receivers cannot be constructed in the native storage surface.
        with self.assertRaisesRegex(RuntimeError, "device 'meta' is not supported"):
            native.empty((2,), device='meta').add_(1)

    def test_alpha_decided_before_float32_narrowing_and_no_coercion(self):
        x = native.tensor([1.0])
        for alpha in (0, 2, -1, 1 + 2**-40, np.float64(1 - 2**-40),
                      float('nan'), float('inf'), np.bool_(False)):
            with self.subTest(alpha=alpha), self.assertRaisesRegex(NotImplementedError, 'alpha'):
                x.add_(1, alpha=alpha)
        for alpha in (False, True):
            with self.assertRaisesRegex(RuntimeError, 'Boolean alpha'):
                x.add_(1, alpha=alpha)
        effects = []
        class Coerce:
            def __float__(self):
                effects.append('float')
                return 1.0
            def __index__(self):
                effects.append('index')
                return 1
        for call in (lambda: x.add_(Coerce()), lambda: x.add_(1, alpha=Coerce())):
            with self.assertRaises(TypeError):
                call()
        self.assertEqual(effects, [])
        for other in (2**64, -(2**63)-1):
            with self.assertRaises(OverflowError):
                x.add_(other)
        with self.assertRaises(TypeError):
            x.add_(np.uint64(2**63))
        self.assertEqual(x.tolist(), [1.0])

    def test_modes_and_ordered_notimplemented_receive_add_descriptor(self):
        from torch_rs.overrides import TorchFunctionMode
        x = native.tensor([1.0])
        descriptor = native.Tensor.add_
        calls = []
        marker = object()
        class Mode(TorchFunctionMode):
            def __torch_function__(self, fn, dispatch_types, args=(), kwargs=None):
                calls.append(('mode', fn, dispatch_types, args, kwargs))
                return NotImplemented
        class First:
            @classmethod
            def __torch_function__(cls, fn, dispatch_types, args=(), kwargs=None):
                calls.append(('first', fn, dispatch_types, args, kwargs))
                return NotImplemented
        class Second(First):
            @classmethod
            def __torch_function__(cls, fn, dispatch_types, args=(), kwargs=None):
                calls.append(('second', fn, dispatch_types, args, kwargs))
                return marker
        other, alpha = First(), Second()
        with Mode():
            result = x.add_(other=other, alpha=alpha)
        self.assertIs(result, marker)
        self.assertEqual([entry[0] for entry in calls], ['mode', 'second'])
        for _, fn, dispatch_types, args, kwargs in calls:
            self.assertIs(fn, descriptor)
            self.assertEqual(dispatch_types, (Second, First))
            self.assertIs(args[0], x)
            self.assertIs(kwargs['other'], other)
            self.assertIs(kwargs['alpha'], alpha)
        calls.clear()
        with Mode(), self.assertRaisesRegex(TypeError, 'torch.Tensor.add_'):
            x.add_(First())
        self.assertEqual([entry[0] for entry in calls], ['mode', 'first'])
        self.assertEqual(x.tolist(), [1.0])

    def test_invalid_binding_precedes_override_callbacks(self):
        events = []
        class Override:
            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                events.append('override')
                return object()
        x = native.tensor([1.0])
        for call in (lambda: x.add_(Override(), 1), lambda: x.add_(x2=Override()),
                     lambda: x.add_(Override(), other=1)):
            with self.assertRaises(TypeError):
                call()
        self.assertEqual(events, [])


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CudaAddInplaceTests(unittest.TestCase):
    def assert_current_device_zero(self):
        ordinal = ctypes.c_int()
        self.assertEqual(runtime().cudaGetDevice(ctypes.byref(ordinal)), 0)
        self.assertEqual(ordinal.value, 0)

    def compare_storage(self, actual, expected):
        a, b = read_bits(actual), read_bits(expected)
        mask = np.isnan(b.view(np.float32))
        np.testing.assert_array_equal(np.isnan(a.view(np.float32)), mask)
        np.testing.assert_array_equal(a[~mask], b[~mask])

    def test_dense_rank_permutations_offset_sentinels_and_exact_identity(self):
        rng = np.random.default_rng(936401)
        shapes = [(), (1,), (257,), (2, 3), (2, 3, 4), (2, 1, 3, 1, 4)]
        shapes += [tuple(map(int, rng.integers(1, 5, size=rank))) for rank in range(3, 7)]
        for shape in shapes:
            n = int(np.prod(shape, dtype=np.int64))
            values = rng.normal(size=n + 13).astype(np.float32)
            a, b = upload(native, values, values.shape), upload(torch, values, values.shape)
            orders = [tuple(range(len(shape))), tuple(reversed(range(len(shape))))]
            if shape:
                orders.append(tuple(map(int, rng.permutation(len(shape)))))
            for order in orders:
                x, tx = a[5:5+n].reshape(shape).permute(order), b[5:5+n].reshape(shape).permute(order)
                alias = x.detach()
                identity = metadata(x), x.data_ptr()
                self.assertEqual(metadata(x), metadata(tx))
                self.assertTrue(alias.is_set_to(x))
                for scalar in (-1.375, 0.0, 2.718281828):
                    with self.subTest(shape=shape, order=order, scalar=scalar):
                        self.assertIs(x.add_(other=scalar, alpha=1), x)
                        tx.add_(other=scalar, alpha=1)
                        self.assertEqual((metadata(x), x.data_ptr()), identity)
                        self.assertEqual(alias.data_ptr(), x.data_ptr())
                        self.assertTrue(alias.is_set_to(x))
                        self.compare_storage(a, b)
            np.testing.assert_array_equal(read_bits(a)[:5], values[:5].view(np.uint32))
            np.testing.assert_array_equal(read_bits(a)[5+n:], values[5+n:].view(np.uint32))

    def test_scalar_types_and_alpha_one_reference(self):
        class Int(int):
            pass
        class Float(float):
            pass
        scalars = (True, False, -7, 2**63, 2**64-1, -(2**63), Int(3), Float(0.125),
                   np.bool_(True), np.int64(-3), np.uint32(7), np.float32(-0.), np.float64(1.1))
        for scalar in scalars:
            for alpha in (1, 1.0, Int(1), Float(1), np.int64(1), np.float32(1), np.bool_(True)):
                with self.subTest(scalar=repr(scalar), alpha=repr(alpha)):
                    a = native.tensor([1.25, -2.0, 0.0, -0.0]).to('cuda:0')
                    b = torch.tensor([1.25, -2.0, 0.0, -0.0], device='cuda:0')
                    self.assertIs(native.Tensor.add_(a, scalar, alpha=alpha), a)
                    b.add_(scalar, alpha=alpha)
                    self.compare_storage(a, b)

    def test_ieee_zero_subnormal_nan_rounding_and_no_zero_shortcut(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                         0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
                         0x7f800000, 0xff800000, 0x7fc12345, 0x7f800001,
                         0xff800001, 0x3f800001, 0x4b800000, 0xcb800000], dtype=np.uint32)
        for scalar in (0.0, -0.0, 1e-45, -1e-45, 1.0, -1.0, 2**-24, 1e40,
                       float('inf'), -float('inf'), float('nan')):
            with self.subTest(scalar=scalar):
                a = native.zeros((len(bits),), device='cuda:0')
                b = torch.zeros((len(bits),), device='cuda:0')
                write_bits(a, bits); write_bits(b, bits)
                a.add_(scalar); b.add_(scalar)
                self.compare_storage(a, b)

    def test_grid_stride_tail_and_repeated_calls(self):
        # More than 65535 blocks of 256 values exercises another grid-stride pass.
        n = 17_000_003
        values = np.random.default_rng(6692).uniform(-3, 5, n).astype(np.float32)
        a, b = upload(native, values, values.shape), upload(torch, values, values.shape)
        pointer = a.data_ptr()
        for scalar in (0.375, -0.625):
            self.assertIs(a.add_(scalar), a)
            b.add_(scalar)
            self.assertEqual(a.data_ptr(), pointer)
            self.compare_storage(a, b)

    def test_empty_views_out_of_allocation_offsets_no_launch(self):
        a, b = native.ones((3, 5)).to('cuda:0'), torch.ones((3, 5), device='cuda:0')
        for view in (lambda x: x[3:3][:, 5:5], lambda x: x[:, 5:5].t(),
                     lambda x: x[3:3].reshape(2, 0, 3).permute(2, 0, 1)):
            x, tx = view(a), view(b)
            before = metadata(x)
            for scalar in (1.25, float('nan'), -0.0):
                self.assertIs(x.add_(scalar), x)
                self.assert_current_device_zero()
                tx.add_(scalar)
                self.assertEqual(metadata(x), before)
                self.assertEqual(metadata(x), metadata(tx))
            self.compare_storage(a, b)
        self.assertGreater(a[3:3][:, 5:5].storage_offset(), a.numel())
        for shape in ((0,), (2, 0, 3), (0, 2**40)):
            x = native.empty(shape).to('cuda:0')
            self.assertIs(x.add_(2.0), x)
            self.assert_current_device_zero()

    def test_invalid_calls_preserve_bits_in_all_aliases(self):
        bits = np.array([0x80000000, 0x7f800001, 1, 0x3f800000] * 6, dtype=np.uint32)
        base = native.zeros((4, 6), device='cuda:0')
        write_bits(base, bits)
        sibling = base.detach()
        failures = (
            lambda: base.add_(base), lambda: base.add_(native.tensor(1.0)),
            lambda: base.add_(2, alpha=True), lambda: base.add_(2, alpha=False),
            lambda: base.add_(2, alpha=1+2**-40), lambda: base.add_(2, alpha=np.bool_(False)),
            lambda: base.add_(2**64), lambda: base.add_(-(2**63)-1),
            lambda: base.add_(np.uint64(2**63)), lambda: base.add_(None),
            lambda: base.add_(1j), lambda: base.add_([], alpha=1),
            lambda: base[:, 1:5].add_(1), lambda: base.select(1, 2).add_(1),
            lambda: base.add_(x2=1), lambda: base.add_(1, 2),
            lambda: base.add_(1, out=base), lambda: base.requires_grad_(),
        )
        for call in failures:
            with self.subTest(call=call), self.assertRaises((NotImplementedError, TypeError, RuntimeError, OverflowError)):
                call()
            self.assert_current_device_zero()
            np.testing.assert_array_equal(read_bits(base), bits)
            np.testing.assert_array_equal(read_bits(sibling), bits)
        self.assertIs(base.add_(0.5), base)

    def test_callbacks_finish_before_borrow_and_have_their_own_effects(self):
        x = native.tensor([1.0, 2.0]).to('cuda:0')
        events = []
        class Scalar(np.float32):
            def __float__(self):
                events.append('convert')
                x.add_(2.0)
                return 3.0
        self.assertIs(x.add_(Scalar(3)), x)
        self.assertEqual(events, ['convert'])
        self.assertEqual(x.cpu().tolist(), [6.0, 7.0])
        class RejectingScalar(np.float32):
            def __float__(self):
                x.add_(4.0)
                raise ValueError('callback rejected conversion')
        with self.assertRaisesRegex(ValueError, 'callback rejected'):
            x.add_(RejectingScalar(3))
        # Native rejection cannot undo the callback's completed independent call.
        self.assertEqual(x.cpu().tolist(), [10.0, 11.0])
        class RejectingAlpha(np.float32):
            def __float__(self):
                x.add_(1.0)
                return 2.0
        with self.assertRaisesRegex(NotImplementedError, 'alpha'):
            x.add_(7.0, alpha=RejectingAlpha(2))
        self.assertEqual(x.cpu().tolist(), [11.0, 12.0])
        from torch_rs.overrides import TorchFunctionMode
        class Forward(TorchFunctionMode):
            def __torch_function__(self, fn, types, args=(), kwargs=None):
                self_outer.assertIs(fn, native.Tensor.add_)
                return fn(*args, **(kwargs or {}))
        self_outer = self
        with Forward():
            self.assertIs(x.add_(other=0.5), x)
        self.assertEqual(x.cpu().tolist(), [11.5, 12.5])

    def test_alias_lifetime_cold_threads_and_independent_stream_visibility(self):
        base = native.full((1031,), 1.5).to('cuda:0')
        x, sibling = base[2:1029], base[2:1029].detach()
        pointer = x.data_ptr()
        del base
        gc.collect()
        def work(value):
            local = sibling.detach()
            self.assertIs(local.add_(value), local)
            return local
        with ThreadPoolExecutor(max_workers=4) as pool:
            retained = list(pool.map(work, (0.25, 0.5, 0.75, 1.0)))
        self.assertEqual(x.data_ptr(), pointer)
        self.assertEqual(x.cpu().tolist(), [4.0] * 1027)
        lib = runtime()
        stream = torch.cuda.Stream()
        out = torch.empty((1027,), device='cuda:0')
        with torch.cuda.stream(stream):
            self.assertIs(x.add_(-0.5), x)
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            self.assertEqual(lib.cudaMemcpyAsync(out.data_ptr(), x.data_ptr(), 1027 * 4,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, 3.5), rtol=0, atol=0)
        ordinal = ctypes.c_int()
        self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
        self.assertEqual(ordinal.value, 0)
        del x, sibling
        gc.collect()
        self.assertEqual(retained[0].cpu().tolist(), [3.5] * 1027)

    def test_default_compile_returned_alias_mutation_and_reuse(self):
        native.compiler.reset(); torch.compiler.reset()
        fn = program('def f(x):\n v=x.transpose(0,1)\n y=x+2\n return (v,y,v,x)')
        compiled, reference = native.compile(fn), torch.compile(fn)
        a = native.tensor([[1., 2., 3.], [4., 5., 6.]]).to('cuda:0')
        b = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda:0')
        try:
            for source, target in ((a, b), (a, b),
                                   (native.full((2, 3), -4.).to('cuda:0'),
                                    torch.full((2, 3), -4., device='cuda:0')),
                                   (native.tensor([9., 1., 2., 3., 4., 5., 6.]).to('cuda:0')[1:].reshape(2, 3),
                                    torch.tensor([9., 1., 2., 3., 4., 5., 6.], device='cuda:0')[1:].reshape(2, 3))):
                out, expected = compiled(source), reference(target)
                independent = read_bits(out[1]).copy()
                self.assertIs(out[0], out[2]); self.assertIs(out[3], source)
                with native.no_grad(), torch.no_grad():
                    self.assertIs(out[0].add_(0.375), out[0])
                    expected[0].add_(0.375)
                self.compare_storage(source, target)
                np.testing.assert_array_equal(read_bits(out[1]), independent)
                self.compare_storage(out[1], expected[1])
            for body in ('y=-x\n return x.add_(1)', 'x.add_(1)\n return x+1'):
                rejected = native.compile(program('def f(x):\n '+body))
                before = read_bits(a).copy()
                with self.assertRaises(NotImplementedError):
                    rejected(a)
                np.testing.assert_array_equal(read_bits(a), before)
                self.assertFalse(cache(rejected).graphs)
        finally:
            native.compiler.reset(); torch.compiler.reset()

    def test_native_only_without_python_copy_or_pytorch(self):
        script = '''
import sys
from unittest.mock import patch
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('native add_ imported PyTorch')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as n
base = n.tensor([[1., 2., 3.], [4., 5., 6.]]).to('cuda:0')
x = base.transpose(0, 1)
ptr = x.data_ptr()
with patch.object(n.Tensor, 'cpu', side_effect=AssertionError('CPU replay')), \\
     patch.object(n.Tensor, 'to', side_effect=AssertionError('copy replay')), \\
     patch.object(n.Tensor, 'clone', side_effect=AssertionError('clone replay')), \\
     patch.object(n.Tensor, 'contiguous', side_effect=AssertionError('materialization replay')):
    assert x.add_(other=1.25, alpha=1) is x
    assert x.data_ptr() == ptr
assert base.cpu().tolist() == [[2.25, 3.25, 4.25], [5.25, 6.25, 7.25]]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True,
                                text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
