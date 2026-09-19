"""Coalesced admission contracts; mocks establish crossings, never CUDA speed."""
import gc
import unittest
import sys
from unittest.mock import patch

import torch_rs as native
from torch_rs import torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, program
from tests import test_compile_pointwise_structured_outputs as structured_tests


CPU_ERROR = ("torch.compile(): native CUDA pointwise: default backend does not compile CPU tensors; "
             "use backend='eager' for the documented CPU capture subset; see docs/compile-pointwise-jit.md")


def old_admission(inputs):
    metadata = tuple(bridge._compile_trace_tensor_metadata(x) for x in inputs)
    if any(m[4] == 'cpu' for m in metadata):
        raise NotImplementedError(CPU_ERROR)
    bridge._pointwise_validate_inputs(inputs)
    return metadata


class BridgeContract(unittest.TestCase):
    def test_private_exports(self):
        name = '_pointwise_admit_inputs'
        self.assertTrue(callable(getattr(bridge, name)))
        for module in (bridge, native):
            self.assertNotIn(name, module.__all__)
        self.assertFalse(hasattr(native, name))
        for statement in ('from torch_rs import *', 'from torch_rs.torch_rs import *'):
            namespace = {}
            exec(statement, namespace)
            self.assertNotIn(name, namespace)

    def assert_metadata(self, x):
        metadata = bridge._compile_trace_tensor_metadata(x)
        self.assertEqual(metadata, (tuple(x.shape), tuple(x.stride()), x.requires_grad,
                                    str(x.dtype), str(x.device), x.storage_offset()))
        self.assertEqual(tuple(map(type, metadata)), (tuple, tuple, bool, str, str, int))
        self.assertTrue(all(type(item) is int for field in metadata[:2] for item in field))
        return metadata

    def test_trace_projection_still_accepts_cpu_noncontiguous_and_gradients(self):
        for x in (native.ones((2, 3)).t(), native.ones(4)[1:],
                  native.ones(2, requires_grad=True), native.tensor(1.), native.ones(0)):
            self.assert_metadata(x)
            with self.assertRaises(NotImplementedError) as error:
                bridge._pointwise_admit_inputs((x,))
            self.assertEqual(str(error.exception), CPU_ERROR)

    def test_exact_types_and_missing_inputs(self):
        effects = []

        class TensorLike:
            def __getattr__(self, name):
                effects.append(name)
                raise AssertionError('callback')

        # Native Tensor cannot be subclassed; lookalikes must not run callbacks.
        for item in (TensorLike(), object(), None, 1.):
            with self.subTest(kind=type(item)), self.assertRaises(TypeError):
                bridge._pointwise_admit_inputs((item,))
        self.assertEqual(effects, [])
        with self.assertRaises(TypeError):
            bridge._pointwise_admit_inputs([])
        with self.assertRaises(NotImplementedError) as old:
            old_admission(())
        with self.assertRaises(type(old.exception)) as new:
            bridge._pointwise_admit_inputs(())
        self.assertEqual(str(new.exception), str(old.exception))


class FrontendCrossings(unittest.TestCase):
    setUp = structured_tests.StructuredCache.setUp
    snapshot = structured_tests.StructuredCache.snapshot

    def test_complete_flattened_inputs_once_per_invocation_and_abi_transition(self):
        compiled = native.compile(program('def f(unused,items):\n return -items[0]'))
        x, y = native.ones(3), native.ones(3)
        histories = ((False, [x], (x,)), (y, [x], (y, x)),
                     (False, [y], (y,)), (x, [x], (x, x)))
        for unused, items, expected in histories:
            for _ in range(2):
                self.admit.reset_mock()
                # Fake host planning/preparation use their metadata adapter. Neither
                # obsolete Python bridge entry point may be called by the frontend.
                with patch.object(bridge, '_compile_trace_tensor_metadata', side_effect=AssertionError('frontend metadata')), \
                        patch.object(bridge, '_pointwise_validate_inputs', side_effect=AssertionError('frontend validation')):
                    compiled(unused, items)
                self.admit.assert_called_once_with(expected)
        self.assertEqual(len(cache(compiled).graphs), 1)
        self.assertEqual(len(next(iter(cache(compiled).graphs.values())).lowerings), 3)

    def test_type_and_method_guards_still_precede_admission(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.ones(3)
        with self.assertRaises(NotImplementedError):
            compiled(object())
        self.admit.assert_not_called()
        with patch.object(native.Tensor, '__neg__', lambda self: self):
            with self.assertRaisesRegex(NotImplementedError, 'patched Tensor operation binding'):
                compiled(x)
        self.admit.assert_not_called()
        self.assertEqual(self.snapshot(compiled), ([], [], [], 0))

    def test_metadata_and_admission_failures_preserve_caches_accounting_and_recover(self):
        compiled = native.compile(program('def f(x):\n return -x'))
        x = native.ones(3)
        compiled(x)
        compiled(native.ones(5))
        before = self.snapshot(compiled)
        for error in (MemoryError('metadata allocation'), TypeError('borrow/type'),
                      NotImplementedError('whole-input admission')):
            for current in (x, native.ones(7)):
                with self.subTest(error=str(error)), patch.object(bridge, '_pointwise_admit_inputs', side_effect=error):
                    with self.assertRaises(type(error)):
                        compiled(current)
                self.assertEqual(self.snapshot(compiled), before)
            cold = native.compile(program('def f(x):\n return -x'))
            with patch.object(bridge, '_pointwise_admit_inputs', side_effect=error):
                with self.assertRaises(type(error)):
                    cold(x)
            self.assertEqual(self.snapshot(cold), ([], [], [], 0))
        compiled(x)
        native.compiler.reset()
        self.assertEqual(self.snapshot(compiled), ([], [], [], 0))
        compiled(x)
        self.assertTrue(cache(compiled).prepared)


@unittest.skipUnless(available(), 'requires real native CUDA')
class CudaBridgeContract(unittest.TestCase):
    assert_metadata = BridgeContract.assert_metadata
    def test_direct_old_api_parity_order_duplicates_offsets_and_fresh_storage(self):
        for shape in ((), (1,), (0,), (2, 3)):
            x = native.ones(shape).to('cuda:0')
            self.assert_metadata(x)
            y = native.full(shape, 2.).to('cuda:0')
            for inputs in ((x,), (x, y), (y, x), (x, x)):
                actual = bridge._pointwise_admit_inputs(inputs)
                self.assertEqual(type(actual), tuple)
                self.assertEqual(actual, old_admission(inputs))
                self.assertEqual(bridge._pointwise_validate_inputs(inputs), 0)
        x = native.ones(7).to('cuda:0')[2:]
        self.assertEqual(bridge._pointwise_admit_inputs((x,)), (self.assert_metadata(x),))
        references = sys.getrefcount(x)
        copied = bridge._pointwise_admit_inputs((x,))
        self.assertEqual(sys.getrefcount(x), references)
        del x
        gc.collect()
        self.assertEqual(copied[0][-1], 2)

    def test_invalid_whole_inputs_and_cpu_precedence_in_either_slot(self):
        valid = native.ones(3).to('cuda:0')
        # Float32 is the sole native dtype; public CUDA gradient creation is
        # rejected before this bridge. Core gradient rejection has Rust coverage.
        invalid = (native.ones((2, 3)).to('cuda:0').t(),
                   native.ones(3, requires_grad=True))
        cpu = native.ones(3)
        for bad in invalid:
            self.assert_metadata(bad)  # Tracing's contract stays broader.
            for inputs in ((bad,), (valid, bad), (bad, valid), (cpu, bad), (bad, cpu)):
                with self.subTest(metadata=tuple(bridge._compile_trace_tensor_metadata(x) for x in inputs)):
                    with self.assertRaises(NotImplementedError) as old:
                        old_admission(inputs)
                    with self.assertRaises(type(old.exception)) as new:
                        bridge._pointwise_admit_inputs(inputs)
                    self.assertEqual(str(new.exception), str(old.exception))
            for inputs in ((cpu, bad), (bad, cpu)):
                with self.assertRaises(NotImplementedError) as error:
                    bridge._pointwise_admit_inputs(inputs)
                self.assertEqual(str(error.exception), CPU_ERROR)
        with self.assertRaises(NotImplementedError) as old:
            old_admission((valid, valid, valid))
        with self.assertRaises(type(old.exception)) as new:
            bridge._pointwise_admit_inputs((valid, valid, valid))
        self.assertEqual(str(new.exception), str(old.exception))

    def test_real_warm_frontend_crossings_and_current_values(self):
        compiled = native.compile(program('def f(unused,x):\n return -x'))
        admit = bridge._pointwise_admit_inputs
        for value in (1., 2., -3.):
            x = native.full((3,), value).to('cuda:0')
            for unused, inputs in ((False, (x,)), (x, (x, x))):
                with patch.object(bridge, '_pointwise_admit_inputs', wraps=admit) as call, \
                        patch.object(bridge, '_compile_trace_tensor_metadata', side_effect=AssertionError('metadata crossing')), \
                        patch.object(bridge, '_pointwise_validate_inputs', side_effect=AssertionError('validation crossing')):
                    result = compiled(unused, x)
                call.assert_called_once_with(inputs)
                self.assertEqual(result.cpu().tolist(), [-value] * 3)
                self.assertNotEqual(result.data_ptr(), x.data_ptr())
        native.compiler.reset()
