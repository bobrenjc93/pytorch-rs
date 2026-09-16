"""Whole-input native admission metadata and independent prepared-run guards."""
from contextlib import ExitStack, contextmanager
import sys
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend, torch_rs as bridge
from tests.test_compile_pointwise_jit import available, cache, program
from tests import test_compile_pointwise_structured_outputs as structured_tests


CPU_ERROR = (
    "torch.compile(): native CUDA pointwise: default backend does not compile CPU tensors; "
    "use backend='eager' for the documented CPU capture subset; see docs/compile-pointwise-jit.md"
)


@contextmanager
def no_body(fn):
    previous = sys.getprofile()

    def reject(frame, event, arg):
        if event == "call" and frame.f_code is fn.__code__:
            raise AssertionError("original body executed")

    sys.setprofile(reject)
    try:
        yield
    finally:
        sys.setprofile(previous)


class NoCacheEntry:
    def __enter__(self):
        raise AssertionError("invalid input reached the cache lock")

    def __exit__(self, *args):
        raise AssertionError("invalid input exited the cache lock")


class AdmissionPortable(unittest.TestCase):
    def tearDown(self):
        native.compiler.reset()

    def test_cpu_diagnostic_is_exact_and_precedes_cache_work(self):
        cpu = native.tensor([1.], requires_grad=True)
        compiled = native.compile(program("def f(x):\n return -x"))
        with self.assertRaises(NotImplementedError) as caught:
            bridge._pointwise_admit_inputs((cpu,))
        self.assertEqual(str(caught.exception), CPU_ERROR)
        with patch.object(cache(compiled), "lock", NoCacheEntry()):
            with self.assertRaises(NotImplementedError) as caught:
                compiled(cpu)
        self.assertEqual(str(caught.exception), CPU_ERROR)
        self.assertEqual(structured_tests.StructuredCache.snapshot(self, compiled), ([], [], [], 0))

    def test_empty_and_foreign_inputs_reject_without_hooks(self):
        effects = []

        class Foreign:
            def __getattribute__(self, name):
                effects.append(name)
                raise AssertionError("foreign attribute hook")

            def __iter__(self):
                effects.append("iter")
                raise AssertionError("foreign iterator hook")

            def __float__(self):
                effects.append("float")
                raise AssertionError("foreign scalar hook")

            @classmethod
            def __torch_function__(cls, *args, **kwargs):
                effects.append("torch_function")
                raise AssertionError("foreign tensor hook")

        with self.assertRaisesRegex(NotImplementedError, "missing tensor input"):
            bridge._pointwise_admit_inputs(())
        cpu = native.tensor([1.])
        foreign = Foreign()
        for inputs in ((foreign,), (cpu, foreign), (foreign, cpu), (1.,), (True,)):
            with self.subTest(count=len(inputs)), self.assertRaisesRegex(
                    TypeError, "pointwise JIT requires exact native tensors"):
                bridge._pointwise_admit_inputs(inputs)
        for inputs in ([], [cpu], foreign):
            with self.assertRaises(TypeError):
                bridge._pointwise_admit_inputs(inputs)
        self.assertEqual(effects, [])

    def test_tree_mode_and_method_guards_precede_native_admission(self):
        from torch_rs.overrides import TorchFunctionMode

        class Mode(TorchFunctionMode):
            def __torch_function__(self, *args, **kwargs):
                raise AssertionError("mode callback executed")

        compiled = native.compile(program("def f(x):\n return x.sin()"))
        cpu = native.tensor([1.])
        with patch.object(bridge, "_pointwise_admit_inputs",
                          side_effect=AssertionError("early admission")) as admit:
            with self.assertRaises(NotImplementedError):
                compiled(object())
            with Mode(), self.assertRaisesRegex(NotImplementedError, "active.*mode"):
                compiled(cpu)
            with patch.object(native.Tensor, "sin", lambda self: self):
                with self.assertRaisesRegex(NotImplementedError, "patched Tensor operation"):
                    compiled(cpu)
            admit.assert_not_called()


@unittest.skipUnless(available(), "requires native CUDA and reference PyTorch CUDA")
class AdmissionHardware(unittest.TestCase):
    snapshot = structured_tests.StructuredCache.snapshot

    def tearDown(self):
        native.compiler.reset()

    def test_metadata_preserves_legacy_schema_and_occurrences_without_hooks(self):
        vector = native.tensor([1., 2., 3., 4.]).to("cuda:0")
        cases = [
            (vector,),
            (vector, native.tensor([5., 6.]).to("cuda:0")),
            (vector, vector),
            (native.tensor(2.).to("cuda:0"),),
            (native.zeros(2, 0, 3).to("cuda:0"),),
            (vector[1:3],),
            (vector[2:2],),
        ]
        for inputs in cases:
            expected = tuple(bridge._compile_trace_tensor_metadata(x) for x in inputs)
            with self.subTest(metadata=expected), ExitStack() as stack:
                # The bridge reads native fields, never overridable Python metadata.
                for name in ("shape", "stride", "requires_grad", "dtype", "device", "storage_offset"):
                    def fail(*args, **kwargs):
                        raise AssertionError("Python metadata hook executed")
                    stack.enter_context(patch.object(native.Tensor, name, property(fail)))
                actual = bridge._pointwise_admit_inputs(inputs)
            self.assertIs(type(actual), tuple)
            self.assertEqual(actual, expected)
            self.assertEqual(len(actual), len(inputs))
            for record in actual:
                self.assertIs(type(record), tuple)
                self.assertEqual(len(record), 6)
                self.assertEqual(tuple(type(value) for value in record),
                                 (tuple, tuple, bool, str, str, int))
                self.assertEqual(record[2:5], (False, "torch.float32", "cuda:0"))
        self.assertEqual(bridge._pointwise_admit_inputs((vector[1:3],)),
                         (((2,), (1,), False, "torch.float32", "cuda:0", 1),))

    def test_cardinality_layout_and_cpu_precedence_match_whole_input_contract(self):
        x = native.ones(2, 2).to("cuda:0")
        bad = x.transpose(0, 1)
        with self.assertRaisesRegex(NotImplementedError, "expected one or two inputs"):
            bridge._pointwise_admit_inputs((x, x, x))
        for inputs in ((bad,), (x, bad), (bad, x)):
            with self.assertRaisesRegex(NotImplementedError, "contiguous CUDA float32"):
                bridge._pointwise_admit_inputs(inputs)
        cpu = native.ones(2, 2, requires_grad=True)
        for inputs in ((bad, cpu), (cpu, bad)):
            with self.assertRaises(NotImplementedError) as caught:
                bridge._pointwise_admit_inputs(inputs)
            self.assertEqual(str(caught.exception), CPU_ERROR)

    def test_public_cold_warm_and_receipt_enter_admission_once(self):
        fn = program("def f(x,y):\n return x+y")
        compiled = native.compile(fn)
        x = native.tensor([1., 2.]).to("cuda:0")
        y = native.tensor([3., 4.]).to("cuda:0")
        for receipt, inputs, expected in [(False, (x, y), [4., 6.]),
                                          (False, (y, x), [4., 6.]),
                                          (True, (x, x), [2., 4.]),
                                          (True, (x, x), [2., 4.])]:
            with ExitStack() as stack:
                admit = stack.enter_context(patch.object(
                    bridge, "_pointwise_admit_inputs", wraps=bridge._pointwise_admit_inputs))
                for name in ("_compile_trace_tensor_metadata", "_pointwise_validate_inputs"):
                    stack.enter_context(patch.object(bridge, name,
                                                    side_effect=AssertionError("legacy admission called")))
                stack.enter_context(no_body(fn))
                result = compiled._torch_rs_pointwise_receipt(*inputs) if receipt else compiled(*inputs)
                if receipt:
                    result, prepared = result
                    self.assertTrue(any(prepared.belongs_to(owner) for owner in cache(compiled).executors.values()))
                self.assertEqual(result.cpu().tolist(), expected)
                self.assertEqual(admit.call_count, 1)
                admitted = admit.call_args.args[0]
                self.assertEqual(len(admitted), 2)
                self.assertTrue(all(a is b for a, b in zip(admitted, inputs)))

    def test_unused_invalid_input_never_enters_or_mutates_cold_or_warm_cache(self):
        x = native.ones(2, 2).to("cuda:0")
        bad = x.transpose(0, 1)
        cpu = native.ones(2, 2)
        for source, invalid in [("def f(x,unused):\n return -x", (x, bad)),
                                ("def f(unused,x):\n return -x", (bad, x)),
                                ("def f(x,unused):\n return -x", (x, cpu))]:
            for warm in (False, True):
                fn = program(source)
                compiled = native.compile(fn)
                if warm:
                    compiled(x, x)
                before = self.snapshot(compiled)
                with patch.object(cache(compiled), "lock", NoCacheEntry()), no_body(fn):
                    with self.assertRaises(NotImplementedError):
                        compiled(*invalid)
                self.assertEqual(self.snapshot(compiled), before)

    def test_retained_receipt_run_revalidates_inputs_and_scalars_independently(self):
        x = native.tensor([[1., 2.], [3., 4.]]).to("cuda:0")
        compiled = native.compile(program("def f(x):\n return -x"))
        _, prepared = compiled._torch_rs_pointwise_receipt(x)
        grad_cpu = native.ones(2, 2, requires_grad=True)
        with self.assertRaises(NotImplementedError):
            x.requires_grad_(True)
        invalid = [((x.reshape(4),), "shape guard"),
                   ((x.transpose(0, 1),), "contiguous CUDA float32"),
                   ((grad_cpu,), "native CUDA inputs")]
        effects = []

        class Scalar:
            def __float__(self):
                effects.append("float")
                return 1.

        with patch.object(bridge, "_pointwise_admit_inputs",
                          side_effect=AssertionError("receipt delegated validation")):
            for inputs, message in invalid:
                with self.assertRaisesRegex(RuntimeError, message):
                    prepared.run(inputs)
            for value in (Scalar(), 1, True):
                with self.assertRaises(TypeError):
                    prepared.run((x,), (value,))
            with self.assertRaises(RuntimeError):
                prepared.run((x,), (1.,))
            self.assertEqual(prepared.run((x,))[0].cpu().tolist(), [[-1., -2.], [-3., -4.]])
            native.compiler.reset()
            self.assertEqual(prepared.run((x,))[0].cpu().tolist(), [[-1., -2.], [-3., -4.]])
        self.assertEqual(effects, [])


if __name__ == "__main__":
    unittest.main()
