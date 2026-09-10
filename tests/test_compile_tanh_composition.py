"""Eager tanh on owned CPU outputs of the bounded native compiler."""

import ctypes
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native

from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_mul_scalar import call_without_python, make_program

try:
    import torch as reference
except ImportError:
    reference = None


def apply_tanh(module, value, form):
    if form == "method":
        return value.tanh()
    if form == "top level":
        return module.tanh(input=value, out=None)
    if form == "functional":
        return module.nn.functional.tanh(value)
    return module.nn.functional.tanhshrink(value)


@unittest.skipIf(reference is None, "install the reference dependency group")
class CompileTanhCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("tanh differentials require pinned PyTorch 2.13.0")

    def compare(self, actual, expected, *, tanhshrink=False):
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        # Match each operator's existing differential tolerance. Tanhshrink's
        # subtraction near zero amplifies tanh's float32 rounding difference;
        # see test_nn_functional_tanhshrink_reference.py.
        a = np.asarray(actual, dtype=np.float32)
        b = expected.detach().numpy()
        np.testing.assert_allclose(
            a, b, rtol=2e-6 if tanhshrink else 3 * np.finfo(np.float32).eps,
            atol=2e-7 if tanhshrink else np.nextafter(np.float32(0), np.float32(1)),
            equal_nan=True,
        )
        zeros = b == 0
        np.testing.assert_array_equal(a[zeros].view(np.uint32), b[zeros].view(np.uint32))

    def backward_and_release(self, actual, expected, leaves, refs, *, tanhshrink=False):
        self.compare(actual, expected, tanhshrink=tanhshrink)
        weights = np.linspace(-2.5, 1.5, actual.numel(), dtype=np.float32).reshape(actual.shape)
        weight = native.tensor(weights.tolist()).reshape(actual.shape)
        ref_weight = reference.tensor(weights.tolist()).reshape(expected.shape)
        loss = ((actual + actual) * weight).sum()
        ref_loss = ((expected + expected) * ref_weight).sum()
        sibling, ref_sibling = actual.sum(), expected.sum()
        loss.backward()
        ref_loss.backward()
        for leaf, ref in zip(leaves, refs):
            if ref.grad is None:
                self.assertIsNone(leaf.grad)
            else:
                self.compare(leaf.grad, ref.grad, tanhshrink=tanhshrink)
        before = [None if leaf.grad is None else np.asarray(leaf.grad).copy() for leaf in leaves]
        for released, ref_released in ((loss, ref_loss), (sibling, ref_sibling)):
            with self.assertRaises(RuntimeError) as expected_error:
                ref_released.backward()
            with self.assertRaises(RuntimeError) as actual_error:
                released.backward()
            self.assertEqual(str(actual_error.exception), str(expected_error.exception))
        for leaf, previous in zip(leaves, before):
            if previous is not None:
                np.testing.assert_array_equal(np.asarray(leaf.grad), previous)

    def test_owned_outputs_changed_values_shared_operands_and_repeated_backward(self):
        sources = (
            "def program(x):\n    return -x\n",
            "def program(x):\n    return x + x\n",
            "def program(x):\n    return -(x + bias)\n",
        )
        shapes = ((), (4,), (2, 2), (2, 1, 2), (0,), (2, 0), (2, 0, 1))
        for shape in shapes:
            for source in sources:
                for form in ("method", "top level", "functional", "tanhshrink"):
                    for shared_capture in (False, True) if "bias" in source else (False,):
                        with self.subTest(shape=shape, source=source, form=form, shared=shared_capture):
                            x = native.full(shape, 0.25, requires_grad=True)
                            rx = reference.full(shape, 0.25, requires_grad=True)
                            bias = x if shared_capture else native.full(shape, -0.5, requires_grad=True)
                            rbias = rx if shared_capture else reference.full(shape, -0.5, requires_grad=True)
                            program = make_program(source, bias=bias)
                            ref_program = make_program(source, module=reference, bias=rbias)
                            compiled, cache = compile_with_cache(program)
                            for value in (0.25, -0.75):
                                # Change existing allocations after releasing the previous graph.
                                (ctypes.c_float * x.numel()).from_address(x.data_ptr())[:] = [value] * x.numel()
                                with reference.no_grad():
                                    rx.fill_(value)
                                    if not shared_capture:
                                        rbias.fill_(-0.5 * value)
                                if not shared_capture:
                                    (ctypes.c_float * bias.numel()).from_address(bias.data_ptr())[:] = [-0.5 * value] * bias.numel()
                                parent = call_without_python(compiled, {program.__code__}, x)
                                ref_parent = ref_program(rx)
                                self.compare(parent, ref_parent)
                                self.assertFalse(parent.is_leaf)
                                if parent.numel():
                                    self.assertNotEqual(parent.data_ptr(), x.data_ptr())
                                self.backward_and_release(
                                    apply_tanh(native, parent, form),
                                    apply_tanh(reference, ref_parent, form),
                                    [x] if shared_capture else [x, bias],
                                    [rx] if shared_capture else [rx, rbias],
                                    tanhshrink=form == "tanhshrink",
                                )
                            self.assertEqual(len(cache.graphs), 1)

    def test_live_grad_flags_capture_rebinding_helper_and_root_code_replacement(self):
        source = "def helper(x):\n    return -x\ndef program(x):\n    return helper(x) + bias\n"
        bias, rbias = native.full((2,), 0.5), reference.full((2,), 0.5)
        program = make_program(source, bias=bias)
        ref_program = make_program(source, module=reference, bias=rbias)
        compiled, cache = compile_with_cache(program)
        x, rx = native.full((2,), 0.25), reference.full((2,), 0.25)

        def check():
            helper = program.__globals__["helper"]
            parent = call_without_python(compiled, {program.__code__, helper.__code__}, x)
            actual, expected = parent.tanh(), ref_program(rx).tanh()
            if expected.requires_grad:
                self.backward_and_release(actual, expected, [x, bias], [rx, rbias])
            else:
                self.compare(actual, expected)

        for input_grad, capture_grad in ((False, False), (True, False), (False, True),
                                         (True, True), (False, False), (True, True)):
            x.requires_grad_(input_grad)
            rx.requires_grad_(input_grad)
            bias.requires_grad_(capture_grad)
            rbias.requires_grad_(capture_grad)
            check()
        self.assertEqual(len(cache.graphs), 4)

        bias = native.full((2,), -0.75, requires_grad=True)
        rbias = reference.full((2,), -0.75, requires_grad=True)
        program.__globals__["bias"], ref_program.__globals__["bias"] = bias, rbias
        check()
        for root, module in ((program, native), (ref_program, reference)):
            root.__globals__["helper"].__code__ = make_program(
                "def program(x):\n    return x + x\n", module=module).__code__
        check()
        for root, module in ((program, native), (ref_program, reference)):
            replacement = make_program(
                "def helper(x):\n    return -(x + x)\ndef program(x):\n    return helper(x)\n",
                module=module)
            root.__globals__["helper"] = types.FunctionType(
                replacement.__globals__["helper"].__code__, root.__globals__, "helper")
        check()
        for root, module in ((program, native), (ref_program, reference)):
            root.__code__ = make_program(
                "def program(x):\n    if x.requires_grad:\n        return x + bias\n    return -x\n",
                module=module).__code__
        check()
        x.requires_grad_(False)
        rx.requires_grad_(False)
        with patch.dict(program.__globals__, bias=object()):
            check()  # The inactive capture must not be resolved.
        x.requires_grad_(True)
        rx.requires_grad_(True)
        before = dict(cache.graphs)
        with patch.dict(program.__globals__, bias=object()), self.assertRaises(NotImplementedError):
            compiled(x)
        self.assertEqual(cache.graphs, before)
        check()


class CompileTanhBoundaryTests(unittest.TestCase):
    def test_unsupported_outputs_leave_compiled_parent_graph_usable(self):
        program = make_program("def program(x):\n    return x + x\n")
        compiled, _ = compile_with_cache(program)
        for shape, value in (((1, 1, 1, 2), 0.5), ((2,), float("inf"))):
            with self.subTest(shape=shape, value=value):
                x = native.full(shape, value, requires_grad=True)
                parent = call_without_python(compiled, {program.__code__}, x)
                for form in ("method", "top level", "functional", "tanhshrink"):
                    with self.assertRaisesRegex(RuntimeError, "tanh.*autograd recording is not supported"):
                        apply_tanh(native, parent, form)
                    self.assertIsNone(x.grad)
                parent.sum().backward()
                np.testing.assert_array_equal(np.asarray(x.grad), np.full(shape, 2.0))

    def test_tanh_inside_compiled_body_stays_unsupported_and_recovers(self):
        program = make_program("def program(x):\n    return x + bias\n", bias=native.ones(2))
        compiled, cache = compile_with_cache(program)
        x = native.full((2,), 0.5, requires_grad=True)
        original = program.__code__
        compiled(x).tanh().sum().backward()
        before = dict(cache.graphs)
        for expression in ("x.tanh()", "m.tanh(x)", "m.nn.functional.tanh(x)"):
            program.__code__ = make_program(f"def program(x):\n    return {expression}\n").__code__
            with self.subTest(expression=expression), self.assertRaises(NotImplementedError):
                call_without_python(compiled, {program.__code__}, x)
            self.assertEqual(cache.graphs, before)
        program.__code__ = original
        compiled(x).tanh().sum().backward()
        self.assertEqual(cache.graphs, before)


if __name__ == "__main__":
    unittest.main()
