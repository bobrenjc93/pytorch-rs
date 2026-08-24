import inspect
import pickle
import subprocess
import sys
import types
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch_rs as torch


EXP_DOC = """
exp() -> Tensor

See :func:`torch.exp`
"""


class TensorExpTests(unittest.TestCase):
    def assert_tensor_bits_match(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    def test_no_mode_preserves_bits_layouts_and_fresh_storage(self):
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0xC2C8_0000,
                0xC2D0_0000,
                0x42B0_0000,
                0x42B2_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("strided", strided),
            (
                "numerical edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

        for case, source in cases:
            actual = source.exp()
            expected = torch.exp(source)
            self.assert_tensor_bits_match(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(source))
            self.assertFalse(expected.is_set_to(source))
            self.assertFalse(actual.is_set_to(expected))

    def test_autograd_name_lifecycle_accumulation_and_no_grad(self):
        probability = torch.tensor([1.0], requires_grad=True).exp()
        with self.assertRaisesRegex(ValueError, r"grad_fn=<ExpBackward0>"):
            torch.nn.functional.dropout(
                torch.tensor([1.0]), p=probability, training=False
            )

        freed = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        loss = freed.exp().sum()
        loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        accumulated = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        accumulated.exp().sum().backward()
        first = np.asarray(accumulated.grad).copy()
        accumulated.exp().sum().backward()
        np.testing.assert_array_equal(np.asarray(accumulated.grad), first * 2.0)

        tracked = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        with torch.no_grad():
            untracked = tracked.exp()
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertFalse(untracked.is_set_to(tracked))
        self.assertTrue(tracked.exp().requires_grad)

        detached = tracked.detach()
        self.assert_tensor_bits_match(
            detached.exp(), torch.exp(detached), case="detached input"
        )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([0.5])
        descriptor = inspect.getattr_static(torch.Tensor, "exp")
        bound = tensor.exp

        self.assertIs(torch.Tensor.exp, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'exp' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "exp")
        self.assertEqual(descriptor.__qualname__, "TensorBase.exp")
        self.assertEqual(bound.__name__, "exp")
        self.assertEqual(bound.__qualname__, "Tensor.exp")
        self.assertEqual(descriptor.__doc__, EXP_DOC)
        self.assertEqual(bound.__doc__, EXP_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)

        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        cases = (
            (lambda: tensor.exp(1), "TensorBase.exp() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.exp() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.exp() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.exp(1, 2),
                "TensorBase.exp() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.exp(input=tensor),
                (
                    "Tensor.exp() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.exp() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.exp() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.exp() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.exp() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'exp' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.exp() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_tensorbase_descriptor_round_trips_through_both_picklers(self):
        descriptor = inspect.getattr_static(torch.Tensor, "exp")
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, pickler="pickle"):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol)), descriptor
                )
            with self.subTest(protocol=protocol, pickler="ForkingPickler"):
                self.assertIs(
                    pickle.loads(ForkingPickler.dumps(descriptor, protocol)),
                    descriptor,
                )

    def test_registration_preserves_preexisting_descriptor_reducers(self):
        source = r'''
import copyreg
import importlib
import inspect
import pickle
import types
from multiprocessing.reduction import ForkingPickler

def standard_reducer(descriptor):
    return str, ("standard reducer",)

def forking_reducer(descriptor):
    return str, ("forking reducer",)

copyreg.pickle(types.MethodDescriptorType, standard_reducer)
ForkingPickler.register(types.MethodDescriptorType, forking_reducer)

import torch_rs as torch

def check_reducers():
    descriptor = inspect.getattr_static(torch.Tensor, "exp")
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
        assert pickle.loads(ForkingPickler.dumps(descriptor, protocol)) is descriptor
        assert pickle.loads(pickle.dumps(str.upper, protocol)) == "standard reducer"
        assert (
            pickle.loads(ForkingPickler.dumps(str.upper, protocol))
            == "forking reducer"
        )

check_reducers()
torch = importlib.reload(torch)
check_reducers()
'''
        subprocess.run([sys.executable, "-c", source], check=True)

    def test_torch_function_modes_dispatch_before_native_and_restore_stack(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "exp")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tensor.exp()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.exp()
        bypass = RecordingMode(marker)
        with bypass:
            self.assertIs(extreme.exp(), marker)
        self.assertEqual(len(bypass.calls), 1)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.exp()
        self.assertEqual(order, ["upper", "lower"])
        forwarded.sum().backward()
        np.testing.assert_allclose(
            np.asarray(tensor.grad),
            np.exp(np.asarray([0.5], dtype=np.float32)),
            rtol=2.0e-6,
            atol=0.0,
        )

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise ValueError("exp mode failed")

        order.clear()
        with ForwardingMode("lower"):
            with self.assertRaisesRegex(ValueError, "^exp mode failed$"):
                with RaisingMode():
                    tensor.exp()
            self.assertEqual(
                len(torch.overrides._get_current_function_mode_stack()), 1
            )
            recovered = tensor.exp()
        self.assertEqual(order, ["lower"])
        self.assertEqual(recovered.tolist(), tensor.detach().exp().tolist())
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    tensor.exp()
                self.assertEqual(
                    len(torch.overrides._get_current_function_mode_stack()), 1
                )
        finally:
            sys.setrecursionlimit(old_recursion_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                tensor.exp(1)
        self.assertEqual(invalid.calls, [])

    def test_inplace_and_method_out_forms_remain_unsupported(self):
        tensor = torch.tensor([0.5])
        self.assertFalse(hasattr(torch.Tensor, "exp_"))
        self.assertFalse(hasattr(tensor, "exp_"))
        with self.assertRaises(TypeError):
            tensor.exp(out=None)


if __name__ == "__main__":
    unittest.main()
