import hashlib
import inspect
import pickle
import re
import subprocess
import sys
import types
import unittest

import torch_rs as torch


TO_DOC_LENGTH = 4025
TO_DOC_SHA256 = "8807a956bc5277bd22c3893d9f7e3b0116d691c60bb40d8f72a8d92de059e979"


class TensorToTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(no_grad_output.requires_grad)
        return leaf, tracked, (
            ("scalar", torch.tensor(-3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("strided view", strided),
            ("offset strided view", offset),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf", produced),
            ("autograd non-leaf view", tracked),
            ("no-grad output", no_grad_output),
            ("no-grad view", no_grad_view),
        )

    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.grad,
        )

    def test_no_argument_call_returns_exact_receiver_for_all_supported_states(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)
                self.assertIs(tensor.to(), tensor)
                self.assertIs(tensor.to(*()), tensor)
                self.assertIs(tensor.to(**{}), tensor)
                self.assertEqual(self.metadata(tensor), metadata)

        tracked.to().sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        gradient = leaf.grad
        self.assertIs(leaf.to(), leaf)
        self.assertIs(leaf.grad, gradient)

    def test_calling_to_inside_no_grad_keeps_existing_autograd_history(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)
        graph = (
            tracked.requires_grad,
            tracked.is_leaf,
            tracked.shape,
            tracked.stride(),
            tracked.storage_offset(),
            tracked.data_ptr(),
        )

        with torch.no_grad():
            result = tracked.to()

        self.assertIs(result, tracked)
        self.assertEqual(
            (
                result.requires_grad,
                result.is_leaf,
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.data_ptr(),
            ),
            graph,
        )
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])

    def test_tensorbase_ownership_documentation_and_pickle_match(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        bound = tensor.to

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'to' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "to")
        self.assertEqual(descriptor.__qualname__, "TensorBase.to")
        self.assertEqual(bound.__name__, "to")
        self.assertEqual(bound.__qualname__, "Tensor.to")
        self.assertEqual(descriptor.__doc__, bound.__doc__)
        self.assertEqual(len(descriptor.__doc__), TO_DOC_LENGTH)
        self.assertEqual(
            hashlib.sha256(descriptor.__doc__.encode()).hexdigest(),
            TO_DOC_SHA256,
        )
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(), tensor)

        reducer, (owner, name) = descriptor.__reduce__()
        self.assertIs(reducer, getattr)
        self.assertIs(owner, descriptor.__objclass__)
        self.assertEqual(name, "to")
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                    descriptor,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(str.upper, protocol=protocol)),
                    str.upper,
                )

    def test_descriptor_pickle_survives_package_reinitialization(self):
        source = r"""
import importlib
import inspect
import pickle
import sys

first = importlib.import_module("torch_rs")
descriptor = inspect.getattr_static(first.Tensor, "to")
saved_modules = {
    name: module
    for name, module in tuple(sys.modules.items())
    if name == "torch_rs" or name.startswith("torch_rs.")
}
for name in saved_modules:
    sys.modules.pop(name, None)
second = importlib.import_module("torch_rs")
assert second.Tensor is first.Tensor
for name in tuple(sys.modules):
    if name == "torch_rs" or name.startswith("torch_rs."):
        sys.modules.pop(name, None)
sys.modules.update(saved_modules)
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(descriptor, protocol=protocol)) is descriptor
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_import_preserves_a_prior_method_descriptor_reducer(self):
        source = r"""
import copyreg
import inspect
import pickle
import types

calls = []

def restore_prior_result():
    return str.lower

def prior_reducer(descriptor):
    calls.append(descriptor)
    return restore_prior_result, ()

copyreg.pickle(types.MethodDescriptorType, prior_reducer)

import torch_rs

tensor_to = inspect.getattr_static(torch_rs.Tensor, "to")
for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(tensor_to, protocol=protocol)) is tensor_to
assert calls == []

for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
    assert pickle.loads(pickle.dumps(str.upper, protocol=protocol)) is str.lower
assert calls == [str.upper] * (pickle.HIGHEST_PROTOCOL + 1)
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_reloads_do_not_chain_method_descriptor_reducers(self):
        source = r"""
import copyreg
import importlib
import inspect
import pickle
import types

calls = []

def restore_prior_result():
    return str.lower

def prior_reducer(descriptor):
    calls.append(descriptor)
    return restore_prior_result, ()

copyreg.pickle(types.MethodDescriptorType, prior_reducer)

import torch_rs

for _ in range(1100):
    torch_rs = importlib.reload(torch_rs)

registered = copyreg.dispatch_table[types.MethodDescriptorType]
assert (
    registered._torch_rs_prior_method_descriptor_reducer is prior_reducer
)
assert pickle.loads(pickle.dumps(str.upper)) is str.lower
assert calls == [str.upper]

tensor_to = inspect.getattr_static(torch_rs.Tensor, "to")
assert pickle.loads(pickle.dumps(tensor_to)) is tensor_to
assert calls == [str.upper]
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_reinitialization_releases_the_previous_package_and_reducer(self):
        source = r"""
import copyreg
import gc
import importlib
import pickle
import sys
import types
import weakref

calls = []

def restore_prior_result():
    return str.lower

def prior_reducer(descriptor):
    calls.append(descriptor)
    return restore_prior_result, ()

copyreg.pickle(types.MethodDescriptorType, prior_reducer)

import torch_rs

old_package = weakref.ref(torch_rs)
old_reducer_object = copyreg.dispatch_table[types.MethodDescriptorType]
old_reducer = weakref.ref(old_reducer_object)
old_helper = weakref.ref(torch_rs._get_tensor_to_descriptor)

for name in tuple(sys.modules):
    if name == "torch_rs" or name.startswith("torch_rs."):
        sys.modules.pop(name, None)
del torch_rs, old_reducer_object

reinitialized = importlib.import_module("torch_rs")
for _ in range(3):
    gc.collect()

assert old_package() is None
assert old_reducer() is None
assert old_helper() is None

registered = copyreg.dispatch_table[types.MethodDescriptorType]
assert (
    registered._torch_rs_prior_method_descriptor_reducer is prior_reducer
)
assert pickle.loads(pickle.dumps(str.upper)) is str.lower
assert calls == [str.upper]
assert reinitialized.Tensor.to is not None
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_unrelated_descriptor_pickle_does_not_import_torch_rs(self):
        source = r"""
import builtins
import pickle
import sys

import torch_rs

for name in tuple(sys.modules):
    if name == "torch_rs" or name.startswith("torch_rs."):
        sys.modules.pop(name, None)

import_attempts = []
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "torch_rs" or name.startswith("torch_rs."):
        import_attempts.append(name)
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
try:
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        assert pickle.loads(pickle.dumps(str.upper, protocol=protocol)) is str.upper
finally:
    builtins.__import__ = original_import

assert import_attempts == []
assert not any(
    name == "torch_rs" or name.startswith("torch_rs.")
    for name in sys.modules
)
"""
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_unbound_no_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.to() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'to' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.to() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_conversion_argument_forms_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        cases = (
            (
                lambda: tensor.to(torch.float32),
                "to() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.to(torch.device("cpu"), torch.float32),
                "to() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: descriptor(tensor, other),
                "to() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.to(dtype=torch.float32),
                "to() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: tensor.to(device=torch.device("cpu")),
                "to() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: tensor.to(copy=False),
                "to() got an unexpected keyword argument 'copy'",
            ),
            (
                lambda: tensor.to(non_blocking=False),
                "to() got an unexpected keyword argument 'non_blocking'",
            ),
            (
                lambda: tensor.to(memory_format=torch.preserve_format),
                "to() got an unexpected keyword argument 'memory_format'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
        self.assertFalse(hasattr(torch, "to"))

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for label, call, expected_kwargs in (
            ("plain", lambda: tensor.to(), None),
            ("empty kwargs", lambda: tensor.to(**{}), {}),
        ):
            recording = RecordingMode(marker)
            with recording:
                result = call()
            with self.subTest(label=label):
                self.assertIs(result, marker)
                self.assertEqual(len(recording.calls), 1)
                function, dispatch_types, args, kwargs = recording.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(len(args), 1)
                self.assertIs(args[0], tensor)
                self.assertEqual(kwargs, expected_kwargs)
                self.assertIs(
                    pickle.loads(pickle.dumps(function)),
                    descriptor,
                )

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.to()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.to()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.to'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        rejected = RecordingMode(marker)
        with rejected:
            with self.assertRaises(TypeError):
                tensor.to(dtype=torch.float32)
        self.assertEqual(rejected.calls, [])


if __name__ == "__main__":
    unittest.main()
