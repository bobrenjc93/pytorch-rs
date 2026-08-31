import copy
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC_PREFIX = (
    "\nas_tensor(data: Any, *, dtype: Optional[dtype] = None, "
    "device: Optional[DeviceLikeType]) -> Tensor\n\n"
    "Converts :attr:`data` into a tensor, sharing data and preserving autograd\n"
    "history if possible."
)


class AsTensorTests(unittest.TestCase):
    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
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
        special_bits = np.asarray(
            (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345),
            dtype=np.uint32,
        )
        return (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            ("empty view", torch.zeros((2, 0, 3), dtype=torch.float32)[1]),
            ("strided view", strided[1]),
            ("leaf", leaf),
            ("tracked view", tracked),
            ("special bits", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

    def tensor_state(self, tensor):
        return (
            np.asarray(tensor).reshape(-1).view(np.uint32).tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def assert_default_tensor(self, tensor, expected_values, *, shape, stride, tolist):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertEqual(tensor.output_nr, 0)
        self.assertFalse(tensor.is_pinned())
        actual_bits = np.asarray(tensor).reshape(-1).view(np.uint32).tolist()
        expected_bits = (
            np.asarray(expected_values, dtype=np.float32).reshape(-1).view(np.uint32).tolist()
        )
        self.assertEqual(actual_bits, expected_bits)
        self.assertEqual(tensor.tolist(), tolist)

    def test_exact_native_cpu_float32_tensors_return_identical_object(self):
        option_cases = (
            {},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": None},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"dtype": torch.float32, "device": torch.device("cpu")},
        )
        for case, tensor in self.tensor_cases():
            before = self.tensor_state(tensor)
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    result = torch.as_tensor(tensor, **options)
                    self.assertIs(result, tensor)
                    self.assertEqual(result.data_ptr(), before[4])
                    self.assertEqual(self.tensor_state(tensor), before)

    def test_python_real_scalars_and_rectangular_sequences_create_default_tensors(self):
        cases = (
            ("float scalar", -0.0, (), (), [-0.0], -0.0),
            ("int scalar", 7, (), (), [7.0], 7.0),
            (
                "flat list",
                [1.0, -0.0, 2.5],
                (3,),
                (1,),
                [1.0, -0.0, 2.5],
                [1.0, -0.0, 2.5],
            ),
            ("flat tuple", (1.0, 2.0), (2,), (1,), [1.0, 2.0], [1.0, 2.0]),
            (
                "nested list tuple",
                [[1.0, 2.0], (3, 4.5)],
                (2, 2),
                (2, 1),
                [1.0, 2.0, 3.0, 4.5],
                [[1.0, 2.0], [3.0, 4.5]],
            ),
            ("empty list", [], (0,), (1,), [], []),
            ("nested empty", [[], [1.0]], (2, 0), (1, 1), [], [[], []]),
        )
        for case, data, shape, stride, expected_values, expected_tolist in cases:
            with self.subTest(case=case):
                result = torch.as_tensor(data)
                self.assert_default_tensor(
                    result,
                    expected_values,
                    shape=shape,
                    stride=stride,
                    tolist=expected_tolist,
                )

        with torch.no_grad():
            no_grad_result = torch.as_tensor([1.0])
        self.assert_default_tensor(
            no_grad_result,
            [1.0],
            shape=(1,),
            stride=(1,),
            tolist=[1.0],
        )

    def test_sequence_inputs_are_copied_into_fresh_storage(self):
        source = [1.0, 2.0]
        first = torch.as_tensor(source)
        second = torch.as_tensor(source)

        source[0] = 9.0
        self.assertEqual(first.tolist(), [1.0, 2.0])
        self.assertEqual(second.tolist(), [1.0, 2.0])
        self.assertIsNot(first, second)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_recursive_sequences_raise_value_error_without_crashing(self):
        script = r"""
import torch_rs as torch


def report(name, data):
    try:
        torch.as_tensor(data)
    except Exception as error:
        print(f"{name}|{type(error).__name__}|{error}")
    else:
        print(f"{name}|OK|")


self_referential = []
self_referential.append(self_referential)
report("self-referential list", self_referential)

mutual = []
mutual.append([mutual])
report("mutually recursive list", mutual)

empty_branch_cycle = []
empty_branch_cycle.append(empty_branch_cycle)
report("recursive empty branch", [[], empty_branch_cycle])

tuple_list = []
tuple_cycle = (tuple_list,)
tuple_list.append(tuple_cycle)
report("recursive tuple branch", tuple_cycle)

too_deep = 1.0
for _ in range(129):
    too_deep = [too_deep]
report("too many list dimensions", too_deep)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            completed.stdout.splitlines(),
            [
                "self-referential list|ValueError|too many dimensions 'list'",
                "mutually recursive list|ValueError|too many dimensions 'list'",
                "recursive empty branch|ValueError|too many dimensions 'list'",
                "recursive tuple branch|ValueError|too many dimensions 'tuple'",
                "too many list dimensions|ValueError|too many dimensions 'list'",
            ],
        )

    def test_identity_preserves_autograd_graph_and_gradient_object(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        result = torch.as_tensor(source, dtype=torch.float32, device="cpu")

        self.assertIs(result, source)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.output_nr, source.output_nr)

        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(torch.as_tensor(leaf.grad), gradient)

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.as_tensor

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "as_tensor")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.as_tensor"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertTrue(function.__doc__.startswith(FUNCTION_DOC_PREFIX))
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method as_tensor of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.as_tensor, function)
        self.assertIs(native.as_tensor, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(package.__all__.count("as_tensor"), 1)
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["as_tensor"], function)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.as_tensor, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.as_tensor, function)
        self.assertEqual(package.__all__.count("as_tensor"), 1)

    def test_torch_function_mode_dispatches_before_native_conversion(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        cases = (
            ("positional tensor", lambda: torch.as_tensor(tensor), (tensor,), None),
            (
                "keyword tensor",
                lambda: torch.as_tensor(data=tensor, dtype=torch.float32),
                (),
                {"data": tensor, "dtype": torch.float32},
            ),
            (
                "unsupported sequence",
                lambda: torch.as_tensor([1.0, 2.0]),
                ([1.0, 2.0],),
                None,
            ),
            (
                "unsupported device string",
                lambda: torch.as_tensor([1.0], device="cuda"),
                ([1.0],),
                {"device": "cuda"},
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode()
            with mode:
                result = call()
            with self.subTest(case=case):
                self.assertIs(result, marker)
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, torch.as_tensor)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.as_tensor(data=tensor, dtype=torch.float32)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError, r"^Multiple dispatch failed for 'torch\.as_tensor'"
        ):
            with DecliningMode():
                torch.as_tensor(tensor)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_binding_errors_and_unsupported_scope_are_explicit(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        unsupported_conversion = (
            "as_tensor(): only exact native CPU float32 Tensor inputs, Python real scalars, and rectangular list/tuple inputs are supported; "
            "NumPy arrays, buffers, dtype conversions, CUDA/meta/indexed CPU devices, copy, pinned memory, tensor subclasses, and __torch_function__ argument dispatch are not implemented"
        )
        cases = (
            (
                lambda: torch.as_tensor(),
                TypeError,
                'as_tensor() missing 1 required positional arguments: "data"',
            ),
            (
                lambda: torch.as_tensor(tensor, tensor),
                TypeError,
                "as_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.as_tensor(tensor, data=tensor),
                TypeError,
                "as_tensor() got multiple values for argument 'data'",
            ),
            (
                lambda: torch.as_tensor(tensor, out=None),
                TypeError,
                "as_tensor() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.as_tensor(tensor, pin_memory=False),
                TypeError,
                "as_tensor() got an unexpected keyword argument 'pin_memory'",
            ),
            (
                lambda: torch.as_tensor(tensor, copy=False),
                TypeError,
                "as_tensor() got an unexpected keyword argument 'copy'",
            ),
            (
                lambda: torch.as_tensor(tensor, dtype=1),
                TypeError,
                "as_tensor(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: torch.as_tensor(tensor, device=1.5),
                TypeError,
                "as_tensor(): argument 'device' must be torch.device, not float",
            ),
            (
                lambda: torch.as_tensor(tensor, device=""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.as_tensor(tensor, device="banana"),
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, "
                "ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, "
                "mtia, privateuseone device type at start of device string: banana",
            ),
            (
                lambda: torch.as_tensor(tensor, device="cuda"),
                RuntimeError,
                "as_tensor(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.as_tensor(tensor, device="cpu:0"),
                NotImplementedError,
                "as_tensor(): explicit indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.as_tensor(tensor, device=torch.device("cpu", 1)),
                NotImplementedError,
                "as_tensor(): indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: torch.as_tensor(np.asarray([1.0], dtype=np.float32)),
                NotImplementedError,
                unsupported_conversion,
            ),
            (
                lambda: torch.as_tensor(memoryview(np.asarray([1.0], dtype=np.float32))),
                NotImplementedError,
                unsupported_conversion,
            ),
            (
                lambda: torch.as_tensor(np.float32(1.0)),
                NotImplementedError,
                unsupported_conversion,
            ),
            (lambda: torch.as_tensor(True), NotImplementedError, unsupported_conversion),
            (
                lambda: torch.as_tensor([[1.0], [2.0, 3.0]]),
                ValueError,
                "expected sequence of length 1 at dim 1 (got 2)",
            ),
            (
                lambda: torch.as_tensor([1.0, [2.0]]),
                TypeError,
                "must be real number, not list",
            ),
            (
                lambda: torch.as_tensor([[1.0], 2.0]),
                TypeError,
                "not a sequence",
            ),
            (
                lambda: torch.as_tensor([[], object()]),
                RuntimeError,
                "Could not infer dtype of object",
            ),
            (
                lambda: torch.as_tensor([[[]], [[object()]]]),
                RuntimeError,
                "Could not infer dtype of object",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)

        class ListSubclass(list):
            pass

        self.assert_error(
            lambda: torch.as_tensor(ListSubclass([1.0])),
            NotImplementedError,
            unsupported_conversion,
        )

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return object()

        self.assert_error(
            lambda: torch.as_tensor(Override()),
            NotImplementedError,
            unsupported_conversion,
        )
        self.assertEqual(Override.calls, [])
        self.assertFalse(hasattr(torch, "float64"))


if __name__ == "__main__":
    unittest.main()
