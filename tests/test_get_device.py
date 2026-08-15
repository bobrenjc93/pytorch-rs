import inspect
import pickle
import re
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\nget_device() -> Device ordinal (Integer)\n\n"
    "For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.\n"
    "For CPU tensors, this function returns `-1`.\n\n"
    "Example::\n\n"
    "    >>> x = torch.randn(3, 4, 5, device='cuda:0')\n"
    "    >>> x.get_device()\n"
    "    0\n"
    "    >>> x.cpu().get_device()\n"
    "    -1\n"
)


class GetDeviceTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        offset_view = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            ("scalar", torch.tensor(3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", leaf.grad),
        )

    def test_cpu_tensors_use_device_metadata_without_materializing_values(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                results = (
                    tensor.get_device(),
                    torch.get_device(tensor),
                    torch.get_device(input=tensor),
                    torch.get_device(x=tensor),
                    torch.get_device(a=tensor),
                )
                self.assertEqual(results, (-1, -1, -1, -1, -1))
                self.assertTrue(all(type(result) is int for result in results))
                self.assertIsNone(tensor.device.index)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_top_level_queries_do_not_mutate_a_pending_autograd_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)

        self.assertEqual(
            (
                torch.get_device(tracked),
                torch.get_device(input=tracked),
                torch.get_device(x=tracked),
                torch.get_device(a=tracked),
            ),
            (-1, -1, -1, -1),
        )
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_tensorbase_descriptor_documentation_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "get_device")
        bound = tensor.get_device

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object, python_313_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            self.assertEqual(callable_object.__name__, "get_device")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)),
                    python_313_signature,
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        self.assertEqual(descriptor.__qualname__, "TensorBase.get_device")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor(tensor), -1)

    def test_top_level_callable_metadata_and_exports_match_pytorch(self):
        function = torch.get_device
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "get_device")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.get_device"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method get_device of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.get_device, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertEqual(torch.__all__.count("get_device"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["get_device"], function)

    def test_no_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "get_device")
        bound = tensor.get_device
        calls = (
            (
                lambda: tensor.get_device(1),
                "TensorBase.get_device() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1, 2),
                "Tensor.get_device() takes no arguments (2 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.get_device() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.get_device(dim=0),
                (
                    "Tensor.get_device() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.get_device() takes no keyword arguments"
                ),
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.get_device() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.get_device() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'get_device' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_top_level_binding_and_type_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.get_device(),
                'get_device() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.get_device(tensor, tensor),
                "get_device() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.get_device(tensor, input=tensor),
                "get_device() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.get_device(tensor, x=tensor),
                "get_device() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.get_device(tensor, a=tensor),
                "get_device() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.get_device(input=tensor, a=tensor),
                "get_device() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.get_device(foo=tensor),
                'get_device() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.get_device(tensor, extra=True),
                "get_device() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.get_device(1),
                "get_device(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.get_device(input=[]),
                "get_device(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.get_device(x=None),
                "get_device(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.get_device(a=1),
                "get_device(): argument 'input' must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
