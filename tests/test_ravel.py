import gc
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


FUNCTION_DOC = (
    "\nravel(input) -> Tensor\n\n"
    "Return a contiguous flattened tensor. A copy is made only if needed.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> t = torch.tensor([[[1, 2],\n"
    "    ...                    [3, 4]],\n"
    "    ...                   [[5, 6],\n"
    "    ...                    [7, 8]]])\n"
    "    >>> torch.ravel(t)\n"
    "    tensor([1, 2, 3, 4, 5, 6, 7, 8])\n"
)


class TensorRavelTests(unittest.TestCase):
    def assert_tensor(self, tensor, values, *, shape, stride, offset=0):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), offset)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        np.testing.assert_array_equal(
            np.asarray(tensor), np.asarray(values, dtype=np.float32).reshape(shape)
        )

    def test_scalar_vector_ordinary_and_empty_inputs_return_new_vectors(self):
        cases = (
            (torch.tensor(-0.0), [-0.0], (1,)),
            (torch.tensor([1.0, 2.0, 3.0]), [1.0, 2.0, 3.0], (1,)),
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                [1.0, 2.0, 3.0, 4.0],
                (1,),
            ),
            (torch.zeros((2, 0, 3)), [], (1,)),
        )
        for source, values, stride in cases:
            with self.subTest(shape=source.shape):
                output = source.ravel()
                self.assertIsNot(output, source)
                self.assert_tensor(
                    output,
                    values,
                    shape=(source.numel(),),
                    stride=stride,
                    offset=source.storage_offset(),
                )

        scalar_bits = np.asarray(cases[0][0].ravel()).view(np.uint32).item()
        self.assertEqual(scalar_bits, 0x8000_0000)

    def test_contiguous_offsets_alias_and_strided_inputs_materialize(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())

        offset_matrix = source[1]
        offset_ravel = offset_matrix.ravel()
        self.assertIsNot(offset_ravel, offset_matrix)
        self.assert_tensor(
            offset_ravel,
            values[1].reshape(-1),
            shape=(12,),
            stride=(1,),
            offset=12,
        )

        strided_vector = source.transpose(0, 2)[0][0]
        self.assertEqual(strided_vector.stride(), (12,))
        packed_vector = strided_vector.ravel()
        self.assert_tensor(
            packed_vector,
            values[:, 0, 0],
            shape=(2,),
            stride=(1,),
        )

        transposed = source.transpose(0, 2)
        packed = transposed.ravel()
        self.assert_tensor(
            packed,
            values.transpose(2, 1, 0).reshape(-1),
            shape=(24,),
            stride=(1,),
        )

        singleton_source = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        singleton = singleton_source.transpose(0, 1)[2]
        self.assertEqual(singleton.stride(), (4,))
        singleton_ravel = singleton.ravel()
        self.assert_tensor(
            singleton_ravel,
            [2.0],
            shape=(1,),
            stride=(4,),
            offset=2,
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        empty_ravel = empty.ravel()
        self.assert_tensor(
            empty_ravel,
            [],
            shape=(0,),
            stride=(1,),
            offset=1,
        )

        def ravel_after_source_drops():
            temporary = torch.tensor(values.tolist())
            return temporary.transpose(0, 2).ravel()

        surviving_copy = ravel_after_source_drops()
        gc.collect()
        np.testing.assert_array_equal(
            np.asarray(surviving_copy), values.transpose(2, 1, 0).reshape(-1)
        )

    def test_autograd_and_no_grad_follow_view_or_copy_behavior(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = leaf.transpose(0, 1).ravel()
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
        )

        scalar = torch.tensor(2.0, requires_grad=True)
        (scalar.ravel() * 7.0).sum().backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.ravel().sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        non_contiguous = no_grad_source.transpose(0, 1)
        with torch.no_grad():
            alias = no_grad_source.ravel()
            copied = non_contiguous.ravel()
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertFalse(copied.requires_grad)
        self.assertTrue(copied.is_leaf)

    def test_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "ravel")
        bound = tensor.ravel
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "ravel")
        self.assertEqual(bound.__name__, "ravel")
        self.assertEqual(
            descriptor.__doc__, "\nravel() -> Tensor\n\nsee :func:`torch.ravel`\n"
        )
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        output = descriptor(tensor)
        self.assertIsNot(output, tensor)
        self.assertEqual(output.shape, (6,))

        calls = (
            (lambda: tensor.ravel(1), "Tensor.ravel() takes no arguments (1 given)"),
            (lambda: tensor.ravel(1, 2), "Tensor.ravel() takes no arguments (2 given)"),
            (lambda: tensor.ravel(dim=0), "Tensor.ravel() takes no keyword arguments"),
            (lambda: descriptor(), "unbound method Tensor.ravel() needs an argument"),
            (
                lambda: descriptor(tensor, 1),
                "Tensor.ravel() takes no arguments (1 given)",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaises(TypeError):
            descriptor([1.0])


class TopLevelRavelTests(unittest.TestCase):
    def assert_matches_method(self, result, source, expected):
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.stride(), expected.stride())
        self.assertEqual(result.storage_offset(), expected.storage_offset())
        self.assertEqual(result.is_contiguous(), expected.is_contiguous())
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result.requires_grad, expected.requires_grad)
        self.assertEqual(result.is_leaf, expected.is_leaf)
        self.assertEqual(
            result.data_ptr() == source.data_ptr(),
            expected.data_ptr() == source.data_ptr(),
        )
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def ravel_calls(self, source):
        return (
            ("positional", torch.ravel(source)),
            ("input", torch.ravel(input=source)),
            ("x", torch.ravel(x=source)),
            ("a", torch.ravel(a=source)),
            ("x1", torch.ravel(x1=source)),
        )

    def test_call_forms_delegate_all_layouts_to_tensor_ravel(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        singleton_base = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("vector", base[0][1]),
            ("ordinary", base),
            ("offset", base[1]),
            ("transpose", base.transpose(0, 2)),
            ("strided-vector", base.transpose(0, 2)[0][0]),
            ("singleton-stride", singleton_base.transpose(0, 1)[2]),
            ("empty-offset", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
        )
        for case, source in cases:
            expected = source.ravel()
            for form, result in self.ravel_calls(source):
                with self.subTest(case=case, form=form):
                    self.assert_matches_method(result, source, expected)

    def test_autograd_no_grad_and_source_lifetime_use_native_ravel(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        output = torch.ravel(leaf.transpose(0, 1))
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]],
        )

        scalar = torch.tensor(2.0, requires_grad=True)
        (torch.ravel(input=scalar) * 7.0).sum().backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.ravel(a=empty).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            alias = torch.ravel(no_grad_source)
            copied = torch.ravel(no_grad_source.transpose(0, 1))
        self.assertEqual((alias.requires_grad, alias.is_leaf), (True, True))
        self.assertEqual((copied.requires_grad, copied.is_leaf), (False, True))

        def outputs_after_source_drops():
            source_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            source = torch.tensor(source_values.tolist())
            return torch.ravel(source[1]), torch.ravel(source.transpose(0, 2))

        surviving_alias, surviving_copy = outputs_after_source_drops()
        gc.collect()
        expected_alias = np.arange(12, 24, dtype=np.float32)
        np.testing.assert_array_equal(
            np.asarray(surviving_alias), expected_alias
        )
        np.testing.assert_array_equal(
            np.asarray(surviving_copy),
            np.arange(24, dtype=np.float32)
            .reshape(2, 3, 4)
            .transpose(2, 1, 0)
            .reshape(-1),
        )

    def test_callable_metadata_documentation_exports_and_pickling(self):
        function = torch.ravel
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "ravel")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.ravel")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method ravel of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.ravel, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("ravel"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["ravel"], function)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.ravel(),
                'ravel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ravel(tensor, tensor),
                "ravel() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.ravel(tensor, input=tensor),
                "ravel() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.ravel(tensor, extra=True, input=tensor),
                "ravel() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.ravel(tensor, input=tensor, extra=True),
                "ravel() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.ravel(extra=tensor),
                'ravel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.ravel(1, extra=True),
                "ravel(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.ravel(input=[]),
                "ravel(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.ravel(a=1),
                "ravel(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.ravel(x=[]),
                "ravel(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.ravel(x1=None),
                "ravel(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.ravel(a=tensor, x=tensor),
                "ravel() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.ravel(x=tensor, a=tensor),
                "ravel() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.ravel(input=tensor, x1=tensor),
                "ravel() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_and_overrides_receive_the_top_level_builtin(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        for keyword in (None, "input", "x", "a", "x1"):
            mode = RecordingMode()
            with mode:
                result = (
                    torch.ravel(tensor)
                    if keyword is None
                    else torch.ravel(**{keyword: tensor})
                )
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.ravel)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,) if keyword is None else ())
            self.assertEqual(kwargs, None if keyword is None else {keyword: tensor})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.ravel(a=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor.ravel()))

        calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.ravel(x=value), marker)
        function, dispatch_types, args, kwargs = calls[0]
        self.assertIs(function, torch.ravel)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"x": value})


if __name__ == "__main__":
    unittest.main()
