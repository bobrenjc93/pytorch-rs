import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


LOG_DOC = """
log() -> Tensor

See :func:`torch.log`
"""

TOP_LEVEL_LOG_DOC = """
log(input, *, out=None) -> Tensor

Returns a new tensor with the natural logarithm of the elements
of :attr:`input`.

.. math::
    y_{i} = \\log_{e}(\\text{input}_{i})

The current native implementation supports exact CPU ``float32`` tensors,
including first-order autograd for tracked inputs. Concrete output tensors,
in-place ``log_``, dtype/device expansion, tensor subclasses, and higher-order
gradients remain unsupported.

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.log(torch.tensor([1., math.e]))
    tensor([ 0.,  1.])
"""

SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EAA_AAAB,
        0xBEAA_AAAB,
        0x3F80_0000,
        0xBF80_0000,
        0x4000_0000,
        0xC000_0000,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7F81_2345,
        0xFF81_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)

SPECIAL_OUTPUT_BITS = np.asarray(
    (
        0xFF80_0000,
        0xFF80_0000,
        0xC2CE_8ED0,
        0x7FC0_0000,
        0xC2AE_AC50,
        0x7FC0_0000,
        0xC2AE_AC50,
        0x7FC0_0000,
        0xBF8C_9F54,
        0x7FC0_0000,
        0x0000_0000,
        0x7FC0_0000,
        0x3F31_7218,
        0x7FC0_0000,
        0x42B1_7218,
        0x7FC0_0000,
        0x7F80_0000,
        0x7FC0_0000,
        0x7FC1_2345,
        0xFFC1_2345,
        0x7FC1_2345,
        0xFFC5_4321,
    ),
    dtype=np.uint32,
)


def tensor_bits(tensor):
    return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)


def make_cases(module):
    base = module.tensor(
        np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
        dtype=module.float32,
    )
    strided = base.transpose(0, 2)
    return (
        ("scalar", module.tensor(1.0, dtype=module.float32), ()),
        (
            "empty offset",
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            (2, 1),
        ),
        ("empty singleton trailing", module.zeros((0, 1)), (1, 1)),
        ("contiguous", base, base.stride()),
        ("offset", strided[1], (1, 3)),
        ("noncontiguous", strided, (1, 4, 12)),
        (
            "numerical edges",
            module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            (1,),
        ),
    )


class TensorLogTests(unittest.TestCase):
    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.log(source)),
            ("input", lambda: torch.log(input=source)),
            ("x", lambda: torch.log(x=source)),
            ("a", lambda: torch.log(a=source)),
            ("x1", lambda: torch.log(x1=source)),
            ("out none", lambda: torch.log(source, out=None)),
            ("alias and out none", lambda: torch.log(x=source, out=None)),
        )

    def assert_untracked_result(self, output, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(output.shape, source.shape)
            self.assertEqual(output.stride(), expected_stride)
            self.assertEqual(output.storage_offset(), 0)
            self.assertFalse(output.requires_grad)
            self.assertTrue(output.is_leaf)
            self.assertIs(output.dtype, torch.float32)
            self.assertEqual(output.device, torch.device("cpu"))
            self.assertFalse(output.is_set_to(source))
            if source.numel():
                self.assertNotEqual(output.data_ptr(), source.data_ptr())

    def assert_matches_tensor(self, actual, expected, *, case):
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
            np.testing.assert_array_equal(tensor_bits(actual), tensor_bits(expected))

    @staticmethod
    def make_autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(2.0, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None

        leaf = torch.tensor(
            np.linspace(0.25, 6.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        if case == "offset":
            source = leaf[1]
            weights = torch.tensor(
                np.linspace(-2.0, 2.0, 12, dtype=np.float32)
                .reshape(3, 4)
                .tolist()
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)[1]
            weights = torch.tensor(
                np.linspace(-2.0, 2.0, 6, dtype=np.float32)
                .reshape(3, 2)
                .tolist()
            )
            return leaf, source, weights
        raise AssertionError(f"unknown log autograd case: {case}")

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            source_bits = tensor_bits(source).copy()
            output = source.log()
            self.assert_untracked_result(
                output, source, expected_stride, case=case
            )

            if case == "numerical edges":
                expected_bits = SPECIAL_OUTPUT_BITS
            else:
                with np.errstate(all="ignore"):
                    expected_bits = (
                        np.log(np.asarray(source, dtype=np.float32))
                        .astype(np.float32)
                        .reshape(-1)
                        .view(np.uint32)
                    )
            with self.subTest(case=case, values=True):
                if case == "numerical edges":
                    np.testing.assert_array_equal(
                        tensor_bits(output), expected_bits
                    )
                else:
                    np.testing.assert_allclose(
                        np.asarray(output, dtype=np.float32),
                        expected_bits.view(np.float32).reshape(source.shape),
                        rtol=2.0e-6,
                        atol=np.nextafter(np.float32(0), np.float32(1)),
                        equal_nan=True,
                    )
            with self.subTest(case=case, input=True):
                np.testing.assert_array_equal(tensor_bits(source), source_bits)

    def test_top_level_calls_reuse_tensor_log_path(self):
        for case, source, expected_stride in make_cases(torch):
            expected = source.log()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_untracked_result(
                    actual, source, expected_stride, case=(case, form)
                )
                with self.subTest(case=case, form=form, values=True):
                    np.testing.assert_array_equal(
                        tensor_bits(actual), tensor_bits(expected)
                    )

    def test_autograd_scalar_empty_offset_and_noncontiguous_reuse_tensor_log_vjp(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                function_leaf, function_source, function_weights = (
                    self.make_autograd_case(case)
                )
                method_leaf, method_source, method_weights = self.make_autograd_case(
                    case
                )
                output = dict(self.top_level_calls(function_source))[form]()
                method_output = method_source.log()

                self.assert_matches_tensor(
                    output, method_output, case=(case, form, "output")
                )
                if function_weights is None:
                    function_loss = output if case == "scalar" else output.sum()
                    method_loss = (
                        method_output if case == "scalar" else method_output.sum()
                    )
                else:
                    function_loss = (output * function_weights).sum()
                    method_loss = (method_output * method_weights).sum()
                function_loss.backward()
                method_loss.backward()
                self.assert_matches_tensor(
                    function_leaf.grad,
                    method_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_special_value_gradients_reuse_tensor_log_vjp_bitwise(self):
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        function_leaf = torch.tensor(
            memoryview(SPECIAL_INPUT_BITS.view(np.float32)), requires_grad=True
        )
        method_leaf = torch.tensor(
            memoryview(SPECIAL_INPUT_BITS.view(np.float32)), requires_grad=True
        )
        function_output = torch.log(function_leaf, out=None)
        method_output = method_leaf.log()
        self.assert_matches_tensor(
            function_output, method_output, case="special forward"
        )

        weights = torch.tensor(memoryview(weight_bits.view(np.float32)))
        (function_output * weights).sum().backward()
        (method_output * weights).sum().backward()
        self.assert_matches_tensor(
            function_leaf.grad, method_leaf.grad, case="special gradient"
        )

    def test_accumulation_graph_freeing_no_grad_and_detach_reuse_method_path(self):
        values = [[0.25, 1.0, 2.0], [4.0, 8.0, 16.0]]
        weights = [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]
        function_leaf = torch.tensor(values, requires_grad=True)
        method_leaf = torch.tensor(values, requires_grad=True)
        function_source = function_leaf.transpose(0, 1)
        method_source = method_leaf.transpose(0, 1)

        function_loss = (
            torch.log(function_source, out=None) * torch.tensor(weights)
        ).sum()
        method_loss = (method_source.log() * torch.tensor(weights)).sum()
        function_loss.backward()
        method_loss.backward()
        self.assert_matches_tensor(
            function_leaf.grad, method_leaf.grad, case="first gradient"
        )

        torch.log(input=function_source).sum().backward()
        method_source.log().sum().backward()
        self.assert_matches_tensor(
            function_leaf.grad, method_leaf.grad, case="accumulated gradient"
        )

        with self.assertRaises(RuntimeError) as function_raised:
            function_loss.backward()
        with self.assertRaises(RuntimeError) as method_raised:
            method_loss.backward()
        self.assertEqual(str(function_raised.exception), str(method_raised.exception))

        no_grad_function_leaf = torch.tensor(values, requires_grad=True)
        no_grad_source = no_grad_function_leaf.transpose(0, 1)[1]
        source_bits = tensor_bits(no_grad_source).copy()
        detached = no_grad_source.detach()
        expected = detached.log()
        with torch.no_grad():
            actual_method = no_grad_source.log()
            actual_function = torch.log(no_grad_source, out=None)

            extreme = torch.zeros((0,), requires_grad=True).reshape(
                (0, sys.maxsize, 3)
            )
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.log()
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.log(extreme)

        for label, actual in (
            ("method no_grad", actual_method),
            ("top-level no_grad", actual_function),
            ("detached", detached.log()),
        ):
            with self.subTest(label=label):
                self.assert_untracked_result(
                    actual, detached, expected.stride(), case=label
                )
                np.testing.assert_array_equal(
                    tensor_bits(actual), tensor_bits(expected)
                )
        np.testing.assert_array_equal(tensor_bits(no_grad_source), source_bits)
        self.assertIsNone(no_grad_function_leaf.grad)
        self.assertTrue(torch.log(no_grad_function_leaf).requires_grad)

        higher_order_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        higher_order_loss = torch.log(higher_order_leaf).sum()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)

    def test_tensorbase_descriptor_metadata_errors_copy_pickle_and_mode_dispatch(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        plain = tensor.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "log")
        bound = plain.log

        self.assertIs(torch.Tensor.log, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'log' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "log")
        self.assertEqual(descriptor.__qualname__, "TensorBase.log")
        self.assertEqual(bound.__name__, "log")
        self.assertEqual(bound.__qualname__, "Tensor.log")
        self.assertEqual(descriptor.__doc__, LOG_DOC)
        self.assertEqual(bound.__doc__, LOG_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: plain.log(1), "TensorBase.log() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.log() takes no arguments (1 given)"),
            (
                lambda: descriptor(plain, 1),
                "TensorBase.log() takes no arguments (1 given)",
            ),
            (
                lambda: plain.log(1, 2),
                "TensorBase.log() takes no arguments (2 given)",
            ),
            (
                lambda: plain.log(input=plain),
                (
                    "Tensor.log() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.log() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.log() takes no keyword arguments"),
            (
                lambda: descriptor(plain, unexpected=True),
                "TensorBase.log() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.log() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'log' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=plain),
                "unbound method TensorBase.log() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
        self.assertIs(copy.copy(bound), bound)
        self.assertIs(copy.deepcopy(bound), bound)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, pickler="pickle"):
                self.assertIs(
                    pickle.loads(pickle.dumps(descriptor, protocol)),
                    descriptor,
                )
            with self.subTest(protocol=protocol, pickler="ForkingPickler"):
                self.assertIs(
                    pickle.loads(ForkingPickler.dumps(descriptor, protocol)),
                    descriptor,
                )

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
            result = tensor.log()
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.log()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.0])

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.log()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.requires_grad)
        self.assertFalse(forwarded.is_leaf)
        forwarded.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [1.0])

    def test_top_level_out_binding_import_wildcard_reload_copy_and_pickle(self):
        source = torch.tensor([1.0, np.e], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_bits = tensor_bits(destination).copy()
        for form, call in (
            ("positional", lambda: torch.log(source, out=destination)),
            ("keyword", lambda: torch.log(input=source, out=destination)),
            ("alias", lambda: torch.log(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^log\(\): the 'out' argument is not supported$",
                ):
                    call()
                np.testing.assert_array_equal(
                    tensor_bits(destination), destination_bits
                )

        function = torch.log
        self.assertIs(function, torch._C.log)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "log")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.log")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_LOG_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method log of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.log, function)
        for action in (
            lambda: setattr(owner, "log", None),
            lambda: delattr(owner, "log"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.log, function)

        imported = importlib.import_module("torch_rs").log
        self.assertIs(imported, function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["log"], function)
        self.assertEqual(torch.__all__.count("log"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.log, function)
        self.assertIs(torch._C.log, function)

    def test_binding_type_and_unsupported_extension_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.log(),
                'log() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.log(tensor, tensor),
                "log() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.log(tensor, input=tensor),
                "log() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.log(out=tensor),
                'log() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.log(extra=tensor),
                'log() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.log(1, extra=True),
                "log(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.log(input=[]),
                "log(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.log(tensor, out=[]),
                "log(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.log(tensor, extra=True, out=[]),
                "log(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.log(tensor, extra=True),
                "log() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.log(input=tensor, a=tensor),
                "log() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.log(a=tensor, x=tensor, out=None),
                "log() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.log(x=tensor, a=tensor, out=None),
                "log() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.log(tensor, dtype=torch.float32),
                "log() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.log(tensor, device=torch.device("cpu")),
                "log() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.log(np.zeros((2, 3), dtype=np.float32)),
                "log(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertTrue(hasattr(torch, "log"))
        self.assertIn("log", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "log_"))
        self.assertFalse(hasattr(tensor, "log_"))
        self.assertFalse(hasattr(torch, "log_"))
        self.assertNotIn("log_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.log(out=None)
        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^tensor\(\): argument 'dtype' must be torch.dtype, not object$",
        ):
            torch.tensor([1.0], dtype=object()).log()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda").log()

    def test_top_level_modes_and_overrides_dispatch_before_native_boundaries(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.log(input=tensor, out=destination), marker)
        self.assertEqual(
            mode.calls,
            [(torch.log, (), (), {"input": tensor, "out": destination})],
        )

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.log(Override()), marker)
        self.assertIs(torch.log(torch.tensor([1.0]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.log)
            self.assertEqual(dispatch_types, (Override,))

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = tensor.detach()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.log(input=plain, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
