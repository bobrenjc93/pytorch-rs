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
    y_{i} = \\log_{e} (x_{i})


Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.rand(5) * 5
    >>> a
    tensor([4.7767, 4.3234, 1.2156, 0.2411, 4.5739])
    >>> torch.log(a)
    tensor([ 1.5637,  1.4640,  0.1952, -1.4226,  1.5204])
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


def make_cases(module):
    base = module.tensor(
        np.linspace(0.25, 24.0, 24, dtype=np.float32)
        .reshape(2, 3, 4)
        .tolist(),
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
        ("contiguous", base, (12, 4, 1)),
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
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

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

    def assert_log_result(
        self, output, source, expected_stride, expected_values, *, case, exact_bits=False
    ):
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
        with self.subTest(case=case, values=True):
            if exact_bits:
                np.testing.assert_array_equal(
                    self.tensor_bits(output), expected_values
                )
            else:
                np.testing.assert_allclose(
                    np.asarray(output, dtype=np.float32),
                    expected_values,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            if case == "numerical edges":
                expected_values = SPECIAL_OUTPUT_BITS
                exact_bits = True
            else:
                with np.errstate(all="ignore"):
                    expected_values = (
                        np.log(np.asarray(source, dtype=np.float32))
                        .astype(np.float32)
                    )
                exact_bits = False
            source_bits = self.tensor_bits(source).copy()

            output = source.log()
            self.assert_log_result(
                output,
                source,
                expected_stride,
                expected_values,
                case=(case, "method"),
                exact_bits=exact_bits,
            )
            np.testing.assert_array_equal(self.tensor_bits(source), source_bits)

    def test_top_level_calls_reuse_tensor_log_bits_layouts_and_storage(self):
        for case, source, expected_stride in make_cases(torch):
            expected = source.log()
            source_bits = self.tensor_bits(source).copy()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_log_result(
                    actual,
                    source,
                    expected_stride,
                    np.asarray(expected, dtype=np.float32)
                    if case != "numerical edges"
                    else self.tensor_bits(expected),
                    case=(case, form),
                    exact_bits=case == "numerical edges",
                )
            np.testing.assert_array_equal(self.tensor_bits(source), source_bits)

    def test_active_autograd_rejected_before_layout_planning(self):
        leaf = torch.tensor(
            [[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        source_bits = self.tensor_bits(source).copy()

        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): autograd recording is not supported$",
        ):
            source.log()

        for form, call in self.top_level_calls(source):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^log\(\): autograd recording is not supported$",
                ):
                    call()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): autograd recording is not supported$",
        ):
            extreme.log()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): autograd recording is not supported$",
        ):
            torch.log(extreme)

        np.testing.assert_array_equal(self.tensor_bits(source), source_bits)
        self.assertIsNone(leaf.grad)

    def test_no_grad_and_detached_tracked_inputs_use_inference_path(self):
        leaf = torch.tensor(
            [[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]
        expected = source.detach().log()

        with torch.no_grad():
            method_output = source.log()
            function_output = torch.log(input=source, out=None)

        for case, output in (
            ("method", method_output),
            ("top-level", function_output),
            ("detached", source.detach().log()),
        ):
            self.assert_log_result(
                output,
                source,
                expected.stride(),
                np.asarray(expected, dtype=np.float32),
                case=case,
            )
        self.assertIsNone(leaf.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.log()
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                torch.log(extreme)

    def test_tensorbase_descriptor_metadata_copying_pickling_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "log")
        bound = tensor.log

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

        self.assertIs(copy.copy(descriptor), descriptor)
        self.assertIs(copy.deepcopy(descriptor), descriptor)
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

        cases = (
            (lambda: tensor.log(1), "TensorBase.log() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.log() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.log() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.log(1, 2),
                "TensorBase.log() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.log(input=tensor),
                (
                    "Tensor.log() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.log() takes no keyword arguments"
                ),
            ),
            (
                lambda: tensor.log(out=None),
                (
                    "Tensor.log() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.log() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.log() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.log() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.log() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'log' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.log() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_boundaries(self):
        tracked = torch.tensor([1.0], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "log")
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
            result = tracked.log()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
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
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): autograd recording is not supported$",
        ):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tracked.log()
        self.assertEqual(order, ["upper", "lower"])

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.log(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        expected_bits = self.tensor_bits(destination).copy()
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
                self.assertEqual(destination.data_ptr(), destination_pointer)
                np.testing.assert_array_equal(
                    self.tensor_bits(destination), expected_bits
                )
                self.assertIsNone(source.grad)

    def test_top_level_modes_and_overrides_observe_original_call(self):
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

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(torch.log(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = tensor.detach()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.log(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [0.0])

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.log(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid:
                torch.log(tensor, tensor)
        self.assertEqual(invalid.calls, [])

    def test_top_level_builtin_metadata_exports_reload_copying_and_pickling(self):
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

        imported_module = importlib.import_module("torch_rs")
        self.assertIs(imported_module.log, function)
        from torch_rs import log as imported_log

        self.assertIs(imported_log, function)
        self.assertEqual(torch.__all__.count("log"), 1)
        self.assertNotIn("log_", torch.__all__)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["log"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )
                self.assertIs(
                    pickle.loads(ForkingPickler.dumps(function, protocol)),
                    function,
                )

        namespace = imported_module.__dict__
        reloaded = importlib.reload(imported_module)
        self.assertIs(reloaded, imported_module)
        self.assertIs(imported_module.__dict__, namespace)
        self.assertIs(imported_module.log, function)

    def test_binding_type_and_unsupported_surface_errors(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
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
                lambda: torch.log(np.ones((2, 3), dtype=np.float32)),
                "log(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        for form, call in (
            ("positional", lambda: torch.log(tensor, out=destination)),
            ("keyword", lambda: torch.log(input=tensor, out=destination)),
            ("alias", lambda: torch.log(x=tensor, out=destination)),
        ):
            with self.subTest(form=form, boundary="out"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^log\(\): the 'out' argument is not supported$",
                ):
                    call()

        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(torch.Tensor, "log"))
        self.assertFalse(hasattr(torch.Tensor, "log_"))
        self.assertFalse(hasattr(tensor, "log_"))
        self.assertFalse(hasattr(torch, "log_"))
        with self.assertRaisesRegex(
            TypeError, "type 'torch_rs.Tensor' is not an acceptable base type"
        ):
            class TensorSubclass(torch.Tensor):
                pass

        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="cuda")
        with self.assertRaises(TypeError):
            tensor.log(out=None)


if __name__ == "__main__":
    unittest.main()
