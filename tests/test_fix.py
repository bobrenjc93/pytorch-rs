import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
    from .test_trunc import SPECIAL_OUTPUT_BITS, make_cases
else:
    from signature_utils import assert_no_argument_signature
    from test_trunc import SPECIAL_OUTPUT_BITS, make_cases


TENSOR_FIX_DOC = """
fix() -> Tensor

See :func:`torch.fix`.
"""

TOP_LEVEL_FIX_DOC = """
fix(input, *, out=None) -> Tensor

Alias for :func:`torch.trunc`
"""


class TensorFixTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    def assert_matches(self, actual, expected, *, case):
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
                self.tensor_bits(actual), self.tensor_bits(expected)
            )

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty = empty_leaf.transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        channels_last_leaf = torch.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            requires_grad=True,
        )
        channels_last = channels_last_leaf.contiguous(
            memory_format=torch.channels_last
        )
        channels_last_3d_leaf = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            requires_grad=True,
        )
        channels_last_3d = channels_last_3d_leaf.contiguous(
            memory_format=torch.channels_last_3d
        )
        return (
            (scalar, scalar),
            (empty_leaf, empty),
            (leaf, strided[1]),
            (leaf, strided),
            (channels_last_leaf, channels_last),
            (channels_last_3d_leaf, channels_last_3d),
        )

    @staticmethod
    def supported_calls(source):
        return (
            ("method", source.fix),
            ("positional", lambda: torch.fix(source)),
            ("input", lambda: torch.fix(input=source)),
            ("x", lambda: torch.fix(x=source)),
            ("a", lambda: torch.fix(a=source)),
            ("x1", lambda: torch.fix(x1=source)),
            ("out none", lambda: torch.fix(source, out=None)),
            ("alias and out none", lambda: torch.fix(x=source, out=None)),
        )

    def test_values_ieee_bits_layouts_aliases_and_fresh_storage(self):
        for case, source, expected_stride in make_cases(torch):
            expected = source.trunc()
            for form, call in self.supported_calls(source):
                output = call()
                self.assert_matches(output, expected, case=(case, form))
                self.assertEqual(output.stride(), expected_stride)
                self.assertFalse(output.is_set_to(source))
                self.assertFalse(output.is_set_to(expected))
                if source.numel():
                    self.assertNotEqual(output.data_ptr(), source.data_ptr())
                if case == "numerical edges":
                    np.testing.assert_array_equal(
                        self.tensor_bits(output), SPECIAL_OUTPUT_BITS
                    )

    def test_autograd_records_reusable_trunc_backward_for_every_call_form(self):
        for case, (leaf, source) in enumerate(self.make_tracked_cases()):
            expected = source.trunc()
            for form, call in self.supported_calls(source):
                with self.subTest(case=case, form=form):
                    output = call()
                    self.assert_matches(output, expected, case=(case, form))
                    self.assertTrue(output.requires_grad)
                    self.assertFalse(output.is_leaf)
                    self.assertFalse(output.is_set_to(source))
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            output
                        ),
                        ", grad_fn=<TruncBackward0>",
                    )

                    loss = output if output.numel() == 1 else output.sum()
                    loss.backward()
                    loss.backward()
                    gradient_bits = self.tensor_bits(leaf.grad)
                    np.testing.assert_array_equal(
                        gradient_bits,
                        np.zeros(gradient_bits.shape, dtype=np.uint32),
                    )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in (
            ("method", extreme.fix),
            ("function", lambda: torch.fix(extreme, out=None)),
        ):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    call()
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, "Stride calculation overflowed"
                    ):
                        call()

    def test_zero_vjp_ignores_upstreams_composes_accumulates_and_repeats(self):
        for form, apply in (
            ("method", lambda tensor: tensor.fix()),
            ("function", lambda tensor: torch.fix(tensor, out=None)),
        ):
            with self.subTest(form=form, case="special upstream"):
                leaf = torch.tensor([-1.25, -0.0, 1.75, 4.5], requires_grad=True)
                weights = torch.tensor(
                    [float("nan"), float("inf"), -float("inf"), -0.0]
                )
                (apply(leaf) * weights).sum().backward()
                np.testing.assert_array_equal(
                    self.tensor_bits(leaf.grad), np.zeros((4,), dtype=np.uint32)
                )

            with self.subTest(form=form, case="accumulation and repeated backward"):
                accumulated = torch.tensor([-2.0, 0.0, 3.0], requires_grad=True)
                (accumulated * 3.0).sum().backward()
                before_zero = self.tensor_bits(accumulated.grad).copy()
                reusable_loss = apply(accumulated).sum()
                reusable_loss.backward()
                np.testing.assert_array_equal(
                    self.tensor_bits(accumulated.grad), before_zero
                )
                reusable_loss.backward()
                np.testing.assert_array_equal(
                    self.tensor_bits(accumulated.grad), before_zero
                )

            with self.subTest(form=form, case="composition"):
                composed = torch.tensor([-0.5, 0.5], requires_grad=True)
                composed_loss = apply(composed.sin()).sum()
                composed_loss.backward()
                np.testing.assert_array_equal(
                    self.tensor_bits(composed.grad), np.zeros((2,), dtype=np.uint32)
                )
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    composed_loss.backward()

    def test_no_grad_and_detached_inputs_use_the_inference_path(self):
        for case, (_, source) in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.trunc()
            for form, call in self.supported_calls(source):
                with self.subTest(case=case, form=form, mode="no_grad"):
                    with torch.no_grad():
                        actual = call()
                    self.assert_matches(actual, expected, case=(case, form))
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertFalse(actual.is_set_to(source))

            for form, call in self.supported_calls(detached):
                with self.subTest(case=case, form=form, mode="detached"):
                    actual = call()
                    self.assert_matches(actual, expected, case=(case, form))
                    self.assertFalse(actual.is_set_to(detached))

    def test_concrete_out_is_rejected_before_autograd_or_planning(self):
        source = torch.tensor([1.25, -1.25], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        destination_pointer = destination.data_ptr()
        for form, call in (
            ("positional", lambda: torch.fix(source, out=destination)),
            ("keyword", lambda: torch.fix(input=source, out=destination)),
            ("alias", lambda: torch.fix(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^fix\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.data_ptr(), destination_pointer)
                self.assertEqual(destination.tolist(), [17.0, 19.0])
                self.assertIsNone(source.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^fix\(\): the 'out' argument is not supported$",
        ):
            torch.fix(extreme, out=destination)

    def test_tensorbase_descriptor_metadata_is_distinct_and_no_argument_only(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "fix")
        trunc_descriptor = inspect.getattr_static(torch.Tensor, "trunc")
        bound = tensor.fix

        self.assertIs(torch.Tensor.fix, descriptor)
        self.assertIsNot(descriptor, trunc_descriptor)
        self.assertIsNot(descriptor, torch.fix)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'fix' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "fix")
        self.assertEqual(descriptor.__qualname__, "TensorBase.fix")
        self.assertEqual(bound.__name__, "fix")
        self.assertEqual(bound.__qualname__, "Tensor.fix")
        self.assertEqual(descriptor.__doc__, TENSOR_FIX_DOC)
        self.assertEqual(bound.__doc__, TENSOR_FIX_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (
                lambda: tensor.fix(1),
                "TensorBase.fix() takes no arguments (1 given)",
            ),
            (lambda: bound(1), "Tensor.fix() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.fix() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.fix(1, 2),
                "TensorBase.fix() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.fix(input=tensor),
                (
                    "Tensor.fix() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.fix() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.fix() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.fix() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.fix() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'fix' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.fix() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_tensor_method_modes_dispatch_through_the_fix_descriptor(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "fix")
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
            result = tracked.fix()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertIsNot(function, inspect.getattr_static(torch.Tensor, "trunc"))
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
                forwarded = plain.fix()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0])

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_forwarded = tracked.fix()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_forwarded.requires_grad)
        self.assertFalse(tracked_forwarded.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                tracked_forwarded
            ),
            ", grad_fn=<TruncBackward0>",
        )

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.fix()
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
                plain.fix(1)
        self.assertEqual(invalid.calls, [])

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([1.25], requires_grad=True)
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
            self.assertIs(torch.fix(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.fix)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.fix(Override()), marker)
        self.assertIs(torch.fix(torch.tensor([1.25]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.fix)
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

        self.assertIs(torch.fix(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([1.25, -1.25])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.fix(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [1.0, -1.0])

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
            self.assertIs(torch.fix(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_distinct_builtin_metadata_copying_pickling_and_exports(self):
        function = torch.fix
        self.assertIs(function, torch._C.fix)
        self.assertIsNot(function, torch.trunc)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "fix")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.fix")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_FIX_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method fix of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.fix, function)
        self.assertIsNot(owner.fix, owner.trunc)
        for action in (
            lambda: setattr(owner, "fix", None),
            lambda: delattr(owner, "fix"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.fix, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertEqual(torch.__all__.count("fix"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["fix"], function)

    def test_inplace_forms_remain_unsupported_without_mutation(self):
        tensor = torch.tensor([1.25, -1.25], requires_grad=True)
        pointer = tensor.data_ptr()
        bits = self.tensor_bits(tensor).copy()
        self.assertTrue(hasattr(torch.Tensor, "fix"))
        self.assertFalse(hasattr(torch.Tensor, "fix_"))
        self.assertTrue(hasattr(tensor, "fix"))
        self.assertFalse(hasattr(tensor, "fix_"))
        self.assertFalse(hasattr(torch, "fix_"))
        self.assertNotIn("fix_", torch.__all__)
        for call in (
            lambda: tensor.fix_(),
            lambda: torch.fix_(tensor),
            lambda: tensor.fix(out=None),
        ):
            with self.assertRaises((AttributeError, TypeError)):
                call()
            self.assertEqual(tensor.data_ptr(), pointer)
            np.testing.assert_array_equal(self.tensor_bits(tensor), bits)
            self.assertIsNone(tensor.grad)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.fix(),
                'fix() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.fix(tensor, tensor),
                "fix() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.fix(tensor, input=tensor),
                "fix() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.fix(out=tensor),
                'fix() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.fix(1, extra=True),
                "fix(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.fix(input=[]),
                "fix(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.fix(tensor, out=[]),
                "fix(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.fix(tensor, extra=True),
                "fix() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.fix(input=tensor, a=tensor),
                "fix() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.fix(a=tensor, x=tensor, out=None),
                "fix() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.fix(x=tensor, a=tensor, out=None),
                "fix() got an unexpected keyword argument 'x'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
