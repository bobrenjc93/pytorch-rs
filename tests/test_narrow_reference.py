import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorNarrowReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("narrow differentials require pinned PyTorch 2.13.0")

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("narrow unexpectedly accepted an invalid call")

    def observe_view(self, source, selected, direct):
        if source.data_ptr() == 0 or selected.data_ptr() == 0:
            data_pointer_delta = None
        else:
            data_pointer_delta = selected.data_ptr() - source.data_ptr()
        return {
            "values": selected.tolist(),
            "shape": tuple(selected.shape),
            "stride": selected.stride(),
            "storage_offset": selected.storage_offset(),
            "data_pointer_delta": data_pointer_delta,
            "direct_data_ptr_matches": selected.data_ptr() == direct.data_ptr(),
            "direct_view_matches": selected.is_set_to(direct),
            "source_view_matches": selected.is_set_to(source),
            "same_dtype": selected.dtype is source.dtype,
            "same_device": selected.device == source.device,
        }

    def offset_noncontiguous_source(self, module, *, requires_grad=False):
        values = [float(value) for value in range(120)]
        return module.tensor(values, requires_grad=requires_grad).reshape(2, 3, 4, 5)[
            1
        ].transpose(0, 1)

    def view_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        contiguous = module.tensor(values.tolist(), dtype=module.float32)
        transposed = contiguous.transpose(0, 2)
        offset = self.offset_noncontiguous_source(module)
        cases = (
            (contiguous, contiguous.narrow(1, 1, 2), contiguous[:, 1:3]),
            (transposed, transposed.narrow(0, 1, 2), transposed[1:3]),
            (offset, offset.narrow(1, 1, 2), offset[:, 1:3]),
            (contiguous, contiguous.narrow(1, 3, 0), contiguous[:, 3:3]),
            (contiguous, contiguous.narrow(-2, -3, 3), contiguous[:, 0:3]),
            (contiguous, module.narrow(contiguous, 1, 1, 2), contiguous[:, 1:3]),
            (
                contiguous,
                module.narrow(input=contiguous, dim=1, start=3, length=0),
                contiguous[:, 3:3],
            ),
            (
                contiguous,
                module.narrow(x=contiguous, dim=-2, start=-3, length=3),
                contiguous[:, 0:3],
            ),
        )
        return tuple(
            self.observe_view(source, selected, direct)
            for source, selected, direct in cases
        )

    def test_values_layout_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(self.view_contract(torch), self.view_contract(reference_torch))

    def autograd_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        selected = source.narrow(0, 1, 2)
        metadata = (
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
        )
        selected.sum().backward()

        top_level_leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        top_level_source = (top_level_leaf * 2.0).reshape(2, 3, 4)
        module.narrow(top_level_source, 1, -2, 2).sum().backward()

        no_grad_source = module.tensor(
            values.tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            untracked = no_grad_source.narrow(1, 1, 2)

        empty = module.zeros((2, 3, 4), dtype=module.float32, requires_grad=True)
        empty.narrow(1, 3, 0).sum().backward()
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "top_level_gradient": top_level_leaf.grad.tolist(),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient_stride": empty.grad.stride(),
            "empty_gradient_offset": empty.grad.storage_offset(),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_backward_through_sum_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def binding_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)

        class IntegerSubclass(int):
            pass

        conversion_order = []

        class StatefulIndex:
            def __init__(self, name, values):
                self.name = name
                self.values = values
                self.calls = 0

            def __index__(self):
                conversion_order.append(self.name)
                value = self.values[self.calls]
                self.calls += 1
                return value

        start = StatefulIndex("start", [1, 2, 1])
        length = StatefulIndex("length", [2, 1, 2])
        selected = tensor.narrow(1, start, length)
        return {
            "numpy_and_subclass": tuple(
                tensor.narrow(np.int64(-2), IntegerSubclass(1), np.uint32(2)).shape
            ),
            "stateful": (
                conversion_order,
                tuple(selected.shape),
                selected.storage_offset(),
            ),
            "errors": (
                self.error(lambda: tensor.narrow()),
                self.error(lambda: tensor.narrow(0)),
                self.error(lambda: tensor.narrow(0, 0)),
                self.error(lambda: tensor.narrow(0, 0, 1, 2)),
                self.error(lambda: tensor.narrow(0, 0, 1, dim=0)),
                self.error(lambda: tensor.narrow(slice(None), 0, 1)),
                self.error(lambda: tensor.narrow(0, True, 1)),
                self.error(lambda: tensor.narrow(0, 0, True)),
                self.error(lambda: tensor.narrow(2**100, 0, 1)),
                self.error(lambda: tensor.narrow(0, 2**100, 1)),
                self.error(lambda: tensor.narrow(0, 0, 2**100)),
                self.error(lambda: tensor.narrow(0, 0, -1)),
                self.error(lambda: tensor.narrow(0, 3, 0)),
                self.error(lambda: tensor.narrow(0, 1, 2)),
                self.error(lambda: tensor.narrow(3, 0, 1)),
                self.error(lambda: module.tensor(1.0, dtype=module.float32).narrow(0, 0, 1)),
                self.error(lambda: module.narrow()),
                self.error(lambda: module.narrow(tensor)),
                self.error(lambda: module.narrow(tensor, 0)),
                self.error(lambda: module.narrow(tensor, 0, 0)),
                self.error(lambda: module.narrow(tensor, 0, 0, 1, 2)),
                self.error(lambda: module.narrow([], 0, 0, 1)),
                self.error(lambda: module.narrow(tensor, 0, 0, -1)),
            ),
        }

    def test_binding_bounds_and_scalar_errors_match_pytorch_2_13(self):
        self.assertEqual(self.binding_contract(torch), self.binding_contract(reference_torch))

    def descriptor_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "narrow")
        bound = tensor.narrow

        def signature_error(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            self.fail("narrow unexpectedly exposed an inspectable signature")

        return {
            "descriptor_type": type(descriptor).__name__,
            "is_method_descriptor": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "repr": repr(descriptor),
            "class_identity": module.Tensor.narrow is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "descriptor_signature": signature_error(descriptor),
            "bound_signature": signature_error(bound),
            "call_shape": tuple(descriptor(tensor, 0, 0, 1).shape),
            "no_receiver": self.error(lambda: descriptor()),
            "wrong_receiver": self.error(lambda: descriptor(1, 0, 0, 1)),
            "keyword_receiver": self.error(
                lambda: descriptor(self=tensor, dim=0, start=0, length=1)
            ),
            "function_doc": module.narrow.__doc__,
            "function_module": module.narrow.__module__,
            "function_qualname": module.narrow.__qualname__,
        }

    def test_descriptor_contract_matches_pytorch_2_13(self):
        actual = self.descriptor_contract(torch)
        expected = self.descriptor_contract(reference_torch)
        actual_function_doc = actual.pop("function_doc")
        expected.pop("function_doc")
        actual["repr"] = re.sub(r"0x[0-9a-f]+", "0xADDR", actual["repr"])
        expected["repr"] = re.sub(r"0x[0-9a-f]+", "0xADDR", expected["repr"])
        self.assertIn("tensor-valued ``start`` arguments are not supported", actual_function_doc)
        self.assertNotIn("torch.narrow(x, -1, torch.tensor", actual_function_doc)
        self.assertEqual(actual, expected)

    def mode_dispatch_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        start = module.tensor(0.0, dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        def describe_call(call):
            func, dispatch_types, args, kwargs = call
            described_args = []
            for argument in args:
                if argument is tensor:
                    described_args.append("input")
                elif argument is start:
                    described_args.append("start")
                else:
                    described_args.append(argument)
            return {
                "function_name": func.__name__,
                "function_qualname": func.__qualname__,
                "dispatch_types": tuple(
                    dispatch_type.__name__ for dispatch_type in dispatch_types
                ),
                "args": tuple(described_args),
                "kwargs": kwargs,
            }

        method_mode = RecordingMode()
        with method_mode:
            method_result = tensor.narrow(0, start, 1)

        top_level_mode = RecordingMode()
        with top_level_mode:
            top_level_result = module.narrow(tensor, 0, start, 1)

        return {
            "method_result_is_marker": method_result is marker,
            "method_calls": tuple(describe_call(call) for call in method_mode.calls),
            "top_level_result_is_marker": top_level_result is marker,
            "top_level_calls": tuple(
                describe_call(call) for call in top_level_mode.calls
            ),
        }

    def test_tensor_start_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
        )

    def argument_override_dispatch_contract(self, module):
        tensor = module.zeros((2, 3, 4), dtype=module.float32)
        marker = object()
        events = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((func, types, args, kwargs))
                return marker

        dim = Override()
        start = Override()
        length = Override()
        results = (
            tensor.narrow(dim, 0, 1),
            tensor.narrow(0, start, 1),
            tensor.narrow(0, 0, length),
            module.narrow(tensor, dim, 0, 1),
            module.narrow(tensor, 0, start, 1),
            module.narrow(tensor, 0, 0, length),
        )

        def describe_call(call):
            func, dispatch_types, args, kwargs = call
            described_args = []
            for argument in args:
                if argument is tensor:
                    described_args.append("input")
                elif argument is dim:
                    described_args.append("dim")
                elif argument is start:
                    described_args.append("start")
                elif argument is length:
                    described_args.append("length")
                else:
                    described_args.append(argument)
            return {
                "function_name": func.__name__,
                "function_qualname": func.__qualname__,
                "dispatch_types": tuple(
                    dispatch_type.__name__ for dispatch_type in dispatch_types
                ),
                "args": tuple(described_args),
                "kwargs": kwargs,
            }

        def dispatch_error(action):
            error_type, message = self.error(action)
            return error_type, message.split("\n\n", maxsplit=1)[0]

        class DecliningOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return NotImplemented

        declining_calls = (
            dispatch_error(lambda: tensor.narrow(DecliningOverride(), 0, 1)),
            dispatch_error(lambda: tensor.narrow(0, DecliningOverride(), 1)),
            dispatch_error(lambda: tensor.narrow(0, 0, DecliningOverride())),
            dispatch_error(lambda: module.narrow(tensor, DecliningOverride(), 0, 1)),
            dispatch_error(lambda: module.narrow(tensor, 0, DecliningOverride(), 1)),
            dispatch_error(lambda: module.narrow(tensor, 0, 0, DecliningOverride())),
        )

        mixed_events = []

        class FirstOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                mixed_events.append(cls.__name__)
                return NotImplemented

        class SecondOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                mixed_events.append(cls.__name__)
                return marker

        mixed_method_result = tensor.narrow(FirstOverride(), SecondOverride(), FirstOverride())
        mixed_top_level_result = module.narrow(
            tensor,
            FirstOverride(),
            SecondOverride(),
            FirstOverride(),
        )

        return {
            "results_are_marker": tuple(result is marker for result in results),
            "calls": tuple(describe_call(call) for call in events),
            "declining_calls": declining_calls,
            "declining_call_count": DecliningOverride.calls,
            "mixed_method_result_is_marker": mixed_method_result is marker,
            "mixed_top_level_result_is_marker": mixed_top_level_result is marker,
            "mixed_call_order": tuple(mixed_events),
        }

    def test_argument_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.argument_override_dispatch_contract(torch),
            self.argument_override_dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
