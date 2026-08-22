import inspect
import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference PyTorch package")
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

    def source(self, module, *, requires_grad=False):
        values = [float(value) for value in range(48)]
        return module.tensor(values, requires_grad=requires_grad).reshape(2, 2, 3, 4)[
            1
        ].transpose(0, 1)

    def view_contract(self, module):
        source = self.source(module)
        calls = (
            source.narrow(0, 1, 2),
            source.narrow(0, start=1, length=2),
            source.narrow(dim=0, start=1, length=2),
            source.narrow(length=2, start=1, dim=0),
            source.narrow(-3, 1, 2),
            source.narrow(0, -2, 2),
        )
        rows = []
        for narrowed in calls:
            rows.append(
                {
                    "values": narrowed.tolist(),
                    "shape": tuple(narrowed.shape),
                    "stride": narrowed.stride(),
                    "offset": narrowed.storage_offset(),
                    "pointer_delta": (
                        narrowed.data_ptr() - source.data_ptr()
                    )
                    // source.element_size(),
                    "is_set_to_repeat": narrowed.is_set_to(
                        source.narrow(0, 1, 2)
                    ),
                    "same_dtype": narrowed.dtype is source.dtype,
                    "same_device": narrowed.device == source.device,
                }
            )

        zero_lengths = []
        for start in (0, 1, -1, 3):
            narrowed = source.narrow(0, start, 0)
            zero_lengths.append(
                (
                    tuple(narrowed.shape),
                    narrowed.stride(),
                    narrowed.storage_offset(),
                    narrowed.data_ptr(),
                    narrowed.tolist(),
                    narrowed.is_set_to(source.narrow(0, start, 0)),
                )
            )

        inner_empty_source = module.zeros((4, 0, 3))
        inner_empty = inner_empty_source.narrow(0, -1, 1)
        leading_empty_source = module.zeros((0, 3))
        leading_empty = leading_empty_source.narrow(-2, 0, 0)
        return {
            "rows": rows,
            "zero_lengths": zero_lengths,
            "inner_empty": (
                tuple(inner_empty.shape),
                inner_empty.stride(),
                inner_empty.storage_offset(),
                inner_empty.data_ptr(),
                inner_empty.tolist(),
            ),
            "leading_empty": (
                tuple(leading_empty.shape),
                leading_empty.stride(),
                leading_empty.storage_offset(),
                leading_empty.data_ptr(),
                leading_empty.tolist(),
            ),
        }

    def test_values_layout_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(
            self.view_contract(torch),
            self.view_contract(reference_torch),
        )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [float(value) for value in range(48)], requires_grad=True
        )
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        narrowed = source.narrow(-3, -2, 2)
        metadata = (
            narrowed.requires_grad,
            narrowed.is_leaf,
            narrowed.output_nr,
            tuple(narrowed.shape),
            narrowed.stride(),
            narrowed.storage_offset(),
        )
        weights = module.tensor(
            [float(value) for value in range(1, 17)]
        ).reshape(2, 2, 4)
        (narrowed * weights).sum().backward()

        no_grad_source = module.zeros((4, 3), requires_grad=True)
        with module.no_grad():
            untracked = no_grad_source.narrow(dim=0, start=1, length=2)

        empty = module.zeros((4, 2), requires_grad=True)
        empty_view = empty.narrow(0, 2, 0)
        empty_view.sum().backward()
        diagnostic = module.tensor([2.0], requires_grad=True).narrow(0, 0, 1)
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "node_diagnostic": self.error(
                lambda: module.nn.functional.dropout(
                    None,
                    p=diagnostic,
                    training=False,
                )
            ),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
            ),
            "empty": (
                tuple(empty_view.shape),
                empty_view.stride(),
                empty_view.storage_offset(),
                empty.grad.tolist(),
            ),
        }

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def error_contract(self, module):
        tensor = module.zeros((4, 2, 3))
        scalar = module.tensor(1.0)
        empty = module.zeros((0, 3))
        return (
            self.error(lambda: scalar.narrow(0, 0, 0)),
            self.error(lambda: scalar.narrow(99, 0, -1)),
            self.error(lambda: tensor.narrow(99, 5, -1)),
            self.error(lambda: tensor.narrow(3, 0, 1)),
            self.error(lambda: tensor.narrow(-4, 0, 1)),
            self.error(lambda: tensor.narrow(0, -5, 0)),
            self.error(lambda: tensor.narrow(0, 5, 0)),
            self.error(lambda: tensor.narrow(0, -1, 2)),
            self.error(lambda: empty.narrow(0, 0, 1)),
            self.error(lambda: empty.narrow(0, -1, 0)),
            self.error(lambda: tensor.narrow(2**100, 0, 0)),
            self.error(lambda: tensor.narrow(0, 2**100, 0)),
            self.error(lambda: tensor.narrow(0, 0, 2**100)),
        )

    def test_runtime_and_range_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch),
            self.error_contract(reference_torch),
        )

    def binding_error_contract(self, module):
        tensor = module.zeros((4, 3))
        return (
            self.error(lambda: tensor.narrow()),
            self.error(lambda: tensor.narrow(0)),
            self.error(lambda: tensor.narrow(0, 0)),
            self.error(lambda: tensor.narrow(0, 0, 0, 0)),
            self.error(lambda: tensor.narrow(None, 0, 0)),
            self.error(lambda: tensor.narrow(0, None, 0)),
            self.error(lambda: tensor.narrow(0, 0, None)),
            self.error(lambda: tensor.narrow(True, 0, 0)),
            self.error(lambda: tensor.narrow(0, True, 0)),
            self.error(lambda: tensor.narrow(0, 0, True)),
            self.error(lambda: tensor.narrow(dim=None, start=0, length=1)),
            self.error(lambda: tensor.narrow(dim=0, start=None, length=1)),
            self.error(lambda: tensor.narrow(dim=0, start=0, length=None)),
            self.error(lambda: tensor.narrow(0, 0, dim=0)),
            self.error(lambda: tensor.narrow(0, 0, foo=1)),
            self.error(lambda: tensor.narrow(dim=0, start=0, foo=1)),
            self.error(lambda: tensor.narrow(0, 0, 1, start=0)),
        )

    def test_binding_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.binding_error_contract(torch),
            self.binding_error_contract(reference_torch),
        )

    def integer_conversion_contract(self, module):
        tensor = module.zeros((5, 2))
        calls = []

        class StatefulIndex:
            def __init__(self, label, values):
                self.label = label
                self.values = iter(values)

            def __index__(self):
                calls.append(self.label)
                return next(self.values)

        start = StatefulIndex("start", (1, 2, 3))
        length = StatefulIndex("length", (2, 1, 1))
        narrowed = tensor.narrow(0, start, length)
        numpy_narrowed = tensor.narrow(np.int8(-2), np.int64(-4), np.uint32(2))
        return {
            "calls": calls,
            "shape": tuple(narrowed.shape),
            "stride": narrowed.stride(),
            "offset": narrowed.storage_offset(),
            "numpy_shape": tuple(numpy_narrowed.shape),
            "numpy_offset": numpy_narrowed.storage_offset(),
        }

    def test_integer_index_conversion_matches_pytorch_2_13(self):
        self.assertEqual(
            self.integer_conversion_contract(torch),
            self.integer_conversion_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.zeros((4, 3))
        descriptor = inspect.getattr_static(module.Tensor, "narrow")
        bound = tensor.narrow

        def signature_error_type(callable_object):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                return type(error).__name__
            return None

        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "objclass": (
                descriptor.__objclass__.__module__,
                descriptor.__objclass__.__name__,
            ),
            "descriptor_module": getattr(descriptor, "__module__", "missing"),
            "bound_module": bound.__module__,
            "text_signature": descriptor.__text_signature__,
            "repr": repr(descriptor),
            "same_descriptor": module.Tensor.narrow is descriptor,
            "signature_errors": (
                signature_error_type(descriptor),
                signature_error_type(bound),
            ),
            "unbound": (
                self.error(lambda: descriptor()),
                self.error(lambda: descriptor(1, 0, 0, 1)),
                self.error(
                    lambda: descriptor(
                        self=tensor,
                        dim=0,
                        start=0,
                        length=1,
                    )
                ),
            ),
        }

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.zeros((4, 3))
        tensor_start = module.tensor(1)
        marker = object()
        events = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label, result):
                self.label = label
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        self.label,
                        func.__name__,
                        func.__qualname__,
                        tuple(value.__name__ for value in types),
                        tuple(argument is tensor for argument in args),
                        tuple(type(argument).__name__ for argument in args),
                        kwargs,
                    )
                )
                return self.result

        positional = RecordingMode("positional", marker)
        with positional:
            positional_result = tensor.narrow(0, 1, 2)

        keyword = RecordingMode("keyword", marker)
        with keyword:
            keyword_result = tensor.narrow(length=2, start=1, dim=0)

        tensor_mode = RecordingMode("tensor", marker)
        with tensor_mode:
            tensor_result = tensor.narrow(0, tensor_start, 1)

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.narrow(dim=0, start=-3, length=2)

        calls = []

        class Index:
            def __init__(self, label):
                self.label = label

            def __index__(self):
                calls.append(self.label)
                return 1

        deferred = RecordingMode("deferred", marker)
        with deferred:
            deferred_result = tensor.narrow(2**100, Index("start"), Index("length"))

        invalid = RecordingMode("invalid", marker)
        invalid_error = self.error(
            lambda: self.call_under_mode(invalid, lambda: tensor.narrow(True, 0, 1))
        )
        return {
            "marker_results": (
                positional_result is marker,
                keyword_result is marker,
                tensor_result is marker,
                deferred_result is marker,
            ),
            "events": events,
            "forward_order": order,
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.tolist(),
            ),
            "index_calls": calls,
            "invalid_error": invalid_error,
        }

    @staticmethod
    def call_under_mode(mode, action):
        with mode:
            return action()

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def extreme_empty_contract(self, module):
        maximum = sys.maxsize
        source = module.zeros((2, 0, maximum))
        valid = source.narrow(0, 1, 0)
        offset = module.zeros((maximum, 0))[maximum - 1].reshape((maximum, 0))
        return {
            "valid": (
                tuple(valid.shape),
                valid.stride(),
                valid.storage_offset(),
                valid.data_ptr(),
            ),
            "range_overflow": self.error(
                lambda: module.zeros((maximum, 0)).narrow(
                    0,
                    maximum,
                    maximum,
                )
            ),
            "one_past_offset": self.error(lambda: source.narrow(0, 2, 0)),
            "offset_addition": self.error(
                lambda: offset.narrow(0, maximum, 0)
            ),
        }

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "checked offset boundary requires a 64-bit platform",
    )
    def test_extreme_empty_offsets_match_pytorch_2_13(self):
        self.assertEqual(
            self.extreme_empty_contract(torch),
            self.extreme_empty_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
