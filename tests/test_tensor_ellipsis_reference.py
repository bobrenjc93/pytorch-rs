import gc
import re
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorBareEllipsisReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("ellipsis differentials require pinned PyTorch 2.13.0")

    def tensor_bits(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            detached = detached.cpu()
        return np.asarray(detached).reshape(-1).view(np.uint32).copy()

    def layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("offset", base[1]),
            ("noncontiguous-offset", base.transpose(0, 2)[1]),
        )

    def alias_observation(self, module, source):
        alias = source[...]
        self.assertIsNot(alias, source)
        return (
            (
                tuple(alias.shape),
                alias.stride(),
                alias.storage_offset(),
                alias.numel(),
                alias.is_contiguous(),
                str(alias.dtype),
                str(alias.device),
                alias.requires_grad,
                alias.is_leaf,
                alias.output_nr,
                alias.data_ptr() == source.data_ptr(),
                alias.is_set_to(source),
            ),
            self.tensor_bits(alias, module),
        )

    def test_layout_storage_and_values_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source = actual_case
            expected_name, expected_source = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual_metadata, actual_bits = self.alias_observation(
                    torch, actual_source
                )
                expected_metadata, expected_bits = self.alias_observation(
                    reference_torch, expected_source
                )
                self.assertEqual(actual_metadata, expected_metadata)
                np.testing.assert_array_equal(actual_bits, expected_bits)

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        alias = source[...]
        metadata = (
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
            alias.requires_grad,
            alias.is_leaf,
            alias.output_nr,
            alias.data_ptr() == source.data_ptr(),
            alias.is_set_to(source),
        )
        values = self.tensor_bits(alias, module)
        del source
        gc.collect()
        weights = module.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            dtype=module.float32,
        )
        (alias * weights).sum().backward()
        gradient = self.tensor_bits(leaf.grad, module)

        scalar = module.tensor(-2.0, requires_grad=True)
        scalar_alias = scalar[...]
        scalar_metadata = (
            scalar_alias.requires_grad,
            scalar_alias.is_leaf,
            scalar_alias.output_nr,
            scalar_alias.is_set_to(scalar),
        )
        (scalar_alias * 7.0).backward()

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_alias = empty[...]
        empty_metadata = (
            tuple(empty_alias.shape),
            empty_alias.stride(),
            empty_alias.storage_offset(),
            empty_alias.requires_grad,
            empty_alias.is_leaf,
            empty_alias.output_nr,
            empty_alias.is_set_to(empty),
        )
        empty_alias.sum().backward()
        return (
            metadata,
            values,
            gradient,
            scalar_metadata,
            scalar.grad.item(),
            empty_metadata,
            tuple(empty.grad.shape),
            empty.grad.tolist(),
        )

    def no_grad_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        with module.no_grad():
            alias = source[...]
        metadata = (
            tuple(alias.shape),
            alias.stride(),
            alias.storage_offset(),
            alias.requires_grad,
            alias.is_leaf,
            alias.output_nr,
            alias.data_ptr() == source.data_ptr(),
            alias.is_set_to(source),
        )
        del source, leaf
        gc.collect()
        return metadata, self.tensor_bits(alias, module)

    def test_autograd_no_grad_and_source_lifetime_match_pytorch_2_13(self):
        actual = self.autograd_outcome(torch)
        expected = self.autograd_outcome(reference_torch)
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        self.assertEqual(actual[3:], expected[3:])

        actual_metadata, actual_bits = self.no_grad_outcome(torch)
        expected_metadata, expected_bits = self.no_grad_outcome(reference_torch)
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_bits, expected_bits)

    def mode_outcome(self, module):
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        descriptor = module.Tensor.__mro__[1].__getitem__
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        records = []
        indices = (
            Ellipsis,
            1,
            (1, 0),
            slice(None),
            None,
            (Ellipsis,),
        )
        for index in indices:
            mode = RecordingMode(marker)
            with mode:
                result = tensor[index]
            func, types, args, kwargs = mode.calls[0]
            records.append(
                (
                    result is marker,
                    len(mode.calls),
                    func is descriptor,
                    type(func).__name__,
                    repr(func),
                    func.__name__,
                    func.__qualname__,
                    types,
                    len(args),
                    args[0] is tensor,
                    args[1] is index,
                    repr(args[1]),
                    kwargs,
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(
                    (
                        self.label,
                        func is descriptor,
                        types,
                        args[0] is tensor,
                        args[1] is Ellipsis,
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor[...]

        conversions = []

        class PoisonIndex:
            def __index__(self):
                conversions.append("index")
                raise RuntimeError("index conversion must be deferred")

        poison = PoisonIndex()
        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor[poison]

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor[...]
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0xADDR", str(error)),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "forwarding": tuple(order),
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.data_ptr() == tensor.data_ptr(),
                forwarded.is_set_to(tensor),
                forwarded is tensor,
            ),
            "deferred": (
                deferred_result is marker,
                len(deferred.calls),
                deferred.calls[0][2][1] is poison,
                tuple(conversions),
            ),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_outcome(torch), self.mode_outcome(reference_torch)
        )
