import gc
import inspect
import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorArbitraryIntegerPrefixFullSliceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "integer-prefix full-slice differentials require pinned PyTorch 2.13.0"
            )

    def view_contract(self, selected, direct):
        values = np.asarray(selected.detach(), dtype=np.float32).reshape(-1)
        return {
            "values": selected.tolist(),
            "value_bits": tuple(values.view(np.uint32).tolist()),
            "shape": tuple(selected.shape),
            "stride": selected.stride(),
            "storage_offset": selected.storage_offset(),
            "same_logical_view": selected.is_set_to(direct),
            "same_data_pointer": selected.data_ptr() == direct.data_ptr(),
            "dtype": str(selected.dtype),
            "device": str(selected.device),
            "requires_grad": selected.requires_grad,
            "is_leaf": selected.is_leaf,
        }

    def six_leading_integer_layout_cases(self, module):
        shape = (2,) * 6 + (3,)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values[1, 0, 1, 0, 1, 0, 0] = -0.0
        base_values = np.arange(2 * np.prod(shape), dtype=np.float32).reshape(
            (2, *shape)
        )
        base = module.tensor(base_values.tolist(), dtype=module.float32)
        return (
            (
                module.tensor(values.tolist(), dtype=module.float32),
                (1, 0, 1, 0, 1, 0),
            ),
            (
                module.zeros((2,) * 6 + (0, 3), dtype=module.float32),
                (1, 0, 1, 0, 1, 0),
            ),
            (base[1].transpose(0, 6), (2, 1, 0, 1, 0, 1)),
            (
                module.zeros(
                    (sys.maxsize, 1, 1, 1, 1, 1, 0, 3),
                    dtype=module.float32,
                ),
                (sys.maxsize - 1, 0, 0, 0, 0, 0),
            ),
        )

    def six_plus_layout_and_protocol_contract(self, module):
        layouts = []
        for source, indices in self.six_leading_integer_layout_cases(module):
            selected = source[indices + (slice(None),)]
            layouts.append(self.view_contract(selected, source[indices]))

        maximum_rank_source = module.zeros((1,) * 64, dtype=module.float32)
        maximum_prefix = (0,) * 63
        maximum_rank = self.view_contract(
            maximum_rank_source[maximum_prefix + (slice(None),)],
            maximum_rank_source[maximum_prefix],
        )

        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first_dynamic = IndexValue(1)
        third_dynamic = IndexValue(1)
        fifth_dynamic = IndexValue(1)
        indices = (
            first_dynamic,
            np.int64(0),
            third_dynamic,
            IntegerSubclass(0),
            fifth_dynamic,
            np.uint64(0),
        )
        source = module.tensor(
            np.arange(192, dtype=np.float32).reshape((2,) * 6 + (3,)).tolist(),
            dtype=module.float32,
        )
        protocol = self.view_contract(
            source[indices + (slice(None),)],
            source[1, 0, 1, 0, 1, 0],
        )

        class RemappedTuple(tuple):
            def __iter__(self):
                return iter((1, 0, 1, 0, 1, 0, slice(None)))

        remapped = source[RemappedTuple((0,))]
        tuple_subclass = self.view_contract(
            remapped,
            source[1, 0, 1, 0, 1, 0],
        )
        return {
            "layouts": layouts,
            "maximum_rank": maximum_rank,
            "protocol": protocol,
            "protocol_calls": (
                first_dynamic.calls,
                third_dynamic.calls,
                fifth_dynamic.calls,
            ),
            "tuple_subclass": tuple_subclass,
        }

    def test_six_plus_layout_protocol_and_tuple_subclass_match_pytorch_2_13(self):
        self.assertEqual(
            self.six_plus_layout_and_protocol_contract(torch),
            self.six_plus_layout_and_protocol_contract(reference_torch),
        )

    def autograd_and_lifetime_contract(self, module):
        shape = (2,) * 8
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        indices = (1, 0, 1, 0, 1, 0)

        def retained_view():
            source = (leaf * 2.0).reshape(shape)[1].transpose(0, 6)
            selected = source[indices + (slice(None),)]
            direct = source[indices]
            return selected, (
                selected.is_set_to(direct),
                selected.data_ptr() == direct.data_ptr(),
            )

        selected, aliasing = retained_view()
        gc.collect()
        metadata = (
            selected.tolist(),
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            aliasing,
        )
        weights = module.tensor([3.0, 5.0], dtype=module.float32)
        (selected * weights).sum().backward()

        empty = module.zeros(
            (2,) * 6 + (0, 3),
            dtype=module.float32,
            requires_grad=True,
        )
        empty[indices + (slice(None),)].sum().backward()

        diagnostic_leaf = module.tensor(
            np.full((1,) * 7, 2.0, dtype=np.float32).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        try:
            module.nn.functional.dropout(
                None,
                p=diagnostic_leaf[(0,) * 6 + (slice(None),)],
                training=False,
            )
        except ValueError as error:
            node_diagnostic = str(error)
        else:
            self.fail("dropout unexpectedly accepted a tensor probability")

        no_grad_leaf = module.tensor(
            np.arange(128, dtype=np.float32).reshape((2,) * 7).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        no_grad_source = no_grad_leaf.transpose(0, 6)
        with module.no_grad():
            untracked = no_grad_source[(1, 0, 1, 0, 1, 0, slice(None))]

        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "empty_gradient": (
                tuple(empty.grad.shape),
                empty.grad.stride(),
                empty.grad.storage_offset(),
                empty.grad.numel(),
            ),
            "node_diagnostic": node_diagnostic,
            "no_grad": (
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                untracked.is_set_to(no_grad_source[1, 0, 1, 0, 1, 0]),
            ),
        }

    def test_six_plus_autograd_and_lifetime_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_and_lifetime_contract(torch),
            self.autograd_and_lifetime_contract(reference_torch),
        )

    def validation_contract(self, module):
        events = []

        class IndexValue:
            def __init__(self, label, value):
                self.label = label
                self.value = value

            def __index__(self):
                events.append(self.label)
                return self.value

        def capture(call):
            try:
                call()
            except Exception as error:
                return type(error).__name__, str(error), tuple(events)
            self.fail(
                "invalid integer-prefix full-slice indexing unexpectedly succeeded"
            )

        excessive = tuple(IndexValue(index, 0) for index in range(6))
        excessive_result = capture(
            lambda: module.zeros((2,) * 6, dtype=module.float32)[
                excessive + (slice(None),)
            ]
        )

        events.clear()
        out_of_bounds = tuple(
            IndexValue(index, 4 if index == 2 else 0) for index in range(6)
        )
        bounds_result = capture(
            lambda: module.zeros(
                (2, 3, 4, 5, 6, 7, 8), dtype=module.float32
            )[out_of_bounds + (slice(None),)]
        )

        events.clear()
        invalid = tuple(
            IndexValue(index, 1.5 if index == 3 else 0) for index in range(6)
        )
        invalid_result = capture(
            lambda: module.zeros((2,) * 7, dtype=module.float32)[
                invalid + (slice(None),)
            ]
        )

        events.clear()
        overflow_result = capture(
            lambda: module.zeros((2,) * 7, dtype=module.float32)[
                (0,) * 5 + (2**100, slice(None))
            ]
        )
        return {
            "excessive": excessive_result,
            "bounds": bounds_result,
            "invalid": invalid_result,
            "overflow": overflow_result,
        }

    def test_six_plus_validation_order_matches_pytorch_2_13(self):
        self.assertEqual(
            self.validation_contract(torch),
            self.validation_contract(reference_torch),
        )

    def dispatch_contract(self, module):
        marker = object()

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        cases = []
        for name, rank, suffix, invalid_position in (
            ("valid", 7, (slice(None),), None),
            ("excessive", 6, (slice(None),), None),
            ("invalid", 7, (slice(None),), 3),
            ("non-full-slice", 7, (slice(None, None, 1),), None),
            ("mixed", 7, (None, slice(None)), None),
        ):
            dynamic = [
                IndexValue(1.5 if index == invalid_position else 0)
                for index in range(6)
            ]
            prefix = tuple(dynamic) if name != "mixed" else tuple(dynamic[:5])
            index = prefix + suffix
            source = module.zeros((2,) * rank, dtype=module.float32)
            mode = RecordingMode()
            with mode:
                result = source[index]
            function, dispatch_types, args, kwargs = mode.calls[0]
            cases.append(
                (
                    name,
                    result is marker,
                    len(mode.calls),
                    type(function).__name__,
                    function.__name__,
                    function.__qualname__,
                    function.__objclass__.__name__,
                    function.__objclass__.__module__,
                    dispatch_types == (),
                    len(args),
                    args[0] is source,
                    args[1] is index,
                    kwargs is None,
                    tuple(item.calls for item in dynamic),
                )
            )
        return tuple(cases)

    def test_six_plus_dispatch_uses_original_tuple_before_validation(self):
        self.assertEqual(
            self.dispatch_contract(torch),
            self.dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
