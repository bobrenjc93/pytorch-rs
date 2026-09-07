import re
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorRangeSliceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "range-slice indexing differentials require pinned PyTorch 2.13.0"
            )

    def view_observation(self, source, index):
        selected = source[index]
        repeated = source[index]
        if selected.data_ptr() == 0 or source.data_ptr() == 0:
            data_pointer_delta = None
        else:
            data_pointer_delta = selected.data_ptr() - source.data_ptr()
        return {
            "values": selected.tolist(),
            "shape": tuple(selected.shape),
            "stride": selected.stride(),
            "storage_offset": selected.storage_offset(),
            "data_pointer_delta": data_pointer_delta,
            "zero_data_pointer": selected.data_ptr() == 0,
            "same_logical_view": selected.is_set_to(repeated),
            "same_storage_as_source": selected.is_set_to(source),
            "same_dtype": selected.dtype is source.dtype,
            "same_device": selected.device == source.device,
        }

    def dropout_probability_node(self, module, value):
        try:
            module.nn.functional.dropout(None, p=value, training=False)
        except ValueError as error:
            match = re.search(r"grad_fn=<([^>]+)>", str(error))
            if match is None:
                return str(error)
            return match.group(1)
        self.fail("dropout unexpectedly accepted a tensor probability")

    def view_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        contiguous = module.tensor(
            np.arange(24, dtype=np.float32).reshape(4, 3, 2).tolist(),
            dtype=module.float32,
        )
        empty_middle = module.zeros((3, 0, 4), dtype=module.float32)
        return (
            self.view_observation(contiguous, slice(1, 3)),
            self.view_observation(contiguous, slice(None, -1)),
            self.view_observation(contiguous, slice(-3, None)),
            self.view_observation(contiguous, slice(None, None, 1)),
            self.view_observation(contiguous, slice(3, 1)),
            self.view_observation(base, (slice(None), slice(1, 3))),
            self.view_observation(base, (slice(1, 2), slice(None), slice(None))),
            self.view_observation(base, (slice(1, 2), Ellipsis)),
            self.view_observation(base, (Ellipsis, slice(1, None))),
            self.view_observation(base, (Ellipsis, slice(1, 3), slice(None))),
            self.view_observation(base, (slice(None), Ellipsis, slice(1, 3))),
            self.view_observation(base, (slice(None), slice(10**5000, None))),
            self.view_observation(base, (slice(None), slice(-2, None))),
            self.view_observation(base, (slice(None), slice(None, -1))),
            self.view_observation(base, (slice(None), slice(0, 3), slice(None))),
            self.view_observation(
                base, (slice(None), slice(None, None, 1), slice(None))
            ),
            self.view_observation(base, (Ellipsis, slice(0, 4))),
            self.view_observation(base.transpose(0, 1), slice(1, 3)),
            self.view_observation(
                base.transpose(0, 1), (slice(None), slice(1, 2))
            ),
            self.view_observation(base[1], slice(1, 3)),
            self.view_observation(base[1], (slice(None), slice(1, None))),
            self.view_observation(empty_middle, slice(2, 3)),
            self.view_observation(empty_middle, (slice(None), slice(0, 1))),
            self.view_observation(module.zeros((5, 2), dtype=module.float32), slice(4, 1)),
            self.view_observation(
                module.zeros((5, 2), dtype=module.float32),
                (slice(None), slice(1, 1)),
            ),
            self.view_observation(base, slice(10**5000, None)),
            self.view_observation(base[1], (1, slice(None, None, 1))),
        )

    def test_values_layout_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(
            self.view_contract(torch),
            self.view_contract(reference_torch),
        )

    def autograd_contract(self, module):
        class SingleUseIndex:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                if self.calls > 1:
                    raise RuntimeError("slice start was converted twice")
                return 0

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        selected = source[1:3]
        metadata = (
            selected.requires_grad,
            selected.is_leaf,
            selected.output_nr,
            tuple(selected.shape),
            selected.stride(),
            selected.storage_offset(),
        )
        selected.sum().backward()

        prefixed_leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        prefixed_source = (prefixed_leaf * 2.0).reshape(2, 3, 4)
        prefixed_source[1, 1:3].sum().backward()

        tuple_leaf = module.tensor(
            values.reshape(-1).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        tuple_source = (tuple_leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        tuple_selected = tuple_source[:, 1:2]
        tuple_metadata = (
            tuple_selected.requires_grad,
            tuple_selected.is_leaf,
            tuple_selected.output_nr,
            tuple(tuple_selected.shape),
            tuple_selected.stride(),
            tuple_selected.storage_offset(),
        )
        tuple_selected.sum().backward()

        no_grad_source = module.tensor(
            values.tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            untracked = no_grad_source[1:3]
            tuple_untracked = no_grad_source[:, 1:3]

        empty = module.zeros((3, 0, 4), dtype=module.float32, requires_grad=True)
        empty[2:3].sum().backward()

        tuple_empty = module.zeros(
            (3, 0, 4), dtype=module.float32, requires_grad=True
        )
        tuple_empty[:, 0:1].sum().backward()

        single_use_start = SingleUseIndex()
        single_use_selected = module.tensor(
            [2.0], dtype=module.float32, requires_grad=True
        )[(slice(single_use_start, None, 1),)]

        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "prefixed_gradient": prefixed_leaf.grad.tolist(),
            "tuple_metadata": tuple_metadata,
            "tuple_gradient": tuple_leaf.grad.tolist(),
            "no_grad": (
                untracked.requires_grad,
                untracked.is_leaf,
                untracked.output_nr,
                tuple(untracked.shape),
                untracked.stride(),
                untracked.storage_offset(),
            ),
            "tuple_no_grad": (
                tuple_untracked.requires_grad,
                tuple_untracked.is_leaf,
                tuple_untracked.output_nr,
                tuple(tuple_untracked.shape),
                tuple_untracked.stride(),
                tuple_untracked.storage_offset(),
            ),
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient_stride": empty.grad.stride(),
            "empty_gradient_offset": empty.grad.storage_offset(),
            "empty_gradient": empty.grad.tolist(),
            "tuple_empty_gradient_shape": tuple(tuple_empty.grad.shape),
            "tuple_empty_gradient_stride": tuple_empty.grad.stride(),
            "tuple_empty_gradient_offset": tuple_empty.grad.storage_offset(),
            "tuple_empty_gradient": tuple_empty.grad.tolist(),
            "node_diagnostics": (
                self.dropout_probability_node(
                    module,
                    module.tensor([2.0], dtype=module.float32, requires_grad=True)[
                        (slice(None, None, 1),)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([2.0], dtype=module.float32, requires_grad=True)[
                        (slice(0, 1),)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([2.0], dtype=module.float32, requires_grad=True)[
                        (slice(np.int64(0), None, 1),)
                    ],
                ),
                self.dropout_probability_node(module, single_use_selected),
                self.dropout_probability_node(
                    module,
                    module.tensor([[2.0]], dtype=module.float32, requires_grad=True)[
                        0, slice(None, None, 1)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([[2.0]], dtype=module.float32, requires_grad=True)[
                        0, slice(0, 1)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([[2.0]], dtype=module.float32, requires_grad=True)[
                        0, slice(np.int64(0), None, 1)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([2.0], dtype=module.float32, requires_grad=True)[
                        (slice(-1, None),)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([[2.0]], dtype=module.float32, requires_grad=True)[
                        0, slice(-1, None)
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([[2.0, 3.0]], dtype=module.float32, requires_grad=True)[
                        :, 1:2
                    ],
                ),
                self.dropout_probability_node(
                    module,
                    module.tensor([[2.0]], dtype=module.float32, requires_grad=True)[
                        :, 0:1
                    ],
                ),
            ),
            "single_use_start_calls": single_use_start.calls,
        }

    def test_backward_through_sum_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def scalar_error_contract(self, module):
        errors = []
        scalar = module.tensor(1.0, dtype=module.float32)
        for index in (slice(None, None, 0), slice(None, None, 2)):
            try:
                scalar[index]
            except Exception as error:
                errors.append((type(error).__name__, str(error).splitlines()[0]))
            else:
                self.fail(f"scalar slice {index!r} unexpectedly succeeded")
        return tuple(errors)

    def test_scalar_slice_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.scalar_error_contract(torch),
            self.scalar_error_contract(reference_torch),
        )

    def tuple_error_contract(self, module):
        errors = []
        cases = (
            (module.tensor(1.0, dtype=module.float32), (slice(1, None),)),
            (
                module.zeros((3,), dtype=module.float32),
                (slice(None), slice(1, 2)),
            ),
            (
                module.zeros((3,), dtype=module.float32),
                (slice(None), Ellipsis, slice(1, 2)),
            ),
            (
                module.zeros((3, 4), dtype=module.float32),
                (slice(None), slice(None, None, 0)),
            ),
        )
        for source, index in cases:
            try:
                source[index]
            except Exception as error:
                errors.append((type(error).__name__, str(error).splitlines()[0]))
            else:
                self.fail(f"tuple slice {index!r} unexpectedly succeeded")
        return tuple(errors)

    def test_tuple_slice_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.tuple_error_contract(torch),
            self.tuple_error_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
