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
            self.view_observation(base.transpose(0, 1), slice(1, 3)),
            self.view_observation(base[1], slice(1, 3)),
            self.view_observation(empty_middle, slice(2, 3)),
            self.view_observation(module.zeros((5, 2), dtype=module.float32), slice(4, 1)),
            self.view_observation(base, slice(10**5000, None)),
            self.view_observation(base[1], (1, slice(None, None, 1))),
        )

    def test_values_layout_aliasing_and_empties_match_pytorch_2_13(self):
        self.assertEqual(
            self.view_contract(torch),
            self.view_contract(reference_torch),
        )

    def autograd_contract(self, module):
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

        no_grad_source = module.tensor(
            values.tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            untracked = no_grad_source[1:3]

        empty = module.zeros((3, 0, 4), dtype=module.float32, requires_grad=True)
        empty[2:3].sum().backward()

        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "prefixed_gradient": prefixed_leaf.grad.tolist(),
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
            ),
        }

    def test_backward_through_sum_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
