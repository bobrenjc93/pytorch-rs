import gc
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorBareEllipsisIndexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "bare-Ellipsis indexing differentials require pinned PyTorch 2.13.0"
            )

    def autograd_suffix(self, module, tensor):
        if module is torch:
            return module._C._nn_functional_dropout_tensor_autograd_suffix(tensor)
        if not tensor.requires_grad:
            return ""
        if tensor.grad_fn is None:
            return ", requires_grad=True"
        return f", grad_fn=<{type(tensor.grad_fn).__name__}>"

    def layout_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist())
        cases = (
            module.tensor(-0.0),
            module.zeros((2, 0, 3)),
            base[1],
            base.transpose(0, 2)[1],
        )
        rows = []
        for source in cases:
            alias = source[...]
            rows.append(
                (
                    alias is source,
                    tuple(alias.shape),
                    alias.stride(),
                    alias.storage_offset(),
                    alias.data_ptr() == source.data_ptr(),
                    alias.is_set_to(source),
                    alias.dtype is source.dtype,
                    str(alias.device),
                    alias.tolist(),
                    tuple(
                        np.asarray(alias).reshape(-1).view(np.uint32).tolist()
                    ),
                )
            )
        return rows

    def test_layout_storage_values_and_distinct_identity_match_pytorch_2_13(self):
        self.assertEqual(
            self.layout_contract(torch),
            self.layout_contract(reference_torch),
        )

    def autograd_contract(self, module):
        leaf = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        source = (leaf * 2.0).transpose(0, 1)
        alias = source[...]
        metadata = (
            alias is source,
            alias.requires_grad,
            alias.is_leaf,
            alias.output_nr,
            alias.is_set_to(source),
            self.autograd_suffix(module, alias),
        )
        del source
        gc.collect()
        weights = module.tensor([[1.0, 2.0], [3.0, 4.0]])
        (alias * weights).sum().backward()

        scalar = module.tensor(-2.0, requires_grad=True)
        scalar_alias = scalar[...]
        scalar_metadata = (
            scalar_alias.requires_grad,
            scalar_alias.is_leaf,
            self.autograd_suffix(module, scalar_alias),
        )
        (scalar_alias * 7.0).backward()

        no_grad_rows = []
        no_grad_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        no_grad_nonleaf = no_grad_leaf * 2.0
        for source in (no_grad_leaf, no_grad_nonleaf):
            with module.no_grad():
                untracked = source[...]
            no_grad_rows.append(
                (
                    untracked is source,
                    untracked.requires_grad,
                    untracked.is_leaf,
                    untracked.output_nr,
                    untracked.is_set_to(source),
                    self.autograd_suffix(module, untracked),
                )
            )

        empty = module.zeros((2, 0, 3), requires_grad=True)
        empty_alias = empty[...]
        empty_suffix = self.autograd_suffix(module, empty_alias)
        empty_alias.sum().backward()
        return {
            "metadata": metadata,
            "gradient": leaf.grad.tolist(),
            "scalar_metadata": scalar_metadata,
            "scalar_gradient": scalar.grad.item(),
            "no_grad": no_grad_rows,
            "empty_suffix": empty_suffix,
            "empty_gradient_shape": tuple(empty.grad.shape),
            "empty_gradient": empty.grad.tolist(),
        }

    def test_alias_autograd_gradient_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def lifetime_contract(self, module):
        def retain_alias():
            temporary = module.tensor(
                np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
            )
            return temporary.transpose(0, 2)[1][...]

        retained = retain_alias()
        gc.collect()
        return (
            tuple(retained.shape),
            retained.stride(),
            retained.storage_offset(),
            retained.tolist(),
            (retained + 1.0).tolist(),
        )

    def test_alias_outlives_temporary_source_like_pytorch_2_13(self):
        self.assertEqual(
            self.lifetime_contract(torch),
            self.lifetime_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
