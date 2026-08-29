import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UnsqueezeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("unsqueeze differentials require pinned PyTorch 2.13.0")

    def layout_cases(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32)),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
        )

    def view_contract(self, source, dim, call):
        result = call(source, dim)
        values = np.asarray(result.detach(), dtype=np.float32).reshape(-1)
        return {
            "distinct_wrapper": result is not source,
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "shared_data_pointer": result.data_ptr() == source.data_ptr(),
            "same_logical_view": result.is_set_to(source),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "values": result.tolist(),
            "value_bits": tuple(values.view(np.uint32).tolist()),
        }

    def test_method_and_top_level_views_match_pytorch_2_13(self):
        for (actual_case, actual), (expected_case, expected) in zip(
            self.layout_cases(torch),
            self.layout_cases(reference_torch),
            strict=True,
        ):
            self.assertEqual(actual_case, expected_case)
            rank = len(actual.shape)
            for dim in range(-rank - 1, rank + 1):
                for spelling, actual_call, expected_call in (
                    (
                        "method",
                        lambda source, dim: source.unsqueeze(dim),
                        lambda source, dim: source.unsqueeze(dim),
                    ),
                    (
                        "top-level",
                        lambda source, dim: torch.unsqueeze(source, dim),
                        lambda source, dim: reference_torch.unsqueeze(source, dim),
                    ),
                ):
                    with self.subTest(case=actual_case, dim=dim, spelling=spelling):
                        self.assertEqual(
                            self.view_contract(actual, dim, actual_call),
                            self.view_contract(expected, dim, expected_call),
                        )

    def error_contract(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error)
        return "ok", None

    def test_dimension_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        cases = (
            (
                lambda module, tensor: tensor.unsqueeze(),
                lambda module, tensor: tensor.unsqueeze(),
            ),
            (
                lambda module, tensor: tensor.unsqueeze(True),
                lambda module, tensor: tensor.unsqueeze(True),
            ),
            (
                lambda module, tensor: tensor.unsqueeze(1.5),
                lambda module, tensor: tensor.unsqueeze(1.5),
            ),
            (
                lambda module, tensor: tensor.unsqueeze(3),
                lambda module, tensor: tensor.unsqueeze(3),
            ),
            (
                lambda module, tensor: module.unsqueeze(),
                lambda module, tensor: module.unsqueeze(),
            ),
            (
                lambda module, tensor: module.unsqueeze(tensor),
                lambda module, tensor: module.unsqueeze(tensor),
            ),
            (
                lambda module, tensor: module.unsqueeze(1, 0),
                lambda module, tensor: module.unsqueeze(1, 0),
            ),
            (
                lambda module, tensor: module.unsqueeze(tensor, "1"),
                lambda module, tensor: module.unsqueeze(tensor, "1"),
            ),
            (
                lambda module, tensor: module.unsqueeze(input=[], dim=0),
                lambda module, tensor: module.unsqueeze(input=[], dim=0),
            ),
            (
                lambda module, tensor: module.unsqueeze(tensor, 0, out=tensor),
                lambda module, tensor: module.unsqueeze(tensor, 0, out=tensor),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(call=actual_call):
                self.assertEqual(
                    self.error_contract(lambda: actual_call(torch, actual)),
                    self.error_contract(lambda: expected_call(reference_torch, expected)),
                )

    def autograd_contract(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=6, dtype=np.float32).reshape(3, 1, 2)
        leaf = module.tensor(values.tolist(), dtype=module.float32, requires_grad=True)
        source = (leaf * 2.0).transpose(0, 2)[1]
        view = module.unsqueeze(source, 1)
        (view * module.tensor(weights.tolist(), dtype=module.float32)).sum().backward()
        return {
            "view": (
                tuple(view.shape),
                view.stride(),
                view.storage_offset(),
                view.data_ptr() == source.data_ptr(),
                view.requires_grad,
                view.is_leaf,
            ),
            "gradient": np.asarray(leaf.grad, dtype=np.float32).tolist(),
        }

    def test_autograd_contract_matches_pytorch_2_13(self):
        self.assertEqual(self.autograd_contract(torch), self.autograd_contract(reference_torch))

    @unittest.skipUnless(
        sys.maxsize == (1 << 63) - 1,
        "signed 64-bit stride wrapping requires a 64-bit Python build",
    )
    def test_extreme_empty_stride_boundaries_match_pytorch_2_13(self):
        def contract(module, leading_dimension, trailing_dimension, dim):
            source = module.zeros((0,), dtype=module.float32).reshape(
                (leading_dimension, 0, trailing_dimension)
            )
            try:
                result = module.unsqueeze(source, dim)
            except Exception as error:
                message = str(error)
                non_concrete = "SymIntArrayRef expected to contain only concrete integers"
                if non_concrete in message:
                    message = non_concrete
                return "error", type(error).__name__, message
            return (
                "result",
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
                result.data_ptr() == source.data_ptr(),
                result.is_set_to(source),
            )

        cases = (
            ((1 << 62) - 1, 2, 0),
            (1 << 62, 2, 0),
            ((1 << 62) + 1, 2, 0),
            ((1 << 62) - 1, 3, 0),
            (1 << 62, 3, 0),
            (sys.maxsize, 3, 1),
            (sys.maxsize, 3, -1),
        )
        for leading_dimension, trailing_dimension, dim in cases:
            with self.subTest(
                leading_dimension=leading_dimension,
                trailing_dimension=trailing_dimension,
                dim=dim,
            ):
                self.assertEqual(
                    contract(torch, leading_dimension, trailing_dimension, dim),
                    contract(reference_torch, leading_dimension, trailing_dimension, dim),
                )


if __name__ == "__main__":
    unittest.main()
