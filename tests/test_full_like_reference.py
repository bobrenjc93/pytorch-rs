import math
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FullLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "full_like differentials require pinned PyTorch 2.13.0"
            )

    def source_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("scalar", module.tensor(-3.0, dtype=module.float32)),
            ("empty", module.ones((0,), dtype=module.float32)),
            (
                "empty multidimensional",
                module.ones((2, 0, 3), dtype=module.float32),
            ),
            ("multidimensional", module.ones((2, 3, 4), dtype=module.float32)),
            ("offset contiguous", base[1]),
        )

    def tensor_observation(self, module, tensor):
        source = tensor.detach() if tensor.requires_grad else tensor
        values = np.asarray(source, dtype=np.float32).reshape(-1).view(np.uint32)
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            values.tolist(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            str(tensor.layout),
            tensor.layout is module.strided,
            tensor.is_pinned(),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.is_contiguous(),
        )

    def test_supported_metadata_and_values_match_pytorch_2_13(self):
        actual_sources = self.source_cases(torch)
        expected_sources = self.source_cases(reference_torch)
        metadata_cases = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.preserve_format},
            lambda module: {"memory_format": module.contiguous_format},
            lambda module: {"requires_grad": True},
        )

        for (case, actual_source), (expected_case, expected_source) in zip(
            actual_sources, expected_sources, strict=True
        ):
            self.assertEqual(case, expected_case)
            for metadata_factory in metadata_cases:
                actual_kwargs = metadata_factory(torch)
                expected_kwargs = metadata_factory(reference_torch)
                with self.subTest(case=case, kwargs=actual_kwargs):
                    actual = torch.full_like(actual_source, -2.25, **actual_kwargs)
                    expected = reference_torch.full_like(
                        expected_source,
                        -2.25,
                        **expected_kwargs,
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )
                    self.assertEqual(
                        actual.is_set_to(actual_source),
                        expected.is_set_to(expected_source),
                    )
                    self.assertFalse(actual.is_set_to(actual_source))

    def test_nonfinite_and_signed_zero_values_match_pytorch_2_13(self):
        actual_source = torch.ones((2,), dtype=torch.float32)
        expected_source = reference_torch.ones((2,), dtype=reference_torch.float32)
        for fill_value in (math.nan, math.inf, -math.inf, 0.0, -0.0):
            with self.subTest(fill_value=repr(fill_value)):
                actual = torch.full_like(actual_source, fill_value)
                expected = reference_torch.full_like(expected_source, fill_value)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

    def test_fresh_storage_matches_pytorch_2_13(self):
        def contract(module):
            source = module.ones((2, 3), dtype=module.float32)
            first = module.full_like(source, 4.5)
            second = module.full_like(source, 4.5)
            return {
                "first": self.tensor_observation(module, first),
                "second": self.tensor_observation(module, second),
                "fresh_pair_storage": first.data_ptr() != second.data_ptr(),
                "fresh_pair_view": not first.is_set_to(second),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_no_grad_requires_grad_behavior_matches_pytorch_2_13(self):
        actual_source = torch.ones((2, 3), dtype=torch.float32, requires_grad=True) * 2.0
        expected_source = (
            reference_torch.ones(
                (2, 3),
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            * 2.0
        )

        with torch.no_grad():
            actual_default = torch.full_like(actual_source, 4.0)
            actual_tracked = torch.full_like(
                actual_source,
                4.0,
                requires_grad=True,
            )
        with reference_torch.no_grad():
            expected_default = reference_torch.full_like(expected_source, 4.0)
            expected_tracked = reference_torch.full_like(
                expected_source,
                4.0,
                requires_grad=True,
            )

        self.assertEqual(
            self.tensor_observation(torch, actual_default),
            self.tensor_observation(reference_torch, expected_default),
        )
        self.assertEqual(
            self.tensor_observation(torch, actual_tracked),
            self.tensor_observation(reference_torch, expected_tracked),
        )

    def test_unsupported_dtype_layout_device_and_out_boundaries_are_pinned(self):
        source = torch.ones((2,), dtype=torch.float32)
        reference_source = reference_torch.ones((2,), dtype=reference_torch.float32)
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))

        with self.assertRaisesRegex(
            TypeError,
            r"^full_like\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.full_like(source, 2.0, dtype=reference_torch.float64)
        self.assertIs(
            reference_torch.full_like(
                reference_source,
                2.0,
                dtype=reference_torch.float64,
            ).dtype,
            reference_torch.float64,
        )

        with self.assertRaisesRegex(
            TypeError,
            r"^full_like\(\): argument 'layout' must be torch\.layout, not ",
        ):
            torch.full_like(source, 2.0, layout=reference_torch.sparse_coo)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^full_like\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.full_like(source, 2.0, device="meta")
        meta = reference_torch.full_like(reference_source, 2.0, device="meta")
        self.assertEqual(str(meta.device), "meta")
        self.assertIs(meta.dtype, reference_torch.float32)
        self.assertIs(meta.layout, reference_torch.strided)

        for out in (None, torch.zeros((2,))):
            with self.subTest(out=out):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^full_like\(\) got an unexpected keyword argument 'out'$",
                ):
                    torch.full_like(source, 2.0, out=out)


if __name__ == "__main__":
    unittest.main()
