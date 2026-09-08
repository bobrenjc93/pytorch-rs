import os
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def _cuda_test_unavailable():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        return "run with CUDA_VISIBLE_DEVICES=0"
    if reference_torch is None:
        return "install PyTorch for CUDA differentials"
    if not reference_torch.cuda.is_available():
        return "reference PyTorch CUDA is unavailable"
    return None


_CUDA_SKIP_REASON = _cuda_test_unavailable()


@unittest.skipIf(_CUDA_SKIP_REASON is not None, _CUDA_SKIP_REASON)
class CudaZeroRoundtripTests(unittest.TestCase):
    def assert_cuda_probe_matches_pytorch(self):
        self.assertIs(type(torch.cuda.device_count()), int)
        self.assertEqual(torch.cuda.device_count(), reference_torch.cuda.device_count())
        self.assertIs(torch.cuda.is_available(), reference_torch.cuda.is_available())
        self.assertTrue(torch.cuda.is_available())

    @staticmethod
    def cuda_metadata(module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "device_type": tensor.device.type,
            "device_index": tensor.device.index,
            "is_cuda": tensor.is_cuda,
            "get_device": tensor.get_device(),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "data_ptr_is_zero": tensor.data_ptr() == 0,
        }

    @staticmethod
    def cpu_copy_observation(tensor):
        copied = tensor.cpu()
        return {
            "shape": tuple(copied.shape),
            "stride": tuple(copied.stride()),
            "numel": copied.numel(),
            "dtype": str(copied.dtype),
            "device": str(copied.device),
            "is_cuda": copied.is_cuda,
            "get_device": copied.get_device(),
            "values": copied.tolist(),
            "same_object": copied is tensor,
        }

    def test_empty_and_non_empty_cuda_zero_metadata_matches_pytorch(self):
        self.assert_cuda_probe_matches_pytorch()
        for elements in (0, 5):
            with self.subTest(elements=elements):
                actual = torch.zeros(
                    (elements,),
                    device="cuda:0",
                    dtype=torch.float32,
                    requires_grad=False,
                )
                expected = reference_torch.zeros(
                    (elements,),
                    device="cuda:0",
                    dtype=reference_torch.float32,
                    requires_grad=False,
                )
                reference_torch.cuda.synchronize(0)

                self.assertEqual(
                    self.cuda_metadata(torch, actual),
                    self.cuda_metadata(reference_torch, expected),
                )

    def test_cpu_and_to_cpu_copy_values_match_pytorch(self):
        for elements in (0, 7):
            with self.subTest(elements=elements):
                actual = torch.zeros((elements,), device=torch.device("cuda:0"))
                expected = reference_torch.zeros(
                    (elements,), device=reference_torch.device("cuda:0")
                )
                reference_torch.cuda.synchronize(0)

                self.assertEqual(
                    self.cpu_copy_observation(actual),
                    self.cpu_copy_observation(expected),
                )

                actual_to_cpu = actual.to("cpu")
                expected_to_cpu = expected.to("cpu")
                self.assertEqual(actual_to_cpu.tolist(), expected_to_cpu.tolist())
                self.assertEqual(str(actual_to_cpu.device), str(expected_to_cpu.device))
                self.assertFalse(actual_to_cpu.is_cuda)
                self.assertIsNot(actual_to_cpu, actual)

    def test_cuda_zero_unsupported_cases_fail_closed(self):
        unsupported_cases = (
            (
                "requires_grad",
                lambda module: module.zeros(
                    (1,), device="cuda:0", requires_grad=True
                ),
                lambda: reference_torch.zeros(
                    (1,), device="cuda:0", requires_grad=True
                ),
            ),
            (
                "rank_zero",
                lambda module: module.zeros((), device="cuda:0"),
                lambda: reference_torch.zeros((), device="cuda:0"),
            ),
            (
                "rank_two",
                lambda module: module.zeros((1, 1), device="cuda:0"),
                lambda: reference_torch.zeros((1, 1), device="cuda:0"),
            ),
            (
                "float64_dtype",
                lambda module: module.zeros(
                    (1,), device="cuda:0", dtype=reference_torch.float64
                ),
                lambda: reference_torch.zeros(
                    (1,), device="cuda:0", dtype=reference_torch.float64
                ),
            ),
            (
                "ones_factory",
                lambda module: module.ones((1,), device="cuda:0"),
                lambda: reference_torch.ones((1,), device="cuda:0"),
            ),
            (
                "empty_factory",
                lambda module: module.empty((1,), device="cuda:0"),
                lambda: reference_torch.empty((1,), device="cuda:0"),
            ),
            (
                "tensor_factory",
                lambda module: module.tensor([0.0], device="cuda:0"),
                lambda: reference_torch.tensor([0.0], device="cuda:0"),
            ),
            (
                "cpu_to_cuda_transfer",
                lambda module: module.zeros((1,)).to("cuda:0"),
                lambda: reference_torch.zeros((1,)).to("cuda:0"),
            ),
            (
                "cuda_math",
                lambda module: module.zeros((1,), device="cuda:0")
                + module.zeros((1,), device="cuda:0"),
                lambda: reference_torch.zeros((1,), device="cuda:0")
                + reference_torch.zeros((1,), device="cuda:0"),
            ),
            (
                "direct_tolist",
                lambda module: module.zeros((1,), device="cuda:0").tolist(),
                lambda: reference_torch.zeros((1,), device="cuda:0").tolist(),
            ),
        )

        for name, actual_call, reference_call in unsupported_cases:
            with self.subTest(name=name):
                reference_result = reference_call()
                if hasattr(reference_result, "device") and reference_result.is_cuda:
                    reference_torch.cuda.synchronize(0)
                with self.assertRaises(Exception):
                    actual_call(torch)


if __name__ == "__main__":
    unittest.main()
