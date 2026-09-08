import os
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def _cuda_roundtrip_test_unavailable():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        return "run with CUDA_VISIBLE_DEVICES=0"
    if reference_torch is None:
        return "install PyTorch for CUDA differentials"
    if not reference_torch.cuda.is_available():
        return "reference PyTorch CUDA is unavailable"
    return None


def _cuda_two_gpu_test_unavailable():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
        return "run with CUDA_VISIBLE_DEVICES=0,1"
    if reference_torch is None:
        return "install PyTorch for CUDA differentials"
    if not reference_torch.cuda.is_available():
        return "reference PyTorch CUDA is unavailable"
    if reference_torch.cuda.device_count() < 2:
        return "two CUDA devices are required"
    return None


_CUDA_ROUNDTRIP_SKIP_REASON = _cuda_roundtrip_test_unavailable()
_CUDA_TWO_GPU_SKIP_REASON = _cuda_two_gpu_test_unavailable()


@unittest.skipIf(_CUDA_ROUNDTRIP_SKIP_REASON is not None, _CUDA_ROUNDTRIP_SKIP_REASON)
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

    def test_to_device_none_keeps_cuda_tensor_on_device(self):
        actual = torch.zeros((2,), device="cuda:0")
        expected = reference_torch.zeros((2,), device="cuda:0")

        self.assertIs(actual.to(device=None), actual)
        self.assertIs(expected.to(device=None), expected)
        self.assertIs(actual.to(dtype=torch.float32, device=None), actual)
        self.assertIs(expected.to(dtype=reference_torch.float32, device=None), expected)
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertTrue(actual.is_cuda)

    def test_cuda_zero_capacity_overflow_fails_before_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.zeros((2**62,), device="cuda:0")

    def test_unindexed_cuda_devices_fail_closed(self):
        expected = reference_torch.zeros((1,), device="cuda")
        self.assertEqual(str(expected.device), "cuda:0")

        for device in ("cuda", torch.device("cuda")):
            with self.subTest(device=device):
                with self.assertRaisesRegex(NotImplementedError, "unindexed CUDA"):
                    torch.zeros((1,), device=device)

        tensor = torch.zeros((1,), device="cuda:0")
        with self.assertRaisesRegex(NotImplementedError, "unindexed CUDA"):
            tensor.to("cuda")

    def test_equal_on_cuda_tensors_fails_closed_without_panic(self):
        tensor = torch.zeros((1,), device="cuda:0")
        expected = reference_torch.zeros((1,), device="cuda:0")
        self.assertTrue(reference_torch.equal(expected, expected))
        self.assertTrue(expected.equal(expected))

        with self.assertRaisesRegex(NotImplementedError, "CUDA tensor equality"):
            torch.equal(tensor, tensor)
        with self.assertRaisesRegex(NotImplementedError, "CUDA tensor equality"):
            tensor.equal(tensor)

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


@unittest.skipIf(_CUDA_TWO_GPU_SKIP_REASON is not None, _CUDA_TWO_GPU_SKIP_REASON)
class CudaCurrentDeviceGuardTests(unittest.TestCase):
    def setUp(self):
        self._previous_device = reference_torch.cuda.current_device()
        self.addCleanup(reference_torch.cuda.set_device, self._previous_device)

    def test_allocation_and_cpu_copy_preserve_current_cuda_device(self):
        reference_torch.cuda.set_device(0)

        tensor = torch.zeros((1,), device="cuda:1")
        self.assertEqual(reference_torch.cuda.current_device(), 0)

        self.assertEqual(tensor.cpu().tolist(), [0.0])
        self.assertEqual(reference_torch.cuda.current_device(), 0)

        self.assertEqual(tensor.to("cpu").tolist(), [0.0])
        self.assertEqual(reference_torch.cuda.current_device(), 0)


if __name__ == "__main__":
    unittest.main()
