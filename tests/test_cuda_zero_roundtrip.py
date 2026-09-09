import os
import subprocess
import sys
import unittest
import warnings

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
    def test_tensor_copy_construction_rejects_cuda_before_sequence_conversion(self):
        expected = reference_torch.zeros((0,), device="cuda:0")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            copied = reference_torch.tensor(expected)
        self.assertEqual(copied.device, expected.device)
        self.assertEqual(copied.shape, expected.shape)

        empty = torch.zeros((0,), device="cuda:0")
        sources = (empty, empty.reshape(0, 3), empty.reshape(3, 0),
                   torch.zeros((4,), device="cuda:0")[2:2],
                   torch.zeros((1,), device="cuda:0").select(0, 0),
                   torch.zeros((1,), device="cuda:0"),
                   torch.zeros((3,), device="cuda:0"))
        for source in sources:
            before = self.cuda_metadata(torch, source)
            for data in (source, [source]):
                for kwargs in ({}, {"dtype": torch.float32}, {"device": "cpu"},
                               {"requires_grad": True}):
                    with self.subTest(shape=source.shape, nested=isinstance(data, list), kwargs=kwargs):
                        with self.assertRaisesRegex(
                            NotImplementedError,
                            r"tensor\(\): copy construction from CUDA tensors is not supported",
                        ):
                            torch.tensor(data, **kwargs)
            self.assertEqual(self.cuda_metadata(torch, source), before)
            self.assertIs(torch.as_tensor(source), source)
            self.assertEqual(source.cpu().device.type, "cpu")

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
            "is_shared": tensor.is_shared(),
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
            "is_shared": copied.is_shared(),
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

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_host_copy_allocation_failure_is_recoverable(self):
        script = r"""
import os
import resource
import torch_rs as torch

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
torch.zeros((1,), device="cuda:0").cpu()
elements = 4 * 1024 * 1024
tensor = torch.zeros((elements,), device="cuda:0")
original_limit = resource.getrlimit(resource.RLIMIT_AS)
with open("/proc/self/statm", encoding="ascii") as statm:
    virtual_bytes = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
limit = virtual_bytes + 4 * 1024 * 1024
if original_limit[1] != resource.RLIM_INFINITY and limit > original_limit[1]:
    os._exit(77)

for copy in (tensor.cpu, lambda: tensor.to("cpu")):
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, original_limit[1]))
        copy()
    except RuntimeError as error:
        assert str(error) == f"failed to allocate storage for {elements} elements", str(error)
    else:
        raise AssertionError("host allocation unexpectedly succeeded")
    finally:
        resource.setrlimit(resource.RLIMIT_AS, original_limit)

assert tensor.cpu().numel() == elements
assert torch.zeros((2,), device="cuda:0").cpu().tolist() == [0.0, 0.0]
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            # Keep large native allocations on mmap, not reusable arena memory
            # that could satisfy a reservation without growing address space.
            env={**os.environ, "MALLOC_MMAP_THRESHOLD_": "65536"},
        )
        if completed.returncode == 77:
            self.skipTest("process hard address-space limit is too low")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_cuda13_wheel_runtime_without_system_libraries_or_pytorch_import(self):
        script = r"""
import importlib.util
import pathlib
import sys
import torch_rs as torch

assert "torch" not in sys.modules
try:
    spec = importlib.util.find_spec("nvidia.cu13")
except ModuleNotFoundError:
    spec = None
if spec is None:
    raise SystemExit(77)
cuda13_libraries = {
    str(path)
    for location in spec.submodule_search_locations
    for path in (pathlib.Path(location) / "lib").glob("libcudart.so*")
}
assert cuda13_libraries
# An explicit wheel path forces the native loader to use this runtime.
import os
os.environ["TORCH_RS_CUDART"] = sorted(cuda13_libraries)[0]
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert torch.zeros((3,), device="cuda:0").cpu().tolist() == [0.0] * 3
with open("/proc/self/maps", encoding="utf8") as maps:
    assert os.environ["TORCH_RS_CUDART"] in maps.read()
assert "torch" not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode == 77:
            self.skipTest("requires the nvidia CUDA 13 runtime wheel")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_requires_grad_mutation_on_cuda_fails_closed(self):
        for elements in (0, 2):
            for args in ((), (True,), (False,)):
                with self.subTest(elements=elements, args=args):
                    actual = torch.zeros((elements,), device="cuda:0")
                    expected = reference_torch.zeros((elements,), device="cuda:0")
                    self.assertIs(expected.requires_grad_(*args), expected)
                    with self.assertRaisesRegex(
                        NotImplementedError, "device 'cuda:0' is not supported"
                    ):
                        actual.requires_grad_(*args)
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertIsNone(actual.grad)
                    self.assertEqual(actual.cpu().tolist(), [0.0] * elements)

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

    def test_cuda_tensor_type_queries_and_cpu_type_target_match_pytorch(self):
        actual = torch.zeros((2,), device="cuda:0")
        expected = reference_torch.zeros((2,), device="cuda:0")
        reference_torch.cuda.synchronize(0)

        self.assertEqual(actual.type(), expected.type())
        self.assertEqual(actual.type(None), expected.type(None))
        self.assertIs(actual.type(torch.float32), actual)
        self.assertIs(expected.type(reference_torch.float32), expected)
        self.assertIs(actual.type("torch.cuda.FloatTensor"), actual)
        self.assertIs(expected.type("torch.cuda.FloatTensor"), expected)

        actual_cpu = actual.type("torch.FloatTensor")
        expected_cpu = expected.type("torch.FloatTensor")
        self.assertIsNot(actual_cpu, actual)
        self.assertEqual(str(actual_cpu.device), str(expected_cpu.device))
        self.assertFalse(actual_cpu.is_cuda)
        self.assertEqual(actual_cpu.tolist(), expected_cpu.tolist())

        actual_type_as_cpu = actual.type_as(torch.zeros((2,)))
        expected_type_as_cpu = expected.type_as(reference_torch.zeros((2,)))
        self.assertEqual(str(actual_type_as_cpu.device), str(expected_type_as_cpu.device))
        self.assertFalse(actual_type_as_cpu.is_cuda)
        self.assertEqual(actual_type_as_cpu.tolist(), expected_type_as_cpu.tolist())

        with self.assertRaisesRegex(NotImplementedError, "CUDA tensor conversions"):
            torch.zeros((2,)).type_as(actual)

    def test_as_tensor_cpu_target_copies_cuda_tensor_to_host(self):
        actual = torch.zeros((2,), device="cuda:0")
        expected = reference_torch.zeros((2,), device="cuda:0")
        reference_torch.cuda.synchronize(0)

        for actual_options, expected_options in (
            ({}, {}),
            ({"device": None}, {"device": None}),
            ({"dtype": torch.float32}, {"dtype": reference_torch.float32}),
        ):
            with self.subTest(options=actual_options):
                actual_alias = torch.as_tensor(actual, **actual_options)
                expected_alias = reference_torch.as_tensor(expected, **expected_options)
                self.assertIs(actual_alias, actual)
                self.assertIs(expected_alias, expected)
                self.assertEqual(str(actual_alias.device), str(expected_alias.device))

        for actual_options, expected_options in (
            ({"device": "cpu"}, {"device": "cpu"}),
            ({"device": torch.device("cpu")}, {"device": reference_torch.device("cpu")}),
            (
                {"dtype": torch.float32, "device": "cpu"},
                {"dtype": reference_torch.float32, "device": "cpu"},
            ),
        ):
            with self.subTest(options=actual_options):
                actual_cpu = torch.as_tensor(actual, **actual_options)
                expected_cpu = reference_torch.as_tensor(expected, **expected_options)
                self.assertIsNot(actual_cpu, actual)
                self.assertEqual(str(actual_cpu.device), str(expected_cpu.device))
                self.assertFalse(actual_cpu.is_cuda)
                self.assertEqual(actual_cpu.tolist(), expected_cpu.tolist())

    def test_cuda_tensor_repr_is_inspectable(self):
        actual = torch.zeros((2,), device="cuda:0")
        expected = reference_torch.zeros((2,), device="cuda:0")
        reference_torch.cuda.synchronize(0)

        actual_repr = repr(actual)
        expected_repr = repr(expected)
        self.assertIn("tensor(", actual_repr)
        self.assertIn("0.0", actual_repr)
        self.assertIn("device='cuda:0'", actual_repr)
        self.assertIn("shape=[2]", actual_repr)
        self.assertIn("device='cuda:0'", expected_repr)

    def test_cuda_zero_capacity_overflow_fails_before_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.zeros((2**62,), device="cuda:0")

    def test_unindexed_cuda_devices_fail_closed(self):
        expected = reference_torch.zeros((1,), device="cuda")
        self.assertEqual(str(expected.device), "cuda:0")

        for device in (
            "cuda",
            "cuda:255",
            "cuda:511",
            torch.device("cuda"),
            torch.device("cuda:255"),
        ):
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

    def test_reflected_scalar_division_on_cuda_fails_closed_without_panic(self):
        tensor = torch.zeros((1,), device="cuda:0")
        expected = reference_torch.zeros((1,), device="cuda:0")

        reference_results = (
            1 / expected,
            reference_torch.div(1, expected),
            reference_torch.divide(1, expected),
        )
        reference_torch.cuda.synchronize(0)
        for result in reference_results:
            self.assertEqual(str(result.device), "cuda:0")

        with self.assertRaisesRegex(
            NotImplementedError, "device 'cuda:0' is not supported"
        ):
            1 / tensor

        for call in (
            lambda: torch.div(1, tensor),
            lambda: torch.divide(1, tensor),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError, "only exact native CPU float32"
                ):
                    call()

    def test_biased_linear_rejects_cuda_operands_without_panicking(self):
        cuda_singleton = torch.zeros((1,), device="cuda:0")
        cuda_empty = torch.zeros((0,), device="cuda:0")
        cases = (
            ("rank_one_singleton_bias", torch.ones((1,)), torch.ones((1, 1)), cuda_singleton),
            ("rank_two_singleton_bias", torch.ones((1, 1)), torch.ones((1, 1)), cuda_singleton),
            ("broadcast_bias", torch.ones((2, 1)), torch.ones((3, 1)), cuda_singleton),
            ("empty_rows_cuda_bias", torch.ones((0, 1)), torch.ones((1, 1)), cuda_singleton),
            ("empty_cuda_bias", torch.ones((1, 1)), torch.ones((0, 1)), cuda_empty),
            ("cuda_input", cuda_singleton, torch.ones((1, 1)), torch.ones((1,))),
            ("empty_cuda_input", cuda_empty, torch.ones((1, 0)), torch.ones((1,))),
        )
        for name, input, weight, bias in cases:
            with self.subTest(case=name):
                with self.assertRaisesRegex(
                    NotImplementedError, "device 'cuda:0' is not supported"
                ):
                    torch.nn.functional.linear(input, weight, bias)

        # Public CUDA weights are currently 1-D, so the binding rejects their
        # rank before reaching the native matrix multiplication device guard.
        for weight in (cuda_singleton, cuda_empty):
            with self.subTest(cuda_weight_numel=weight.numel()):
                with self.assertRaises(NotImplementedError):
                    torch.nn.functional.linear(
                        torch.ones((1,)), weight, torch.ones((1,))
                    )

        self.assertEqual(cuda_singleton.cpu().tolist(), [0.0])
        self.assertEqual(cuda_empty.cpu().tolist(), [])

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

        # Exercise both pitched transfers and bounded staging on another GPU.
        for rows in (3, 20003):
            view = torch.zeros((rows * 7,), device="cuda:1").reshape(rows, 7).select(1, 2)
            self.assertEqual(reference_torch.cuda.current_device(), 0)
            for copy in (view.cpu, lambda: view.to("cpu")):
                self.assertEqual(copy().tolist(), [0.0] * rows)
                self.assertEqual(reference_torch.cuda.current_device(), 0)


if __name__ == "__main__":
    unittest.main()
