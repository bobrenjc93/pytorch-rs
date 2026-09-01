import inspect
import json
import math
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


PROPERTY_NAMES = ("is_ipu", "is_mtia", "is_maia")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAcceleratorLocationFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "accelerator-location flag differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module, device):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            device=device,
            requires_grad=True,
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        detached = tracked.detach()
        detached_view = tracked_view.detach()
        tracked.sum().backward()

        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            device=device,
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,), device=device)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            *(
                (f"scalar {value!r}", module.tensor(value, device=device))
                for value in (
                    -math.inf,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    math.inf,
                    math.nan,
                )
            ),
            ("ordinary tensor", source),
            ("empty", module.zeros((2, 0, 3), device=device)),
            ("strided view", strided_view),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("detached non-leaf", detached),
            ("detached strided view", detached_view),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
            ("autograd non-leaf view", tracked_view),
            ("accumulated gradient", leaf.grad),
        )

    def location_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        data_pointer = tensor.data_ptr()
        values = {
            property_name: getattr(tensor, property_name)
            for property_name in PROPERTY_NAMES
        }
        return {
            "values": values,
            "value_types": {
                property_name: type(value).__name__
                for property_name, value in values.items()
            },
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            ),
            "storage_identity_unchanged": data_pointer == tensor.data_ptr(),
        }

    def test_supported_cpu_tensor_results_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch, "cpu")
        expected_cases = self.tensor_cases(reference_torch, "cpu")
        for (actual_name, actual), (expected_name, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            with self.subTest(case=actual_name, shape=actual.shape):
                self.assertEqual(actual_name, expected_name)
                self.assertEqual(
                    self.location_contract(actual),
                    self.location_contract(expected),
                )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_reference_pytorch_real_cuda_tensors_report_false(self):
        device = reference_torch.device("cuda", 0)
        device_name = reference_torch.cuda.get_device_name(device)
        self.assertTrue(device_name)

        for case, tensor in self.tensor_cases(reference_torch, device):
            for property_name in PROPERTY_NAMES:
                with self.subTest(
                    case=case,
                    property=property_name,
                    shape=tensor.shape,
                    gpu=device_name,
                ):
                    self.assertEqual(tensor.device.type, "cuda")
                    result = getattr(tensor, property_name)
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
        reference_torch.cuda.synchronize(device)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("accelerator-location descriptor unexpectedly accepted the operation")

    def descriptor_contract(self, module, property_name):
        descriptor = inspect.getattr_static(module.Tensor, property_name)
        tensor = module.tensor([1.0])
        actions = (
            lambda: setattr(tensor, property_name, True),
            lambda: delattr(tensor, property_name),
            lambda: descriptor.__set__(tensor, True),
            lambda: descriptor.__delete__(tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "repr": repr(descriptor),
            "class_identity": getattr(module.Tensor, property_name) is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": descriptor.__get__(tensor, module.Tensor),
            "value_type": type(
                descriptor.__get__(tensor, module.Tensor)
            ).__name__,
            "mutation_errors": tuple(self.error(action) for action in actions),
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        for property_name in PROPERTY_NAMES:
            with self.subTest(property=property_name):
                self.assertEqual(
                    self.descriptor_contract(torch, property_name),
                    self.descriptor_contract(reference_torch, property_name),
                )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
observations = {}

for property_name in PROPERTY_NAMES:
    tensor = module.tensor([1.0], dtype=module.float32)
    descriptor = inspect.getattr_static(module.Tensor, property_name)
    marker = object()

    class RecordingMode(module.overrides.TorchFunctionMode):
        def __init__(self, result):
            self.result = result
            self.calls = []

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls.append((func, types, args, kwargs))
            return self.result

    recording = RecordingMode(marker)
    with recording:
        intercepted = getattr(tensor, property_name)
    function, dispatch_types, args, kwargs = recording.calls[0]

    order = []
    class ForwardingMode(module.overrides.TorchFunctionMode):
        def __init__(self, label):
            self.label = label

        def __torch_function__(self, func, types, args=(), kwargs=None):
            order.append(self.label)
            return func(*args, **(kwargs or {}))

    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            forwarded = getattr(tensor, property_name)

    sys.setrecursionlimit(80)
    class DecliningMode(module.overrides.TorchFunctionMode):
        def __init__(self):
            self.calls = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls += 1
            return NotImplemented

    lower = RecordingMode(marker)
    upper = DecliningMode()
    try:
        with lower:
            with upper:
                getattr(tensor, property_name)
    except Exception as error:
        declining_error = [type(error).__name__, str(error)]
    else:
        declining_error = None

    observations[property_name] = {
        "intercepted": intercepted is marker,
        "call_count": len(recording.calls),
        "function_type": type(function).__name__,
        "function_name": function.__name__,
        "function_qualname": function.__qualname__,
        "function_self": function.__self__ is descriptor,
        "function_equals_descriptor_get": function == descriptor.__get__,
        "types": dispatch_types == (module.Tensor,),
        "args": len(args) == 1 and args[0] is tensor,
        "kwargs_is_none": kwargs is None,
        "forwarding_order": order,
        "forwarded": forwarded,
        "forwarded_type": type(forwarded).__name__,
        "declining_error": declining_error,
        "declining_calls": upper.calls,
        "lower_skipped": len(lower.calls) == 0,
        "stack_depth": len(module.overrides._get_current_function_mode_stack()),
    }

print(json.dumps(observations))
'''
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                f"MODULE = {module_name!r}\n"
                f"PROPERTY_NAMES = {PROPERTY_NAMES!r}\n"
                + source,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_ipu_mtia_and_maia_backends_remain_intentionally_unsupported(self):
        tensor = torch.tensor([1.0])
        self.assertTrue(hasattr(torch.Tensor, "to"))

        for backend in ("ipu", "mtia", "maia"):
            self.assertFalse(hasattr(torch, backend))
            self.assertFalse(hasattr(torch.Tensor, backend))

            for specification in (backend, f"{backend}:0"):
                reference_device = reference_torch.device(specification)
                self.assertEqual(reference_device.type, backend)

                with self.subTest(
                    backend=backend,
                    specification=specification,
                    surface="device",
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, r"only 'cpu' is implemented"
                    ):
                        torch.device(specification)
                with self.subTest(
                    backend=backend,
                    specification=specification,
                    surface="to",
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError, r"device conversions are not supported"
                    ):
                        tensor.to(specification)
                with self.subTest(
                    backend=backend,
                    specification=specification,
                    surface="create",
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, r"only 'cpu' is implemented"
                    ):
                        torch.tensor([1.0], device=specification)


if __name__ == "__main__":
    unittest.main()
