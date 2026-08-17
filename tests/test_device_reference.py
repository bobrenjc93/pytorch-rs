import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "device differentials require pinned PyTorch 2.13.0"
            )

    def normalized_device(self, value):
        return (
            value.type,
            value.index,
            str(value),
            repr(value),
            hash(value),
        )

    def outcome(self, create):
        try:
            return ("return", self.normalized_device(create()))
        except Exception as error:
            return ("error", type(error).__name__, str(error))

    def test_indexed_cpu_constructor_forms_and_normalization_match(self):
        constructors = (
            lambda module: module.device("cpu"),
            lambda module: module.device("cpu", None),
            lambda module: module.device("cpu", 0),
            lambda module: module.device("cpu", 127),
            lambda module: module.device("cpu", 128),
            lambda module: module.device("cpu", 255),
            lambda module: module.device("cpu", 256),
            lambda module: module.device("cpu", 2**63 - 1),
            lambda module: module.device("cpu", np.int64(6)),
            lambda module: module.device("cpu", index=3),
            lambda module: module.device(type="cpu", index=4),
            lambda module: module.device(device=module.device("cpu:5")),
            lambda module: module.device("cpu:0"),
            lambda module: module.device("cpu:127"),
            lambda module: module.device("cpu:128"),
            lambda module: module.device("cpu:255"),
            lambda module: module.device("cpu:256"),
            lambda module: module.device("cpu:2147483647"),
        )
        for create in constructors:
            with self.subTest(create=create):
                self.assertEqual(
                    self.normalized_device(create(torch)),
                    self.normalized_device(create(reference_torch)),
                )

    def test_index_validation_and_errors_match(self):
        class Index:
            def __index__(self):
                return 2

        constructors = (
            lambda module: module.device("cpu", -1),
            lambda module: module.device("cpu", True),
            lambda module: module.device("cpu", 1.5),
            lambda module: module.device("cpu", np.bool_(True)),
            lambda module: module.device("cpu", np.float64(1.5)),
            lambda module: module.device("cpu", Index()),
            lambda module: module.device("cpu", 2**63),
            lambda module: module.device(type="cpu", index=True),
            lambda module: module.device("cpu:01"),
            lambda module: module.device("cpu:-1"),
            lambda module: module.device("cpu:"),
            lambda module: module.device("cpu:2147483648"),
            lambda module: module.device("cpu:2", None),
            lambda module: module.device(type="cpu:2"),
            lambda module: module.device(module.device("cpu"), 1),
        )
        for create in constructors:
            with self.subTest(create=create):
                self.assertEqual(
                    self.outcome(lambda: create(torch)),
                    self.outcome(lambda: create(reference_torch)),
                )

    def test_equality_and_hashing_match(self):
        def contract(module):
            devices = (
                module.device("cpu"),
                module.device("cpu", 0),
                module.device("cpu:0"),
                module.device("cpu", 1),
                module.device("cpu:1"),
                module.device("cpu", 255),
                module.device("cpu", 256),
            )
            return (
                tuple(self.normalized_device(device) for device in devices),
                tuple(
                    tuple(left == right for right in devices)
                    for left in devices
                ),
                tuple(
                    tuple(hash(left) == hash(right) for right in devices)
                    for left in devices
                ),
                tuple((device == "cpu", device != "cpu") for device in devices),
            )

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_factories_normalize_indexed_cpu_descriptors_to_cpu_storage(self):
        factories = (
            lambda module, device: module.tensor([1.25], device=device),
            lambda module, device: module.zeros((1,), device=device),
            lambda module, device: module.ones((1,), device=device),
            lambda module, device: module.eye(1, device=device),
            lambda module, device: module.full((1,), 2.5, device=device),
            lambda module, device: module.scalar_tensor(3.5, device=device),
        )
        specifications = (
            lambda module: "cpu:3",
            lambda module: module.device("cpu", 4),
            lambda module: module.device(type="cpu", index=5),
            lambda module: module.device("cpu:128"),
        )

        def tensor_contract(value):
            return (
                value.shape,
                value.stride(),
                value.tolist(),
                str(value.dtype),
                self.normalized_device(value.device),
                value.get_device(),
                value.is_cpu,
                value.is_cuda,
            )

        for factory in factories:
            for specification in specifications:
                with self.subTest(factory=factory, specification=specification):
                    actual = factory(torch, specification(torch))
                    expected = factory(
                        reference_torch, specification(reference_torch)
                    )
                    self.assertEqual(
                        tensor_contract(actual), tensor_contract(expected)
                    )
                    self.assertEqual(actual.device, torch.device("cpu"))
                    self.assertIsNone(actual.device.index)


if __name__ == "__main__":
    unittest.main()
