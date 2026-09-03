import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorToReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.to differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided = source.transpose(0, 1)
        channels_last = module.zeros(
            (2, 3, 4, 5), dtype=module.float32
        ).contiguous(memory_format=module.channels_last)
        special_bits = np.asarray(
            (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty view", module.zeros((2, 0, 3), dtype=module.float32)[1]),
            ("strided view", strided[1]),
            ("channels last", channels_last),
            ("leaf", leaf),
            ("tracked view", tracked),
            ("special bits", module.tensor(memoryview(special_bits.view(np.float32)))),
        )

    def identity_operations(self, module):
        peer = module.tensor([1.0], dtype=module.float32)
        return (
            ("omitted", lambda tensor: tensor.to()),
            ("dtype positional", lambda tensor: tensor.to(module.float32)),
            ("float alias positional", lambda tensor: tensor.to(module.float)),
            ("dtype keyword", lambda tensor: tensor.to(dtype=module.float32)),
            ("dtype none", lambda tensor: tensor.to(dtype=None)),
            ("device none", lambda tensor: tensor.to(device=None)),
            ("cpu string", lambda tensor: tensor.to("cpu")),
            ("cpu device", lambda tensor: tensor.to(module.device("cpu"))),
            ("non-blocking keyword", lambda tensor: tensor.to(non_blocking=True)),
            ("copy false keyword", lambda tensor: tensor.to(copy=False)),
            ("none positional", lambda tensor: tensor.to(None)),
            ("none dtype positional", lambda tensor: tensor.to(None, None)),
            (
                "none then dtype keyword",
                lambda tensor: tensor.to(None, dtype=module.float32),
            ),
            (
                "device dtype positional",
                lambda tensor: tensor.to("cpu", module.float32),
            ),
            (
                "device dtype options positional",
                lambda tensor: tensor.to("cpu", module.float32, True, False),
            ),
            (
                "keyword options",
                lambda tensor: tensor.to(
                    device="cpu",
                    dtype=module.float32,
                    non_blocking=True,
                    copy=False,
                    memory_format=module.preserve_format,
                ),
            ),
            (
                "preserve memory format",
                lambda tensor: tensor.to(memory_format=module.preserve_format),
            ),
            (
                "memory format none",
                lambda tensor: tensor.to(memory_format=None),
            ),
            ("other tensor", lambda tensor: tensor.to(peer)),
            (
                "tensor keyword",
                lambda tensor: tensor.to(
                    tensor=peer, non_blocking=True, copy=False
                ),
            ),
        )

    def tensor_bits(self, module, tensor):
        if module is reference_torch:
            return tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).tolist()
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist()

    def tensor_state(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "output_nr": tensor.output_nr,
            "values": self.tensor_bits(module, tensor),
        }

    def identity_contract(self, module, tensor, operation):
        before = self.tensor_state(module, tensor)
        before_data_ptr = tensor.data_ptr()
        result = operation(tensor)
        after = self.tensor_state(module, tensor)
        return {
            "same_object": result is tensor,
            "same_pointer": result.data_ptr() == before_data_ptr,
            "same_logical_storage": result.is_set_to(tensor),
            "result_state": self.tensor_state(module, result),
            "source_unchanged": before == after,
        }

    def test_identity_forms_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        actual_operations = self.identity_operations(torch)
        expected_operations = self.identity_operations(reference_torch)

        for (case, actual), (_, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for (form, actual_operation), (_, expected_operation) in zip(
                actual_operations, expected_operations, strict=True
            ):
                with self.subTest(case=case, form=form):
                    self.assertEqual(
                        self.identity_contract(torch, actual, actual_operation),
                        self.identity_contract(
                            reference_torch, expected, expected_operation
                        ),
                    )

    def test_contiguous_memory_format_identity_matches_pytorch_2_13(self):
        actual_cases = (
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1),
        )
        expected_cases = (
            reference_torch.tensor([1.0, 2.0, 3.0]),
            reference_torch.tensor([[1.0, 2.0], [3.0, 4.0]]).transpose(0, 1),
        )
        for actual, expected in zip(actual_cases, expected_cases, strict=True):
            with self.subTest(shape=tuple(actual.shape), stride=actual.stride()):
                self.assertEqual(
                    self.identity_contract(
                        torch,
                        actual,
                        lambda tensor: tensor.to(
                            dtype=torch.float32,
                            memory_format=torch.contiguous_format,
                        ),
                    ),
                    self.identity_contract(
                        reference_torch,
                        expected,
                        lambda tensor: tensor.to(
                            dtype=reference_torch.float32,
                            memory_format=reference_torch.contiguous_format,
                        ),
                    ),
                )

    def test_matching_channels_last_identity_matches_pytorch_2_13(self):
        actual = torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )
        expected = reference_torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=reference_torch.channels_last
        )

        actual_contract = self.identity_contract(
            torch,
            actual,
            lambda tensor: tensor.to(memory_format=torch.channels_last),
        )
        expected_contract = self.identity_contract(
            reference_torch,
            expected,
            lambda tensor: tensor.to(memory_format=reference_torch.channels_last),
        )
        self.assertEqual(actual_contract, expected_contract)

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = (leaf * 3.0).transpose(0, 1)
        result = source.to(module.float32, non_blocking=True)
        result.sum().backward()
        return {
            "same_object": result is source,
            "result_state": self.tensor_state(module, result),
            "grad": self.tensor_bits(module, leaf.grad),
        }

    def test_autograd_identity_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )

    def descriptor_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "to")
        bound = tensor.to
        return {
            "descriptor_type": type(descriptor),
            "bound_type": type(bound),
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "descriptor_doc": descriptor.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "descriptor_objclass_name": descriptor.__objclass__.__name__,
            "descriptor_objclass_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_doc": bound.__doc__,
            "bound_text_signature": bound.__text_signature__,
            "bound_module": bound.__module__,
            "descriptor_result_identity": descriptor(tensor) is tensor,
            "bound_result_identity": bound() is tensor,
        }

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        actual = self.descriptor_contract(torch)
        expected = self.descriptor_contract(reference_torch)
        self.assertIs(actual.pop("descriptor_type"), types.MethodDescriptorType)
        self.assertIs(expected.pop("descriptor_type"), types.MethodDescriptorType)
        self.assertIs(actual.pop("bound_type"), types.BuiltinMethodType)
        self.assertIs(expected.pop("bound_type"), types.BuiltinMethodType)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
