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

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, -0.0, 2.0, 3.0],
                [4.0, 5.0, float("inf"), -7.0],
                [8.0, 9.0, 10.0, float("nan")],
            ],
            dtype=module.float32,
        )
        noncontiguous = source.transpose(0, 1)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1]),
            ("noncontiguous", noncontiguous),
            ("offset", noncontiguous[1]),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
        )

    def identity_calls(self, module):
        return (
            ("no arguments", lambda tensor: tensor.to()),
            ("dtype positional", lambda tensor: tensor.to(module.float32)),
            ("float alias positional", lambda tensor: tensor.to(module.float)),
            ("dtype keyword", lambda tensor: tensor.to(dtype=module.float32)),
            ("device string positional", lambda tensor: tensor.to("cpu")),
            (
                "device object positional",
                lambda tensor: tensor.to(module.device("cpu")),
            ),
            ("device keyword", lambda tensor: tensor.to(device="cpu")),
            (
                "device dtype positional",
                lambda tensor: tensor.to("cpu", module.float32),
            ),
            (
                "full identity keywords",
                lambda tensor: tensor.to(
                    device=module.device("cpu"),
                    dtype=module.float,
                    non_blocking=True,
                    copy=False,
                    memory_format=module.preserve_format,
                ),
            ),
            (
                "explicit none defaults",
                lambda tensor: tensor.to(
                    None, None, False, False, memory_format=None
                ),
            ),
        )

    def tensor_state(self, module, tensor):
        if module is reference_torch:
            values = tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).tolist()
        else:
            values = np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist()
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "data_ptr": tensor.data_ptr(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "output_nr": tensor.output_nr,
            "values": values,
        }

    def comparable_state(self, module, tensor):
        state = self.tensor_state(module, tensor)
        state.pop("data_ptr")
        return state

    def test_identity_forms_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        actual_calls = self.identity_calls(torch)
        expected_calls = self.identity_calls(reference_torch)

        for (case, actual), (_, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for (form, actual_call), (_, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                with self.subTest(case=case, form=form):
                    actual_before = self.tensor_state(torch, actual)
                    expected_before = self.tensor_state(reference_torch, expected)

                    actual_result = actual_call(actual)
                    expected_result = expected_call(expected)

                    self.assertEqual(
                        actual_result is actual, expected_result is expected
                    )
                    self.assertTrue(actual_result.is_set_to(actual.detach()))
                    self.assertEqual(
                        actual_result.data_ptr() == actual_before["data_ptr"],
                        expected_result.data_ptr() == expected_before["data_ptr"],
                    )
                    self.assertEqual(
                        self.comparable_state(torch, actual_result),
                        self.comparable_state(reference_torch, expected_result),
                    )
                    self.assertEqual(self.tensor_state(torch, actual), actual_before)
                    self.assertEqual(
                        self.tensor_state(reference_torch, expected),
                        expected_before,
                    )

    def test_autograd_identity_graph_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = (leaf * 3.0).transpose(0, 1)[1]
            result = source.to(dtype=module.float32, device="cpu")
            result.sum().backward()
            outcomes.append(
                (
                    result is source,
                    source.requires_grad,
                    source.is_leaf,
                    source.output_nr,
                    np.asarray(leaf.grad).copy(),
                )
            )

        self.assertEqual(outcomes[0][:4], outcomes[1][:4])
        np.testing.assert_array_equal(outcomes[0][4], outcomes[1][4])

    def test_descriptor_documentation_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        actual_descriptor = inspect.getattr_static(torch.Tensor, "to")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "to")

        for actual, expected, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual_tensor.to, expected_tensor.to, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertIs(actual_descriptor(actual_tensor), actual_tensor)
        self.assertIs(expected_descriptor(expected_tensor), expected_tensor)
        self.assertIs(actual_descriptor(actual_tensor, torch.float32), actual_tensor)
        self.assertIs(
            expected_descriptor(expected_tensor, reference_torch.float32),
            expected_tensor,
        )

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "to")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "to")

        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
            (
                lambda: actual.to("cpu", True),
                lambda: expected.to("cpu", True),
            ),
            (lambda: actual.to(dtype=1), lambda: expected.to(dtype=1)),
            (
                lambda: actual.to(non_blocking=1),
                lambda: expected.to(non_blocking=1),
            ),
            (
                lambda: actual.to(memory_format=1),
                lambda: expected.to(memory_format=1),
            ),
            (
                lambda: actual.to(torch.float32, dtype=torch.float32),
                lambda: expected.to(
                    reference_torch.float32, dtype=reference_torch.float32
                ),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(call=actual_call):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
