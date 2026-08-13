import inspect
import math
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsCpuReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_cpu differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        offset_view = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        extreme_empty = (
            module.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            *(module.tensor(value) for value in (
                -math.inf,
                -1.0,
                -0.0,
                0.0,
                1.0,
                math.inf,
                math.nan,
            )),
            module.zeros((2, 0, 3)),
            offset_view,
            extreme_empty,
            leaf,
            tracked,
            leaf.grad,
        )

    def test_supported_tensor_results_match_pytorch_2_13(self):
        actual_tensors = self.tensor_cases(torch)
        expected_tensors = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_tensors, expected_tensors, strict=True)
        ):
            with self.subTest(case=case):
                actual_result = actual.is_cpu
                expected_result = expected.is_cpu
                self.assertIs(type(actual_result), type(expected_result))
                self.assertIs(actual_result, expected_result)

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "is_cpu")
        tensor = module.tensor([1.0])
        errors = []
        for action in ("set", "delete"):
            try:
                if action == "set":
                    tensor.is_cpu = False
                else:
                    del tensor.is_cpu
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail(f"{module.__name__}.Tensor.is_cpu allowed {action}")
        try:
            descriptor.__get__(1, int)
        except Exception as error:
            invalid_receiver = (type(error).__name__, str(error))
        else:
            self.fail(f"{module.__name__}.Tensor.is_cpu accepted an int receiver")

        return {
            "type": type(descriptor).__name__,
            "getset_type": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "class_identity": module.Tensor.is_cpu is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor) is descriptor,
            "value": descriptor.__get__(tensor, module.Tensor),
            "value_type": type(descriptor.__get__(tensor, module.Tensor)).__name__,
            "errors": errors,
            "invalid_receiver": invalid_receiver,
        }

    def test_descriptor_and_read_only_contract_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
