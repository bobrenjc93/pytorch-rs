import copy
import math
import pickle
import struct
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


CONSTANT_NAMES = ("e", "pi", "nan", "inf")


def float_bits(value):
    return struct.pack(">d", value)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NumericConstantReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "numeric constant differentials require pinned PyTorch 2.13.0"
            )

    def namespace_contract(self, module):
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        return {
            "relative_all_order": tuple(
                name for name in module.__all__ if name in CONSTANT_NAMES
            ),
            "all_counts": tuple(module.__all__.count(name) for name in CONSTANT_NAMES),
            "wildcard_identities": tuple(
                wildcard_namespace[name] is getattr(module, name)
                for name in CONSTANT_NAMES
            ),
            "native_absence": tuple(
                not hasattr(module._C, name) for name in CONSTANT_NAMES
            ),
        }

    def value_contract(self, module):
        results = []
        for name in CONSTANT_NAMES:
            value = getattr(module, name)
            results.append(
                {
                    "type": type(value).__name__,
                    "repr": repr(value),
                    "hex": value.hex(),
                    "bits": float_bits(value),
                    "math_identity": value is getattr(math, name),
                    "isfinite": math.isfinite(value),
                    "isinf": math.isinf(value),
                    "isnan": math.isnan(value),
                    "positive": value > 0.0,
                    "copy_identity": copy.copy(value) is value,
                    "deepcopy_identity": copy.deepcopy(value) is value,
                    "pickle_payloads": tuple(
                        pickle.dumps(value, protocol=protocol)
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                    ),
                    "pickle_results": tuple(
                        (
                            type(restored).__name__,
                            float_bits(restored),
                            restored is value,
                        )
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                        for restored in (
                            pickle.loads(pickle.dumps(value, protocol=protocol)),
                        )
                    ),
                }
            )
        return results

    def tensor_contract(self, module):
        default_dtype = module.get_default_dtype()
        base = module.tensor([1.0, -2.0, 0.0])

        def tensor_bits(tensor):
            return tuple(struct.pack(">f", value) for value in tensor.tolist())

        outcomes = []
        for name in CONSTANT_NAMES:
            value = getattr(module, name)
            results = (
                base + value,
                value + base,
                base * value,
                value * base,
            )
            outcomes.append(
                {
                    "tensor_dtype": str(module.tensor(value).dtype),
                    "scalar_tensor_dtype": str(module.scalar_tensor(value).dtype),
                    "result_dtypes": tuple(str(result.dtype) for result in results),
                    "result_bits": tuple(tensor_bits(result) for result in results),
                }
            )
        return {
            "default_dtype": str(default_dtype),
            "base_dtype": str(base.dtype),
            "base_bits": tensor_bits(base),
            "outcomes": outcomes,
            "default_dtype_identity_unchanged": module.get_default_dtype()
            is default_dtype,
        }

    def test_namespace_and_values_match_pytorch_2_13(self):
        self.assertEqual(
            self.namespace_contract(torch),
            self.namespace_contract(reference_torch),
        )
        self.assertEqual(
            self.value_contract(torch),
            self.value_contract(reference_torch),
        )

    def test_tensor_defaults_and_arithmetic_match_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_contract(torch),
            self.tensor_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
