import copy
import pickle
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DevicePickleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "device pickle differentials require pinned PyTorch 2.13.0"
            )

    def normalized_reduction(self, module, reduction):
        constructor, arguments = reduction
        return constructor is module.device, arguments

    def normalized_device(self, value, original):
        return (
            value is original,
            type(value) is type(original),
            value == original,
            value.type,
            value.index,
            repr(value),
            str(value),
            hash(value) == hash(original),
        )

    def contract(self, module):
        cpu = module.device("cpu")
        copies = (copy.copy(cpu), copy.deepcopy(cpu))
        pickle_results = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(cpu, protocol=protocol))
            pickle_results.append(self.normalized_device(restored, cpu))

        return {
            "reduce": self.normalized_reduction(module, cpu.__reduce__()),
            "reduce_ex": tuple(
                self.normalized_reduction(module, cpu.__reduce_ex__(protocol))
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "copies": tuple(
                self.normalized_device(value, cpu) for value in copies
            ),
            "pickle_results": tuple(pickle_results),
        }

    def test_cpu_device_pickle_and_copy_match_pytorch_2_13(self):
        self.assertEqual(
            self.contract(torch),
            self.contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
