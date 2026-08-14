import copy
import pickle
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DTypePickleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dtype pickle differentials require pinned PyTorch 2.13.0"
            )

    def metadata(self, value):
        return (
            repr(value),
            str(value),
            value.itemsize,
            value.is_complex,
            value.is_signed,
        )

    def contract(self, module):
        canonical = module.float32
        values = (
            canonical,
            module.float,
            copy.copy(canonical),
            copy.deepcopy(canonical),
            copy.copy(module.float),
            copy.deepcopy(module.float),
        )
        pickle_results = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(canonical, protocol=protocol))
            pickle_results.append(
                (restored is canonical, self.metadata(restored))
            )
        return {
            "reduce": canonical.__reduce__(),
            "reduce_ex": tuple(
                canonical.__reduce_ex__(protocol)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "alias_and_copy_identities": tuple(
                value is canonical for value in values
            ),
            "alias_and_copy_metadata": tuple(
                self.metadata(value) for value in values
            ),
            "pickle_results": tuple(pickle_results),
        }

    def test_float32_pickle_copy_and_alias_match_pytorch_2_13(self):
        self.assertEqual(
            self.contract(torch),
            self.contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
