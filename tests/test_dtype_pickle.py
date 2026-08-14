import copy
import pickle
import unittest

import torch_rs as torch


def dtype_metadata(value):
    return (
        repr(value),
        str(value),
        value.itemsize,
        value.is_complex,
        value.is_signed,
    )


class DTypePickleTests(unittest.TestCase):
    def test_reduce_and_every_pickle_protocol_restore_float32(self):
        self.assertEqual(torch.float32.__reduce__(), "float32")

        expected_metadata = dtype_metadata(torch.float32)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(torch.float32, protocol=protocol)
                )
                self.assertIs(restored, torch.float32)
                self.assertEqual(dtype_metadata(restored), expected_metadata)

    def test_float_alias_and_copies_preserve_identity_and_metadata(self):
        self.assertIs(torch.float, torch.float32)

        expected_metadata = dtype_metadata(torch.float32)
        values = (
            torch.float,
            copy.copy(torch.float32),
            copy.deepcopy(torch.float32),
            copy.copy(torch.float),
            copy.deepcopy(torch.float),
        )
        for value in values:
            with self.subTest(value=value):
                self.assertIs(value, torch.float32)
                self.assertEqual(value.__reduce__(), "float32")
                self.assertEqual(dtype_metadata(value), expected_metadata)


if __name__ == "__main__":
    unittest.main()
