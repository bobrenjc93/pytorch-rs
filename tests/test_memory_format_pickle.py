import copy
import pickle
import unittest

import torch_rs as torch


MEMORY_FORMATS = (
    ("preserve_format", torch.preserve_format),
    ("contiguous_format", torch.contiguous_format),
    ("channels_last", torch.channels_last),
    ("channels_last_3d", torch.channels_last_3d),
)


def memory_format_metadata(value):
    formats = tuple(singleton for _, singleton in MEMORY_FORMATS)
    return (
        repr(value),
        str(value),
        tuple(value == other for other in formats),
        tuple(value != other for other in formats),
        tuple(hash(value) == hash(other) for other in formats),
        type(hash(value)),
    )


class MemoryFormatPickleTests(unittest.TestCase):
    def test_public_reductions_and_every_pickle_protocol_restore_singletons(self):
        self.assertIs(torch.torch, torch)
        self.assertNotIn("torch", torch.__all__)

        for name, singleton in MEMORY_FORMATS:
            public_name = f"torch.{name}"
            expected_metadata = memory_format_metadata(singleton)
            with self.subTest(memory_format=public_name):
                self.assertEqual(singleton.__reduce__(), public_name)
                self.assertIs(getattr(torch.torch, name), singleton)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(memory_format=public_name, protocol=protocol):
                    self.assertEqual(singleton.__reduce_ex__(protocol), public_name)
                    restored = pickle.loads(
                        pickle.dumps(singleton, protocol=protocol)
                    )
                    self.assertIs(restored, singleton)
                    self.assertEqual(
                        memory_format_metadata(restored), expected_metadata
                    )

    def test_shallow_and_deep_copy_preserve_identity_and_metadata(self):
        for name, singleton in MEMORY_FORMATS:
            expected_metadata = memory_format_metadata(singleton)
            for operation, copied in (
                ("copy", copy.copy(singleton)),
                ("deepcopy", copy.deepcopy(singleton)),
            ):
                with self.subTest(memory_format=name, operation=operation):
                    self.assertIs(copied, singleton)
                    self.assertEqual(copied.__reduce__(), f"torch.{name}")
                    self.assertEqual(
                        memory_format_metadata(copied), expected_metadata
                    )


if __name__ == "__main__":
    unittest.main()
