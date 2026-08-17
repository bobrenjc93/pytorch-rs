import copy
import pickle
import unittest

import torch_rs as torch


class DevicePickleTests(unittest.TestCase):
    def assert_cpu_device_clone(self, restored, original):
        self.assertIsNot(restored, original)
        self.assertIs(type(restored), type(original))
        self.assertEqual(restored, original)
        self.assertEqual(restored.type, original.type)
        self.assertEqual(restored.index, original.index)
        self.assertEqual(repr(restored), repr(original))
        self.assertEqual(str(restored), str(original))
        self.assertEqual(hash(restored), hash(original))

    def test_reduce_and_every_pickle_protocol_match_cpu_contract(self):
        cases = (
            (torch.device("cpu"), (torch.device, ("cpu",))),
            (torch.device("cpu", 0), (torch.device, ("cpu", 0))),
            (torch.device("cpu:7"), (torch.device, ("cpu", 7))),
            (torch.device(type="cpu", index=127), (torch.device, ("cpu", 127))),
        )

        for original, expected_reduction in cases:
            with self.subTest(device=original):
                self.assertEqual(original.__reduce__(), expected_reduction)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(device=original, protocol=protocol):
                        self.assertEqual(
                            original.__reduce_ex__(protocol), expected_reduction
                        )
                        restored = pickle.loads(
                            pickle.dumps(original, protocol=protocol)
                        )
                        self.assert_cpu_device_clone(restored, original)

    def test_shallow_and_deep_copy_produce_equal_distinct_devices(self):
        for original in (
            torch.device("cpu"),
            torch.device("cpu", 0),
            torch.device("cpu:7"),
            torch.device(type="cpu", index=127),
        ):
            for name, restored in (
                ("copy", copy.copy(original)),
                ("deepcopy", copy.deepcopy(original)),
            ):
                with self.subTest(device=original, operation=name):
                    self.assert_cpu_device_clone(restored, original)


if __name__ == "__main__":
    unittest.main()
