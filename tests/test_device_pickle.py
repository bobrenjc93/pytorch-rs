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
        cpu = torch.device("cpu")
        expected_reduction = (torch.device, ("cpu",))

        self.assertEqual(cpu.__reduce__(), expected_reduction)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(cpu.__reduce_ex__(protocol), expected_reduction)
                restored = pickle.loads(pickle.dumps(cpu, protocol=protocol))
                self.assert_cpu_device_clone(restored, cpu)

    def test_shallow_and_deep_copy_produce_equal_distinct_devices(self):
        cpu = torch.device("cpu")

        for name, restored in (
            ("copy", copy.copy(cpu)),
            ("deepcopy", copy.deepcopy(cpu)),
        ):
            with self.subTest(operation=name):
                self.assert_cpu_device_clone(restored, cpu)


if __name__ == "__main__":
    unittest.main()
