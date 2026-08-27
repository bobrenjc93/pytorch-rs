import unittest

import torch_rs as torch


class BackendPreferenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.cuda = torch.backends.cuda
        self.cudnn = torch.backends.cudnn
        self.original_flash = self.cuda.flash_sdp_enabled()
        self.original_cudnn = self.cudnn.enabled

    def tearDown(self):
        self.cuda.enable_flash_sdp(self.original_flash)
        self.cudnn.enabled = self.original_cudnn

    def test_flash_and_cudnn_preferences_are_independent(self):
        self.cuda.enable_flash_sdp(True)
        self.cudnn.enabled = True

        self.cudnn.enabled = False
        self.assertIs(self.cudnn.enabled, False)
        self.assertIs(self.cuda.flash_sdp_enabled(), True)

        self.cuda.enable_flash_sdp(False)
        self.assertIs(self.cudnn.enabled, False)
        self.assertIs(self.cuda.flash_sdp_enabled(), False)

        self.cudnn.enabled = True
        self.assertIs(self.cudnn.enabled, True)
        self.assertIs(self.cuda.flash_sdp_enabled(), False)

        self.cuda.enable_flash_sdp(True)
        self.assertIs(self.cudnn.enabled, True)
        self.assertIs(self.cuda.flash_sdp_enabled(), True)

        with self.assertRaises(RuntimeError):
            self.cudnn.enabled = 1
        self.assertIs(self.cudnn.enabled, True)
        self.assertIs(self.cuda.flash_sdp_enabled(), True)

        with self.assertRaises(RuntimeError):
            self.cuda.enable_flash_sdp(1)
        self.assertIs(self.cudnn.enabled, True)
        self.assertIs(self.cuda.flash_sdp_enabled(), True)


if __name__ == "__main__":
    unittest.main()
