import copy
import inspect
import pickle
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetDefaultDeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_default_device differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        # In PyTorch 2.13, "cpu" installs a DeviceContext mode; None clears it.
        torch.set_default_device(None)
        reference_torch.set_default_device(None)

    def tearDown(self):
        # Do not leave PyTorch's default-device mode on the global stack.
        torch.set_default_device(None)
        reference_torch.set_default_device(None)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def default_device_setter_outcome(self, module, value):
        result = module.set_default_device(value)
        first = module.get_default_device()
        second = module.get_default_device()
        factories = (
            module.tensor([1.0, 2.0]),
            module.scalar_tensor(1.0),
            module.zeros((2, 0, 3)),
            module.ones((2, 3)),
            module.eye(2, 3),
            module.full((2,), 3.0),
        )
        return (
            result,
            str(first),
            repr(first),
            first.type,
            first.index,
            first == second,
            first is second,
            tuple(tensor.device == first for tensor in factories),
            tuple((tensor.device.type, tensor.device.index) for tensor in factories),
        )

    def test_default_equivalent_cpu_noops_match_pytorch_2_13(self):
        for actual_value, expected_value in (
            (None, None),
            ("cpu", "cpu"),
            (torch.device("cpu"), reference_torch.device("cpu")),
        ):
            with self.subTest(value=repr(actual_value)):
                self.assertEqual(
                    self.default_device_setter_outcome(torch, actual_value),
                    self.default_device_setter_outcome(reference_torch, expected_value),
                )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device

        self.assertEqual(
            "set_default_device" in torch.__all__,
            "set_default_device" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertEqual(
            torch.__all__.count("set_default_device"),
            reference_torch.__all__.count("set_default_device"),
        )

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs import *", actual_namespace)
        exec("from torch import *", expected_namespace)
        self.assertIs(actual_namespace["set_default_device"], actual)
        self.assertIs(expected_namespace["set_default_device"], expected)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

    def test_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.set_default_device(),
                lambda: reference_torch.set_default_device(),
            ),
            (
                lambda: torch.set_default_device("cpu", "cpu"),
                lambda: reference_torch.set_default_device("cpu", "cpu"),
            ),
            (
                lambda: torch.set_default_device(foo="cpu"),
                lambda: reference_torch.set_default_device(foo="cpu"),
            ),
            (
                lambda: torch.set_default_device("cpu", device="cpu"),
                lambda: reference_torch.set_default_device("cpu", device="cpu"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))
                self.assertEqual(
                    reference_torch.get_default_device(),
                    reference_torch.device("cpu"),
                )

        self.assertIs(torch.set_default_device(device="cpu"), None)
        self.assertIs(reference_torch.set_default_device(device="cpu"), None)
        self.assertEqual(torch.get_default_device(), torch.device("cpu"))
        self.assertEqual(
            reference_torch.get_default_device(),
            reference_torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
