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
        torch.set_default_device(None)
        reference_torch.set_default_device(None)
        self.addCleanup(torch.set_default_device, None)
        self.addCleanup(reference_torch.set_default_device, None)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def default_device_outcome(self, module, value):
        result = module.set_default_device(value(module))
        default = module.get_default_device()
        factories = (
            module.tensor([1.0, 2.0]),
            module.scalar_tensor(1.0),
            module.zeros((2, 0, 3)),
            module.ones((2, 3)),
            module.eye(2, 3),
            module.full((2,), 3.0),
        )
        return (
            result is None,
            str(default),
            repr(default),
            default.type,
            default.index,
            tuple(tensor.device == default for tensor in factories),
            tuple((tensor.device.type, tensor.device.index) for tensor in factories),
        )

    def test_supported_cpu_forms_match_pytorch_2_13(self):
        cases = (
            lambda module: None,
            lambda module: "cpu",
            lambda module: "cpu:0",
            lambda module: "cpu:2",
            lambda module: module.device("cpu"),
            lambda module: module.device("cpu", 0),
            lambda module: module.device("cpu:2"),
            lambda module: copy.copy(module.device("cpu:7")),
            lambda module: pickle.loads(pickle.dumps(module.device("cpu:127"))),
        )
        for value in cases:
            with self.subTest(value=value):
                torch.set_default_device(None)
                reference_torch.set_default_device(None)
                self.assertEqual(
                    self.default_device_outcome(torch, value),
                    self.default_device_outcome(reference_torch, value),
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
        self.assertIsNone(actual.__defaults__)
        self.assertIsNone(expected.__defaults__)
        self.assertIsNone(actual.__kwdefaults__)
        self.assertIsNone(expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            "set_default_device" in torch.__all__,
            "set_default_device" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("set_default_device"), 1)

    def test_imports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs import *", actual_namespace)
        exec("from torch import *", expected_namespace)
        self.assertIs(actual_namespace["set_default_device"], actual)
        self.assertIs(expected_namespace["set_default_device"], expected)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__module__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

    def test_argument_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda module: module.set_default_device(),
                lambda module: module.set_default_device(),
            ),
            (
                lambda module: module.set_default_device("cpu", "cpu"),
                lambda module: module.set_default_device("cpu", "cpu"),
            ),
            (
                lambda module: module.set_default_device(foo="cpu"),
                lambda module: module.set_default_device(foo="cpu"),
            ),
            (
                lambda module: module.set_default_device("cpu", device="cpu"),
                lambda module: module.set_default_device("cpu", device="cpu"),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(
                    lambda: actual_call(torch),
                    lambda: expected_call(reference_torch),
                )
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))

        self.assertIsNone(torch.set_default_device(device="cpu"))
        self.assertIsNone(reference_torch.set_default_device(device="cpu"))
        self.assertEqual(
            torch.get_default_device(),
            torch.device("cpu"),
        )
        self.assertEqual(
            reference_torch.get_default_device(),
            reference_torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
