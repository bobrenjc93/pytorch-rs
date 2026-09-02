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

    def tearDown(self):
        reference_torch.set_default_device(None)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def default_device_outcome(self, module, requested_device):
        result = module.set_default_device(requested_device)
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

    def test_cpu_default_equivalent_requests_match_pytorch_2_13(self):
        requests = (
            lambda module: None,
            lambda module: "cpu",
            lambda module: module.device("cpu"),
            lambda module: module.device("cpu", None),
            lambda module: module.get_default_device(),
        )
        for create_request in requests:
            with self.subTest(create_request=create_request):
                reference_torch.set_default_device(None)
                self.assertEqual(
                    self.default_device_outcome(torch, create_request(torch)),
                    self.default_device_outcome(
                        reference_torch,
                        create_request(reference_torch),
                    ),
                )
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            str(inspect.signature(actual)),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            "set_default_device" in torch.__all__,
            "set_default_device" in reference_torch.__all__,
        )
        self.assertEqual(torch.__all__.count("set_default_device"), 1)
        self.assertEqual(
            inspect.cleandoc(actual.__doc__).splitlines()[0],
            inspect.cleandoc(expected.__doc__).splitlines()[0],
        )
        self.assertIn(
            "Mutable default-device routing",
            inspect.cleandoc(actual.__doc__),
        )

    def test_binding_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.set_default_device(),
                lambda: reference_torch.set_default_device(),
            ),
            (
                lambda: torch.set_default_device(None, None),
                lambda: reference_torch.set_default_device(None, None),
            ),
            (
                lambda: torch.set_default_device(None, None, None),
                lambda: reference_torch.set_default_device(None, None, None),
            ),
            (
                lambda: torch.set_default_device(foo=None),
                lambda: reference_torch.set_default_device(foo=None),
            ),
            (
                lambda: torch.set_default_device(None, device=None),
                lambda: reference_torch.set_default_device(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertEqual(torch.get_default_device(), torch.device("cpu"))

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device

        actual_direct = {}
        expected_direct = {}
        exec("from torch_rs import set_default_device", actual_direct)
        exec("from torch import set_default_device", expected_direct)
        self.assertIs(actual_direct["set_default_device"], actual)
        self.assertIs(expected_direct["set_default_device"], expected)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs import *", actual_wildcard)
        exec("from torch import *", expected_wildcard)
        self.assertIs(actual_wildcard["set_default_device"], actual)
        self.assertIs(expected_wildcard["set_default_device"], expected)

        for function in (actual, expected):
            with self.subTest(module=function.__module__):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(function, protocol)),
                            function,
                        )


if __name__ == "__main__":
    unittest.main()
