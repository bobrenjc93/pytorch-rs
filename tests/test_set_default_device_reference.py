import copy
import inspect
import pickle
import pickletools
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
        torch.set_default_device("cpu")
        reference_torch.set_default_device(None)

    def tearDown(self):
        torch.set_default_device("cpu")
        reference_torch.set_default_device(None)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def cpu_noop_outcome(self, module, value, *, keyword=False):
        if keyword:
            result = module.set_default_device(device=value)
        else:
            result = module.set_default_device(value)
        default_device = module.get_default_device()
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
            str(default_device),
            repr(default_device),
            default_device.type,
            default_device.index,
            str(module.get_default_dtype()),
            tuple(str(tensor.device) for tensor in factories),
            tuple(tensor.device == default_device for tensor in factories),
            tuple(str(tensor.dtype) for tensor in factories),
        )

    def test_unindexed_cpu_noops_match_pytorch_2_13(self):
        for actual, expected, keyword in (
            (None, None, False),
            (None, None, True),
            ("cpu", "cpu", False),
            ("cpu", "cpu", True),
            (torch.device("cpu"), reference_torch.device("cpu"), False),
            (torch.device(type="cpu"), reference_torch.device(type="cpu"), False),
            (
                torch.device("cpu", index=None),
                reference_torch.device("cpu", index=None),
                False,
            ),
            (torch.get_default_device(), reference_torch.get_default_device(), False),
        ):
            with self.subTest(value=repr(expected), keyword=keyword):
                self.assertEqual(
                    self.cpu_noop_outcome(torch, actual, keyword=keyword),
                    self.cpu_noop_outcome(reference_torch, expected, keyword=keyword),
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
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device

        self.assertEqual(
            torch.__all__.count("set_default_device"),
            reference_torch.__all__.count("set_default_device"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            direct_namespace = {}
            exec(f"from {module.__name__} import set_default_device", direct_namespace)
            self.assertIs(direct_namespace["set_default_device"], function)

            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["set_default_device"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

        self.assertFalse(hasattr(torch._C, "set_default_device"))
        self.assertFalse(hasattr(reference_torch._C, "set_default_device"))

    def test_argument_binding_errors_match_pytorch_2_13(self):
        actual = torch.set_default_device
        expected = reference_torch.set_default_device
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual("cpu", "cpu"), lambda: expected("cpu", "cpu")),
            (lambda: actual(foo="cpu"), lambda: expected(foo="cpu")),
            (
                lambda: actual("cpu", device="cpu"),
                lambda: expected("cpu", device="cpu"),
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


if __name__ == "__main__":
    unittest.main()
