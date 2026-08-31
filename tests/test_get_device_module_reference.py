import copy
import functools
import importlib
import inspect
import pickle
import pickletools
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class GetDeviceModuleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "get_device_module differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        torch.get_device_module.cache_clear()
        reference_torch.get_device_module.cache_clear()
        self.addCleanup(torch.get_device_module.cache_clear)
        self.addCleanup(reference_torch.get_device_module.cache_clear)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def cache_outcome(self, module):
        function = module.get_device_module
        function.cache_clear()
        outcomes = []
        calls = (
            lambda: function("cpu"),
            lambda: function("cpu"),
            lambda: function(device="cpu"),
            lambda: function("cpu:0"),
            lambda: function(module.device("cpu")),
            lambda: function(module.device("cpu")),
            lambda: function(module.device("cpu:0")),
        )
        for call in calls:
            result = call()
            info = function.cache_info()
            outcomes.append(
                (
                    result.__name__.replace("torch_rs", "torch"),
                    result is module.cpu,
                    (info.hits, info.misses, info.maxsize, info.currsize),
                )
            )
        return outcomes

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

    def test_cpu_strings_and_descriptors_match_pytorch_2_13(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        specifications = (
            lambda module: "cpu",
            lambda module: "cpu:0",
            lambda module: "cpu:127",
            lambda module: "cpu:128",
            lambda module: "cpu:255",
            lambda module: module.device("cpu"),
            lambda module: module.device("cpu", 0),
            lambda module: module.device("cpu:127"),
            lambda module: copy.copy(module.device("cpu:7")),
            lambda module: pickle.loads(pickle.dumps(module.device("cpu:9"))),
        )

        for specification in specifications:
            with self.subTest(specification=specification):
                actual = torch.get_device_module(specification(torch))
                expected = reference_torch.get_device_module(
                    specification(reference_torch)
                )
                self.assertIs(actual, actual_cpu)
                self.assertIs(expected, expected_cpu)
                self.assertEqual(
                    actual.__name__.replace("torch_rs", "torch"),
                    expected.__name__,
                )
                self.assertIs(type(actual), type(expected))

    def test_signature_documentation_annotations_and_exports_match(self):
        actual = torch.get_device_module
        expected = reference_torch.get_device_module

        self.assertIs(type(actual), type(expected))
        self.assertIs(type(actual), type(functools.cache(lambda: None)))
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertEqual(
            hasattr(actual, "__annotate__"),
            hasattr(expected, "__annotate__"),
        )
        if hasattr(actual, "__annotations__"):
            self.assertEqual(
                str(actual.__annotations__).replace("torch_rs", "torch"),
                str(expected.__annotations__),
            )
        self.assertEqual(
            str(inspect.get_annotations(actual)).replace("torch_rs", "torch"),
            str(inspect.get_annotations(expected)),
        )
        self.assertEqual(
            str(typing.get_type_hints(actual)).replace("torch_rs", "torch"),
            str(typing.get_type_hints(expected)),
        )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), torch)
        self.assertIs(inspect.getmodule(expected), reference_torch)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(tuple(actual.__dict__), tuple(expected.__dict__))
        self.assertEqual(actual.cache_parameters(), expected.cache_parameters())
        self.assertEqual(
            str(inspect.signature(actual.__wrapped__)).replace("torch_rs", "torch"),
            str(inspect.signature(expected.__wrapped__)),
        )
        self.assertEqual(
            str(inspect.get_annotations(actual.__wrapped__)).replace(
                "torch_rs", "torch"
            ),
            str(inspect.get_annotations(expected.__wrapped__)),
        )
        self.assertEqual(
            actual.__wrapped__.__defaults__, expected.__wrapped__.__defaults__
        )
        self.assertEqual(
            actual.__wrapped__.__kwdefaults__, expected.__wrapped__.__kwdefaults__
        )
        self.assertEqual(
            torch.__all__.count("get_device_module"),
            reference_torch.__all__.count("get_device_module"),
        )
        self.assertEqual(torch.__all__.count("get_device_module"), 1)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs import *", actual_namespace)
        exec("from torch import *", expected_namespace)
        self.assertIs(actual_namespace["get_device_module"], actual)
        self.assertIs(expected_namespace["get_device_module"], expected)

    def test_cache_behavior_matches_for_supported_cpu_inputs(self):
        self.assertEqual(self.cache_outcome(torch), self.cache_outcome(reference_torch))

    def test_copy_and_pickle_match_pytorch_2_13(self):
        actual = torch.get_device_module
        expected = reference_torch.get_device_module

        for function in (actual, expected):
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

        for actual_module, expected_module in (
            (actual("cpu"), expected("cpu")),
            (torch.cpu, reference_torch.cpu),
        ):
            for operation in (
                copy.copy,
                copy.deepcopy,
                lambda module: pickle.loads(pickle.dumps(module)),
            ):
                with self.subTest(module=actual_module, operation=operation):
                    self.assert_error_matches(
                        lambda: operation(actual_module),
                        lambda: operation(expected_module),
                    )

    def test_invalid_value_errors_match_pytorch_2_13(self):
        class StringLike:
            def __str__(self):
                return "cpu"

        class BadString:
            def __str__(self):
                raise ValueError("bad device text")

        actual = torch.get_device_module
        expected = reference_torch.get_device_module
        values = (0, False, 1.5, ("cpu",), StringLike(), BadString())
        cases = [
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(foo=None), lambda: expected(foo=None)),
            (
                lambda: actual(None, device=None),
                lambda: expected(None, device=None),
            ),
        ]
        cases.extend(
            (lambda value=value: actual(value), lambda value=value: expected(value))
            for value in values
        )
        cases.extend(
            (
                lambda value=value: actual(value),
                lambda value=value: expected(value),
            )
            for value in (
                [],
                {},
                set(),
                "",
                "cpu:",
                "cpu:-1",
                "cpu:01",
                "cpu:2147483648",
            )
        )

        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_cuda_visible_reference_marks_the_cpu_only_default_boundary(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertEqual(torch.get_default_device(), torch.device("cpu"))
        self.assertEqual(
            reference_torch.get_default_device(), reference_torch.device("cpu")
        )
        self.assertIs(torch.get_device_module(), torch.cpu)
        self.assertIs(reference_torch.get_device_module(), reference_torch.cuda)
        self.assertIs(
            reference_torch.get_device_module("cuda:0"), reference_torch.cuda
        )

        probe = reference_torch.ones(1, device=reference_torch.device("cuda", 0))
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(0)

        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        with self.assertRaises(RuntimeError):
            torch.get_device_module("cuda")
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)


if __name__ == "__main__":
    unittest.main()
