import importlib
import unittest
from collections import OrderedDict

import torch_rs as torch
import torch_rs.nn as nn

try:
    import torch as reference_torch
    import torch.nn as reference_nn
except ImportError:
    reference_torch = None
    reference_nn = None


class _ActualRecorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.events = []

    def forward(self, *args, **kwargs):
        self.events.append((args, kwargs, self.training))
        return self.events[-1]


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ModuleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise unittest.SkipTest(
                "nn.Module differentials require pinned PyTorch 2.13.0"
            )

    def make_expected(self):
        class ExpectedRecorder(reference_nn.Module):
            def __init__(self):
                super().__init__()
                self.events = []

            def forward(self, *args, **kwargs):
                self.events.append((args, kwargs, self.training))
                return self.events[-1]

        return ExpectedRecorder()

    def state_dict_snapshot(self, module):
        state = module.state_dict()
        return (
            type(state),
            list(state.items()),
            getattr(state, "_metadata", None),
        )

    def assert_unsupported_while_reference_supports(self, actual_call, expected_call):
        with self.assertRaises(NotImplementedError):
            actual_call()
        expected_call()

    def test_imports_and_supported_exports_match_reference_shape(self):
        actual_modules = importlib.import_module("torch_rs.nn.modules")
        expected_modules = importlib.import_module("torch.nn.modules")
        actual_module_file = importlib.import_module("torch_rs.nn.modules.module")
        expected_module_file = importlib.import_module("torch.nn.modules.module")

        self.assertIs(torch.nn.Module, nn.Module)
        self.assertIs(nn.modules.Module, nn.Module)
        self.assertIs(actual_modules.Module, nn.Module)
        self.assertIs(actual_module_file.Module, nn.Module)
        self.assertIs(reference_torch.nn.Module, reference_nn.Module)
        self.assertIs(reference_nn.modules.Module, reference_nn.Module)
        self.assertIs(expected_modules.Module, reference_nn.Module)
        self.assertIs(expected_module_file.Module, reference_nn.Module)
        self.assertEqual(
            nn.Module.__module__.replace("torch_rs", "torch", 1),
            reference_nn.Module.__module__,
        )
        self.assertIn("Module", actual_modules.__all__)
        self.assertIn("Module", expected_modules.__all__)
        self.assertFalse(hasattr(nn, "Parameter"))
        self.assertTrue(hasattr(reference_nn, "Parameter"))

    def test_training_call_and_empty_iteration_behavior_match(self):
        actual = _ActualRecorder()
        expected = self.make_expected()

        self.assertIs(actual.training, expected.training)
        self.assertEqual(actual(1, label="x"), expected(1, label="x"))

        for mode in (False, True):
            with self.subTest(mode=mode):
                self.assertIs(actual.train(mode), actual)
                self.assertIs(expected.train(mode), expected)
                self.assertIs(actual.training, expected.training)
                self.assertEqual(actual("value"), expected("value"))

        self.assertIs(actual.eval(), actual)
        self.assertIs(expected.eval(), expected)
        self.assertIs(actual.training, expected.training)

        for make_actual, make_expected in (
            (lambda: actual.parameters(), lambda: expected.parameters()),
            (lambda: actual.parameters(False), lambda: expected.parameters(False)),
            (lambda: actual.named_parameters(), lambda: expected.named_parameters()),
            (
                lambda: actual.named_parameters("prefix.", False, False),
                lambda: expected.named_parameters("prefix.", False, False),
            ),
            (lambda: actual.buffers(), lambda: expected.buffers()),
            (lambda: actual.buffers(False), lambda: expected.buffers(False)),
            (lambda: actual.named_buffers(), lambda: expected.named_buffers()),
            (
                lambda: actual.named_buffers("prefix.", False, False),
                lambda: expected.named_buffers("prefix.", False, False),
            ),
        ):
            with self.subTest(iterator=make_actual):
                self.assertEqual(list(make_actual()), list(make_expected()))

        self.assertEqual(
            self.state_dict_snapshot(actual),
            self.state_dict_snapshot(expected),
        )

        actual_destination = OrderedDict()
        expected_destination = OrderedDict()
        actual_destination._metadata = OrderedDict()
        expected_destination._metadata = OrderedDict()
        self.assertIs(
            actual.state_dict(destination=actual_destination, prefix="child."),
            actual_destination,
        )
        self.assertIs(
            expected.state_dict(destination=expected_destination, prefix="child."),
            expected_destination,
        )
        self.assertEqual(actual_destination, expected_destination)
        self.assertEqual(
            actual_destination._metadata,
            expected_destination._metadata,
        )

    def test_train_argument_errors_match(self):
        actual = _ActualRecorder()
        expected = self.make_expected()
        actual.train(False)
        expected.train(False)

        for mode in (0, 1, None, "training"):
            with self.subTest(mode=mode):
                with self.assertRaises(Exception) as actual_raised:
                    actual.train(mode)
                with self.assertRaises(Exception) as expected_raised:
                    expected.train(mode)
                self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
                self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
                self.assertIs(actual.training, expected.training)

    def test_registration_hooks_conversion_and_loading_remain_unsupported(self):
        actual = _ActualRecorder()
        expected = self.make_expected()

        self.assert_unsupported_while_reference_supports(
            lambda: actual.register_parameter("weight", None),
            lambda: expected.register_parameter("weight", None),
        )

        self.assert_unsupported_while_reference_supports(
            lambda: actual.register_buffer("running", torch.tensor(1.0)),
            lambda: expected.register_buffer("running", reference_torch.tensor(1.0)),
        )

        self.assert_unsupported_while_reference_supports(
            lambda: actual.add_module("child", _ActualRecorder()),
            lambda: expected.add_module("child", self.make_expected()),
        )

        self.assert_unsupported_while_reference_supports(
            lambda: setattr(actual, "assigned_child", _ActualRecorder()),
            lambda: setattr(expected, "assigned_child", self.make_expected()),
        )

        self.assert_unsupported_while_reference_supports(
            lambda: actual.register_forward_hook(lambda *args: None),
            lambda: expected.register_forward_hook(lambda *args: None).remove(),
        )

        self.assert_unsupported_while_reference_supports(
            lambda: actual.register_forward_pre_hook(lambda *args: None),
            lambda: expected.register_forward_pre_hook(lambda *args: None).remove(),
        )

        self.assert_unsupported_while_reference_supports(
            lambda: actual.to("cpu"),
            lambda: expected.to("cpu"),
        )
        self.assert_unsupported_while_reference_supports(
            lambda: actual.cpu(),
            lambda: expected.cpu(),
        )
        self.assert_unsupported_while_reference_supports(
            lambda: actual.load_state_dict({}),
            lambda: expected.load_state_dict({}),
        )
