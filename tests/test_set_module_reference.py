import copy
import importlib
import inspect
import pickle
import pickletools
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _ModuleName(str):
    pass


class _CustomTarget:
    pass


class _SlottedTarget:
    __slots__ = ()


class _RecordingTarget:
    def __init__(self):
        object.__setattr__(self, "writes", [])

    def __setattr__(self, name, value):
        self.writes.append((name, value))
        object.__setattr__(self, name, value)


class _RejectingTarget:
    def __setattr__(self, name, value):
        raise AttributeError(f"blocked {name}: {value!r}")


class _RejectingMeta(type):
    def __setattr__(cls, name, value):
        if name == "__module__":
            raise AttributeError(f"blocked class {name}: {value!r}")
        super().__setattr__(name, value)


class _RejectingClass(metaclass=_RejectingMeta):
    pass


def _make_function():
    def target():
        pass

    return target


def _make_class():
    class Target:
        pass

    return Target


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetModuleReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_module differentials require pinned PyTorch 2.13.0"
            )

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

    def normalize_error(self, error):
        return (
            type(error).__name__,
            re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                "torch_rs", "torch"
            ),
        )

    def test_assignment_and_string_subclass_behavior_match_pytorch_2_13(self):
        target_factories = (_make_function, _make_class, _CustomTarget)
        for target_factory in target_factories:
            for module_factory in (
                lambda: "example.module",
                lambda: _ModuleName("example.subclass"),
            ):
                with self.subTest(
                    target=target_factory.__name__,
                    module_type=module_factory().__class__.__name__,
                ):
                    actual_target = target_factory()
                    expected_target = target_factory()
                    actual_mod = module_factory()
                    expected_mod = module_factory()
                    actual_result = torch.utils.set_module(actual_target, actual_mod)
                    expected_result = reference_torch.utils.set_module(
                        expected_target, expected_mod
                    )
                    self.assertIs(actual_result, expected_result)
                    self.assertIs(actual_result, None)
                    self.assertEqual(
                        actual_target.__module__, expected_target.__module__
                    )
                    self.assertIs(
                        type(actual_target.__module__),
                        type(expected_target.__module__),
                    )
                    self.assertIs(actual_target.__module__, actual_mod)
                    self.assertIs(expected_target.__module__, expected_mod)

    def test_type_and_attribute_error_ordering_matches_pytorch_2_13(self):
        target_factories = (
            _make_function,
            _make_class,
            _CustomTarget,
            _RecordingTarget,
            _SlottedTarget,
            _RejectingTarget,
            lambda: _RejectingClass,
        )
        for target_factory in target_factories:
            for mod in (None, 1, object()):
                with self.subTest(
                    target=target_factory.__name__, module_type=type(mod).__name__
                ):
                    actual_target = target_factory()
                    expected_target = target_factory()
                    self.assert_error_matches(
                        lambda: torch.utils.set_module(actual_target, mod),
                        lambda: reference_torch.utils.set_module(expected_target, mod),
                    )
                    if isinstance(actual_target, _RecordingTarget):
                        self.assertEqual(actual_target.writes, expected_target.writes)
                        self.assertEqual(actual_target.writes, [])

        assignment_failures = (
            (_SlottedTarget, _SlottedTarget),
            (_RejectingTarget, _RejectingTarget),
            (lambda: _RejectingClass, lambda: _RejectingClass),
            (lambda: 1, lambda: 1),
        )
        for actual_factory, expected_factory in assignment_failures:
            with self.subTest(target=actual_factory.__name__):
                self.assert_error_matches(
                    lambda: torch.utils.set_module(
                        actual_factory(), "example.module"
                    ),
                    lambda: reference_torch.utils.set_module(
                        expected_factory(), "example.module"
                    ),
                )

    def test_call_shapes_signature_and_metadata_match_pytorch_2_13(self):
        actual_utils = importlib.import_module("torch_rs.utils")
        expected_utils = importlib.import_module("torch.utils")
        actual = actual_utils.set_module
        expected = expected_utils.set_module

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_utils)
        self.assertIs(inspect.getmodule(expected), expected_utils)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        actual_target = _make_function()
        expected_target = _make_function()
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(actual_target), lambda: expected(expected_target)),
            (
                lambda: actual(actual_target, "module", "extra"),
                lambda: expected(expected_target, "module", "extra"),
            ),
            (
                lambda: actual(object=actual_target, mod="module"),
                lambda: expected(object=expected_target, mod="module"),
            ),
            (
                lambda: actual(actual_target, "module", mod="other"),
                lambda: expected(expected_target, "module", mod="other"),
            ),
        )
        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)

        self.assertIs(actual(obj=actual_target, mod="keyword.module"), None)
        self.assertIs(expected(obj=expected_target, mod="keyword.module"), None)
        self.assertEqual(actual_target.__module__, expected_target.__module__)

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_utils = torch.utils
        expected_utils = reference_torch.utils
        actual = actual_utils.set_module
        expected = expected_utils.set_module

        self.assertEqual(
            hasattr(actual_utils, "__all__"), hasattr(expected_utils, "__all__")
        )
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.utils import *", actual_namespace)
        exec("from torch.utils import *", expected_namespace)
        for name in ("data", "set_module"):
            self.assertIn(name, actual_namespace)
            self.assertIn(name, expected_namespace)
            self.assertIs(actual_namespace[name], getattr(actual_utils, name))
            self.assertIs(expected_namespace[name], getattr(expected_utils, name))

        for package in (torch, reference_torch):
            namespace = {}
            exec(f"from {package.__name__} import *", namespace)
            self.assertFalse(hasattr(package, "set_module"))
            self.assertNotIn("set_module", package.__all__)
            self.assertNotIn("set_module", namespace)

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

    def reload_outcome(self, package):
        utils = package.utils
        data = utils.data
        old_function = utils.set_module
        reloaded = importlib.reload(utils)
        new_function = utils.set_module
        old_target = _CustomTarget()
        new_target = _CustomTarget()
        old_result = old_function(old_target, "old.module")
        new_result = new_function(new_target, "new.module")

        stale_pickle_errors = []
        new_pickle_identities = []
        new_pickle_shapes = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            try:
                pickle.dumps(old_function, protocol=protocol)
            except Exception as error:
                stale_pickle_errors.append(self.normalize_error(error))
            else:
                stale_pickle_errors.append(("success", ""))
            payload = pickle.dumps(new_function, protocol=protocol)
            new_pickle_identities.append(pickle.loads(payload) is new_function)
            new_pickle_shapes.append(self.pickle_shape(new_function, protocol))

        return {
            "reload_identity": reloaded is utils,
            "package_identity": package.utils is utils,
            "data_identity": utils.data is data,
            "data_module": data.__name__.replace("torch_rs", "torch"),
            "function_replaced": new_function is not old_function,
            "old_result": old_result,
            "new_result": new_result,
            "old_module": old_target.__module__,
            "new_module": new_target.__module__,
            "old_copy_identity": copy.copy(old_function) is old_function,
            "new_copy_identity": copy.copy(new_function) is new_function,
            "stale_pickle_errors": stale_pickle_errors,
            "new_pickle_identities": new_pickle_identities,
            "new_pickle_shapes": new_pickle_shapes,
        }

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
