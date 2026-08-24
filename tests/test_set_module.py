import copy
import importlib
import inspect
import pickle
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Set the module attribute on a python object for a given object for nicer printing
    """


class _ModuleName(str):
    pass


def _pickle_function_target():
    pass


class _PickleClassTarget:
    pass


class _PickleCustomTarget:
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


class SetModuleTests(unittest.TestCase):
    def test_assigns_exact_strings_and_subclasses_and_returns_none(self):
        target_factories = (
            ("function", _make_function),
            ("class", _make_class),
            ("custom", _PickleCustomTarget),
        )
        modules = ("example.module", _ModuleName("example.subclass"))

        for target_name, target_factory in target_factories:
            for mod in modules:
                with self.subTest(target=target_name, module_type=type(mod).__name__):
                    target = target_factory()
                    self.assertIs(torch.utils.set_module(target, mod), None)
                    self.assertIs(target.__module__, mod)

    def test_validates_module_before_attempting_assignment(self):
        message = "The mod argument should be a string"
        target_factories = (
            ("function", _make_function),
            ("class", _make_class),
            ("custom", _RecordingTarget),
            ("slotted", _SlottedTarget),
            ("rejecting", _RejectingTarget),
            ("rejecting_class", lambda: _RejectingClass),
        )

        for target_name, target_factory in target_factories:
            for mod in (None, 1, object()):
                with self.subTest(target=target_name, module_type=type(mod).__name__):
                    target = target_factory()
                    original = getattr(target, "__module__", None)
                    with self.assertRaises(TypeError) as raised:
                        torch.utils.set_module(target, mod)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(getattr(target, "__module__", None), original)
                    if isinstance(target, _RecordingTarget):
                        self.assertEqual(target.writes, [])

    def test_assignment_attribute_errors_are_not_rewritten(self):
        cases = (
            (
                _SlottedTarget(),
                "'_SlottedTarget' object attribute '__module__' is read-only",
            ),
            (
                _RejectingTarget(),
                "blocked __module__: 'example.module'",
            ),
            (
                _RejectingClass,
                "blocked class __module__: 'example.module'",
            ),
            (1, "'int' object has no attribute '__module__'"),
        )

        for target, message in cases:
            with self.subTest(target=target):
                with self.assertRaises(AttributeError) as raised:
                    torch.utils.set_module(target, "example.module")
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_signature_and_metadata_match_pytorch_2_13(self):
        utils = importlib.import_module("torch_rs.utils")
        function = utils.set_module

        self.assertIs(torch.utils, utils)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(obj, mod)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__name__, "set_module")
        self.assertEqual(function.__qualname__, "set_module")
        self.assertEqual(function.__module__, "torch_rs.utils")
        self.assertIs(inspect.getmodule(function), utils)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_call_shape_errors_use_the_python_function_signature(self):
        function = torch.utils.set_module
        target = _make_function()
        cases = (
            (
                lambda: function(),
                "set_module() missing 2 required positional arguments: 'obj' and 'mod'",
            ),
            (
                lambda: function(target),
                "set_module() missing 1 required positional argument: 'mod'",
            ),
            (
                lambda: function(target, "module", "extra"),
                "set_module() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: function(object=target, mod="module"),
                "set_module() got an unexpected keyword argument 'object'",
            ),
            (
                lambda: function(target, "module", mod="other"),
                "set_module() got multiple values for argument 'mod'",
            ),
        )

        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(obj=target, mod="keyword.module"), None)
        self.assertEqual(target.__module__, "keyword.module")

    def test_function_class_and_custom_object_copy_and_pickle_behavior(self):
        exact_module = __name__
        subclass_module = _ModuleName(__name__)

        for target in (_pickle_function_target, _PickleClassTarget):
            original_module = target.__module__
            try:
                torch.utils.set_module(target, exact_module)
                self.assertIs(copy.copy(target), target)
                self.assertIs(copy.deepcopy(target), target)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(target=target.__name__, protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(target, protocol=protocol)),
                            target,
                        )

                torch.utils.set_module(target, subclass_module)
                self.assertIs(target.__module__, subclass_module)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(
                        target=target.__name__,
                        subclass=True,
                        protocol=protocol,
                    ):
                        payload = pickle.dumps(target, protocol=protocol)
                        if protocol < 4:
                            self.assertIs(pickle.loads(payload), target)
                        else:
                            with self.assertRaisesRegex(
                                pickle.UnpicklingError,
                                "STACK_GLOBAL requires str",
                            ):
                                pickle.loads(payload)
            finally:
                target.__module__ = original_module

        custom = _PickleCustomTarget()
        torch.utils.set_module(custom, subclass_module)
        shallow = copy.copy(custom)
        deep = copy.deepcopy(custom)
        self.assertIsNot(shallow, custom)
        self.assertIs(shallow.__module__, subclass_module)
        self.assertIsNot(deep, custom)
        self.assertIs(type(deep.__module__), _ModuleName)
        self.assertEqual(deep.__module__, subclass_module)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(target="custom", protocol=protocol):
                restored = pickle.loads(pickle.dumps(custom, protocol=protocol))
                self.assertIs(type(restored), _PickleCustomTarget)
                self.assertIs(type(restored.__module__), _ModuleName)
                self.assertEqual(restored.__module__, subclass_module)

    def test_utils_exports_preserve_data_without_top_level_promotion(self):
        utils = importlib.import_module("torch_rs.utils")
        data = importlib.import_module("torch_rs.utils.data")

        self.assertIs(torch.utils, utils)
        self.assertIs(utils.data, data)
        self.assertFalse(hasattr(utils, "__all__"))
        utils_namespace = {}
        exec("from torch_rs.utils import *", utils_namespace)
        self.assertIs(utils_namespace["data"], data)
        self.assertIs(utils_namespace["set_module"], utils.set_module)

        self.assertFalse(hasattr(torch, "set_module"))
        self.assertNotIn("set_module", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("set_module", top_level_namespace)

    def test_copy_pickle_and_reload_keep_the_canonical_utils_binding(self):
        utils = torch.utils
        data = utils.data
        old_function = utils.set_module

        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(stage="before_reload", protocol=protocol):
                payload = pickle.dumps(old_function, protocol=protocol)
                self.assertIn(b"torch_rs.utils", payload)
                self.assertIs(pickle.loads(payload), old_function)

        self.assertIs(importlib.reload(utils), utils)
        self.assertIs(torch.utils, utils)
        self.assertIs(utils.data, data)
        new_function = utils.set_module
        self.assertIsNot(new_function, old_function)

        old_target = _PickleCustomTarget()
        new_target = _PickleCustomTarget()
        self.assertIs(old_function(old_target, "old.module"), None)
        self.assertIs(new_function(new_target, "new.module"), None)
        self.assertEqual(old_target.__module__, "old.module")
        self.assertEqual(new_target.__module__, "new.module")
        self.assertIs(copy.copy(old_function), old_function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(stage="after_reload", protocol=protocol):
                with self.assertRaisesRegex(
                    pickle.PicklingError,
                    "not the same object as torch_rs.utils.set_module",
                ):
                    pickle.dumps(old_function, protocol=protocol)
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol=protocol)),
                    new_function,
                )


if __name__ == "__main__":
    unittest.main()
