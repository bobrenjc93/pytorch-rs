import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("the overwrite policy must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FutureOverwriteModuleParamsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.__future__ overwrite-policy differentials require "
                "pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.reload(
            importlib.import_module("torch_rs.__future__")
        )
        self.expected = importlib.reload(
            importlib.import_module("torch.__future__")
        )

    def tearDown(self):
        importlib.reload(self.actual)
        importlib.reload(self.expected)

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

    def state_contract(self, module):
        outcomes = [
            module.get_overwrite_module_params_on_conversion() is False,
            module._overwrite_module_params_on_conversion is False,
        ]
        for value in (True, False, None, 0, 1, 0.0, "", [], object(), _RejectTruthiness()):
            outcomes.append(
                module.set_overwrite_module_params_on_conversion(value) is None
            )
            outcomes.append(module._overwrite_module_params_on_conversion is value)
            outcomes.append(
                module.get_overwrite_module_params_on_conversion() is value
            )
        keyword_value = object()
        outcomes.append(
            module.set_overwrite_module_params_on_conversion(
                value=keyword_value
            )
            is None
        )
        outcomes.append(
            module.get_overwrite_module_params_on_conversion() is keyword_value
        )
        return outcomes

    def thread_contract(self, module):
        initial = object()
        worker_value = object()
        main_value = object()
        worker_written = threading.Event()
        main_written = threading.Event()
        outcomes = {}
        errors = []

        module.set_overwrite_module_params_on_conversion(initial)

        def worker():
            try:
                outcomes["initial"] = (
                    module.get_overwrite_module_params_on_conversion() is initial
                )
                outcomes["setter"] = (
                    module.set_overwrite_module_params_on_conversion(worker_value)
                    is None
                )
                worker_written.set()
                if not main_written.wait(timeout=10):
                    raise TimeoutError("main thread did not publish its state")
                outcomes["final"] = (
                    module.get_overwrite_module_params_on_conversion()
                    is main_value
                )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_written.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_written.wait(timeout=10)
        main_saw_worker = (
            module.get_overwrite_module_params_on_conversion() is worker_value
        )
        main_setter = (
            module.set_overwrite_module_params_on_conversion(main_value) is None
        )
        main_written.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            main_setter,
            not thread.is_alive(),
            errors,
            outcomes,
        )

    def reload_contract(self, root, module):
        old_getter = module.get_overwrite_module_params_on_conversion
        old_setter = module.set_overwrite_module_params_on_conversion
        namespace = module.__dict__
        annotations = module.__annotations__
        old_setter(object())

        reloaded = importlib.reload(module)
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            module.__annotations__ is annotations,
            root.__future__ is module,
            sys.modules[module.__name__] is module,
            module._overwrite_module_params_on_conversion is False,
            module.get_overwrite_module_params_on_conversion() is False,
            old_getter() is False,
            module.get_overwrite_module_params_on_conversion is not old_getter,
            module.set_overwrite_module_params_on_conversion is not old_setter,
        )

        replacement = object()
        old_setter(replacement)
        old_function_state = (
            module.get_overwrite_module_params_on_conversion() is replacement
        )

        function_outcomes = []
        for name, old_function in (
            ("get_overwrite_module_params_on_conversion", old_getter),
            ("set_overwrite_module_params_on_conversion", old_setter),
        ):
            new_function = getattr(module, name)
            try:
                pickle.dumps(old_function)
            except Exception as error:
                old_pickle = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                        "torch_rs", "torch"
                    ),
                    tuple(
                        re.sub(r"0x[0-9a-fA-F]+", "0x...", str(argument)).replace(
                            "torch_rs", "torch"
                        )
                        for argument in error.args
                    ),
                )
            else:
                old_pickle = None
            function_outcomes.append(
                (
                    copy.copy(old_function) is old_function,
                    copy.deepcopy(old_function) is old_function,
                    copy.copy(new_function) is new_function,
                    copy.deepcopy(new_function) is new_function,
                    pickle.loads(pickle.dumps(new_function)) is new_function,
                    old_pickle,
                )
            )
        return initial, old_function_state, function_outcomes

    def test_state_identity_and_thread_visibility_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_contract(self.actual),
            self.state_contract(self.expected),
        )
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual = self.actual
        expected = self.expected
        supported = {
            "get_overwrite_module_params_on_conversion",
            "set_overwrite_module_params_on_conversion",
        }

        self.assertIs(torch.__future__, actual)
        self.assertIs(reference_torch.__future__, expected)
        self.assertIs(sys.modules[actual.__name__], actual)
        self.assertIs(sys.modules[expected.__name__], expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(hasattr(actual, "__all__"), hasattr(expected, "__all__"))
        self.assertEqual(
            actual.__annotations__,
            {
                name: annotation
                for name, annotation in expected.__annotations__.items()
                if name == "_overwrite_module_params_on_conversion"
            },
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            supported,
        )
        self.assertTrue(supported.issubset(vars(expected)))

        for name in supported:
            with self.subTest(function=name):
                actual_function = getattr(actual, name)
                expected_function = getattr(expected, name)
                self.assertIs(type(actual_function), types.FunctionType)
                self.assertIs(type(expected_function), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual_function)),
                    str(inspect.signature(expected_function)),
                )
                self.assertEqual(
                    actual_function.__annotations__,
                    expected_function.__annotations__,
                )
                self.assertEqual(
                    typing.get_type_hints(actual_function),
                    typing.get_type_hints(expected_function),
                )
                self.assertEqual(actual_function.__name__, expected_function.__name__)
                self.assertEqual(
                    actual_function.__qualname__, expected_function.__qualname__
                )
                self.assertEqual(
                    actual_function.__module__.replace("torch_rs", "torch"),
                    expected_function.__module__,
                )
                self.assertIs(inspect.getmodule(actual_function), actual)
                self.assertIs(inspect.getmodule(expected_function), expected)
                self.assertEqual(actual_function.__doc__, expected_function.__doc__)
                self.assertEqual(
                    actual_function.__defaults__, expected_function.__defaults__
                )
                self.assertEqual(
                    actual_function.__kwdefaults__, expected_function.__kwdefaults__
                )
                self.assertEqual(actual_function.__dict__, expected_function.__dict__)
                self.assertEqual(
                    hasattr(actual_function, "__text_signature__"),
                    hasattr(expected_function, "__text_signature__"),
                )
                self.assertEqual(
                    actual_function.__code__.co_names,
                    expected_function.__code__.co_names,
                )
                self.assertEqual(
                    actual_function.__code__.co_freevars,
                    expected_function.__code__.co_freevars,
                )
                self.assertEqual(
                    actual_function.__code__.co_cellvars,
                    expected_function.__code__.co_cellvars,
                )

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        supported = {
            "get_overwrite_module_params_on_conversion",
            "set_overwrite_module_params_on_conversion",
        }

        for package, module in ((torch, actual), (reference_torch, expected)):
            package_import = {}
            module_import = {}
            exec(f"from {package.__name__} import __future__", package_import)
            exec(f"import {package.__name__}.__future__ as future", module_import)
            self.assertIs(package_import["__future__"], module)
            self.assertIs(module_import["future"], module)

            for name in supported:
                direct_import = {}
                exec(
                    f"from {package.__name__}.__future__ import {name}",
                    direct_import,
                )
                self.assertIs(direct_import[name], getattr(module, name))

            child_wildcard = {}
            exec(f"from {package.__name__}.__future__ import *", child_wildcard)
            self.assertEqual(
                {
                    name
                    for name in child_wildcard
                    if not name.startswith("__") and name in supported
                },
                supported,
            )

            top_level_wildcard = {}
            exec(f"from {package.__name__} import *", top_level_wildcard)
            self.assertNotIn("__future__", package.__all__)
            self.assertNotIn("__future__", top_level_wildcard)

        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            supported,
        )

        for name in supported:
            actual_function = getattr(actual, name)
            expected_function = getattr(expected, name)
            self.assertIs(copy.copy(actual_function), actual_function)
            self.assertIs(copy.copy(expected_function), expected_function)
            self.assertIs(copy.deepcopy(actual_function), actual_function)
            self.assertIs(copy.deepcopy(expected_function), expected_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_function, protocol)),
                        actual_function,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_function, protocol)),
                        expected_function,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual_function, protocol),
                        self.pickle_shape(expected_function, protocol),
                    )

    def test_reload_reset_and_replaced_function_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_value = object()
        expected_value = object()
        self.assertIsNone(
            actual.set_overwrite_module_params_on_conversion(value=actual_value)
        )
        self.assertIsNone(
            expected.set_overwrite_module_params_on_conversion(value=expected_value)
        )
        self.assertIs(
            actual.get_overwrite_module_params_on_conversion(), actual_value
        )
        self.assertIs(
            expected.get_overwrite_module_params_on_conversion(), expected_value
        )

        cases = (
            (
                lambda: actual.get_overwrite_module_params_on_conversion(None),
                lambda: expected.get_overwrite_module_params_on_conversion(None),
            ),
            (
                lambda: actual.get_overwrite_module_params_on_conversion(None, None),
                lambda: expected.get_overwrite_module_params_on_conversion(None, None),
            ),
            (
                lambda: actual.get_overwrite_module_params_on_conversion(value=True),
                lambda: expected.get_overwrite_module_params_on_conversion(value=True),
            ),
            (
                lambda: actual.set_overwrite_module_params_on_conversion(),
                lambda: expected.set_overwrite_module_params_on_conversion(),
            ),
            (
                lambda: actual.set_overwrite_module_params_on_conversion(True, False),
                lambda: expected.set_overwrite_module_params_on_conversion(True, False),
            ),
            (
                lambda: actual.set_overwrite_module_params_on_conversion(enabled=True),
                lambda: expected.set_overwrite_module_params_on_conversion(enabled=True),
            ),
            (
                lambda: actual.set_overwrite_module_params_on_conversion(
                    True, value=False
                ),
                lambda: expected.set_overwrite_module_params_on_conversion(
                    True, value=False
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(
                    actual.get_overwrite_module_params_on_conversion(), actual_value
                )
                self.assertIs(
                    expected.get_overwrite_module_params_on_conversion(), expected_value
                )

    def test_swap_policy_and_module_conversion_support_boundary(self):
        for name in (
            "_swap_module_params_on_conversion",
            "get_swap_module_params_on_conversion",
            "set_swap_module_params_on_conversion",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.actual, name))
                self.assertTrue(hasattr(self.expected, name))

        self.assertFalse(hasattr(torch.nn, "Module"))
        self.assertTrue(hasattr(reference_torch.nn, "Module"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.nn.modules.module")
        self.assertIsNotNone(importlib.import_module("torch.nn.modules.module"))


if __name__ == "__main__":
    unittest.main()
