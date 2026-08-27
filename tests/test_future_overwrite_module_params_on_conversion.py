import copy
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


SETTER_DOC = """
    Sets whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an ``nn.Module``.

    When enabled, the following methods will assign new parameters to the module:

    #. ``module.{device}()`` (e.g. :meth:`nn.Module.cuda()`) for moving a module between devices
    #. ``module.{dtype}()`` (e.g. :meth:`nn.Module.float()`) for converting a module to a different dtype
    #. :meth:`nn.Module.to`
    #. :meth:`nn.Module.to_empty`

    Args:
        value (bool): Whether to assign new tensors or not.

    """

GETTER_DOC = """
    Returns whether to assign new tensors to the parameters instead of changing the
    existing parameters in-place when converting an :class:`torch.nn.Module`. Defaults to ``False``.

    See :func:`~torch.__future__.set_overwrite_module_params_on_conversion` for more information.
    """


class FutureOverwriteModuleParamsOnConversionTests(unittest.TestCase):
    def setUp(self):
        self.future = importlib.reload(
            importlib.import_module("torch_rs.__future__")
        )

    def tearDown(self):
        importlib.reload(self.future)

    def test_default_and_setter_preserve_exact_objects(self):
        future = self.future
        getter = future.get_overwrite_module_params_on_conversion
        setter = future.set_overwrite_module_params_on_conversion

        self.assertIs(future._overwrite_module_params_on_conversion, False)
        self.assertIs(getter(), False)

        class RejectTruthConversion:
            def __bool__(self):
                raise AssertionError("the policy value must not be coerced")

        for value in (True, False, None, 0, "", [], RejectTruthConversion()):
            with self.subTest(value_type=type(value).__name__):
                self.assertIsNone(setter(value))
                self.assertIs(
                    future._overwrite_module_params_on_conversion,
                    value,
                )
                self.assertIs(getter(), value)

    def test_state_is_process_global_and_visible_across_threads(self):
        future = self.future
        initial = object()
        worker_value = object()
        main_value = object()
        worker_written = threading.Event()
        main_written = threading.Event()
        outcomes = {}
        errors = []

        future.set_overwrite_module_params_on_conversion(initial)

        def worker():
            try:
                outcomes["initial"] = (
                    future.get_overwrite_module_params_on_conversion()
                )
                outcomes["setter"] = (
                    future.set_overwrite_module_params_on_conversion(worker_value)
                )
                worker_written.set()
                if not main_written.wait(timeout=10):
                    raise TimeoutError("main thread did not publish its state")
                outcomes["final"] = (
                    future.get_overwrite_module_params_on_conversion()
                )
            except BaseException as error:
                errors.append(error)
                worker_written.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_written.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(
            future.get_overwrite_module_params_on_conversion(),
            worker_value,
        )
        self.assertIsNone(
            future.set_overwrite_module_params_on_conversion(main_value)
        )
        main_written.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIs(outcomes["initial"], initial)
        self.assertIsNone(outcomes["setter"])
        self.assertIs(outcomes["final"], main_value)

    def test_signature_annotations_documentation_and_module_identity(self):
        future = self.future
        getter = future.get_overwrite_module_params_on_conversion
        setter = future.set_overwrite_module_params_on_conversion

        self.assertIs(torch.__future__, future)
        self.assertIs(sys.modules["torch_rs.__future__"], future)
        self.assertIs(type(future), types.ModuleType)
        self.assertIsNone(future.__doc__)
        self.assertFalse(hasattr(future, "__all__"))
        self.assertEqual(
            {name for name in vars(future) if not name.startswith("_")},
            {
                "get_overwrite_module_params_on_conversion",
                "set_overwrite_module_params_on_conversion",
            },
        )
        self.assertEqual(
            future.__annotations__,
            {"_overwrite_module_params_on_conversion": bool},
        )

        expected = (
            (
                getter,
                "get_overwrite_module_params_on_conversion",
                "() -> bool",
                {"return": bool},
                GETTER_DOC,
                (),
            ),
            (
                setter,
                "set_overwrite_module_params_on_conversion",
                "(value: bool) -> None",
                {"value": bool, "return": None},
                SETTER_DOC,
                ("value",),
            ),
        )
        for function, name, signature, annotations, doc, varnames in expected:
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(inspect.get_annotations(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.__future__")
                self.assertIs(inspect.getmodule(function), future)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__),
                    inspect.cleandoc(doc),
                )
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(
                    function.__code__.co_names,
                    ("_overwrite_module_params_on_conversion",),
                )
                self.assertEqual(function.__code__.co_varnames, varnames)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

        self.assertEqual(typing.get_type_hints(getter), {"return": bool})
        self.assertEqual(
            typing.get_type_hints(setter),
            {"value": bool, "return": type(None)},
        )

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        future = self.future
        functions = {
            "get_overwrite_module_params_on_conversion": (
                future.get_overwrite_module_params_on_conversion
            ),
            "set_overwrite_module_params_on_conversion": (
                future.set_overwrite_module_params_on_conversion
            ),
        }

        package_import = {}
        function_imports = {}
        child_wildcard = {}
        exec("from torch_rs import __future__", package_import)
        for name in functions:
            namespace = {}
            exec(f"from torch_rs.__future__ import {name}", namespace)
            function_imports[name] = namespace[name]
        exec("from torch_rs.__future__ import *", child_wildcard)

        self.assertIs(package_import["__future__"], future)
        for name, function in functions.items():
            self.assertIs(function_imports[name], function)
            self.assertIs(child_wildcard[name], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            set(functions),
        )

        self.assertNotIn("__future__", torch.__all__)
        self.assertNotIn(
            "get_overwrite_module_params_on_conversion",
            torch.__all__,
        )
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("__future__", top_level_wildcard)
        self.assertNotIn(
            "get_overwrite_module_params_on_conversion",
            top_level_wildcard,
        )

        for name, function in functions.items():
            with self.subTest(function=name):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.__future__", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_reload_resets_state_and_replaces_functions(self):
        future = self.future
        old_functions = {
            "get_overwrite_module_params_on_conversion": (
                future.get_overwrite_module_params_on_conversion
            ),
            "set_overwrite_module_params_on_conversion": (
                future.set_overwrite_module_params_on_conversion
            ),
        }
        namespace = future.__dict__
        future.set_overwrite_module_params_on_conversion(object())

        reloaded = importlib.reload(future)

        self.assertIs(reloaded, future)
        self.assertIs(future.__dict__, namespace)
        self.assertIs(torch.__future__, future)
        self.assertIs(sys.modules[future.__name__], future)
        self.assertIs(future._overwrite_module_params_on_conversion, False)
        self.assertIs(
            future.get_overwrite_module_params_on_conversion(),
            False,
        )
        for name, old_function in old_functions.items():
            new_function = getattr(future, name)
            self.assertIsNot(new_function, old_function)
            self.assertIs(copy.copy(new_function), new_function)
            self.assertIs(copy.deepcopy(new_function), new_function)
            self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(old_function)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                f"Can't pickle <function {name} at 0x...>: "
                "it's not the same object as "
                f"torch_rs.__future__.{name}",
            )

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        future = self.future
        value = object()
        self.assertIsNone(
            future.set_overwrite_module_params_on_conversion(value=value)
        )
        self.assertIs(
            future.get_overwrite_module_params_on_conversion(),
            value,
        )

        cases = (
            (
                lambda: future.get_overwrite_module_params_on_conversion(None),
                "get_overwrite_module_params_on_conversion() takes 0 "
                "positional arguments but 1 was given",
            ),
            (
                lambda: future.get_overwrite_module_params_on_conversion(
                    value=True
                ),
                "get_overwrite_module_params_on_conversion() got an unexpected "
                "keyword argument 'value'",
            ),
            (
                lambda: future.set_overwrite_module_params_on_conversion(),
                "set_overwrite_module_params_on_conversion() missing 1 required "
                "positional argument: 'value'",
            ),
            (
                lambda: future.set_overwrite_module_params_on_conversion(
                    True,
                    False,
                ),
                "set_overwrite_module_params_on_conversion() takes 1 positional "
                "argument but 2 were given",
            ),
            (
                lambda: future.set_overwrite_module_params_on_conversion(
                    enabled=True
                ),
                "set_overwrite_module_params_on_conversion() got an unexpected "
                "keyword argument 'enabled'",
            ),
            (
                lambda: future.set_overwrite_module_params_on_conversion(
                    True,
                    value=False,
                ),
                "set_overwrite_module_params_on_conversion() got multiple values "
                "for argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r'''
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

future = torch.__future__
modules_before_call = set(sys.modules)
assert future.get_overwrite_module_params_on_conversion() is False
value = object()
assert future.set_overwrite_module_params_on_conversion(value) is None
assert future.get_overwrite_module_params_on_conversion() is value
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")

    def test_swap_policy_and_module_conversion_remain_unsupported(self):
        for name in (
            "get_swap_module_params_on_conversion",
            "set_swap_module_params_on_conversion",
            "_swap_module_params_on_conversion",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.future, name))

        self.assertFalse(hasattr(torch.nn, "Module"))
        self.assertFalse(hasattr(torch.nn.modules, "Module"))


if __name__ == "__main__":
    unittest.main()
