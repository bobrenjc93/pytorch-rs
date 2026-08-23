import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


GETTER_DOC = """Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
SETTER_DOC = """When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """


class IsWarnAlwaysEnabledTests(unittest.TestCase):
    def test_default_false_is_exact_and_preserves_grad_mode(self):
        function = torch.is_warn_always_enabled
        self.assertEqual(function.__code__.co_names, ("_C", "_get_warnAlways"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), False)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_default_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.is_warn_always_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    False,
                    expected_grad_state,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[3], False)

    def test_mutable_state_is_native_process_global_across_threads_and_reloads(self):
        original = torch.is_warn_always_enabled()
        self.addCleanup(torch.set_warn_always, original)
        torch.set_warn_always(False)

        state_is_true = threading.Event()
        may_reset_state = threading.Event()
        state_is_false = threading.Event()
        observations = []

        def worker():
            torch._C._set_warnAlways(True)
            observations.append(torch.is_warn_always_enabled())
            state_is_true.set()
            if not may_reset_state.wait(timeout=10):
                return
            observations.append(torch.is_warn_always_enabled())
            torch.set_warn_always(False)
            observations.append(torch._C._get_warnAlways())
            state_is_false.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(state_is_true.wait(timeout=10))
        self.assertIs(torch.is_warn_always_enabled(), True)

        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(torch.is_warn_always_enabled(), True)
        package = importlib.reload(torch)
        self.assertIs(package, torch)
        self.assertIs(torch._C, native)
        self.assertIs(torch.is_warn_always_enabled(), True)

        may_reset_state.set()
        self.assertTrue(state_is_false.wait(timeout=10))
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observations, [True, True, False])
        self.assertIs(torch.is_warn_always_enabled(), False)

    def test_signature_annotations_documentation_and_module_identity(self):
        package = importlib.import_module("torch_rs")
        getter = package.is_warn_always_enabled
        setter = package.set_warn_always

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        for function in (getter, setter):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs")
            self.assertIs(inspect.getmodule(function), package)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))
            self.assertEqual(function.__code__.co_freevars, ())
            self.assertEqual(function.__code__.co_cellvars, ())

        self.assertEqual(str(inspect.signature(getter)), "() -> bool")
        self.assertEqual(getter.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(getter), {"return": bool})
        self.assertEqual(getter.__name__, "is_warn_always_enabled")
        self.assertEqual(getter.__qualname__, "is_warn_always_enabled")
        self.assertEqual(getter.__code__.co_names, ("_C", "_get_warnAlways"))
        self.assertEqual(inspect.cleandoc(getter.__doc__), inspect.cleandoc(GETTER_DOC))

        self.assertEqual(str(inspect.signature(setter)), "(b: bool, /) -> None")
        self.assertEqual(setter.__annotations__, {"b": bool, "return": None})
        self.assertEqual(
            typing.get_type_hints(setter), {"b": bool, "return": type(None)}
        )
        self.assertEqual(setter.__name__, "set_warn_always")
        self.assertEqual(setter.__qualname__, "set_warn_always")
        self.assertEqual(setter.__code__.co_names, ("_C", "_set_warnAlways"))
        self.assertEqual(inspect.cleandoc(setter.__doc__), inspect.cleandoc(SETTER_DOC))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        namespace = {}
        exec("from torch_rs import *", namespace)
        for name in ("is_warn_always_enabled", "set_warn_always"):
            with self.subTest(name=name):
                function = getattr(torch, name)
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertIs(namespace[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs", payload)
                    self.assertIs(pickle.loads(payload), function)

        self.assertNotIn("_set_warnAlways", torch._C.__all__)
        self.assertNotIn("_get_warnAlways", torch._C.__all__)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        getter = torch.is_warn_always_enabled
        getter_cases = (
            (
                lambda: getter(None),
                "is_warn_always_enabled() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: getter(None, None),
                "is_warn_always_enabled() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: getter(enabled=True),
                "is_warn_always_enabled() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: getter(None, enabled=True),
                "is_warn_always_enabled() got an unexpected keyword argument "
                "'enabled'",
            ),
        )
        for call, message in getter_cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        setter = torch.set_warn_always
        setter_cases = (
            (
                lambda: setter(),
                "set_warn_always() missing 1 required positional argument: 'b'",
            ),
            (
                lambda: setter(True, False),
                "set_warn_always() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: setter(b=True),
                "set_warn_always() got some positional-only arguments passed as "
                "keyword arguments: 'b'",
            ),
            (
                lambda: setter(enabled=True),
                "set_warn_always() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: setter(True, b=False),
                "set_warn_always() got some positional-only arguments passed as "
                "keyword arguments: 'b'",
            ),
        )
        for call, message in setter_cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        original = torch.is_warn_always_enabled()
        self.addCleanup(torch.set_warn_always, original)
        torch.set_warn_always(False)
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1.0, "float"),
            ("", "str"),
            (torch.tensor(1.0), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.contiguous_format, "torch.memory_format"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for value, type_name in invalid_values:
            with self.subTest(value=value):
                message = f"setWarnOnlyOnce expects a bool, but got {type_name}"
                with self.assertRaises(RuntimeError) as raised:
                    setter(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.is_warn_always_enabled(), False)

    def test_setter_returns_none_and_controls_native_warning_once_markers(self):
        source = r'''
import warnings

import torch_rs as torch


def warning_count(attribute, calls):
    tensor = torch.tensor(1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(calls):
            getattr(tensor, attribute)
    return len(caught)


assert torch.set_warn_always(False) is None
assert torch.is_warn_always_enabled() is False
assert warning_count("T", 2) == 1
assert torch.set_warn_always(True) is None
assert torch.is_warn_always_enabled() is True
assert warning_count("T", 2) == 2
assert torch.set_warn_always(False) is None
assert warning_count("T", 2) == 0

# Always-enabled emissions do not consume a previously untouched marker.
torch.set_warn_always(True)
assert warning_count("mT", 2) == 2
torch.set_warn_always(False)
assert warning_count("mT", 2) == 1


def ordinary_warning():
    warnings.warn("ordinary Python warning", UserWarning)


for enabled in (False, True):
    torch.set_warn_always(enabled)
    globals().pop("__warningregistry__", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        ordinary_warning()
        ordinary_warning()
    assert len(caught) == 1

torch.set_warn_always(False)
'''
        completed = subprocess.run(
            [sys.executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.is_warn_always_enabled() is False
assert torch.set_warn_always(True) is None
assert torch.is_warn_always_enabled() is True
assert torch.set_warn_always(False) is None
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
