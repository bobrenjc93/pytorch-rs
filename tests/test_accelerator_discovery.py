import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


MODULE_DOC = """
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

CURRENT_ACCELERATOR_DOC = """Return the device of the accelerator available at compilation time.
    If no accelerator were available at compilation time, returns None.
    See :ref:`accelerator<accelerators>` for details.

    Args:
        check_available (bool, optional): if True, will also do a runtime check to see
            if the device :func:`torch.accelerator.is_available` on top of the compile-time
            check.
            Default: ``False``

    Returns:
        torch.device: return the current accelerator as :class:`torch.device`.

    .. note:: The index of the returned :class:`torch.device` will be ``None``, please use
        :func:`torch.accelerator.current_device_index` to know the current index being used.
        This API does NOT poison fork. For more details, see :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> # xdoctest:
        >>> # If an accelerator is available, sent the model to it
        >>> model = torch.nn.Linear(2, 2)
        >>> if (current_device := current_accelerator(check_available=True)) is not None:
        >>>     model.to(current_device)
    """

IS_AVAILABLE_DOC = """Check if the current accelerator is available at runtime: it was built, all the
    required drivers are available and at least one device is visible.
    See :ref:`accelerator<accelerators>` for details.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    .. note:: This API delegates to the device-specific version of `is_available`.
        On CUDA, when the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set,
        this function will NOT poison fork. Otherwise, it will. For more details, see
        :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    """

DEVICE_COUNT_DOC = """Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT poison fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """


class ExplosiveBool:
    def __bool__(self):
        raise AssertionError("check_available truthiness was evaluated")


class AcceleratorDiscoveryTests(unittest.TestCase):
    def test_cpu_only_results_come_from_one_probe_free_discovery_helper(self):
        accelerator = torch.accelerator
        discovery_module = importlib.import_module(
            "torch_rs.accelerator._discovery"
        )
        helper = discovery_module._discover_accelerator

        self.assertIs(accelerator._discover_accelerator, helper)
        self.assertEqual(helper.__code__.co_names, ())
        self.assertEqual(helper.__code__.co_freevars, ())
        self.assertEqual(helper.__code__.co_cellvars, ())
        self.assertEqual(helper(), (None, False, 0))
        self.assertIs(helper()[0], None)
        self.assertIs(helper()[1], False)
        self.assertIs(type(helper()[2]), int)
        self.assertEqual(helper()[2], 0)

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch(
                        "os.cpu_count",
                        side_effect=AssertionError("host hardware was probed"),
                    ):
                        self.assertIs(accelerator.current_accelerator(), None)
                        self.assertIs(
                            accelerator.current_accelerator(
                                check_available=ExplosiveBool()
                            ),
                            None,
                        )
                        self.assertIs(accelerator.is_available(), False)
                        count = accelerator.device_count()
                self.assertIs(type(count), int)
                self.assertEqual(count, 0)

        calls = []

        def shared_helper():
            calls.append(threading.get_ident())
            return None, False, 0

        with mock.patch.object(
            accelerator, "_discover_accelerator", shared_helper
        ):
            self.assertIs(accelerator.current_accelerator(), None)
            self.assertIs(accelerator.current_accelerator(True), None)
            self.assertIs(accelerator.is_available(), False)
            self.assertEqual(accelerator.device_count(), 0)
        self.assertEqual(len(calls), 4)

    def test_results_are_stable_across_threads_and_grad_modes(self):
        accelerator = torch.accelerator
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
                        accelerator.current_accelerator(),
                        accelerator.current_accelerator(True),
                        accelerator.is_available(),
                        accelerator.device_count(),
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
                    None,
                    None,
                    False,
                    0,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[3], False)
            self.assertIs(type(result[4]), int)

    def test_signatures_annotations_documentation_and_module_identity(self):
        accelerator = importlib.import_module("torch_rs.accelerator")
        expected = {
            "current_accelerator": (
                "(check_available: bool = False) -> torch_rs.device | None",
                {
                    "check_available": bool,
                    "return": torch.device | None,
                },
                (False,),
                CURRENT_ACCELERATOR_DOC,
            ),
            "is_available": (
                "() -> bool",
                {"return": bool},
                None,
                IS_AVAILABLE_DOC,
            ),
            "device_count": (
                "() -> int",
                {"return": int},
                None,
                DEVICE_COUNT_DOC,
            ),
        }

        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(sys.modules["torch_rs.accelerator"], accelerator)
        self.assertEqual(accelerator.__doc__, MODULE_DOC)
        for name, (signature, annotations, defaults, documentation) in expected.items():
            with self.subTest(name=name):
                function = getattr(accelerator, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.accelerator")
                self.assertIs(inspect.getmodule(function), accelerator)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__),
                    inspect.cleandoc(documentation),
                )
                self.assertEqual(function.__defaults__, defaults)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        accelerator = torch.accelerator

        self.assertEqual(
            accelerator.__all__,
            ["current_accelerator", "device_count", "is_available"],
        )

        package_import = {}
        exec("from torch_rs import accelerator", package_import)
        self.assertIs(package_import["accelerator"], accelerator)

        direct_import = {}
        exec(
            "from torch_rs.accelerator import "
            "current_accelerator, device_count, is_available",
            direct_import,
        )
        for name in accelerator.__all__:
            self.assertIs(direct_import[name], getattr(accelerator, name))

        namespace = {}
        exec("from torch_rs.accelerator import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            {"current_accelerator", "device_count", "is_available"},
        )
        for name in accelerator.__all__:
            self.assertIs(namespace[name], getattr(accelerator, name))

        self.assertNotIn("accelerator", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("accelerator", top_level_namespace)

        for name in accelerator.__all__:
            function = getattr(accelerator, name)
            with self.subTest(name=name):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.accelerator", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        current_accelerator = torch.accelerator.current_accelerator
        for call in (
            lambda: current_accelerator(),
            lambda: current_accelerator(False),
            lambda: current_accelerator(True),
            lambda: current_accelerator(check_available=True),
            lambda: current_accelerator(**{}),
            lambda: current_accelerator(ExplosiveBool()),
        ):
            self.assertIs(call(), None)

        current_errors = (
            (
                lambda: current_accelerator(None, None),
                "current_accelerator() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: current_accelerator(device=True),
                "current_accelerator() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: current_accelerator(None, device=True),
                "current_accelerator() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: current_accelerator(False, check_available=True),
                "current_accelerator() got multiple values for argument 'check_available'",
            ),
        )
        no_argument_errors = {
            "is_available": (
                "is_available() takes 0 positional arguments but 1 was given",
                "is_available() takes 0 positional arguments but 2 were given",
                "is_available() got an unexpected keyword argument 'device'",
            ),
            "device_count": (
                "device_count() takes 0 positional arguments but 1 was given",
                "device_count() takes 0 positional arguments but 2 were given",
                "device_count() got an unexpected keyword argument 'device'",
            ),
        }

        cases = list(current_errors)
        for name, messages in no_argument_errors.items():
            function = getattr(torch.accelerator, name)
            cases.extend(
                (
                    (lambda function=function: function(None), messages[0]),
                    (
                        lambda function=function: function(None, None),
                        messages[1],
                    ),
                    (
                        lambda function=function: function(device=True),
                        messages[2],
                    ),
                )
            )

        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_reload_and_reimport_rebind_functions_and_keep_results(self):
        original_module = torch.accelerator
        original_functions = {
            name: getattr(original_module, name)
            for name in original_module.__all__
        }
        module_name = original_module.__name__

        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.accelerator, original_module)
        for name, original_function in original_functions.items():
            with self.subTest(stage="reload", name=name):
                reloaded_function = getattr(original_module, name)
                self.assertIsNot(reloaded_function, original_function)
                self.assertEqual(reloaded_function.__module__, module_name)
                self.assertIs(
                    pickle.loads(pickle.dumps(reloaded_function)),
                    reloaded_function,
                )
        self.assertIs(original_module.current_accelerator(), None)
        self.assertIs(original_module.is_available(), False)
        self.assertEqual(original_module.device_count(), 0)

        reloaded_functions = {
            name: getattr(original_module, name)
            for name in original_module.__all__
        }
        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)

            self.assertIsNot(replacement_module, original_module)
            self.assertIs(sys.modules[module_name], replacement_module)
            self.assertIs(torch.accelerator, replacement_module)
            for name, reloaded_function in reloaded_functions.items():
                with self.subTest(stage="reimport", name=name):
                    self.assertIsNot(
                        getattr(replacement_module, name), reloaded_function
                    )
            self.assertIs(replacement_module.current_accelerator(), None)
            self.assertIs(replacement_module.is_available(), False)
            self.assertEqual(replacement_module.device_count(), 0)
        finally:
            sys.modules[module_name] = original_module
            torch.accelerator = original_module

    def test_device_selection_streams_memory_graphs_and_execution_are_unsupported(self):
        accelerator = torch.accelerator
        self.assertEqual(
            {name for name in vars(accelerator) if not name.startswith("_")},
            {"current_accelerator", "device_count", "is_available"},
        )

        unsupported = (
            "Graph",
            "current_device_idx",
            "current_device_index",
            "current_stream",
            "device_index",
            "empty_cache",
            "empty_host_cache",
            "get_device_capability",
            "get_memory_info",
            "max_memory_allocated",
            "max_memory_reserved",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
            "reset_accumulated_memory_stats",
            "reset_peak_memory_stats",
            "set_device_idx",
            "set_device_index",
            "set_stream",
            "synchronize",
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(accelerator, name))
                self.assertNotIn(name, accelerator.__all__)

        for module_name in (
            "torch_rs.accelerator.graphs",
            "torch_rs.accelerator.memory",
        ):
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

        for name in ("cuda", "Event", "Stream"):
            self.assertFalse(hasattr(torch, name))
        with self.assertRaisesRegex(
            RuntimeError,
            "device 'cuda' is not supported; only 'cpu' is implemented",
        ):
            torch.device("cuda")
        with self.assertRaisesRegex(
            RuntimeError,
            "device 'cuda' is not supported; only 'cpu' is implemented",
        ):
            torch.ones(1, device="cuda")

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
        script = r"""
import copy
import importlib
import os
import pickle
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class ExplosiveBool:
    def __bool__(self):
        raise AssertionError("check_available truthiness was evaluated")

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch

accelerator = torch.accelerator
assert accelerator.current_accelerator() is None
assert accelerator.current_accelerator(True) is None
assert accelerator.current_accelerator(ExplosiveBool()) is None
assert accelerator.is_available() is False
assert type(accelerator.device_count()) is int
assert accelerator.device_count() == 0
for name in accelerator.__all__:
    function = getattr(accelerator, name)
    assert copy.copy(function) is function
    assert copy.deepcopy(function) is function
    assert pickle.loads(pickle.dumps(function)) is function

original_functions = tuple(getattr(accelerator, name) for name in accelerator.__all__)
assert importlib.reload(accelerator) is accelerator
assert all(
    getattr(accelerator, name) is not original
    for name, original in zip(accelerator.__all__, original_functions)
)
del sys.modules["torch_rs.accelerator"]
replacement = importlib.import_module("torch_rs.accelerator")
assert replacement is not accelerator
assert torch.accelerator is replacement
assert replacement.current_accelerator() is None
assert replacement.is_available() is False
assert replacement.device_count() == 0
assert not hasattr(replacement, "current_stream")
assert not hasattr(replacement, "empty_cache")
assert not hasattr(replacement, "Graph")
assert not any(
    name == "torch" or name.startswith("torch.") for name in sys.modules
)
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
