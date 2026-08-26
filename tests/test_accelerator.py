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

NO_ACCELERATOR_DEVICE_ERROR = (
    "Cannot access accelerator device when none is available."
)

FUNCTION_DOCS = {
    "current_accelerator": """Return the device of the accelerator available at compilation time.
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
    """,
    "is_available": """Check if the current accelerator is available at runtime: it was built, all the
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
    """,
    "device_count": """Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT poison fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """,
    "current_device_index": """Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.
    """,
}


class AcceleratorTests(unittest.TestCase):
    def test_cpu_only_results_come_from_one_probe_without_hardware_discovery(self):
        accelerator = torch.accelerator
        discovery = accelerator._discover_accelerator

        self.assertEqual(discovery.__code__.co_names, ())
        self.assertEqual(discovery.__code__.co_freevars, ())
        self.assertEqual(discovery.__code__.co_cellvars, ())
        state = discovery()
        self.assertEqual(state, (None, False, 0, None))
        self.assertIs(state[0], None)
        self.assertIs(state[1], False)
        self.assertIs(type(state[2]), int)
        self.assertIs(state[3], None)

        with mock.patch.object(
            accelerator,
            "_discover_accelerator",
            wraps=discovery,
        ) as shared_discovery:
            self.assertIs(accelerator.current_accelerator(), None)
            self.assertIs(
                accelerator.current_accelerator(check_available=True), None
            )
            self.assertIs(accelerator.is_available(), False)
            count = accelerator.device_count()
            self.assertIs(type(count), int)
            self.assertEqual(count, 0)
            for _ in range(2):
                with self.assertRaises(RuntimeError) as raised:
                    accelerator.current_device_index()
                self.assertEqual(str(raised.exception), NO_ACCELERATOR_DEVICE_ERROR)
                self.assertEqual(raised.exception.args, (NO_ACCELERATOR_DEVICE_ERROR,))

        self.assertEqual(shared_discovery.call_count, 6)
        self.assertEqual(shared_discovery.call_args_list, [mock.call()] * 6)
        for function in (
            accelerator.current_accelerator,
            accelerator.is_available,
            accelerator.device_count,
        ):
            self.assertEqual(function.__code__.co_names, ("_discover_accelerator",))
        self.assertEqual(
            accelerator.current_device_index.__code__.co_names,
            ("_discover_accelerator", "RuntimeError"),
        )

        class ExplodingTruth:
            def __bool__(self):
                raise AssertionError("check_available was evaluated without an accelerator")

        self.assertIs(accelerator.current_accelerator(ExplodingTruth()), None)

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
                        side_effect=AssertionError("hardware was probed"),
                    ):
                        self.assertIs(accelerator.current_accelerator(), None)
                        with self.assertRaises(RuntimeError) as raised:
                            accelerator.current_device_index()
                        self.assertEqual(
                            str(raised.exception), NO_ACCELERATOR_DEVICE_ERROR
                        )
                        self.assertIs(accelerator.is_available(), False)
                        self.assertEqual(accelerator.device_count(), 0)

    def test_signature_annotations_documentation_and_module_identity(self):
        accelerator = importlib.import_module("torch_rs.accelerator")
        expected_signatures = {
            "current_accelerator": inspect.Signature(
                parameters=(
                    inspect.Parameter(
                        "check_available",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=False,
                        annotation=bool,
                    ),
                ),
                return_annotation=torch.device | None,
            ),
            "current_device_index": inspect.Signature(return_annotation=int),
            "is_available": inspect.Signature(return_annotation=bool),
            "device_count": inspect.Signature(return_annotation=int),
        }
        expected_annotations = {
            "current_accelerator": {
                "check_available": bool,
                "return": torch.device | None,
            },
            "current_device_index": {"return": int},
            "is_available": {"return": bool},
            "device_count": {"return": int},
        }

        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(sys.modules["torch_rs.accelerator"], accelerator)
        self.assertEqual(accelerator.__doc__, MODULE_DOC)
        for name in (
            "current_accelerator",
            "current_device_index",
            "device_count",
            "is_available",
        ):
            with self.subTest(name=name):
                function = getattr(accelerator, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(inspect.signature(function), expected_signatures[name])
                self.assertEqual(
                    inspect.get_annotations(function), expected_annotations[name]
                )
                self.assertEqual(
                    typing.get_type_hints(function), expected_annotations[name]
                )
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.accelerator")
                self.assertIs(inspect.getmodule(function), accelerator)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__),
                    inspect.cleandoc(FUNCTION_DOCS[name]),
                )
                self.assertEqual(
                    function.__defaults__,
                    (False,) if name == "current_accelerator" else None,
                )
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(
                    function.__dict__,
                    (
                        {"__deprecated__": "Use `current_device_index` instead."}
                        if name == "current_device_index"
                        else {}
                    ),
                )
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        accelerator = torch.accelerator
        supported = {
            "current_accelerator",
            "current_device_index",
            "device_count",
            "is_available",
        }

        self.assertEqual(
            accelerator.__all__,
            [
                "current_accelerator",
                "current_device_index",
                "device_count",
                "is_available",
            ],
        )
        self.assertEqual(
            {name for name in vars(accelerator) if not name.startswith("_")},
            supported,
        )

        package_import = {}
        direct_import = {}
        wildcard_import = {}
        exec("from torch_rs import accelerator", package_import)
        exec(
            "from torch_rs.accelerator import current_accelerator, "
            "current_device_index, device_count, is_available",
            direct_import,
        )
        exec("from torch_rs.accelerator import *", wildcard_import)
        self.assertIs(package_import["accelerator"], accelerator)
        self.assertEqual(
            {name for name in wildcard_import if not name.startswith("__")},
            supported,
        )
        for name in supported:
            function = getattr(accelerator, name)
            with self.subTest(name=name):
                self.assertIs(direct_import[name], function)
                self.assertIs(wildcard_import[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.accelerator", payload)
                    self.assertIs(pickle.loads(payload), function)

        for name in ("accelerator", *supported):
            self.assertNotIn(name, torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in ("accelerator", *supported):
            self.assertNotIn(name, top_level_namespace)

    def test_reload_replaces_functions_but_preserves_the_canonical_module(self):
        accelerator = torch.accelerator
        old_all = accelerator.__all__
        old_discovery = accelerator._discover_accelerator
        old_functions = {
            name: getattr(accelerator, name)
            for name in (
                "current_accelerator",
                "current_device_index",
                "device_count",
                "is_available",
            )
        }

        reloaded = importlib.reload(accelerator)

        self.assertIs(reloaded, accelerator)
        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(sys.modules["torch_rs.accelerator"], accelerator)
        self.assertIsNot(accelerator.__all__, old_all)
        self.assertIsNot(accelerator._discover_accelerator, old_discovery)
        self.assertEqual(accelerator._discover_accelerator(), (None, False, 0, None))
        self.assertIs(accelerator.current_accelerator(), None)
        with self.assertRaises(RuntimeError) as raised:
            accelerator.current_device_index()
        self.assertEqual(str(raised.exception), NO_ACCELERATOR_DEVICE_ERROR)
        self.assertEqual(raised.exception.args, (NO_ACCELERATOR_DEVICE_ERROR,))
        self.assertIs(accelerator.is_available(), False)
        self.assertEqual(accelerator.device_count(), 0)

        for name, old_function in old_functions.items():
            with self.subTest(name=name):
                new_function = getattr(accelerator, name)
                self.assertIsNot(new_function, old_function)
                self.assertIs(copy.copy(new_function), new_function)
                self.assertIs(copy.deepcopy(new_function), new_function)
                self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(old_function)

    def test_argument_errors_match_python_3_binding_used_by_pytorch_2_13(self):
        accelerator = torch.accelerator
        cases = (
            (
                lambda: accelerator.current_accelerator(False, False),
                "current_accelerator() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.current_accelerator(
                    False, check_available=True
                ),
                "current_accelerator() got multiple values for argument 'check_available'",
            ),
            (
                lambda: accelerator.current_accelerator(unexpected=True),
                "current_accelerator() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: accelerator.current_device_index(None),
                "current_device_index() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: accelerator.current_device_index(None, None),
                "current_device_index() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.current_device_index(device=True),
                "current_device_index() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: accelerator.is_available(None),
                "is_available() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: accelerator.is_available(None, None),
                "is_available() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.is_available(device=True),
                "is_available() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: accelerator.device_count(None),
                "device_count() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: accelerator.device_count(None, None),
                "device_count() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.device_count(device=True),
                "device_count() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_results_are_thread_safe_and_preserve_grad_mode(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def current_device_index_outcome():
            try:
                return ("return", torch.accelerator.current_device_index())
            except Exception as error:
                return ("raise", type(error), str(error), error.args)

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        torch.accelerator.current_accelerator(),
                        torch.accelerator.current_accelerator(True),
                        current_device_index_outcome(),
                        current_device_index_outcome(),
                        torch.accelerator.is_available(),
                        torch.accelerator.device_count(),
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
                    (
                        "raise",
                        RuntimeError,
                        NO_ACCELERATOR_DEVICE_ERROR,
                        (NO_ACCELERATOR_DEVICE_ERROR,),
                    ),
                    (
                        "raise",
                        RuntimeError,
                        NO_ACCELERATOR_DEVICE_ERROR,
                        (NO_ACCELERATOR_DEVICE_ERROR,),
                    ),
                    False,
                    0,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[5], False)
            self.assertIs(type(result[6]), int)

    def test_selection_stream_memory_graph_and_execution_apis_stay_unsupported(self):
        accelerator = torch.accelerator
        unsupported = {
            "Graph",
            "current_device_idx",
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
        }
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(accelerator, name))
                self.assertNotIn(name, accelerator.__all__)

        for module_name in (
            "torch_rs.accelerator.graphs",
            "torch_rs.accelerator.memory",
        ):
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

        self.assertFalse(hasattr(torch, "cuda"))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.cuda")
        for specification in ("cuda", "cuda:0"):
            with self.subTest(specification=specification):
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.device(specification)
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.tensor([1.0], device=specification)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r'''
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch

modules_before_calls = set(sys.modules)
assert torch.accelerator._discover_accelerator() == (None, False, 0, None)
assert torch.accelerator.current_accelerator() is None
assert torch.accelerator.current_accelerator(check_available=True) is None
for _ in range(2):
    try:
        torch.accelerator.current_device_index()
    except RuntimeError as error:
        assert str(error) == "Cannot access accelerator device when none is available."
        assert error.args == (
            "Cannot access accelerator device when none is available.",
        )
    else:
        raise AssertionError("current_device_index() unexpectedly returned")
assert torch.accelerator.is_available() is False
count = torch.accelerator.device_count()
assert type(count) is int and count == 0
assert set(sys.modules) == modules_before_calls
assert not hasattr(torch, "cuda")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
'''
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
