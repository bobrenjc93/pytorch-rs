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
from collections import OrderedDict
from unittest import mock

import torch_rs as torch


MODULE_DOC = """
This package introduces support for the current :ref:`accelerator<accelerators>` in python.
"""

NO_ACCELERATOR_ERROR = "Cannot access accelerator device when none is available."

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
    "empty_cache": """Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other application.

    .. note:: This function is a no-op if the memory allocator for the current
        :ref:`accelerator <accelerators>` has not been initialized.
    """,
    "memory_allocated": """Return the current :ref:`accelerator<accelerators>` device memory occupied by tensors
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory occupied by live tensors (in bytes) within the current process.
    """,
    "max_memory_allocated": """Return the current :ref:`accelerator<accelerators>` maximum device memory occupied by tensors
    in bytes for a given device index.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory occupied by live tensors (in bytes) within the current process.
    """,
    "max_memory_reserved": """Return the current :ref:`accelerator<accelerators>` maximum device memory managed by the caching allocator
    in bytes for a given device index.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.accelerator.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the peak memory reserved by PyTorch (in bytes) within the current process.
    """,
    "memory_reserved": """Return the current :ref:`accelerator<accelerators>` device memory managed by the caching allocator
    in bytes for a given device index.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        int: the current memory reserved by PyTorch (in bytes) within the current process.
    """,
    "memory_stats": """Return a dictionary of accelerator device memory allocator statistics for a given device index.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer.

    Core statistics:

    - ``"allocated.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of allocation requests received by the memory allocator.
    - ``"allocated_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of allocated memory.
    - ``"segment.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of reserved segments from device memory allocation.
    - ``"reserved_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of reserved memory.
    - ``"active.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of active memory blocks.
    - ``"active_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of active memory.
    - ``"inactive_split.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      number of inactive, non-releasable memory blocks.
    - ``"inactive_split_bytes.{all,large_pool,small_pool}.{current,peak,allocated,freed}"``:
      amount of inactive, non-releasable memory.

    For these core statistics, values are broken down as follows.

    Pool type:

    - ``all``: combined statistics across all memory pools.
    - ``large_pool``: statistics for the large allocation pool
      (as of June 2025, for size >= 1MB allocations).
    - ``small_pool``: statistics for the small allocation pool
      (as of June 2025, for size < 1MB allocations).

    Metric type:

    - ``current``: current value of this metric.
    - ``peak``: maximum value of this metric.
    - ``allocated``: historical total increase in this metric.
    - ``freed``: historical total decrease in this metric.

    In addition to the core statistics, we also provide some simple event
    counters:

    - ``"num_alloc_retries"``: number of failed device memory allocation calls that
      result in a cache flush and retry.
    - ``"num_ooms"``: number of out-of-memory errors thrown.
    - ``"num_sync_all_streams"``: number of ``synchronize_and_free_events`` calls.
    - ``"num_device_alloc"``: number of device memory allocation calls.
    - ``"num_device_free"``: number of device memory free calls.

    Args:
        device_index (:class:`torch.device`, str, int, optional): the index of the device to target.
            If not given, use :func:`torch.accelerator.current_device_index` by default.
            If a :class:`torch.device` or str is provided, its type must match the current
            :ref:`accelerator<accelerators>` device type.

    Returns:
        OrderedDict[str, Any]: an ordered dictionary mapping statistic names to their values.
    """,
}


class AcceleratorTests(unittest.TestCase):
    def assert_current_device_index_unavailable(self, call):
        with self.assertRaises(RuntimeError) as raised:
            call()
        self.assertEqual(str(raised.exception), NO_ACCELERATOR_ERROR)
        self.assertEqual(raised.exception.args, (NO_ACCELERATOR_ERROR,))
        return raised.exception

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
            self.assert_current_device_index_unavailable(
                accelerator.current_device_index
            )
            self.assertIs(accelerator.empty_cache(), None)
            self.assertIs(type(accelerator.memory_allocated()), int)
            self.assertEqual(accelerator.memory_allocated(), 0)
            self.assertIs(type(accelerator.max_memory_allocated()), int)
            self.assertEqual(accelerator.max_memory_allocated(), 0)
            self.assertIs(type(accelerator.max_memory_reserved()), int)
            self.assertEqual(accelerator.max_memory_reserved(), 0)
            self.assertIs(type(accelerator.memory_reserved()), int)
            self.assertEqual(accelerator.memory_reserved(), 0)
            self.assertEqual(accelerator.memory_stats(), OrderedDict())
            self.assertIs(accelerator.is_available(), False)
            count = accelerator.device_count()
            self.assertIs(type(count), int)
            self.assertEqual(count, 0)

        self.assertEqual(shared_discovery.call_count, 5)
        self.assertEqual(shared_discovery.call_args_list, [mock.call()] * 5)
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
        self.assertEqual(accelerator.empty_cache.__code__.co_names, ())
        self.assertEqual(
            accelerator.memory_allocated.__code__.co_names,
            ("memory_stats", "get"),
        )
        self.assertEqual(
            accelerator.max_memory_allocated.__code__.co_names,
            ("memory_stats", "get"),
        )
        self.assertEqual(
            accelerator.max_memory_reserved.__code__.co_names,
            ("memory_stats", "get"),
        )
        self.assertEqual(
            accelerator.memory_reserved.__code__.co_names,
            ("memory_stats", "get"),
        )
        self.assertEqual(accelerator.memory_stats.__code__.co_names, ("_OrderedDict",))

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
                        self.assert_current_device_index_unavailable(
                            accelerator.current_device_index
                        )
                        self.assertIs(accelerator.empty_cache(), None)
                        self.assertEqual(accelerator.memory_allocated(), 0)
                        self.assertEqual(accelerator.max_memory_allocated(), 0)
                        self.assertEqual(accelerator.max_memory_reserved(), 0)
                        self.assertEqual(accelerator.memory_reserved(), 0)
                        self.assertEqual(accelerator.memory_stats(), OrderedDict())
                        self.assertIs(accelerator.is_available(), False)
                        self.assertEqual(accelerator.device_count(), 0)

    def test_current_device_index_repeated_calls_raise_fresh_exact_errors(self):
        errors = [
            self.assert_current_device_index_unavailable(
                torch.accelerator.current_device_index
            )
            for _ in range(8)
        ]

        self.assertEqual(len({id(error) for error in errors}), len(errors))

    def test_empty_cache_is_a_repeatable_probe_free_no_op(self):
        accelerator = torch.accelerator

        with mock.patch.object(
            accelerator,
            "_discover_accelerator",
            side_effect=AssertionError("accelerator discovery was attempted"),
        ):
            results = tuple(accelerator.empty_cache() for _ in range(16))

        self.assertEqual(results, (None,) * 16)
        for result in results:
            self.assertIs(result, None)

    def test_memory_queries_are_probe_free_and_ignore_device_tokens(self):
        accelerator = torch.accelerator

        class ExplodingDeviceToken:
            def __bool__(self):
                raise AssertionError("device token truth value was inspected")

            def __index__(self):
                raise AssertionError("device token index was inspected")

            def __int__(self):
                raise AssertionError("device token integer value was inspected")

            def __str__(self):
                raise AssertionError("device token string value was inspected")

        tokens = (
            None,
            0,
            -1,
            True,
            1.5,
            "cpu",
            "cuda:0",
            torch.device("cpu"),
            object(),
            [],
            {},
            ExplodingDeviceToken(),
        )
        with mock.patch.object(
            accelerator,
            "_discover_accelerator",
            side_effect=AssertionError("accelerator discovery was attempted"),
        ):
            results = tuple(accelerator.memory_stats(token) for token in tokens)
            results += tuple(accelerator.memory_stats() for _ in range(8))
            allocated = tuple(
                accelerator.memory_allocated(token) for token in tokens
            )
            allocated += tuple(accelerator.memory_allocated() for _ in range(8))
            max_allocated = tuple(
                accelerator.max_memory_allocated(token) for token in tokens
            )
            max_allocated += tuple(
                accelerator.max_memory_allocated() for _ in range(8)
            )
            max_reserved = tuple(
                accelerator.max_memory_reserved(token) for token in tokens
            )
            max_reserved += tuple(
                accelerator.max_memory_reserved() for _ in range(8)
            )
            reserved = tuple(
                accelerator.memory_reserved(token) for token in tokens
            )
            reserved += tuple(accelerator.memory_reserved() for _ in range(8))

        self.assertTrue(all(type(result) is OrderedDict for result in results))
        self.assertTrue(all(result == OrderedDict() for result in results))
        self.assertEqual(len({id(result) for result in results}), len(results))
        results[0]["mutated"] = 1
        self.assertEqual(results[0], OrderedDict((("mutated", 1),)))
        self.assertTrue(all(not result for result in results[1:]))
        self.assertEqual(allocated, (0,) * len(allocated))
        self.assertTrue(all(type(result) is int for result in allocated))
        self.assertEqual(max_allocated, (0,) * len(max_allocated))
        self.assertTrue(all(type(result) is int for result in max_allocated))
        self.assertEqual(max_reserved, (0,) * len(max_reserved))
        self.assertTrue(all(type(result) is int for result in max_reserved))
        self.assertEqual(reserved, (0,) * len(reserved))
        self.assertTrue(all(type(result) is int for result in reserved))

        sentinel = object()
        with mock.patch.object(
            accelerator.memory,
            "memory_stats",
            return_value=OrderedDict(
                (
                    ("allocated_bytes.all.current", 37),
                    ("allocated_bytes.all.peak", 43),
                    ("reserved_bytes.all.current", 53),
                    ("reserved_bytes.all.peak", 59),
                )
            ),
        ) as memory_stats:
            self.assertEqual(accelerator.memory_allocated(sentinel), 37)
            self.assertEqual(accelerator.max_memory_allocated(sentinel), 43)
            self.assertEqual(accelerator.max_memory_reserved(sentinel), 59)
            self.assertEqual(accelerator.memory_reserved(sentinel), 53)
        self.assertEqual(memory_stats.call_args_list, [mock.call(sentinel)] * 4)

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
            "empty_cache": inspect.Signature(return_annotation=None),
            "is_available": inspect.Signature(return_annotation=bool),
            "device_count": inspect.Signature(return_annotation=int),
            "memory_allocated": inspect.Signature(
                parameters=(
                    inspect.Parameter(
                        "device_index",
                        inspect.Parameter.POSITIONAL_ONLY,
                        default=None,
                        annotation=torch.device | str | int | None,
                    ),
                ),
                return_annotation=int,
            ),
            "max_memory_allocated": inspect.Signature(
                parameters=(
                    inspect.Parameter(
                        "device_index",
                        inspect.Parameter.POSITIONAL_ONLY,
                        default=None,
                        annotation=torch.device | str | int | None,
                    ),
                ),
                return_annotation=int,
            ),
            "max_memory_reserved": inspect.Signature(
                parameters=(
                    inspect.Parameter(
                        "device_index",
                        inspect.Parameter.POSITIONAL_ONLY,
                        default=None,
                        annotation=torch.device | str | int | None,
                    ),
                ),
                return_annotation=int,
            ),
            "memory_reserved": inspect.Signature(
                parameters=(
                    inspect.Parameter(
                        "device_index",
                        inspect.Parameter.POSITIONAL_ONLY,
                        default=None,
                        annotation=torch.device | str | int | None,
                    ),
                ),
                return_annotation=int,
            ),
            "memory_stats": inspect.Signature(
                parameters=(
                    inspect.Parameter(
                        "device_index",
                        inspect.Parameter.POSITIONAL_ONLY,
                        default=None,
                        annotation=torch.device | str | int | None,
                    ),
                ),
                return_annotation=OrderedDict[str, typing.Any],
            ),
        }
        expected_annotations = {
            "current_accelerator": {
                "check_available": bool,
                "return": torch.device | None,
            },
            "current_device_index": {"return": int},
            "empty_cache": {"return": None},
            "is_available": {"return": bool},
            "device_count": {"return": int},
            "memory_allocated": {
                "device_index": torch.device | str | int | None,
                "return": int,
            },
            "max_memory_allocated": {
                "device_index": torch.device | str | int | None,
                "return": int,
            },
            "max_memory_reserved": {
                "device_index": torch.device | str | int | None,
                "return": int,
            },
            "memory_reserved": {
                "device_index": torch.device | str | int | None,
                "return": int,
            },
            "memory_stats": {
                "device_index": torch.device | str | int | None,
                "return": OrderedDict[str, typing.Any],
            },
        }
        expected_type_hints = {
            **expected_annotations,
            "empty_cache": {"return": type(None)},
        }

        self.assertIs(torch.accelerator, accelerator)
        self.assertIs(sys.modules["torch_rs.accelerator"], accelerator)
        self.assertEqual(accelerator.__doc__, MODULE_DOC)
        for name in (
            "current_accelerator",
            "current_device_index",
            "device_count",
            "empty_cache",
            "is_available",
            "max_memory_allocated",
            "max_memory_reserved",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
        ):
            with self.subTest(name=name):
                function = getattr(accelerator, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(inspect.signature(function), expected_signatures[name])
                self.assertEqual(
                    inspect.get_annotations(function), expected_annotations[name]
                )
                self.assertEqual(
                    typing.get_type_hints(function), expected_type_hints[name]
                )
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                defining_module = (
                    accelerator.memory
                    if name
                    in {
                        "empty_cache",
                        "max_memory_allocated",
                        "max_memory_reserved",
                        "memory_allocated",
                        "memory_reserved",
                        "memory_stats",
                    }
                    else accelerator
                )
                self.assertEqual(function.__module__, defining_module.__name__)
                self.assertIs(inspect.getmodule(function), defining_module)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__),
                    inspect.cleandoc(FUNCTION_DOCS[name]),
                )
                self.assertEqual(
                    function.__defaults__,
                    (False,)
                    if name == "current_accelerator"
                    else (None,)
                    if name
                    in {
                        "max_memory_allocated",
                        "max_memory_reserved",
                        "memory_allocated",
                        "memory_reserved",
                        "memory_stats",
                    }
                    else None,
                )
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(
                    function.__dict__,
                    {"__deprecated__": "Use `current_device_index` instead."}
                    if name == "current_device_index"
                    else {},
                )
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        accelerator = torch.accelerator
        supported = {
            "current_accelerator",
            "current_device_index",
            "device_count",
            "empty_cache",
            "is_available",
            "max_memory_allocated",
            "max_memory_reserved",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
        }
        memory = importlib.import_module("torch_rs.accelerator.memory")

        self.assertIs(accelerator.memory, memory)
        self.assertIs(accelerator.empty_cache, memory.empty_cache)
        self.assertIs(
            accelerator.max_memory_allocated,
            memory.max_memory_allocated,
        )
        self.assertIs(
            accelerator.max_memory_reserved,
            memory.max_memory_reserved,
        )
        self.assertIs(accelerator.memory_allocated, memory.memory_allocated)
        self.assertIs(accelerator.memory_reserved, memory.memory_reserved)
        self.assertIs(accelerator.memory_stats, memory.memory_stats)
        self.assertIs(sys.modules["torch_rs.accelerator.memory"], memory)
        self.assertIsNone(memory.__doc__)
        self.assertEqual(
            memory.__all__,
            [
                "empty_cache",
                "max_memory_allocated",
                "max_memory_reserved",
                "memory_allocated",
                "memory_reserved",
                "memory_stats",
            ],
        )
        self.assertEqual(
            {name for name in vars(memory) if not name.startswith("_")},
            {
                "empty_cache",
                "max_memory_allocated",
                "max_memory_reserved",
                "memory_allocated",
                "memory_reserved",
                "memory_stats",
            },
        )

        self.assertEqual(
            accelerator.__all__,
            [
                "current_accelerator",
                "current_device_index",
                "device_count",
                "empty_cache",
                "is_available",
                "max_memory_allocated",
                "max_memory_reserved",
                "memory_allocated",
                "memory_reserved",
                "memory_stats",
            ],
        )
        self.assertEqual(
            {name for name in vars(accelerator) if not name.startswith("_")},
            supported | {"memory"},
        )

        package_import = {}
        direct_import = {}
        wildcard_import = {}
        memory_wildcard_import = {}
        exec("from torch_rs import accelerator", package_import)
        exec(
            "from torch_rs.accelerator import current_accelerator, current_device_index, device_count, empty_cache, is_available, max_memory_allocated, max_memory_reserved, memory_allocated, memory_reserved, memory_stats",
            direct_import,
        )
        exec("from torch_rs.accelerator import *", wildcard_import)
        exec("from torch_rs.accelerator.memory import *", memory_wildcard_import)
        self.assertIs(package_import["accelerator"], accelerator)
        self.assertEqual(
            {name for name in wildcard_import if not name.startswith("__")},
            supported,
        )
        self.assertEqual(
            {
                name
                for name in memory_wildcard_import
                if not name.startswith("__")
            },
            {
                "empty_cache",
                "max_memory_allocated",
                "max_memory_reserved",
                "memory_allocated",
                "memory_reserved",
                "memory_stats",
            },
        )
        for name in supported:
            function = getattr(accelerator, name)
            with self.subTest(name=name):
                self.assertIs(direct_import[name], function)
                self.assertIs(wildcard_import[name], function)
                if name in {
                    "empty_cache",
                    "max_memory_allocated",
                    "max_memory_reserved",
                    "memory_allocated",
                    "memory_reserved",
                    "memory_stats",
                }:
                    self.assertIs(memory_wildcard_import[name], function)
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
        old_empty_cache = accelerator.empty_cache
        old_max_memory_allocated = accelerator.max_memory_allocated
        old_max_memory_reserved = accelerator.max_memory_reserved
        old_memory_allocated = accelerator.memory_allocated
        old_memory_reserved = accelerator.memory_reserved
        old_memory_stats = accelerator.memory_stats
        memory = accelerator.memory
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
        self.assertIs(accelerator.memory, memory)
        self.assertIs(accelerator.empty_cache, old_empty_cache)
        self.assertIs(accelerator.empty_cache, memory.empty_cache)
        self.assertIs(accelerator.empty_cache(), None)
        self.assertIs(
            accelerator.max_memory_allocated,
            old_max_memory_allocated,
        )
        self.assertIs(
            accelerator.max_memory_allocated,
            memory.max_memory_allocated,
        )
        self.assertEqual(accelerator.max_memory_allocated(), 0)
        self.assertIs(
            accelerator.max_memory_reserved,
            old_max_memory_reserved,
        )
        self.assertIs(
            accelerator.max_memory_reserved,
            memory.max_memory_reserved,
        )
        self.assertEqual(accelerator.max_memory_reserved(), 0)
        self.assertIs(accelerator.memory_allocated, old_memory_allocated)
        self.assertIs(accelerator.memory_allocated, memory.memory_allocated)
        self.assertEqual(accelerator.memory_allocated(), 0)
        self.assertIs(accelerator.memory_reserved, old_memory_reserved)
        self.assertIs(accelerator.memory_reserved, memory.memory_reserved)
        self.assertEqual(accelerator.memory_reserved(), 0)
        self.assertIs(accelerator.memory_stats, old_memory_stats)
        self.assertIs(accelerator.memory_stats, memory.memory_stats)
        self.assertEqual(accelerator.memory_stats(), OrderedDict())
        self.assertEqual(
            accelerator._discover_accelerator(), (None, False, 0, None)
        )
        self.assertIs(accelerator.current_accelerator(), None)
        self.assert_current_device_index_unavailable(
            accelerator.current_device_index
        )
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

    def test_memory_reload_keeps_old_and_new_cpu_build_functions_usable(self):
        accelerator = torch.accelerator
        memory = accelerator.memory
        old_all = memory.__all__
        old_functions = {
            "empty_cache": memory.empty_cache,
            "max_memory_allocated": memory.max_memory_allocated,
            "max_memory_reserved": memory.max_memory_reserved,
            "memory_allocated": memory.memory_allocated,
            "memory_reserved": memory.memory_reserved,
            "memory_stats": memory.memory_stats,
        }

        reloaded = importlib.reload(memory)
        new_functions = {
            "empty_cache": memory.empty_cache,
            "max_memory_allocated": memory.max_memory_allocated,
            "max_memory_reserved": memory.max_memory_reserved,
            "memory_allocated": memory.memory_allocated,
            "memory_reserved": memory.memory_reserved,
            "memory_stats": memory.memory_stats,
        }

        self.assertIs(reloaded, memory)
        self.assertIs(accelerator.memory, memory)
        self.assertIs(sys.modules["torch_rs.accelerator.memory"], memory)
        self.assertIsNot(memory.__all__, old_all)
        for name in (
            "empty_cache",
            "max_memory_allocated",
            "max_memory_reserved",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
        ):
            with self.subTest(name=name):
                old_function = old_functions[name]
                new_function = new_functions[name]
                self.assertIsNot(new_function, old_function)
                self.assertIs(getattr(accelerator, name), old_function)
                self.assertIsNot(getattr(accelerator, name), new_function)
                self.assertIs(copy.copy(new_function), new_function)
                self.assertIs(copy.deepcopy(new_function), new_function)
                self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(old_function)

        self.assertEqual(
            (old_functions["empty_cache"](), new_functions["empty_cache"]()),
            (None, None),
        )
        self.assertEqual(
            (
                old_functions["max_memory_allocated"](object()),
                new_functions["max_memory_allocated"](object()),
            ),
            (0, 0),
        )
        self.assertEqual(
            (
                old_functions["max_memory_reserved"](object()),
                new_functions["max_memory_reserved"](object()),
            ),
            (0, 0),
        )
        self.assertEqual(
            (
                old_functions["memory_allocated"](object()),
                new_functions["memory_allocated"](object()),
            ),
            (0, 0),
        )
        self.assertEqual(
            (
                old_functions["memory_reserved"](object()),
                new_functions["memory_reserved"](object()),
            ),
            (0, 0),
        )
        old_stats = old_functions["memory_stats"](object())
        new_stats = new_functions["memory_stats"](object())
        self.assertIs(type(old_stats), OrderedDict)
        self.assertIs(type(new_stats), OrderedDict)
        self.assertEqual((old_stats, new_stats), (OrderedDict(), OrderedDict()))
        self.assertIsNot(old_stats, new_stats)

        self.assertIs(importlib.reload(accelerator), accelerator)
        self.assertIs(accelerator.empty_cache, new_functions["empty_cache"])
        self.assertIs(accelerator.empty_cache, memory.empty_cache)
        self.assertIs(accelerator.empty_cache(), None)
        self.assertIs(
            accelerator.max_memory_allocated,
            new_functions["max_memory_allocated"],
        )
        self.assertIs(
            accelerator.max_memory_allocated,
            memory.max_memory_allocated,
        )
        self.assertEqual(accelerator.max_memory_allocated(), 0)
        self.assertIs(
            accelerator.max_memory_reserved,
            new_functions["max_memory_reserved"],
        )
        self.assertIs(
            accelerator.max_memory_reserved,
            memory.max_memory_reserved,
        )
        self.assertEqual(accelerator.max_memory_reserved(), 0)
        self.assertIs(
            accelerator.memory_allocated,
            new_functions["memory_allocated"],
        )
        self.assertIs(accelerator.memory_allocated, memory.memory_allocated)
        self.assertEqual(accelerator.memory_allocated(), 0)
        self.assertIs(
            accelerator.memory_reserved,
            new_functions["memory_reserved"],
        )
        self.assertIs(accelerator.memory_reserved, memory.memory_reserved)
        self.assertEqual(accelerator.memory_reserved(), 0)
        self.assertIs(accelerator.memory_stats, new_functions["memory_stats"])
        self.assertIs(accelerator.memory_stats, memory.memory_stats)
        self.assertEqual(accelerator.memory_stats(), OrderedDict())

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
                lambda: accelerator.empty_cache(None),
                "empty_cache() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: accelerator.empty_cache(None, None),
                "empty_cache() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.empty_cache(device=True),
                "empty_cache() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: accelerator.memory_allocated(device_index=None),
                "memory_allocated() got some positional-only arguments passed as keyword arguments: 'device_index'",
            ),
            (
                lambda: accelerator.memory_allocated(None, None),
                "memory_allocated() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.memory_allocated(unexpected=True),
                "memory_allocated() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: accelerator.max_memory_allocated(device_index=None),
                "max_memory_allocated() got some positional-only arguments passed as keyword arguments: 'device_index'",
            ),
            (
                lambda: accelerator.max_memory_allocated(None, None),
                "max_memory_allocated() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.max_memory_allocated(unexpected=True),
                "max_memory_allocated() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: accelerator.max_memory_reserved(device_index=None),
                "max_memory_reserved() got some positional-only arguments passed as keyword arguments: 'device_index'",
            ),
            (
                lambda: accelerator.max_memory_reserved(None, None),
                "max_memory_reserved() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.max_memory_reserved(unexpected=True),
                "max_memory_reserved() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: accelerator.memory_reserved(device_index=None),
                "memory_reserved() got some positional-only arguments passed as keyword arguments: 'device_index'",
            ),
            (
                lambda: accelerator.memory_reserved(None, None),
                "memory_reserved() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.memory_reserved(unexpected=True),
                "memory_reserved() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: accelerator.memory_stats(device_index=None),
                "memory_stats() got some positional-only arguments passed as keyword arguments: 'device_index'",
            ),
            (
                lambda: accelerator.memory_stats(None, None),
                "memory_stats() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: accelerator.memory_stats(unexpected=True),
                "memory_stats() got an unexpected keyword argument 'unexpected'",
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

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    try:
                        torch.accelerator.current_device_index()
                    except RuntimeError as error:
                        index_outcome = (type(error), str(error), error.args)
                    else:
                        raise AssertionError(
                            "current_device_index() unexpectedly returned"
                        )
                    results[index] = (
                        torch.is_grad_enabled(),
                        torch.accelerator.current_accelerator(),
                        torch.accelerator.current_accelerator(True),
                        index_outcome,
                        tuple(torch.accelerator.empty_cache() for _ in range(4)),
                        tuple(
                            torch.accelerator.memory_allocated(index)
                            for _ in range(4)
                        ),
                        tuple(
                            torch.accelerator.max_memory_allocated(index)
                            for _ in range(4)
                        ),
                        tuple(
                            torch.accelerator.max_memory_reserved(index)
                            for _ in range(4)
                        ),
                        tuple(
                            torch.accelerator.memory_reserved(index)
                            for _ in range(4)
                        ),
                        tuple(torch.accelerator.memory_stats(index) for _ in range(4)),
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
                    (RuntimeError, NO_ACCELERATOR_ERROR, (NO_ACCELERATOR_ERROR,)),
                    (None,) * 4,
                    (0,) * 4,
                    (0,) * 4,
                    (0,) * 4,
                    (0,) * 4,
                    (OrderedDict(),) * 4,
                    False,
                    0,
                    expected_grad_state,
                ),
            )
            self.assertTrue(all(type(value) is int for value in result[5]))
            self.assertTrue(all(type(value) is int for value in result[6]))
            self.assertTrue(all(type(value) is int for value in result[7]))
            self.assertTrue(all(type(value) is int for value in result[8]))
            self.assertTrue(all(type(stats) is OrderedDict for stats in result[9]))
            self.assertEqual(len({id(stats) for stats in result[9]}), 4)
            self.assertIs(result[10], False)
            self.assertIs(type(result[11]), int)

        all_stats = [stats for result in results for stats in result[9]]
        self.assertEqual(len({id(stats) for stats in all_stats}), len(all_stats))

    def test_selection_stream_memory_graph_and_execution_apis_stay_unsupported(self):
        accelerator = torch.accelerator
        unsupported = {
            "Graph",
            "current_device_idx",
            "current_stream",
            "device_index",
            "empty_host_cache",
            "get_device_capability",
            "get_memory_info",
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

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.accelerator.graphs")

        memory = importlib.import_module("torch_rs.accelerator.memory")
        self.assertIs(accelerator.memory, memory)
        self.assertEqual(
            memory.__all__,
            [
                "empty_cache",
                "max_memory_allocated",
                "max_memory_reserved",
                "memory_allocated",
                "memory_reserved",
                "memory_stats",
            ],
        )
        for name in unsupported:
            with self.subTest(memory_name=name):
                self.assertFalse(hasattr(memory, name))
                self.assertNotIn(name, memory.__all__)

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

    def test_importing_and_calling_does_not_import_external_runtimes(self):
        script = r'''
import os
import sys
from collections import OrderedDict

class RejectExternalRuntimeImport:
    blocked = {
        "amdsmi",
        "cupy",
        "intel_extension_for_pytorch",
        "nvidia",
        "numpy",
        "pyamdgpuinfo",
        "pynvml",
        "torch",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch
from torch_rs.accelerator.memory import (
    empty_cache,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
)

class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")

modules_before_calls = set(sys.modules)
assert torch.accelerator.empty_cache is empty_cache
assert torch.accelerator.max_memory_allocated is max_memory_allocated
assert torch.accelerator.max_memory_reserved is max_memory_reserved
assert torch.accelerator.memory_allocated is memory_allocated
assert torch.accelerator.memory_reserved is memory_reserved
assert torch.accelerator.memory_stats is memory_stats
assert torch.accelerator._discover_accelerator() == (None, False, 0, None)
assert torch.accelerator.current_accelerator() is None
assert torch.accelerator.current_accelerator(check_available=True) is None
for _ in range(3):
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
for _ in range(8):
    assert torch.accelerator.empty_cache() is None
    assert empty_cache() is None
stats = [
    torch.accelerator.memory_stats(),
    torch.accelerator.memory_stats("cuda:0"),
    memory_stats(ExplodingDeviceToken()),
]
assert all(type(value) is OrderedDict and not value for value in stats)
assert len({id(value) for value in stats}) == len(stats)
allocated = [
    torch.accelerator.memory_allocated(),
    torch.accelerator.memory_allocated("cuda:0"),
    memory_allocated(ExplodingDeviceToken()),
]
assert allocated == [0, 0, 0]
assert all(type(value) is int for value in allocated)
max_allocated = [
    torch.accelerator.max_memory_allocated(),
    torch.accelerator.max_memory_allocated("cuda:0"),
    max_memory_allocated(ExplodingDeviceToken()),
]
assert max_allocated == [0, 0, 0]
assert all(type(value) is int for value in max_allocated)
max_reserved = [
    torch.accelerator.max_memory_reserved(),
    torch.accelerator.max_memory_reserved("cuda:0"),
    max_memory_reserved(ExplodingDeviceToken()),
]
assert max_reserved == [0, 0, 0]
assert all(type(value) is int for value in max_reserved)
reserved = [
    torch.accelerator.memory_reserved(),
    torch.accelerator.memory_reserved("cuda:0"),
    memory_reserved(ExplodingDeviceToken()),
]
assert reserved == [0, 0, 0]
assert all(type(value) is int for value in reserved)
assert set(sys.modules) == modules_before_calls
assert not hasattr(torch, "cuda")
assert not any(
    name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    for name in sys.modules
)
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
