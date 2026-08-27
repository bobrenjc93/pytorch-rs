import copy
import importlib
import inspect
import pickle
import sys
import threading
import types
import typing
import unittest
from contextlib import AbstractContextManager

import torch_rs as torch


CONTEXT_DOC = """Context-manager that selects a given stream.

    N.B. This class only exists to facilitate device-agnostic code

    """

STREAM_DOC = """Wrapper around the Context-manager StreamContext that
    selects a given stream.

    N.B. This function only exists to facilitate device-agnostic code
    """


class MarkerError(Exception):
    pass


class CpuStreamContextTests(unittest.TestCase):
    def setUp(self):
        torch.cpu._current_stream = torch.cpu._default_cpu_stream

    def tearDown(self):
        torch.cpu._current_stream = torch.cpu._default_cpu_stream

    def test_selects_exact_objects_and_restores_nested_exceptional_state(self):
        cpu = torch.cpu
        default = cpu.current_stream()
        outer = cpu.Stream()
        inner = object()
        context = cpu.StreamContext(outer)

        self.assertEqual(
            vars(context),
            {"stream": outer, "prev_stream": cpu._default_cpu_stream},
        )
        with context as entered:
            self.assertIsNone(entered)
            self.assertIs(cpu.current_stream(), outer)
            self.assertIs(context.prev_stream, default)

            with cpu.stream(None) as none_entered:
                self.assertIsNone(none_entered)
                self.assertIs(cpu.current_stream(), outer)

            with self.assertRaises(MarkerError):
                with cpu.stream(inner) as inner_entered:
                    self.assertIsNone(inner_entered)
                    self.assertIs(cpu.current_stream(), inner)
                    raise MarkerError("restore through exceptional exit")

            self.assertIs(cpu.current_stream(), outer)

        self.assertIs(cpu.current_stream(), default)

        none_context = cpu.StreamContext(None)
        with none_context as entered:
            self.assertIsNone(entered)
            self.assertIs(cpu.current_stream(), default)
        self.assertEqual(
            vars(none_context),
            {"stream": None, "prev_stream": cpu._default_cpu_stream},
        )

    def test_selection_is_process_global_across_threads(self):
        cpu = torch.cpu
        default = cpu.current_stream()
        selected = cpu.Stream()
        entered = threading.Event()
        release = threading.Event()
        results = []
        errors = []

        def worker():
            try:
                with cpu.stream(selected):
                    results.append(cpu.current_stream())
                    entered.set()
                    if not release.wait(timeout=10):
                        raise TimeoutError("main thread did not release worker")
                results.append(cpu.current_stream())
            except BaseException as error:
                errors.append(error)
                entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=10))
            self.assertEqual(errors, [])
            self.assertIs(cpu.current_stream(), selected)
        finally:
            release.set()
            thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 2)
        self.assertIs(results[0], selected)
        self.assertIs(results[1], default)
        self.assertIs(cpu.current_stream(), default)

    def test_tensor_views_and_autograd_are_unchanged(self):
        cpu = torch.cpu
        selected = cpu.Stream()
        leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        view = leaf.transpose(0, 1)
        before = (
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.data_ptr(),
            view.requires_grad,
            view.is_leaf,
        )

        with cpu.stream(selected):
            self.assertIs(cpu.current_stream(), selected)
            result = (view * 3.0).sum()
            self.assertEqual(
                (
                    view.shape,
                    view.stride(),
                    view.storage_offset(),
                    view.data_ptr(),
                    view.requires_grad,
                    view.is_leaf,
                ),
                before,
            )

        self.assertIs(cpu.current_stream(), cpu._default_cpu_stream)
        self.assertIsNone(leaf.grad)
        result.backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0]])

    def test_metadata_signatures_annotations_and_exports(self):
        cpu = importlib.import_module("torch_rs.cpu")
        context_type = cpu.StreamContext
        function = cpu.stream

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertTrue(issubclass(context_type, AbstractContextManager))
        self.assertEqual(str(inspect.signature(context_type)), "(stream)")
        self.assertEqual(context_type.__name__, "StreamContext")
        self.assertEqual(context_type.__qualname__, "StreamContext")
        self.assertEqual(context_type.__module__, "torch_rs.cpu")
        self.assertIs(inspect.getmodule(context_type), cpu)
        self.assertEqual(
            inspect.cleandoc(context_type.__doc__), inspect.cleandoc(CONTEXT_DOC)
        )
        self.assertEqual(
            context_type.__annotations__, {"cur_stream": cpu.Stream | None}
        )

        method_metadata = {
            "__init__": ("(self, stream)", {}),
            "__enter__": ("(self)", {}),
            "__exit__": (
                "(self, type: Any, value: Any, traceback: Any) -> None",
                {
                    "type": typing.Any,
                    "value": typing.Any,
                    "traceback": typing.Any,
                    "return": None,
                },
            ),
        }
        for name, (signature, annotations) in method_metadata.items():
            method = getattr(context_type, name)
            with self.subTest(method=name):
                self.assertIs(type(method), types.FunctionType)
                self.assertEqual(str(inspect.signature(method)), signature)
                self.assertEqual(method.__annotations__, annotations)
                self.assertEqual(method.__name__, name)
                self.assertEqual(method.__qualname__, f"StreamContext.{name}")
                self.assertEqual(method.__module__, "torch_rs.cpu")
                self.assertIsNone(method.__doc__)
                self.assertEqual(method.__dict__, {})
                self.assertIsNone(method.__defaults__)
                self.assertIsNone(method.__kwdefaults__)
                self.assertFalse(hasattr(method, "__text_signature__"))

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(stream: torch_rs.cpu.Stream) -> contextlib.AbstractContextManager",
        )
        self.assertEqual(
            function.__annotations__,
            {"stream": cpu.Stream, "return": AbstractContextManager},
        )
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(function.__name__, "stream")
        self.assertEqual(function.__qualname__, "stream")
        self.assertEqual(function.__module__, "torch_rs.cpu")
        self.assertIs(inspect.getmodule(function), cpu)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(STREAM_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(
            cpu.__all__,
            [
                "is_available",
                "is_initialized",
                "synchronize",
                "current_device",
                "current_stream",
                "stream",
                "device_count",
                "Stream",
                "StreamContext",
                "Event",
            ],
        )
        direct = {}
        exec("from torch_rs.cpu import StreamContext, stream", direct)
        self.assertIs(direct["StreamContext"], context_type)
        self.assertIs(direct["stream"], function)
        namespace = {}
        exec("from torch_rs.cpu import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(cpu.__all__),
        )
        self.assertIs(namespace["StreamContext"], context_type)
        self.assertIs(namespace["stream"], function)
        self.assertNotIn("StreamContext", torch.__all__)
        self.assertNotIn("stream", torch.__all__)
        self.assertFalse(hasattr(torch, "StreamContext"))
        self.assertFalse(hasattr(torch, "stream"))

    def test_argument_errors_match_pytorch_2_13(self):
        context_type = torch.cpu.StreamContext
        function = torch.cpu.stream
        cases = (
            (
                lambda: context_type(),
                "StreamContext.__init__() missing 1 required positional argument: "
                "'stream'",
            ),
            (
                lambda: context_type(None, None),
                "StreamContext.__init__() takes 2 positional arguments but 3 "
                "were given",
            ),
            (
                lambda: context_type(None, stream=None),
                "StreamContext.__init__() got multiple values for argument 'stream'",
            ),
            (
                lambda: context_type(unexpected=None),
                "StreamContext.__init__() got an unexpected keyword argument "
                "'unexpected'",
            ),
            (
                lambda: function(),
                "stream() missing 1 required positional argument: 'stream'",
            ),
            (
                lambda: function(None, None),
                "stream() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(None, stream=None),
                "stream() got multiple values for argument 'stream'",
            ),
            (
                lambda: function(unexpected=None),
                "stream() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_copy_and_pickle_use_python_object_semantics(self):
        cpu = torch.cpu
        context_type = cpu.StreamContext
        function = cpu.stream
        selected = cpu.Stream()
        selected.payload = [1, 2, 3]
        context = context_type(selected)

        for value in (
            context_type,
            function,
            context_type.__init__,
            context_type.__enter__,
            context_type.__exit__,
        ):
            with self.subTest(value=getattr(value, "__qualname__", repr(value))):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)

        shallow = copy.copy(context)
        deep = copy.deepcopy(context)
        self.assertIs(type(shallow), context_type)
        self.assertIsNot(shallow, context)
        self.assertIs(shallow.stream, selected)
        self.assertIs(shallow.prev_stream, cpu._default_cpu_stream)
        self.assertIs(type(deep), context_type)
        self.assertIsNot(deep, context)
        self.assertIsNot(deep.stream, selected)
        self.assertEqual(deep.stream.payload, selected.payload)
        self.assertIsNot(deep.stream.payload, selected.payload)
        self.assertIsNot(deep.prev_stream, cpu._default_cpu_stream)

        canonical_values = (
            context_type,
            function,
            context_type.__init__,
            context_type.__enter__,
            context_type.__exit__,
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for value in canonical_values:
                with self.subTest(protocol=protocol, value=value.__qualname__):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.cpu", payload)
                    self.assertIs(pickle.loads(payload), value)

            with self.subTest(protocol=protocol, value="context"):
                payload = pickle.dumps(context, protocol=protocol)
                self.assertIn(b"torch_rs.cpu", payload)
                restored = pickle.loads(payload)
                self.assertIs(type(restored), context_type)
                self.assertIsNot(restored, context)
                self.assertEqual(restored.stream.payload, selected.payload)
                self.assertIsNot(restored.stream, selected)
                self.assertIsNot(restored.prev_stream, cpu._default_cpu_stream)

    def test_reload_replaces_canonical_objects_and_preserves_old_context_use(self):
        cpu = torch.cpu
        old_context_type = cpu.StreamContext
        old_function = cpu.stream
        old_methods = tuple(
            getattr(old_context_type, name)
            for name in ("__init__", "__enter__", "__exit__")
        )
        old_selected = cpu.Stream()
        old_context = old_context_type(old_selected)
        type_payload = pickle.dumps(old_context_type)
        function_payload = pickle.dumps(old_function)
        method_payloads = tuple(pickle.dumps(method) for method in old_methods)
        context_payload = pickle.dumps(old_context)

        self.assertIs(importlib.reload(cpu), cpu)
        new_context_type = cpu.StreamContext
        new_function = cpu.stream
        new_methods = tuple(
            getattr(new_context_type, name)
            for name in ("__init__", "__enter__", "__exit__")
        )
        new_default = cpu.current_stream()

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIsNot(new_context_type, old_context_type)
        self.assertIsNot(new_function, old_function)
        for old_method, new_method in zip(old_methods, new_methods, strict=True):
            self.assertIsNot(old_method, new_method)

        self.assertIs(type(old_function(old_selected)), new_context_type)
        with old_context:
            self.assertIs(cpu.current_stream(), old_selected)
        self.assertIs(cpu.current_stream(), new_default)

        self.assertIs(pickle.loads(type_payload), new_context_type)
        self.assertIs(pickle.loads(function_payload), new_function)
        for payload, new_method in zip(method_payloads, new_methods, strict=True):
            self.assertIs(pickle.loads(payload), new_method)

        restored = pickle.loads(context_payload)
        self.assertIs(type(restored), new_context_type)
        self.assertIs(type(restored.stream), cpu.Stream)
        self.assertIs(type(restored.prev_stream), cpu.Stream)
        self.assertIsNot(restored.stream, old_selected)
        self.assertIsNot(restored.prev_stream, new_default)

        for value in (
            old_context_type,
            old_function,
            *old_methods,
            old_context,
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(value)


if __name__ == "__main__":
    unittest.main()
