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


class MarkerError(Exception):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuStreamContextReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cpu stream-context differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        torch.cpu._current_stream = torch.cpu._default_cpu_stream
        reference_torch.cpu._current_stream = reference_torch.cpu._default_cpu_stream

    def tearDown(self):
        torch.cpu._current_stream = torch.cpu._default_cpu_stream
        reference_torch.cpu._current_stream = reference_torch.cpu._default_cpu_stream

    def normalized(self, value):
        return str(value).replace("torch_rs", "torch")

    def normalized_error(self, value):
        return re.sub(r"0x[0-9a-fA-F]+", "<address>", self.normalized(value))

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def selection_outcome(self, module):
        cpu = module.cpu
        cpu._current_stream = cpu._default_cpu_stream
        default = cpu.current_stream()
        outer = cpu.Stream()
        inner = object()
        context = cpu.StreamContext(outer)
        states = [
            (
                context.stream is outer,
                context.prev_stream is cpu._default_cpu_stream,
                tuple(vars(context)),
            )
        ]

        with context as entered:
            states.append(
                (
                    entered is None,
                    cpu.current_stream() is outer,
                    context.prev_stream is default,
                )
            )
            with cpu.stream(None) as none_entered:
                states.append(
                    (none_entered is None, cpu.current_stream() is outer)
                )
            try:
                with cpu.stream(inner) as inner_entered:
                    states.append(
                        (inner_entered is None, cpu.current_stream() is inner)
                    )
                    raise MarkerError("restore")
            except MarkerError:
                states.append(cpu.current_stream() is outer)

        states.append(cpu.current_stream() is default)
        none_context = cpu.StreamContext(None)
        with none_context as entered:
            states.append(
                (
                    entered is None,
                    cpu.current_stream() is default,
                    none_context.prev_stream is cpu._default_cpu_stream,
                )
            )
        states.append(cpu.current_stream() is default)
        return states

    def threaded_outcome(self, module):
        cpu = module.cpu
        cpu._current_stream = cpu._default_cpu_stream
        default = cpu.current_stream()
        selected = cpu.Stream()
        entered = threading.Event()
        release = threading.Event()
        results = []
        errors = []

        def worker():
            try:
                with cpu.stream(selected):
                    results.append(cpu.current_stream() is selected)
                    entered.set()
                    if not release.wait(timeout=10):
                        raise TimeoutError("main thread did not release worker")
                results.append(cpu.current_stream() is default)
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            entered_in_main = entered.wait(timeout=10)
            globally_selected = cpu.current_stream() is selected
        finally:
            release.set()
            thread.join(timeout=10)

        return (
            entered_in_main,
            globally_selected,
            thread.is_alive(),
            results,
            errors,
            cpu.current_stream() is default,
        )

    def tensor_outcome(self, module):
        cpu = module.cpu
        selected = cpu.Stream()
        leaf = module.tensor([[1.0, 2.0]], requires_grad=True)
        view = leaf.transpose(0, 1)
        before = (
            tuple(view.shape),
            view.stride(),
            view.storage_offset(),
            view.data_ptr(),
            view.requires_grad,
            view.is_leaf,
        )
        with cpu.stream(selected):
            selected_exactly = cpu.current_stream() is selected
            result = (view * 3.0).sum()
            unchanged = before == (
                tuple(view.shape),
                view.stride(),
                view.storage_offset(),
                view.data_ptr(),
                view.requires_grad,
                view.is_leaf,
            )
        restored = cpu.current_stream() is cpu._default_cpu_stream
        grad_was_none = leaf.grad is None
        result.backward()
        return (
            selected_exactly,
            unchanged,
            restored,
            grad_was_none,
            leaf.grad.tolist(),
        )

    def test_selection_nesting_none_exceptions_and_process_global_state_match(self):
        actual = self.selection_outcome(torch)
        expected = self.selection_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            [
                (True, True, ("stream", "prev_stream")),
                (True, True, True),
                (True, True),
                (True, True),
                True,
                True,
                (True, True, True),
                True,
            ],
        )

        actual_threaded = self.threaded_outcome(torch)
        expected_threaded = self.threaded_outcome(reference_torch)
        self.assertEqual(actual_threaded, expected_threaded)
        self.assertEqual(
            actual_threaded,
            (True, True, False, [True, True], [], True),
        )

    def test_tensor_view_and_autograd_behavior_matches_pytorch_2_13(self):
        actual = self.tensor_outcome(torch)
        expected = self.tensor_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, (True, True, True, True, [[3.0, 3.0]]))

    def test_metadata_signatures_annotations_and_exports_match(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual_type = actual_cpu.StreamContext
        expected_type = expected_cpu.StreamContext
        actual_function = actual_cpu.stream
        expected_function = expected_cpu.stream

        self.assertEqual(
            self.normalized(inspect.signature(actual_type)),
            self.normalized(inspect.signature(expected_type)),
        )
        self.assertEqual(
            self.normalized(actual_type.__bases__),
            self.normalized(expected_type.__bases__),
        )
        self.assertEqual(set(actual_type.__dict__), set(expected_type.__dict__))
        self.assertEqual(
            self.normalized(actual_type.__annotations__),
            self.normalized(expected_type.__annotations__),
        )
        self.assertEqual(actual_type.__name__, expected_type.__name__)
        self.assertEqual(actual_type.__qualname__, expected_type.__qualname__)
        self.assertEqual(
            actual_type.__module__.replace("torch_rs", "torch"),
            expected_type.__module__,
        )
        self.assertEqual(actual_type.__doc__, expected_type.__doc__)

        for name in ("__init__", "__enter__", "__exit__"):
            actual = getattr(actual_type, name)
            expected = getattr(expected_type, name)
            with self.subTest(method=name):
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    self.normalized(inspect.signature(actual)),
                    self.normalized(inspect.signature(expected)),
                )
                self.assertEqual(
                    self.normalized(actual.__annotations__),
                    self.normalized(expected.__annotations__),
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
                self.assertEqual(
                    actual.__code__.co_freevars, expected.__code__.co_freevars
                )
                self.assertEqual(
                    actual.__code__.co_cellvars, expected.__code__.co_cellvars
                )

        self.assertIs(type(actual_function), types.FunctionType)
        self.assertIs(type(expected_function), types.FunctionType)
        self.assertEqual(
            self.normalized(inspect.signature(actual_function)),
            self.normalized(inspect.signature(expected_function)),
        )
        self.assertEqual(
            self.normalized(actual_function.__annotations__),
            self.normalized(expected_function.__annotations__),
        )
        self.assertEqual(
            self.normalized(typing.get_type_hints(actual_function)),
            self.normalized(typing.get_type_hints(expected_function)),
        )
        self.assertEqual(actual_function.__name__, expected_function.__name__)
        self.assertEqual(actual_function.__qualname__, expected_function.__qualname__)
        self.assertEqual(
            actual_function.__module__.replace("torch_rs", "torch"),
            expected_function.__module__,
        )
        self.assertEqual(actual_function.__doc__, expected_function.__doc__)
        self.assertEqual(actual_function.__defaults__, expected_function.__defaults__)
        self.assertEqual(
            actual_function.__kwdefaults__, expected_function.__kwdefaults__
        )
        self.assertEqual(actual_function.__dict__, expected_function.__dict__)
        self.assertEqual(
            actual_function.__code__.co_names, expected_function.__code__.co_names
        )

        supported = {
            "current_device",
            "current_stream",
            "stream",
            "set_device",
            "device_count",
            "Event",
            "is_available",
            "is_initialized",
            "Stream",
            "StreamContext",
            "synchronize",
        }
        self.assertEqual(
            actual_cpu.__all__,
            [name for name in expected_cpu.__all__ if name in supported],
        )
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.cpu import *", actual_namespace)
        exec("from torch.cpu import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            supported,
        )
        for name in supported:
            with self.subTest(export=name):
                self.assertIs(actual_namespace[name], getattr(actual_cpu, name))
                self.assertIs(expected_namespace[name], getattr(expected_cpu, name))

    def test_argument_errors_match_pytorch_2_13(self):
        actual_type = torch.cpu.StreamContext
        expected_type = reference_torch.cpu.StreamContext
        actual_function = torch.cpu.stream
        expected_function = reference_torch.cpu.stream
        cases = (
            (lambda: actual_type(), lambda: expected_type()),
            (
                lambda: actual_type(None, None),
                lambda: expected_type(None, None),
            ),
            (
                lambda: actual_type(None, stream=None),
                lambda: expected_type(None, stream=None),
            ),
            (
                lambda: actual_type(unexpected=None),
                lambda: expected_type(unexpected=None),
            ),
            (lambda: actual_function(), lambda: expected_function()),
            (
                lambda: actual_function(None, None),
                lambda: expected_function(None, None),
            ),
            (
                lambda: actual_function(None, stream=None),
                lambda: expected_function(None, stream=None),
            ),
            (
                lambda: actual_function(unexpected=None),
                lambda: expected_function(unexpected=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def copy_outcome(self, module):
        cpu = module.cpu
        selected = cpu.Stream()
        selected.payload = [1, 2, 3]
        context = cpu.StreamContext(selected)
        shallow = copy.copy(context)
        deep = copy.deepcopy(context)
        return (
            type(shallow) is cpu.StreamContext,
            shallow is context,
            shallow.stream is selected,
            shallow.prev_stream is cpu._default_cpu_stream,
            type(deep) is cpu.StreamContext,
            deep is context,
            deep.stream is selected,
            deep.stream.payload,
            deep.stream.payload is selected.payload,
            deep.prev_stream is cpu._default_cpu_stream,
        )

    def test_copy_and_pickle_behavior_matches_pytorch_2_13(self):
        actual_copy = self.copy_outcome(torch)
        expected_copy = self.copy_outcome(reference_torch)
        self.assertEqual(actual_copy, expected_copy)
        self.assertEqual(
            actual_copy,
            (True, False, True, True, True, False, False, [1, 2, 3], False, False),
        )

        actual_type = torch.cpu.StreamContext
        expected_type = reference_torch.cpu.StreamContext
        actual_selected = torch.cpu.Stream()
        expected_selected = reference_torch.cpu.Stream()
        actual_selected.payload = [1, 2, 3]
        expected_selected.payload = [1, 2, 3]
        actual_context = actual_type(actual_selected)
        expected_context = expected_type(expected_selected)
        actual_objects = (
            actual_type,
            torch.cpu.stream,
            actual_type.__init__,
            actual_type.__enter__,
            actual_type.__exit__,
            actual_context,
        )
        expected_objects = (
            expected_type,
            reference_torch.cpu.stream,
            expected_type.__init__,
            expected_type.__enter__,
            expected_type.__exit__,
            expected_context,
        )
        for actual_value, expected_value in zip(
            actual_objects[:-1], expected_objects[:-1], strict=True
        ):
            self.assertIs(copy.copy(actual_value), actual_value)
            self.assertIs(copy.copy(expected_value), expected_value)
            self.assertIs(copy.deepcopy(actual_value), actual_value)
            self.assertIs(copy.deepcopy(expected_value), expected_value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for index, (actual_value, expected_value) in enumerate(
                zip(actual_objects, expected_objects, strict=True)
            ):
                with self.subTest(protocol=protocol, value=index):
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )
                    actual_restored = pickle.loads(
                        pickle.dumps(actual_value, protocol=protocol)
                    )
                    expected_restored = pickle.loads(
                        pickle.dumps(expected_value, protocol=protocol)
                    )
                    if index < 5:
                        self.assertIs(actual_restored, actual_value)
                        self.assertIs(expected_restored, expected_value)
                    else:
                        self.assertIs(type(actual_restored), actual_type)
                        self.assertIs(type(expected_restored), expected_type)
                        self.assertEqual(
                            tuple(vars(actual_restored)),
                            tuple(vars(expected_restored)),
                        )
                        self.assertEqual(
                            actual_restored.stream.payload,
                            expected_restored.stream.payload,
                        )
                        self.assertIsNot(actual_restored.stream, actual_selected)
                        self.assertIsNot(expected_restored.stream, expected_selected)
                        self.assertIsNot(
                            actual_restored.prev_stream,
                            torch.cpu._default_cpu_stream,
                        )
                        self.assertIsNot(
                            expected_restored.prev_stream,
                            reference_torch.cpu._default_cpu_stream,
                        )

    def reload_outcome(self, module):
        cpu = module.cpu
        old_type = cpu.StreamContext
        old_function = cpu.stream
        old_methods = tuple(
            getattr(old_type, name) for name in ("__init__", "__enter__", "__exit__")
        )
        old_selected = cpu.Stream()
        old_context = old_type(old_selected)
        type_payload = pickle.dumps(old_type)
        function_payload = pickle.dumps(old_function)
        method_payloads = tuple(pickle.dumps(method) for method in old_methods)
        context_payload = pickle.dumps(old_context)

        reloaded = importlib.reload(cpu)
        new_type = reloaded.StreamContext
        new_function = reloaded.stream
        new_methods = tuple(
            getattr(new_type, name) for name in ("__init__", "__enter__", "__exit__")
        )
        new_default = reloaded.current_stream()
        old_wrapper_result = old_function(old_selected)
        with old_context as entered:
            old_context_state = (
                entered is None,
                reloaded.current_stream() is old_selected,
            )
        restored_context = pickle.loads(context_payload)

        outcome = (
            reloaded is cpu,
            module.cpu is cpu,
            sys.modules[cpu.__name__] is cpu,
            new_type is old_type,
            new_function is old_function,
            tuple(
                new is old
                for new, old in zip(new_methods, old_methods, strict=True)
            ),
            type(old_wrapper_result) is new_type,
            old_context_state,
            reloaded.current_stream() is new_default,
            pickle.loads(type_payload) is new_type,
            pickle.loads(function_payload) is new_function,
            tuple(
                pickle.loads(payload) is method
                for payload, method in zip(method_payloads, new_methods, strict=True)
            ),
            type(restored_context) is new_type,
            type(restored_context.stream) is reloaded.Stream,
            type(restored_context.prev_stream) is reloaded.Stream,
            restored_context.stream is old_selected,
            restored_context.prev_stream is new_default,
        )

        errors = []
        for value in (old_type, old_function, *old_methods, old_context):
            try:
                pickle.dumps(value)
            except BaseException as error:
                errors.append(
                    (
                        type(error).__name__,
                        self.normalized_error(error),
                        tuple(
                            self.normalized_error(item)
                            if isinstance(item, str)
                            else item
                            for item in error.args
                        ),
                    )
                )
            else:
                errors.append(None)
        return outcome, errors

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_outcome(torch)
        expected = self.reload_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual[0],
            (
                True,
                True,
                True,
                False,
                False,
                (False, False, False),
                True,
                (True, True),
                True,
                True,
                True,
                (True, True, True),
                True,
                True,
                True,
                False,
                False,
            ),
        )
        self.assertTrue(all(error[0] == "PicklingError" for error in actual[1]))


if __name__ == "__main__":
    unittest.main()
