import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


FUNCTION_DOC = """
    A common function to skip guards on the inbuilt nn modules like
    torch.nn.Linear. This is unsafe to use by default. But for majority of
    torch.compile users, the model code does not modify the inbuilt nn module
    attributes. They can benefit from reduction in guard latency overhead using
    this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """

COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


class _TruthProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __bool__(self):
        self.events.append(("bool", self.label))
        return self.result


class _LengthProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __len__(self):
        self.events.append(("len", self.label))
        return self.result


class _Method:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __call__(self):
        self.events.append(("call", self.label))
        return self.result


class _Source:
    def __init__(self, label, result, events):
        self.label = label
        self.method = _Method(label, result, events)
        self.events = events

    @property
    def is_unspecialized_builtin_nn_module(self):
        self.events.append(("method_attribute", self.label))
        return self.method


class _OrigGuard:
    def __init__(self, label, result, events):
        self.label = label
        self._source = _Source(label, result, events)
        self.events = events

    @property
    def source(self):
        self.events.append(("source", self.label))
        return self._source


class _Entry:
    def __init__(self, label, result, events):
        self.label = label
        self._orig_guard = _OrigGuard(label, result, events)
        self.events = events

    @property
    def orig_guard(self):
        self.events.append(("orig_guard", self.label))
        return self._orig_guard


class _CountingIterator:
    def __init__(self, owner):
        self.owner = owner
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.owner.next_calls += 1
        if self.index == len(self.owner.entries):
            raise StopIteration
        entry = self.owner.entries[self.index]
        self.index += 1
        return entry


class _CountingIterable:
    def __init__(self, entries):
        self.entries = entries
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        if self.iter_calls != 1:
            raise AssertionError("iterable was consumed more than once")
        return _CountingIterator(self)

    def __len__(self):
        raise AssertionError("iterable length was inspected")

    def __getitem__(self, index):
        raise AssertionError("iterable was indexed")


class _IterFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __iter__(self):
        self.calls += 1
        raise self.error


class _NextFailure:
    def __init__(self, entry, error):
        self.entry = entry
        self.error = error
        self.calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.calls += 1
        if self.calls == 1:
            return self.entry
        raise self.error


class _AttributeFailure:
    def __init__(self, attribute, error):
        self.attribute = attribute
        self.error = error
        self.calls = 0

    def __getattr__(self, name):
        if name == self.attribute:
            self.calls += 1
            raise self.error
        raise AttributeError(name)


class _CallFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise self.error


class _TruthFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        raise self.error


def _simple_entry(result):
    source = types.SimpleNamespace(is_unspecialized_builtin_nn_module=lambda: result)
    return types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))


def _entry_with_method(method):
    source = types.SimpleNamespace(is_unspecialized_builtin_nn_module=method)
    return types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))


def _raised_outcome(call):
    try:
        call()
    except BaseException as error:
        return type(error).__name__, error.args, str(error)
    return None


def _walk_code_objects(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _walk_code_objects(constant)


class CompilerSkipGuardOnInbuiltNnModulesUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, actual, expected):
        self.assertIs(type(actual), list)
        self.assertEqual(len(actual), len(expected))
        for value, expected_value in zip(actual, expected):
            self.assertIs(value, expected_value)

    def test_nested_access_call_order_and_truth_conversion(self):
        events = []
        labels_and_results = [
            ("true", True),
            ("false", False),
            ("one", 1),
            ("zero", 0),
            ("none", None),
            ("object", object()),
            ("custom-true", _TruthProbe("custom-true", True, events)),
            ("custom-false", _TruthProbe("custom-false", False, events)),
            ("length-true", _LengthProbe("length-true", 2, events)),
            ("length-false", _LengthProbe("length-false", 0, events)),
        ]
        entries = [
            _Entry(label, result, events) for label, result in labels_and_results
        ]

        result = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe(entries)

        self.assert_exact_bool_list(
            result,
            [False, True, False, True, True, False, False, True, False, True],
        )
        expected_events = []
        for label, _ in labels_and_results:
            expected_events.extend(
                [
                    ("orig_guard", label),
                    ("source", label),
                    ("method_attribute", label),
                    ("call", label),
                ]
            )
            if label.startswith("custom"):
                expected_events.append(("bool", label))
            elif label.startswith("length"):
                expected_events.append(("len", label))
        self.assertEqual(events, expected_events)

    def test_arbitrary_iterables_are_consumed_once_into_fresh_lists(self):
        entries = [_simple_entry(True), _simple_entry(False), _simple_entry(None)]
        iterable = _CountingIterable(entries)

        result = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe(iterable)

        self.assert_exact_bool_list(result, [False, True, True])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, 4)

        events = []

        def generator():
            for index, entry in enumerate(entries):
                events.append(("yield", index))
                yield entry
            events.append(("finished", 3))

        values = generator()
        generated = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe(values)
        empty_one = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe([])
        empty_two = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe(())

        self.assert_exact_bool_list(generated, [False, True, True])
        self.assertEqual(
            events,
            [("yield", 0), ("yield", 1), ("yield", 2), ("finished", 3)],
        )
        with self.assertRaises(StopIteration):
            next(values)
        self.assert_exact_bool_list(empty_one, [])
        self.assert_exact_bool_list(empty_two, [])
        self.assertIsNot(empty_one, empty_two)

    def test_iteration_access_call_and_truth_exceptions_propagate(self):
        function = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe

        iter_error = RuntimeError("iter")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as raised:
            function(iter_failure)
        self.assertIs(raised.exception, iter_error)
        self.assertEqual(iter_failure.calls, 1)

        next_error = LookupError("next")
        next_failure = _NextFailure(_simple_entry(False), next_error)
        with self.assertRaises(LookupError) as raised:
            function(next_failure)
        self.assertIs(raised.exception, next_error)
        self.assertEqual(next_failure.calls, 2)

        orig_error = KeyError("orig_guard")
        orig_failure = _AttributeFailure("orig_guard", orig_error)
        untouched = _AttributeFailure("orig_guard", AssertionError("untouched"))
        with self.assertRaises(KeyError) as raised:
            function((orig_failure, untouched))
        self.assertIs(raised.exception, orig_error)
        self.assertEqual(orig_failure.calls, 1)
        self.assertEqual(untouched.calls, 0)

        source_error = OSError("source")
        source_failure = _AttributeFailure("source", source_error)
        with self.assertRaises(OSError) as raised:
            function((types.SimpleNamespace(orig_guard=source_failure),))
        self.assertIs(raised.exception, source_error)
        self.assertEqual(source_failure.calls, 1)

        method_error = ZeroDivisionError("method")
        method_failure = _AttributeFailure(
            "is_unspecialized_builtin_nn_module", method_error
        )
        with self.assertRaises(ZeroDivisionError) as raised:
            function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(source=method_failure)
                    ),
                )
            )
        self.assertIs(raised.exception, method_error)
        self.assertEqual(method_failure.calls, 1)

        call_error = ArithmeticError("call")
        call_failure = _CallFailure(call_error)
        with self.assertRaises(ArithmeticError) as raised:
            function((_entry_with_method(call_failure),))
        self.assertIs(raised.exception, call_error)
        self.assertEqual(call_failure.calls, 1)

        truth_error = ValueError("truth")
        truth_failure = _TruthFailure(truth_error)
        with self.assertRaises(ValueError) as raised:
            function((_simple_entry(truth_failure),))
        self.assertIs(raised.exception, truth_error)
        self.assertEqual(truth_failure.calls, 1)

    def test_signature_documentation_imports_copying_and_pickling(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.skip_guard_on_inbuilt_nn_modules_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(guard_entries)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "skip_guard_on_inbuilt_nn_modules_unsafe")
        self.assertEqual(
            function.__qualname__, "skip_guard_on_inbuilt_nn_modules_unsafe"
        )
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        code_objects = tuple(_walk_code_objects(function.__code__))
        self.assertEqual(
            tuple(name for code in code_objects for name in code.co_names),
            ("orig_guard", "source", "is_unspecialized_builtin_nn_module"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        self.assertIs(namespace["skip_guard_on_inbuilt_nn_modules_unsafe"], function)
        self.assertNotIn("skip_guard_on_inbuilt_nn_modules_unsafe", torch.__all__)
        self.assertFalse(hasattr(torch, "skip_guard_on_inbuilt_nn_modules_unsafe"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

    def test_calls_and_reload_do_not_change_compiler_or_grad_state(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (torch.no_grad(), False),
            ):
                with context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertEqual(
                        compiler.skip_guard_on_inbuilt_nn_modules_unsafe(
                            iter((_simple_entry(False),))
                        ),
                        [True],
                    )
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertIs(compiler.get_default_backend(), backend)
                    self.assertEqual(
                        (
                            compiler.is_compiling(),
                            compiler.is_dynamo_compiling(),
                            compiler.is_exporting(),
                        ),
                        expected_queries,
                    )

            old_function = compiler.skip_guard_on_inbuilt_nn_modules_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_inbuilt_nn_modules_unsafe

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(new_function((_simple_entry(True),)), [False])
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)),
                    new_function,
                )
        finally:
            compiler.set_default_backend(original_backend)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys
import types

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
source = types.SimpleNamespace(
    is_unspecialized_builtin_nn_module=lambda: True
)
entry = types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))
result = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe(iter((entry,)))
assert result == [False]
assert set(sys.modules) == modules_before_call
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


@unittest.skipIf(reference_torch is None, "PyTorch reference package is unavailable")
class CompilerSkipGuardOnInbuiltNnModulesUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.skip_guard_on_inbuilt_nn_modules_unsafe differentials "
                "require pinned PyTorch 2.13.0"
            )

    def behavior_outcome(self, function):
        events = []
        entries = [
            _Entry("true", True, events),
            _Entry("false", False, events),
            _Entry("one", 1, events),
            _Entry("none", None, events),
            _Entry("probe", _TruthProbe("probe", False, events), events),
        ]
        iterable = _CountingIterable(entries)
        result = function(iterable)
        empty_one = function([])
        empty_two = function(())
        return (
            type(result).__name__,
            tuple(value is True for value in result),
            events,
            iterable.iter_calls,
            iterable.next_calls,
            type(empty_one).__name__,
            empty_one,
            empty_two,
            empty_one is empty_two,
        )

    def exception_outcome(self, function):
        cases = (
            lambda: function(_IterFailure(RuntimeError("iter"))),
            lambda: function(_NextFailure(_simple_entry(False), LookupError("next"))),
            lambda: function(
                (_AttributeFailure("orig_guard", KeyError("orig_guard")),)
            ),
            lambda: function(
                (
                    types.SimpleNamespace(
                        orig_guard=_AttributeFailure("source", OSError("source"))
                    ),
                )
            ),
            lambda: function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(
                            source=_AttributeFailure(
                                "is_unspecialized_builtin_nn_module",
                                ZeroDivisionError("method"),
                            )
                        )
                    ),
                )
            ),
            lambda: function(
                (_entry_with_method(_CallFailure(ArithmeticError("call"))),)
            ),
            lambda: function((_simple_entry(_TruthFailure(ValueError("truth"))),)),
            lambda: function(),
            lambda: function([], []),
            lambda: function(entries=[]),
            lambda: function(None),
        )
        return tuple(_raised_outcome(case) for case in cases)

    def test_behavior_and_exceptions_match_pytorch_2_13(self):
        actual = torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
        expected = reference_torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe
        self.assertEqual(self.behavior_outcome(actual), self.behavior_outcome(expected))
        self.assertEqual(
            self.exception_outcome(actual), self.exception_outcome(expected)
        )

    def test_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.skip_guard_on_inbuilt_nn_modules_unsafe
        expected = expected_compiler.skip_guard_on_inbuilt_nn_modules_unsafe

        self.assertIs(type(actual), type(expected))
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            inspect.cleandoc(actual.__doc__), inspect.cleandoc(expected.__doc__)
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)
        self.assertEqual(
            actual_compiler.__all__,
            [name for name in expected_compiler.__all__ if name in COMPILER_EXPORTS],
        )
        self.assertEqual(
            torch.__all__.count("skip_guard_on_inbuilt_nn_modules_unsafe"),
            reference_torch.__all__.count("skip_guard_on_inbuilt_nn_modules_unsafe"),
        )

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

    def state_and_reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            calls = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    grad_before = module.is_grad_enabled()
                    result = compiler.skip_guard_on_inbuilt_nn_modules_unsafe(
                        iter((_simple_entry(False),))
                    )
                    calls.append(
                        (
                            result,
                            grad_before,
                            module.is_grad_enabled(),
                            compiler.get_default_backend() is backend,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
                        )
                    )

            old_function = compiler.skip_guard_on_inbuilt_nn_modules_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_inbuilt_nn_modules_unsafe
            try:
                pickle.dumps(old_function)
            except pickle.PicklingError:
                old_pickles = False
            else:
                old_pickles = True
            return (
                calls,
                reloaded is compiler,
                module.compiler is compiler,
                new_function is old_function,
                compiler.__all__ is old_exports,
                compiler.get_default_backend() is backend,
                new_function((_simple_entry(True),)),
                old_pickles,
                all(
                    pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_state_and_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.state_and_reload_outcome(torch, "torch_rs.compiler"),
            self.state_and_reload_outcome(reference_torch, "torch.compiler"),
        )


if __name__ == "__main__":
    unittest.main()
