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


FUNCTION_DOC = """
    A common function to skip guards on all nn modules, both user defined as
    well inbuilt nn modules (like torch.nn.Linear). This is unsafe to use by
    default. But for majority of torch.compile users, the model code does not
    modify the nn module attributes. They can benefit from reduction in guard
    latency overhead using this API.

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


class _LengthTruthProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __len__(self):
        self.events.append(("len", self.label))
        return self.result


class _CallProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __call__(self, *args, **kwargs):
        self.events.append(("call", self.label, args, kwargs))
        return self.result


class _Source:
    def __init__(self, label, result, events):
        self.label = label
        self.callable = _CallProbe(label, result, events)
        self.events = events

    @property
    def is_unspecialized_nn_module(self):
        self.events.append(("method", self.label))
        return self.callable


class _OrigGuard:
    def __init__(self, label, source, events):
        self.label = label
        self._source = source
        self.events = events

    @property
    def source(self):
        self.events.append(("source", self.label))
        return self._source


class _GuardEntry:
    def __init__(self, label, result, events):
        self.label = label
        self._orig_guard = _OrigGuard(label, _Source(label, result, events), events)
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
    def __init__(self, first_entry, error):
        self.first_entry = first_entry
        self.error = error
        self.iter_calls = 0
        self.next_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return self

    def __next__(self):
        self.next_calls += 1
        if self.next_calls == 1:
            return self.first_entry
        raise self.error


class _OrigGuardFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    @property
    def orig_guard(self):
        self.calls += 1
        raise self.error


class _SourceFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    @property
    def source(self):
        self.calls += 1
        raise self.error


class _MethodFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    @property
    def is_unspecialized_nn_module(self):
        self.calls += 1
        raise self.error


class _CallFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def is_unspecialized_nn_module(self):
        self.calls += 1
        raise self.error


class _TruthFailure:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        raise self.error


class _InvalidTruth:
    def __bool__(self):
        return 1


class _MissingOrigGuard:
    pass


class _MissingSource:
    pass


class _MissingMethod:
    pass


class _InvalidIterator:
    def __iter__(self):
        return []


def _entry_with_source(source):
    return types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))


def _walk_code_objects(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _walk_code_objects(constant)


class CompilerSkipGuardOnAllNnModulesUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, result, expected):
        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(expected))
        for actual, expected_value in zip(result, expected):
            self.assertIs(actual, expected_value)

    def test_exact_booleans_follow_nested_access_call_and_truth_order(self):
        events = []
        entries = [
            _GuardEntry("true", True, events),
            _GuardEntry("false", False, events),
            _GuardEntry("one", 1, events),
            _GuardEntry("zero", 0, events),
            _GuardEntry("none", None, events),
            _GuardEntry("object", object(), events),
            _GuardEntry(
                "custom-true", _TruthProbe("custom-true", True, events), events
            ),
            _GuardEntry(
                "custom-false", _TruthProbe("custom-false", False, events), events
            ),
            _GuardEntry(
                "length-true", _LengthTruthProbe("length-true", 2, events), events
            ),
            _GuardEntry(
                "length-false", _LengthTruthProbe("length-false", 0, events), events
            ),
        ]
        original_entries = tuple(entries)

        result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(entries)

        self.assert_exact_bool_list(
            result,
            [False, True, False, True, True, False, False, True, False, True],
        )
        self.assertIsNot(result, entries)
        self.assertTrue(
            all(
                actual is original
                for actual, original in zip(entries, original_entries)
            )
        )
        expected_events = []
        for label in (
            "true",
            "false",
            "one",
            "zero",
            "none",
            "object",
            "custom-true",
            "custom-false",
            "length-true",
            "length-false",
        ):
            expected_events.extend(
                [
                    ("orig_guard", label),
                    ("source", label),
                    ("method", label),
                    ("call", label, (), {}),
                ]
            )
            if label.startswith("custom-"):
                expected_events.append(("bool", label))
            elif label.startswith("length-"):
                expected_events.append(("len", label))
        self.assertEqual(events, expected_events)

    def test_arbitrary_iterables_are_consumed_once_into_fresh_lists(self):
        entries = [
            _entry_with_source(
                types.SimpleNamespace(is_unspecialized_nn_module=lambda: True)
            ),
            _entry_with_source(
                types.SimpleNamespace(is_unspecialized_nn_module=lambda: False)
            ),
        ]
        first = torch.compiler.skip_guard_on_all_nn_modules_unsafe(entries)
        second = torch.compiler.skip_guard_on_all_nn_modules_unsafe(tuple(entries))
        first_empty = torch.compiler.skip_guard_on_all_nn_modules_unsafe([])
        second_empty = torch.compiler.skip_guard_on_all_nn_modules_unsafe(())
        iterable = _CountingIterable(entries)
        iterable_result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(iterable)

        yielded = []

        def generated_entries():
            for index, entry in enumerate(entries):
                yielded.append(index)
                yield entry
            yielded.append("finished")

        generator = generated_entries()
        generator_result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(generator)

        for result in (first, second, iterable_result, generator_result):
            self.assert_exact_bool_list(result, [False, True])
        self.assert_exact_bool_list(first_empty, [])
        self.assert_exact_bool_list(second_empty, [])
        self.assertIsNot(first, second)
        self.assertIsNot(first_empty, second_empty)
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, len(entries) + 1)
        self.assertEqual(yielded, [0, 1, "finished"])
        with self.assertRaises(StopIteration):
            next(generator)

    def test_iteration_access_call_and_truth_exceptions_propagate(self):
        function = torch.compiler.skip_guard_on_all_nn_modules_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as raised:
            function(iter_failure)
        self.assertIs(raised.exception, iter_error)
        self.assertEqual(iter_failure.calls, 1)

        first_events = []
        next_error = LookupError("next failure")
        next_failure = _NextFailure(
            _GuardEntry("first", False, first_events), next_error
        )
        with self.assertRaises(LookupError) as raised:
            function(next_failure)
        self.assertIs(raised.exception, next_error)
        self.assertGreaterEqual(next_failure.iter_calls, 1)
        self.assertEqual(next_failure.next_calls, 2)
        self.assertEqual(
            first_events,
            [
                ("orig_guard", "first"),
                ("source", "first"),
                ("method", "first"),
                ("call", "first", (), {}),
            ],
        )

        access_cases = []
        orig_error = KeyError("orig_guard failure")
        orig_failure = _OrigGuardFailure(orig_error)
        access_cases.append((orig_failure, orig_error, orig_failure))

        source_error = IndexError("source failure")
        source_failure = _SourceFailure(source_error)
        access_cases.append(
            (
                types.SimpleNamespace(orig_guard=source_failure),
                source_error,
                source_failure,
            )
        )

        method_error = OSError("method failure")
        method_failure = _MethodFailure(method_error)
        access_cases.append(
            (_entry_with_source(method_failure), method_error, method_failure)
        )

        call_error = ArithmeticError("call failure")
        call_failure = _CallFailure(call_error)
        access_cases.append(
            (_entry_with_source(call_failure), call_error, call_failure)
        )

        for entry, error, probe in access_cases:
            untouched = _OrigGuardFailure(AssertionError("should not be reached"))
            with self.subTest(error=error):
                with self.assertRaises(type(error)) as raised:
                    function((entry, untouched))
                self.assertIs(raised.exception, error)
                self.assertEqual(probe.calls, 1)
                self.assertEqual(untouched.calls, 0)

        truth_error = ValueError("truth failure")
        truth_failure = _TruthFailure(truth_error)
        with self.assertRaises(ValueError) as raised:
            function(
                (
                    _entry_with_source(
                        types.SimpleNamespace(
                            is_unspecialized_nn_module=lambda: truth_failure
                        )
                    ),
                )
            )
        self.assertIs(raised.exception, truth_error)
        self.assertEqual(truth_failure.calls, 1)

        with self.assertRaises(TypeError) as raised:
            function(_InvalidIterator())
        self.assertEqual(
            str(raised.exception), "iter() returned non-iterator of type 'list'"
        )

    def test_missing_noncallable_and_invalid_truth_errors_match_pytorch(self):
        function = torch.compiler.skip_guard_on_all_nn_modules_unsafe
        cases = (
            (
                _MissingOrigGuard(),
                AttributeError,
                "'_MissingOrigGuard' object has no attribute 'orig_guard'",
            ),
            (
                types.SimpleNamespace(orig_guard=_MissingSource()),
                AttributeError,
                "'_MissingSource' object has no attribute 'source'",
            ),
            (
                _entry_with_source(_MissingMethod()),
                AttributeError,
                "'_MissingMethod' object has no attribute "
                "'is_unspecialized_nn_module'",
            ),
            (
                _entry_with_source(
                    types.SimpleNamespace(is_unspecialized_nn_module=None)
                ),
                TypeError,
                "'NoneType' object is not callable",
            ),
            (
                _entry_with_source(
                    types.SimpleNamespace(
                        is_unspecialized_nn_module=lambda: _InvalidTruth()
                    )
                ),
                TypeError,
                "__bool__ should return bool, returned int",
            ),
        )
        for entry, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    function((entry,))
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.skip_guard_on_all_nn_modules_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(guard_entries)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "skip_guard_on_all_nn_modules_unsafe")
        self.assertEqual(function.__qualname__, "skip_guard_on_all_nn_modules_unsafe")
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
            ("orig_guard", "source", "is_unspecialized_nn_module"),
        )
        self.assertEqual(function.__code__.co_varnames[0], "guard_entries")
        self.assertIn(
            "entry", {name for code in code_objects for name in code.co_varnames}
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.skip_guard_on_all_nn_modules_unsafe

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("skip_guard_on_all_nn_modules_unsafe", torch.__all__)
        self.assertFalse(hasattr(torch, "skip_guard_on_all_nn_modules_unsafe"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("skip_guard_on_all_nn_modules_unsafe", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.skip_guard_on_all_nn_modules_unsafe
        cases = (
            (
                lambda: function(),
                "skip_guard_on_all_nn_modules_unsafe() missing 1 required "
                "positional argument: 'guard_entries'",
            ),
            (
                lambda: function([], []),
                "skip_guard_on_all_nn_modules_unsafe() takes 1 positional "
                "argument but 2 were given",
            ),
            (
                lambda: function([], guard_entries=[]),
                "skip_guard_on_all_nn_modules_unsafe() got multiple values for "
                "argument 'guard_entries'",
            ),
            (
                lambda: function(entries=[]),
                "skip_guard_on_all_nn_modules_unsafe() got an unexpected keyword "
                "argument 'entries'",
            ),
            (lambda: function(None), "'NoneType' object is not iterable"),
            (lambda: function(1), "'int' object is not iterable"),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_calls_and_reload_preserve_compiler_and_grad_state(self):
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
            entry = _entry_with_source(
                types.SimpleNamespace(is_unspecialized_nn_module=lambda: False)
            )

            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (torch.no_grad(), False),
            ):
                with context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertEqual(
                        compiler.skip_guard_on_all_nn_modules_unsafe(iter((entry,))),
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

            old_function = compiler.skip_guard_on_all_nn_modules_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_all_nn_modules_unsafe

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(new_function((entry,)), [True])
            self.assertIs(copy.copy(old_function), old_function)
            self.assertIs(copy.deepcopy(old_function), old_function)
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
guard_entries = (
    types.SimpleNamespace(orig_guard=types.SimpleNamespace(
        source=types.SimpleNamespace(is_unspecialized_nn_module=lambda: True)
    )),
    types.SimpleNamespace(orig_guard=types.SimpleNamespace(
        source=types.SimpleNamespace(is_unspecialized_nn_module=lambda: False)
    )),
)
result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(iter(guard_entries))
assert result == [False, True]
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


if __name__ == "__main__":
    unittest.main()
