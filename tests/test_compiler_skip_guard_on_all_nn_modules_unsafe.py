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
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "cudagraph_mark_step_begin",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
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


class _MethodProbe:
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
        self.method = _MethodProbe(label, result, events)
        self.events = events

    @property
    def is_unspecialized_nn_module(self):
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


class _GuardEntry:
    def __init__(self, label, result, events):
        self.label = label
        self._orig_guard = _OrigGuard(label, result, events)
        self.events = events

    @property
    def orig_guard(self):
        self.events.append(("orig_guard", self.label))
        return self._orig_guard

    def _fail(self, operation):
        raise AssertionError(f"guard entry was inspected through {operation}")

    def __bool__(self):
        self._fail("bool")

    def __eq__(self, other):
        self._fail("equality")

    def __hash__(self):
        self._fail("hash")

    def __iter__(self):
        self._fail("iteration")

    def __len__(self):
        self._fail("length")

    def __repr__(self):
        self._fail("repr")

    def __str__(self):
        self._fail("str")


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
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
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


class _OrigGuardFailureEntry:
    def __init__(self, error):
        self.error = error
        self.attribute_calls = 0

    @property
    def orig_guard(self):
        self.attribute_calls += 1
        raise self.error


class _SourceFailureOrigGuard:
    def __init__(self, error):
        self.error = error
        self.attribute_calls = 0

    @property
    def source(self):
        self.attribute_calls += 1
        raise self.error


class _MethodFailureSource:
    def __init__(self, error):
        self.error = error
        self.attribute_calls = 0

    @property
    def is_unspecialized_nn_module(self):
        self.attribute_calls += 1
        raise self.error


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
        self.bool_calls = 0

    def __bool__(self):
        self.bool_calls += 1
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


def _walk_code_objects(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _walk_code_objects(constant)


def _simple_entry(result):
    source = types.SimpleNamespace(is_unspecialized_nn_module=lambda: result)
    return types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))


class CompilerSkipGuardOnAllNnModulesUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, result, expected):
        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(expected))
        for actual, expected_value in zip(result, expected):
            self.assertIs(actual, expected_value)

    def test_exact_booleans_follow_the_full_attribute_and_call_chain(self):
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
        original_entries = entries.copy()

        result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(entries)

        self.assert_exact_bool_list(
            result,
            [False, True, False, True, True, False, False, True, False, True],
        )
        self.assertIsNot(result, entries)
        for actual, original in zip(entries, original_entries):
            self.assertIs(actual, original)

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
                    ("method_attribute", label),
                    ("call", label),
                ]
            )
            if label.startswith("custom"):
                expected_events.append(("bool", label))
            elif label.startswith("length"):
                expected_events.append(("len", label))
        self.assertEqual(events, expected_events)

    def test_arbitrary_iterables_are_consumed_once_and_return_fresh_lists(self):
        events = []
        entries = [
            _GuardEntry("first", True, events),
            _GuardEntry("second", False, events),
            _GuardEntry("third", None, events),
        ]
        iterable = _CountingIterable(entries)

        result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(iterable)

        self.assert_exact_bool_list(result, [False, True, True])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, len(entries) + 1)

        generator_events = []

        def guard_entries():
            for index, entry in enumerate(entries):
                generator_events.append(("yield", index))
                yield entry
            generator_events.append(("finished", len(entries)))

        generator = guard_entries()
        generated = torch.compiler.skip_guard_on_all_nn_modules_unsafe(generator)
        first_empty = torch.compiler.skip_guard_on_all_nn_modules_unsafe([])
        second_empty = torch.compiler.skip_guard_on_all_nn_modules_unsafe(())

        self.assert_exact_bool_list(generated, [False, True, True])
        self.assertEqual(
            generator_events,
            [("yield", 0), ("yield", 1), ("yield", 2), ("finished", 3)],
        )
        with self.assertRaises(StopIteration):
            next(generator)
        self.assert_exact_bool_list(first_empty, [])
        self.assert_exact_bool_list(second_empty, [])
        self.assertIsNot(first_empty, second_empty)

    def test_iteration_access_call_and_truth_exceptions_propagate_unchanged(self):
        function = torch.compiler.skip_guard_on_all_nn_modules_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as iter_raised:
            function(iter_failure)
        self.assertIs(iter_raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        events = []
        next_error = LookupError("next failure")
        next_failure = _NextFailure(_GuardEntry("first", False, events), next_error)
        with self.assertRaises(LookupError) as next_raised:
            function(next_failure)
        self.assertIs(next_raised.exception, next_error)
        self.assertGreaterEqual(next_failure.iter_calls, 1)
        self.assertEqual(next_failure.next_calls, 2)

        orig_guard_error = KeyError("orig_guard failure")
        orig_guard_failure = _OrigGuardFailureEntry(orig_guard_error)
        untouched = _OrigGuardFailureEntry(AssertionError("should not be reached"))
        with self.assertRaises(KeyError) as orig_guard_raised:
            function((orig_guard_failure, untouched))
        self.assertIs(orig_guard_raised.exception, orig_guard_error)
        self.assertEqual(orig_guard_failure.attribute_calls, 1)
        self.assertEqual(untouched.attribute_calls, 0)

        source_error = OSError("source failure")
        source_failure = _SourceFailureOrigGuard(source_error)
        with self.assertRaises(OSError) as source_raised:
            function((types.SimpleNamespace(orig_guard=source_failure),))
        self.assertIs(source_raised.exception, source_error)
        self.assertEqual(source_failure.attribute_calls, 1)

        method_error = ZeroDivisionError("method attribute failure")
        method_failure = _MethodFailureSource(method_error)
        with self.assertRaises(ZeroDivisionError) as method_raised:
            function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(source=method_failure)
                    ),
                )
            )
        self.assertIs(method_raised.exception, method_error)
        self.assertEqual(method_failure.attribute_calls, 1)

        call_error = ArithmeticError("call failure")
        call_failure = _CallFailure(call_error)
        with self.assertRaises(ArithmeticError) as call_raised:
            function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(
                            source=types.SimpleNamespace(
                                is_unspecialized_nn_module=call_failure
                            )
                        )
                    ),
                )
            )
        self.assertIs(call_raised.exception, call_error)
        self.assertEqual(call_failure.calls, 1)

        truth_error = ValueError("truth failure")
        truth_failure = _TruthFailure(truth_error)
        with self.assertRaises(ValueError) as truth_raised:
            function((_simple_entry(truth_failure),))
        self.assertIs(truth_raised.exception, truth_error)
        self.assertEqual(truth_failure.bool_calls, 1)

        with self.assertRaises(TypeError) as invalid_iterator_raised:
            function(_InvalidIterator())
        invalid_iterator_message = "iter() returned non-iterator of type 'list'"
        self.assertEqual(str(invalid_iterator_raised.exception), invalid_iterator_message)
        self.assertEqual(
            invalid_iterator_raised.exception.args, (invalid_iterator_message,)
        )

    def test_missing_noncallable_and_invalid_truth_errors_match_pytorch_2_13(self):
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
                types.SimpleNamespace(
                    orig_guard=types.SimpleNamespace(source=_MissingMethod())
                ),
                AttributeError,
                "'_MissingMethod' object has no attribute "
                "'is_unspecialized_nn_module'",
            ),
            (
                types.SimpleNamespace(
                    orig_guard=types.SimpleNamespace(
                        source=types.SimpleNamespace(
                            is_unspecialized_nn_module=object()
                        )
                    )
                ),
                TypeError,
                "'object' object is not callable",
            ),
            (
                _simple_entry(_InvalidTruth()),
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
                "skip_guard_on_all_nn_modules_unsafe() takes 1 positional argument "
                "but 2 were given",
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

            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (torch.no_grad(), False),
            ):
                with context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertEqual(
                        compiler.skip_guard_on_all_nn_modules_unsafe(
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
            self.assertEqual(new_function((_simple_entry(True),)), [False])
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
source = types.SimpleNamespace(is_unspecialized_nn_module=lambda: True)
entry = types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))
result = torch.compiler.skip_guard_on_all_nn_modules_unsafe(iter((entry,)))
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


if __name__ == "__main__":
    unittest.main()
