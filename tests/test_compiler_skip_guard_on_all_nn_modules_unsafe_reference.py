import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_COMPILER_EXPORTS = {
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
}


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


def _normalized_code_metadata(code):
    return (
        code.co_names,
        code.co_varnames,
        tuple(
            (
                _normalized_code_metadata(constant)
                if isinstance(constant, types.CodeType)
                else constant
            )
            for constant in code.co_consts
        ),
        code.co_freevars,
        code.co_cellvars,
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerSkipGuardOnAllNnModulesUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.skip_guard_on_all_nn_modules_unsafe differentials "
                "require pinned PyTorch 2.13.0"
            )

    def captured_error(self, call, expected_error=None):
        try:
            call()
        except BaseException as error:
            return (
                type(error).__name__,
                str(error),
                error.args,
                error is expected_error,
            )
        self.fail("expected an exception")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(BaseException) as actual_raised:
            actual_call()
        with self.assertRaises(BaseException) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def behavior_outcome(self, function):
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
        list_result = function(entries)
        tuple_result = function(tuple(entries))
        first_empty = function([])
        second_empty = function(())

        iterable = _CountingIterable(entries)
        iterable_result = function(iterable)

        generator_events = []
        generator_entries = [
            _GuardEntry("generator-true", True, generator_events),
            _GuardEntry("generator-false", False, generator_events),
        ]

        def generated_entries():
            for index, entry in enumerate(generator_entries):
                generator_events.append(("yield", index))
                yield entry
            generator_events.append(("finished", len(generator_entries)))

        generator = generated_entries()
        generator_result = function(generator)
        try:
            next(generator)
        except StopIteration:
            generator_exhausted = True
        else:
            generator_exhausted = False

        results = (
            list_result,
            tuple_result,
            first_empty,
            second_empty,
            iterable_result,
            generator_result,
        )
        return {
            "results": results,
            "result_types": tuple(type(result).__name__ for result in results),
            "exact_bools": tuple(
                tuple(value is True if value else value is False for value in result)
                for result in results
            ),
            "fresh": (
                list_result is not entries,
                list_result is not tuple_result,
                first_empty is not second_empty,
            ),
            "source_preserved": all(
                actual is original
                for actual, original in zip(entries, original_entries)
            ),
            "events": events,
            "iter_calls": iterable.iter_calls,
            "next_calls": iterable.next_calls,
            "generator_events": generator_events,
            "generator_exhausted": generator_exhausted,
        }

    def exception_outcome(self, function):
        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        iter_outcome = self.captured_error(
            lambda: function(iter_failure), iter_error
        ) + (iter_failure.calls,)

        next_error = LookupError("next failure")
        next_failure = _NextFailure(
            _entry_with_source(
                types.SimpleNamespace(is_unspecialized_nn_module=lambda: False)
            ),
            next_error,
        )
        next_outcome = self.captured_error(
            lambda: function(next_failure), next_error
        ) + (next_failure.iter_calls, next_failure.next_calls)

        orig_error = KeyError("orig_guard failure")
        orig_failure = _OrigGuardFailure(orig_error)
        orig_outcome = self.captured_error(
            lambda: function((orig_failure,)), orig_error
        ) + (orig_failure.calls,)

        source_error = IndexError("source failure")
        source_failure = _SourceFailure(source_error)
        source_outcome = self.captured_error(
            lambda: function((types.SimpleNamespace(orig_guard=source_failure),)),
            source_error,
        ) + (source_failure.calls,)

        method_error = OSError("method failure")
        method_failure = _MethodFailure(method_error)
        method_outcome = self.captured_error(
            lambda: function((_entry_with_source(method_failure),)), method_error
        ) + (method_failure.calls,)

        call_error = ArithmeticError("call failure")
        call_failure = _CallFailure(call_error)
        call_outcome = self.captured_error(
            lambda: function((_entry_with_source(call_failure),)), call_error
        ) + (call_failure.calls,)

        truth_error = ValueError("truth failure")
        truth_failure = _TruthFailure(truth_error)
        truth_outcome = self.captured_error(
            lambda: function(
                (
                    _entry_with_source(
                        types.SimpleNamespace(
                            is_unspecialized_nn_module=lambda: truth_failure
                        )
                    ),
                )
            ),
            truth_error,
        ) + (truth_failure.calls,)

        return (
            iter_outcome,
            next_outcome,
            orig_outcome,
            source_outcome,
            method_outcome,
            call_outcome,
            truth_outcome,
            self.captured_error(lambda: function((_MissingOrigGuard(),))),
            self.captured_error(
                lambda: function((types.SimpleNamespace(orig_guard=_MissingSource()),))
            ),
            self.captured_error(
                lambda: function((_entry_with_source(_MissingMethod()),))
            ),
            self.captured_error(
                lambda: function(
                    (
                        _entry_with_source(
                            types.SimpleNamespace(is_unspecialized_nn_module=None)
                        ),
                    )
                )
            ),
            self.captured_error(
                lambda: function(
                    (
                        _entry_with_source(
                            types.SimpleNamespace(
                                is_unspecialized_nn_module=lambda: _InvalidTruth()
                            )
                        ),
                    )
                )
            ),
            self.captured_error(lambda: function(_InvalidIterator())),
        )

    def test_values_access_calls_truth_and_iteration_match_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_outcome(torch.compiler.skip_guard_on_all_nn_modules_unsafe),
            self.behavior_outcome(
                reference_torch.compiler.skip_guard_on_all_nn_modules_unsafe
            ),
        )

    def test_exceptions_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.skip_guard_on_all_nn_modules_unsafe
        expected = reference_torch.compiler.skip_guard_on_all_nn_modules_unsafe
        self.assertEqual(
            self.exception_outcome(actual), self.exception_outcome(expected)
        )
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual([], []), lambda: expected([], [])),
            (
                lambda: actual([], guard_entries=[]),
                lambda: expected([], guard_entries=[]),
            ),
            (lambda: actual(entries=[]), lambda: expected(entries=[])),
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(1), lambda: expected(1)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.skip_guard_on_all_nn_modules_unsafe
        expected = expected_compiler.skip_guard_on_all_nn_modules_unsafe

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            _normalized_code_metadata(actual.__code__),
            _normalized_code_metadata(expected.__code__),
        )

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.skip_guard_on_all_nn_modules_unsafe
        expected = expected_compiler.skip_guard_on_all_nn_modules_unsafe

        self.assertEqual(
            actual_compiler.__all__,
            [
                name
                for name in expected_compiler.__all__
                if name in SUPPORTED_COMPILER_EXPORTS
            ],
        )
        self.assertEqual(
            torch.__all__.count("compiler"),
            reference_torch.__all__.count("compiler"),
        )
        self.assertEqual(
            torch.__all__.count("skip_guard_on_all_nn_modules_unsafe"),
            reference_torch.__all__.count("skip_guard_on_all_nn_modules_unsafe"),
        )

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("skip_guard_on_all_nn_modules_unsafe", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def state_outcome(self, module):
        compiler = module.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            outcomes = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    before_grad = module.is_grad_enabled()
                    before_queries = (
                        compiler.is_compiling(),
                        compiler.is_dynamo_compiling(),
                        compiler.is_exporting(),
                    )
                    entry = _entry_with_source(
                        types.SimpleNamespace(is_unspecialized_nn_module=lambda: False)
                    )
                    result = compiler.skip_guard_on_all_nn_modules_unsafe(
                        iter((entry,))
                    )
                    outcomes.append(
                        (
                            result,
                            tuple(value is True for value in result),
                            module.is_grad_enabled(),
                            before_grad,
                            (
                                compiler.is_compiling(),
                                compiler.is_dynamo_compiling(),
                                compiler.is_exporting(),
                            ),
                            before_queries,
                            compiler.get_default_backend() is backend,
                        )
                    )
            return outcomes
        finally:
            compiler.set_default_backend(original_backend)

    def test_calls_preserve_compiler_and_grad_state_like_pytorch_2_13(self):
        self.assertEqual(self.state_outcome(torch), self.state_outcome(reference_torch))

    def reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            old_function = compiler.skip_guard_on_all_nn_modules_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_all_nn_modules_unsafe

            try:
                pickle.dumps(old_function)
            except BaseException as error:
                old_pickle_error = (
                    type(error).__name__,
                    "not the same object" in str(error),
                )
            else:
                old_pickle_error = None

            new_pickle_results = tuple(
                pickle.loads(pickle.dumps(new_function, protocol)) is new_function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            )
            entry = _entry_with_source(
                types.SimpleNamespace(is_unspecialized_nn_module=lambda: True)
            )
            return (
                reloaded is compiler,
                module.compiler is compiler,
                old_function is new_function,
                old_exports is compiler.__all__,
                compiler.get_default_backend() is backend,
                new_function((entry,)),
                copy.copy(old_function) is old_function,
                copy.deepcopy(old_function) is old_function,
                old_pickle_error,
                new_pickle_results,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch, "torch_rs.compiler"),
            self.reload_outcome(reference_torch, "torch.compiler"),
        )


if __name__ == "__main__":
    unittest.main()
