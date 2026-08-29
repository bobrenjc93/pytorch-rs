import contextlib
import copy
import importlib
import inspect
import pickle
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
}


class _TruthProbe:
    def __init__(self, label, result, events):
        self.label = label
        self.result = result
        self.events = events

    def __bool__(self):
        self.events.append(("bool", self.label))
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

    def __iter__(self):
        raise self.error


class _NextFailure:
    def __init__(self, entry, error):
        self.entry = entry
        self.error = error
        self.next_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        if self.next_calls == 1:
            return self.entry
        raise self.error


class _AttributeFailure:
    def __init__(self, attribute, error):
        self.attribute = attribute
        self.error = error

    def __getattr__(self, name):
        if name == self.attribute:
            raise self.error
        raise AttributeError(name)


class _CallFailure:
    def __init__(self, error):
        self.error = error

    def __call__(self):
        raise self.error


class _TruthFailure:
    def __init__(self, error):
        self.error = error

    def __bool__(self):
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


def _simple_entry(result):
    source = types.SimpleNamespace(is_unspecialized_nn_module=lambda: result)
    return types.SimpleNamespace(orig_guard=types.SimpleNamespace(source=source))


def _raised_outcome(call):
    try:
        call()
    except BaseException as error:
        return type(error).__name__, error.args, str(error)
    return None


@unittest.skipIf(reference_torch is None, "PyTorch reference package is unavailable")
class CompilerSkipGuardOnAllNnModulesUnsafeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.skip_guard_on_all_nn_modules_unsafe differentials "
                "require pinned PyTorch 2.13.0"
            )

    def behavior_outcome(self, function):
        events = []
        entries = [
            _GuardEntry("true", True, events),
            _GuardEntry("false", False, events),
            _GuardEntry("one", 1, events),
            _GuardEntry("zero", 0, events),
            _GuardEntry("none", None, events),
            _GuardEntry("object", object(), events),
            _GuardEntry("probe", _TruthProbe("probe", False, events), events),
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
        first_entry = _simple_entry(False)
        cases = [
            lambda: function(_IterFailure(RuntimeError("iter failure"))),
            lambda: function(_NextFailure(first_entry, LookupError("next failure"))),
            lambda: function((_AttributeFailure("orig_guard", KeyError("orig")),)),
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
                                "is_unspecialized_nn_module",
                                ZeroDivisionError("method"),
                            )
                        )
                    ),
                )
            ),
            lambda: function((_simple_entry(_TruthFailure(ValueError("truth"))),)),
            lambda: function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(
                            source=types.SimpleNamespace(
                                is_unspecialized_nn_module=_CallFailure(
                                    ArithmeticError("call")
                                )
                            )
                        )
                    ),
                )
            ),
            lambda: function(_InvalidIterator()),
            lambda: function((_MissingOrigGuard(),)),
            lambda: function(
                (types.SimpleNamespace(orig_guard=_MissingSource()),)
            ),
            lambda: function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(source=_MissingMethod())
                    ),
                )
            ),
            lambda: function(
                (
                    types.SimpleNamespace(
                        orig_guard=types.SimpleNamespace(
                            source=types.SimpleNamespace(
                                is_unspecialized_nn_module=object()
                            )
                        )
                    ),
                )
            ),
            lambda: function((_simple_entry(_InvalidTruth()),)),
        ]
        return tuple(_raised_outcome(case) for case in cases)

    def argument_outcome(self, function):
        cases = (
            lambda: function(),
            lambda: function([], []),
            lambda: function([], guard_entries=[]),
            lambda: function(entries=[]),
            lambda: function(None),
            lambda: function(1),
        )
        return tuple(_raised_outcome(case) for case in cases)

    def test_values_iteration_access_calls_and_truth_match_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_outcome(
                torch.compiler.skip_guard_on_all_nn_modules_unsafe
            ),
            self.behavior_outcome(
                reference_torch.compiler.skip_guard_on_all_nn_modules_unsafe
            ),
        )

    def test_exceptions_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.compiler.skip_guard_on_all_nn_modules_unsafe
        expected = reference_torch.compiler.skip_guard_on_all_nn_modules_unsafe
        self.assertEqual(self.exception_outcome(actual), self.exception_outcome(expected))
        self.assertEqual(self.argument_outcome(actual), self.argument_outcome(expected))

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.skip_guard_on_all_nn_modules_unsafe
        expected = expected_compiler.skip_guard_on_all_nn_modules_unsafe

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(inspect.signature(actual), inspect.signature(expected))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, "torch_rs.compiler")
        self.assertEqual(expected.__module__, "torch.compiler")
        self.assertIs(inspect.getmodule(actual), actual_compiler)
        self.assertIs(inspect.getmodule(expected), expected_compiler)
        self.assertEqual(
            inspect.cleandoc(actual.__doc__), inspect.cleandoc(expected.__doc__)
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

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
            reference_torch.__all__.count(
                "skip_guard_on_all_nn_modules_unsafe"
            ),
        )

        for compiler in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {compiler.__name__} import *", namespace)
            self.assertIs(
                namespace["skip_guard_on_all_nn_modules_unsafe"],
                compiler.skip_guard_on_all_nn_modules_unsafe,
            )

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("skip_guard_on_all_nn_modules_unsafe", namespace)

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol)), function
                )

    def state_and_reload_outcome(self, module, compiler_module_name):
        compiler = importlib.import_module(compiler_module_name)
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            query_state = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            call_outcomes = []
            for context in (contextlib.nullcontext(), module.no_grad()):
                with context:
                    grad_before = module.is_grad_enabled()
                    result = compiler.skip_guard_on_all_nn_modules_unsafe(
                        iter((_simple_entry(False),))
                    )
                    call_outcomes.append(
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

            old_function = compiler.skip_guard_on_all_nn_modules_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.skip_guard_on_all_nn_modules_unsafe
            old_pickles = True
            try:
                pickle.dumps(old_function)
            except pickle.PicklingError:
                old_pickles = False
            return (
                call_outcomes,
                query_state,
                reloaded is compiler,
                module.compiler is compiler,
                new_function is old_function,
                compiler.__all__ is old_exports,
                compiler.get_default_backend() is backend,
                new_function((_simple_entry(True),)),
                old_pickles,
                all(
                    pickle.loads(pickle.dumps(new_function, protocol))
                    is new_function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_calls_and_reload_preserve_state_like_pytorch_2_13(self):
        self.assertEqual(
            self.state_and_reload_outcome(torch, "torch_rs.compiler"),
            self.state_and_reload_outcome(reference_torch, "torch.compiler"),
        )


if __name__ == "__main__":
    unittest.main()
