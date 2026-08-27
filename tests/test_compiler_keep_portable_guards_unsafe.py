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
    A common function to only keep guards that can be used in both Python and non-Python environments.
    This includes:
    - Tensor metadata and dynamic shape information.
    - Global contexts state (e.g. autocast, no_grad, etc.)

    This is unsafe to use by default.
    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_global_context_and_tensor_guards_unsafe},
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
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _GuardEntry:
    def __init__(self, label, guard_type, is_global, events):
        self.label = label
        self._guard_type = guard_type
        self._is_global = is_global
        self.events = events

    @property
    def guard_type(self):
        self.events.append(("guard_type", self.label))
        return self._guard_type

    @property
    def is_global(self):
        self.events.append(("is_global", self.label))
        return self._is_global


class _ChangingGuardEntry:
    def __init__(self, label, guard_types, is_global, events):
        self.label = label
        self.guard_types = iter(guard_types)
        self._is_global = is_global
        self.events = events

    @property
    def guard_type(self):
        self.events.append(("guard_type", self.label))
        return next(self.guard_types)

    @property
    def is_global(self):
        self.events.append(("is_global", self.label))
        return self._is_global


class _EqualityProbe:
    def __init__(self, label, results, events):
        self.label = label
        self.results = iter(results)
        self.events = events

    def __eq__(self, other):
        self.events.append(("eq", self.label, other))
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


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


class _GuardTypeFailureEntry:
    def __init__(self, error, fail_on_call=1):
        self.error = error
        self.fail_on_call = fail_on_call
        self.guard_type_calls = 0
        self.is_global_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        if self.guard_type_calls == self.fail_on_call:
            raise self.error
        return "OTHER"

    @property
    def is_global(self):
        self.is_global_calls += 1
        raise AssertionError("is_global should not be accessed")


class _GlobalFailureEntry:
    def __init__(self, error):
        self.error = error
        self.guard_type_calls = 0
        self.is_global_calls = 0

    @property
    def guard_type(self):
        self.guard_type_calls += 1
        return "TENSOR_MATCH"

    @property
    def is_global(self):
        self.is_global_calls += 1
        raise self.error


class _InvalidTruth:
    def __bool__(self):
        return 1


class _MissingGuardType:
    pass


class _InvalidIterator:
    def __iter__(self):
        return []


def _walk_code_objects(code):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            yield from _walk_code_objects(constant)


class CompilerKeepPortableGuardsUnsafeTests(unittest.TestCase):
    def assert_exact_bool_list(self, result, expected):
        self.assertIs(type(result), list)
        self.assertEqual(len(result), len(expected))
        for actual, expected_value in zip(result, expected):
            self.assertIs(actual, expected_value)

    def test_portable_predicate_returns_exact_booleans_and_short_circuits(self):
        events = []
        entries = [
            _GuardEntry("global", "GLOBAL_STATE", object(), events),
            _GuardEntry("shape", "SHAPE_ENV", object(), events),
            _GuardEntry("local-tensor", "TENSOR_MATCH", False, events),
            _GuardEntry("global-tensor", "TENSOR_MATCH", True, events),
            _GuardEntry("none-tensor", "TENSOR_MATCH", None, events),
            _GuardEntry("other", "OTHER", object(), events),
        ]
        original_entries = tuple(entries)

        result = torch.compiler.keep_portable_guards_unsafe(entries)

        self.assert_exact_bool_list(result, [True, True, True, False, True, False])
        self.assertIsNot(result, entries)
        self.assertTrue(
            all(
                actual is original
                for actual, original in zip(entries, original_entries)
            )
        )
        self.assertEqual(
            events,
            [
                ("guard_type", "global"),
                ("guard_type", "shape"),
                ("guard_type", "local-tensor"),
                ("guard_type", "local-tensor"),
                ("is_global", "local-tensor"),
                ("guard_type", "global-tensor"),
                ("guard_type", "global-tensor"),
                ("is_global", "global-tensor"),
                ("guard_type", "none-tensor"),
                ("guard_type", "none-tensor"),
                ("is_global", "none-tensor"),
                ("guard_type", "other"),
                ("guard_type", "other"),
            ],
        )

    def test_repeated_attribute_access_and_comparison_truth_match_expression(self):
        events = []
        changing = _ChangingGuardEntry(
            "changing", ("OTHER", "TENSOR_MATCH"), False, events
        )
        self.assert_exact_bool_list(
            torch.compiler.keep_portable_guards_unsafe((changing,)), [True]
        )
        self.assertEqual(
            events,
            [
                ("guard_type", "changing"),
                ("guard_type", "changing"),
                ("is_global", "changing"),
            ],
        )

        events = []
        global_result = _TruthProbe("global-comparison", False, events)
        shape_result = _TruthProbe("shape-comparison", False, events)
        tensor_result = _TruthProbe("tensor-comparison", False, events)
        guard_type = _EqualityProbe(
            "custom", (global_result, shape_result, tensor_result), events
        )
        entry = _GuardEntry("custom", guard_type, False, events)

        result = torch.compiler.keep_portable_guards_unsafe((entry,))

        self.assertIs(result[0], tensor_result)
        self.assertEqual(
            events,
            [
                ("guard_type", "custom"),
                ("eq", "custom", "GLOBAL_STATE"),
                ("bool", "global-comparison"),
                ("eq", "custom", "SHAPE_ENV"),
                ("bool", "shape-comparison"),
                ("guard_type", "custom"),
                ("eq", "custom", "TENSOR_MATCH"),
                ("bool", "tensor-comparison"),
            ],
        )

        events = []
        membership_result = _TruthProbe("membership", True, events)
        guard_type = _EqualityProbe("short-circuit", (membership_result,), events)
        entry = _GuardEntry("short-circuit", guard_type, object(), events)

        result = torch.compiler.keep_portable_guards_unsafe((entry,))

        self.assert_exact_bool_list(result, [True])
        self.assertEqual(
            events,
            [
                ("guard_type", "short-circuit"),
                ("eq", "short-circuit", "GLOBAL_STATE"),
                ("bool", "membership"),
            ],
        )

    def test_arbitrary_iterables_are_consumed_once_in_order(self):
        events = []
        entries = [
            _GuardEntry("global", "GLOBAL_STATE", object(), events),
            _GuardEntry("tensor", "TENSOR_MATCH", False, events),
            _GuardEntry("other", "OTHER", object(), events),
        ]
        iterable = _CountingIterable(entries)

        result = torch.compiler.keep_portable_guards_unsafe(iterable)

        self.assert_exact_bool_list(result, [True, True, False])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, len(entries) + 1)

        generator_events = []

        def guard_entries():
            for index, entry in enumerate(entries):
                generator_events.append(("yield", index))
                yield entry
            generator_events.append(("finished", len(entries)))

        generator = guard_entries()
        self.assert_exact_bool_list(
            torch.compiler.keep_portable_guards_unsafe(generator),
            [True, True, False],
        )
        self.assertEqual(
            generator_events,
            [("yield", 0), ("yield", 1), ("yield", 2), ("finished", 3)],
        )
        with self.assertRaises(StopIteration):
            next(generator)

        first_empty = torch.compiler.keep_portable_guards_unsafe([])
        second_empty = torch.compiler.keep_portable_guards_unsafe(())
        self.assertEqual(first_empty, [])
        self.assertEqual(second_empty, [])
        self.assertIsNot(first_empty, second_empty)

    def test_iteration_attribute_comparison_and_truth_errors_propagate(self):
        function = torch.compiler.keep_portable_guards_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as iter_raised:
            function(iter_failure)
        self.assertIs(iter_raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        first = types.SimpleNamespace(guard_type="GLOBAL_STATE")
        next_error = LookupError("next failure")
        next_failure = _NextFailure(first, next_error)
        with self.assertRaises(LookupError) as next_raised:
            function(next_failure)
        self.assertIs(next_raised.exception, next_error)
        self.assertGreaterEqual(next_failure.iter_calls, 1)
        self.assertEqual(next_failure.next_calls, 2)

        first_attribute_error = KeyError("first guard_type failure")
        first_attribute_failure = _GuardTypeFailureEntry(first_attribute_error)
        untouched = _GuardTypeFailureEntry(AssertionError("should not be reached"))
        with self.assertRaises(KeyError) as first_attribute_raised:
            function((first_attribute_failure, untouched))
        self.assertIs(first_attribute_raised.exception, first_attribute_error)
        self.assertEqual(first_attribute_failure.guard_type_calls, 1)
        self.assertEqual(untouched.guard_type_calls, 0)

        second_attribute_error = IndexError("second guard_type failure")
        second_attribute_failure = _GuardTypeFailureEntry(
            second_attribute_error, fail_on_call=2
        )
        with self.assertRaises(IndexError) as second_attribute_raised:
            function((second_attribute_failure,))
        self.assertIs(second_attribute_raised.exception, second_attribute_error)
        self.assertEqual(second_attribute_failure.guard_type_calls, 2)
        self.assertEqual(second_attribute_failure.is_global_calls, 0)

        comparison_error = ZeroDivisionError("comparison failure")
        comparison_value = _EqualityProbe("comparison", (comparison_error,), [])
        with self.assertRaises(ZeroDivisionError) as comparison_raised:
            function((types.SimpleNamespace(guard_type=comparison_value),))
        self.assertIs(comparison_raised.exception, comparison_error)

        comparison_truth_error = ValueError("comparison truth failure")
        comparison_truth = _TruthProbe(
            "comparison", comparison_truth_error, []
        )
        comparison_value = _EqualityProbe(
            "comparison", (comparison_truth,), []
        )
        with self.assertRaises(ValueError) as comparison_truth_raised:
            function((types.SimpleNamespace(guard_type=comparison_value),))
        self.assertIs(comparison_truth_raised.exception, comparison_truth_error)

        global_error = OSError("is_global failure")
        global_failure = _GlobalFailureEntry(global_error)
        with self.assertRaises(OSError) as global_raised:
            function((global_failure,))
        self.assertIs(global_raised.exception, global_error)
        self.assertEqual(global_failure.guard_type_calls, 2)
        self.assertEqual(global_failure.is_global_calls, 1)

        truth_error = ArithmeticError("is_global truth failure")
        truth_probe = _TruthProbe("is_global", truth_error, [])
        with self.assertRaises(ArithmeticError) as truth_raised:
            function(
                (
                    types.SimpleNamespace(
                        guard_type="TENSOR_MATCH", is_global=truth_probe
                    ),
                )
            )
        self.assertIs(truth_raised.exception, truth_error)

        with self.assertRaises(AttributeError) as missing_raised:
            function((_MissingGuardType(),))
        missing_message = "'_MissingGuardType' object has no attribute 'guard_type'"
        self.assertEqual(str(missing_raised.exception), missing_message)
        self.assertEqual(missing_raised.exception.args, (missing_message,))

        invalid_entry = types.SimpleNamespace(
            guard_type="TENSOR_MATCH", is_global=_InvalidTruth()
        )
        with self.assertRaises(TypeError) as invalid_truth_raised:
            function((invalid_entry,))
        invalid_truth_message = "__bool__ should return bool, returned int"
        self.assertEqual(str(invalid_truth_raised.exception), invalid_truth_message)
        self.assertEqual(invalid_truth_raised.exception.args, (invalid_truth_message,))

        with self.assertRaises(TypeError) as invalid_iterator_raised:
            function(_InvalidIterator())
        invalid_iterator_message = "iter() returned non-iterator of type 'list'"
        self.assertEqual(
            str(invalid_iterator_raised.exception), invalid_iterator_message
        )

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.keep_portable_guards_unsafe

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(guard_entries)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(typing.get_type_hints(function), {})
        self.assertEqual(function.__name__, "keep_portable_guards_unsafe")
        self.assertEqual(function.__qualname__, "keep_portable_guards_unsafe")
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
            ("guard_type", "is_global"),
        )
        self.assertEqual(function.__code__.co_varnames[0], "guard_entries")
        self.assertIn("g", {name for code in code_objects for name in code.co_varnames})
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copying_and_pickling_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.keep_portable_guards_unsafe

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
        self.assertNotIn("keep_portable_guards_unsafe", torch.__all__)
        self.assertFalse(hasattr(torch, "keep_portable_guards_unsafe"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("keep_portable_guards_unsafe", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.keep_portable_guards_unsafe
        cases = (
            (
                lambda: function(),
                "keep_portable_guards_unsafe() missing 1 required positional "
                "argument: 'guard_entries'",
            ),
            (
                lambda: function([], []),
                "keep_portable_guards_unsafe() takes 1 positional argument but 2 "
                "were given",
            ),
            (
                lambda: function([], guard_entries=[]),
                "keep_portable_guards_unsafe() got multiple values for argument "
                "'guard_entries'",
            ),
            (
                lambda: function(entries=[]),
                "keep_portable_guards_unsafe() got an unexpected keyword argument "
                "'entries'",
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
                    entries = (
                        types.SimpleNamespace(guard_type="GLOBAL_STATE"),
                        types.SimpleNamespace(
                            guard_type="TENSOR_MATCH", is_global=False
                        ),
                    )
                    self.assertEqual(
                        compiler.keep_portable_guards_unsafe(iter(entries)),
                        [True, True],
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

            old_function = compiler.keep_portable_guards_unsafe
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.keep_portable_guards_unsafe

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            entry = types.SimpleNamespace(guard_type="SHAPE_ENV")
            self.assertEqual(new_function((entry,)), [True])
            self.assertIs(copy.copy(old_function), old_function)
            self.assertIs(copy.deepcopy(old_function), old_function)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function, protocol)), new_function
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
    types.SimpleNamespace(guard_type="GLOBAL_STATE"),
    types.SimpleNamespace(guard_type="SHAPE_ENV"),
    types.SimpleNamespace(guard_type="TENSOR_MATCH", is_global=False),
    types.SimpleNamespace(guard_type="TENSOR_MATCH", is_global=True),
    types.SimpleNamespace(guard_type="OTHER"),
)
result = torch.compiler.keep_portable_guards_unsafe(iter(guard_entries))
assert result == [True, True, True, False, False]
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
