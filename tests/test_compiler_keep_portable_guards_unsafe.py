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
        return self.result


class _GuardTypeProbe:
    def __init__(self, label, outcomes, events):
        self.label = label
        self.outcomes = outcomes
        self.events = events

    def __eq__(self, other):
        self.events.append(("eq", self.label, other))
        outcome = self.outcomes[other]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _GuardEntry:
    def __init__(self, label, guard_types, is_global, events):
        self.label = label
        self.guard_types = list(guard_types)
        self._is_global = is_global
        self.events = events
        self.guard_type_calls = 0
        self.is_global_calls = 0

    @property
    def guard_type(self):
        self.events.append(("guard_type", self.label))
        index = self.guard_type_calls
        self.guard_type_calls += 1
        value = self.guard_types[index]
        if isinstance(value, BaseException):
            raise value
        return value

    @property
    def is_global(self):
        self.events.append(("is_global", self.label))
        self.is_global_calls += 1
        if isinstance(self._is_global, BaseException):
            raise self._is_global
        return self._is_global


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


class _InvalidTruth:
    def __bool__(self):
        return 1


class _TruthFailure:
    def __init__(self, error):
        self.error = error
        self.bool_calls = 0

    def __bool__(self):
        self.bool_calls += 1
        raise self.error


class _InvalidIterator:
    def __iter__(self):
        return []


class _MissingGuardType:
    pass


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

    def test_canonical_guard_types_return_exact_booleans_in_access_order(self):
        events = []
        entries = [
            _GuardEntry("global-state", ["GLOBAL_STATE"], AssertionError(), events),
            _GuardEntry("shape-env", ["SHAPE_ENV"], AssertionError(), events),
            _GuardEntry(
                "local-tensor", ["TENSOR_MATCH", "TENSOR_MATCH"], False, events
            ),
            _GuardEntry(
                "global-tensor", ["TENSOR_MATCH", "TENSOR_MATCH"], True, events
            ),
            _GuardEntry("other", ["OTHER", "OTHER"], AssertionError(), events),
        ]

        result = torch.compiler.keep_portable_guards_unsafe(entries)

        self.assert_exact_bool_list(result, [True, True, True, False, False])
        self.assertEqual(
            events,
            [
                ("guard_type", "global-state"),
                ("guard_type", "shape-env"),
                ("guard_type", "local-tensor"),
                ("guard_type", "local-tensor"),
                ("is_global", "local-tensor"),
                ("guard_type", "global-tensor"),
                ("guard_type", "global-tensor"),
                ("is_global", "global-tensor"),
                ("guard_type", "other"),
                ("guard_type", "other"),
            ],
        )

    def test_comparison_truth_conversion_and_short_circuiting_are_preserved(self):
        events = []
        false_global = _TruthProbe("global-comparison", False, events)
        false_shape = _TruthProbe("shape-comparison", False, events)
        true_tensor = _TruthProbe("tensor-comparison", True, events)
        local = _TruthProbe("is-global", False, events)
        guard_type = _GuardTypeProbe(
            "type",
            {
                "GLOBAL_STATE": false_global,
                "SHAPE_ENV": false_shape,
                "TENSOR_MATCH": true_tensor,
            },
            events,
        )
        entry = _GuardEntry("custom", [guard_type, guard_type], local, events)

        result = torch.compiler.keep_portable_guards_unsafe((entry,))

        self.assert_exact_bool_list(result, [True])
        self.assertEqual(
            events,
            [
                ("guard_type", "custom"),
                ("eq", "type", "GLOBAL_STATE"),
                ("bool", "global-comparison"),
                ("eq", "type", "SHAPE_ENV"),
                ("bool", "shape-comparison"),
                ("guard_type", "custom"),
                ("eq", "type", "TENSOR_MATCH"),
                ("bool", "tensor-comparison"),
                ("is_global", "custom"),
                ("bool", "is-global"),
            ],
        )

        events.clear()
        false_tensor = _TruthProbe("tensor-comparison", False, events)
        guard_type = _GuardTypeProbe(
            "type",
            {
                "GLOBAL_STATE": False,
                "SHAPE_ENV": False,
                "TENSOR_MATCH": false_tensor,
            },
            events,
        )
        entry = _GuardEntry(
            "false-tensor",
            [guard_type, guard_type],
            AssertionError("is_global must be skipped"),
            events,
        )

        result = torch.compiler.keep_portable_guards_unsafe((entry,))

        self.assertIs(result[0], false_tensor)
        self.assertEqual(entry.is_global_calls, 0)
        self.assertEqual(events[-1], ("bool", "tensor-comparison"))

    def test_inputs_are_materialized_once_into_fresh_lists(self):
        entries = [
            types.SimpleNamespace(guard_type="GLOBAL_STATE"),
            types.SimpleNamespace(guard_type="TENSOR_MATCH", is_global=False),
            types.SimpleNamespace(guard_type="OTHER"),
        ]
        iterable = _CountingIterable(entries)

        iterable_result = torch.compiler.keep_portable_guards_unsafe(iterable)
        tuple_result = torch.compiler.keep_portable_guards_unsafe(tuple(entries))
        first_empty = torch.compiler.keep_portable_guards_unsafe([])
        second_empty = torch.compiler.keep_portable_guards_unsafe(())

        self.assert_exact_bool_list(iterable_result, [True, True, False])
        self.assert_exact_bool_list(tuple_result, [True, True, False])
        self.assert_exact_bool_list(first_empty, [])
        self.assert_exact_bool_list(second_empty, [])
        self.assertEqual(iterable.iter_calls, 1)
        self.assertEqual(iterable.next_calls, len(entries) + 1)
        self.assertIsNot(iterable_result, tuple_result)
        self.assertIsNot(first_empty, second_empty)

    def test_generator_is_consumed_in_order_and_exhausted(self):
        events = []

        def guard_entries():
            for index, entry in enumerate(
                (
                    types.SimpleNamespace(guard_type="GLOBAL_STATE"),
                    types.SimpleNamespace(guard_type="TENSOR_MATCH", is_global=True),
                )
            ):
                events.append(("yield", index))
                yield entry
            events.append(("finished", 2))

        generator = guard_entries()
        result = torch.compiler.keep_portable_guards_unsafe(generator)

        self.assert_exact_bool_list(result, [True, False])
        self.assertEqual(events, [("yield", 0), ("yield", 1), ("finished", 2)])
        with self.assertRaises(StopIteration):
            next(generator)

    def test_iteration_attribute_comparison_and_truth_errors_propagate(self):
        function = torch.compiler.keep_portable_guards_unsafe

        iter_error = RuntimeError("iter failure")
        iter_failure = _IterFailure(iter_error)
        with self.assertRaises(RuntimeError) as raised:
            function(iter_failure)
        self.assertIs(raised.exception, iter_error)
        self.assertEqual(iter_failure.iter_calls, 1)

        events = []
        next_error = LookupError("next failure")
        first = _GuardEntry("first", ["GLOBAL_STATE"], AssertionError(), events)
        next_failure = _NextFailure(first, next_error)
        with self.assertRaises(LookupError) as raised:
            function(next_failure)
        self.assertIs(raised.exception, next_error)
        self.assertGreaterEqual(next_failure.iter_calls, 1)
        self.assertEqual(next_failure.next_calls, 2)
        self.assertEqual(events, [("guard_type", "first")])

        first_error = KeyError("first guard_type failure")
        first_failure = _GuardEntry("first", [first_error], None, events=[])
        untouched = _GuardEntry("untouched", ["GLOBAL_STATE"], None, events=[])
        with self.assertRaises(KeyError) as raised:
            function((first_failure, untouched))
        self.assertIs(raised.exception, first_error)
        self.assertEqual(untouched.guard_type_calls, 0)

        second_error = IndexError("second guard_type failure")
        second_failure = _GuardEntry("second", ["OTHER", second_error], None, [])
        with self.assertRaises(IndexError) as raised:
            function((second_failure,))
        self.assertIs(raised.exception, second_error)
        self.assertEqual(second_failure.guard_type_calls, 2)
        self.assertEqual(second_failure.is_global_calls, 0)

        global_error = ValueError("is_global failure")
        global_failure = _GuardEntry(
            "global", ["TENSOR_MATCH", "TENSOR_MATCH"], global_error, []
        )
        with self.assertRaises(ValueError) as raised:
            function((global_failure,))
        self.assertIs(raised.exception, global_error)

        comparison_error = ArithmeticError("comparison failure")
        comparison_type = _GuardTypeProbe(
            "failure", {"GLOBAL_STATE": comparison_error}, []
        )
        comparison_entry = _GuardEntry("comparison", [comparison_type], None, [])
        with self.assertRaises(ArithmeticError) as raised:
            function((comparison_entry,))
        self.assertIs(raised.exception, comparison_error)

        truth_error = OverflowError("truth failure")
        truth_probe = _TruthFailure(truth_error)
        truth_type = _GuardTypeProbe("truth", {"GLOBAL_STATE": truth_probe}, [])
        truth_entry = _GuardEntry("truth", [truth_type], None, [])
        with self.assertRaises(OverflowError) as raised:
            function((truth_entry,))
        self.assertIs(raised.exception, truth_error)
        self.assertEqual(truth_probe.bool_calls, 1)

        with self.assertRaises(TypeError) as raised:
            function(_InvalidIterator())
        self.assertEqual(
            str(raised.exception), "iter() returned non-iterator of type 'list'"
        )

    def test_missing_attributes_and_invalid_truth_values_match_contract(self):
        function = torch.compiler.keep_portable_guards_unsafe

        with self.assertRaises(AttributeError) as raised:
            function((_MissingGuardType(),))
        message = "'_MissingGuardType' object has no attribute 'guard_type'"
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

        with self.assertRaises(AttributeError) as raised:
            function((types.SimpleNamespace(guard_type="TENSOR_MATCH"),))
        self.assertIn("has no attribute 'is_global'", str(raised.exception))

        invalid = _GuardTypeProbe("invalid", {"GLOBAL_STATE": _InvalidTruth()}, [])
        with self.assertRaises(TypeError) as raised:
            function((_GuardEntry("invalid", [invalid], None, []),))
        message = "__bool__ should return bool, returned int"
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

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
            entries = (
                types.SimpleNamespace(guard_type="GLOBAL_STATE"),
                types.SimpleNamespace(guard_type="TENSOR_MATCH", is_global=False),
            )

            for context, expected_grad_state in (
                (contextlib.nullcontext(), True),
                (torch.no_grad(), False),
            ):
                with context:
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
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
            self.assertEqual(new_function(iter(entries)), [True, True])
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
            raise AssertionError(f"unexpected PyTorch import: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
modules_before_call = set(sys.modules)
guard_entries = (
    types.SimpleNamespace(guard_type="GLOBAL_STATE"),
    types.SimpleNamespace(guard_type="SHAPE_ENV"),
    types.SimpleNamespace(guard_type="TENSOR_MATCH", is_global=False),
)
result = torch.compiler.keep_portable_guards_unsafe(iter(guard_entries))
assert result == [True, True, True]
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
