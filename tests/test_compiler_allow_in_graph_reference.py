import copy
import importlib
import inspect
import json
import pickle
import pickletools
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_COMPILER_EXPORTS = {
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
}


def _actual_picklable_function(value):
    return value + 1


_actual_picklable_function = torch.compiler.allow_in_graph(_actual_picklable_function)


def _reference_picklable_function(value):
    return value + 1


if reference_torch is not None:
    _reference_picklable_function = reference_torch.compiler.allow_in_graph(
        _reference_picklable_function
    )


class _CallableTarget:
    def __call__(self, value):
        return value * 2


class _SlotCallable:
    __slots__ = ()

    def __call__(self):
        return "called"


class _LookupProbe:
    def __init__(self, label, hash_outcomes, module_outcomes, events):
        self.label = label
        self.hash_outcomes = list(hash_outcomes)
        self.module_outcomes = list(module_outcomes)
        self.events = events
        self.hash_index = 0
        self.module_index = 0

    def __call__(self):
        return self.label

    def __hash__(self):
        index = self.hash_index
        self.hash_index += 1
        self.events.append(("hash", self.label, index))
        outcome = self.hash_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def __getattribute__(self, name):
        if name == "__module__":
            index = object.__getattribute__(self, "module_index")
            object.__setattr__(self, "module_index", index + 1)
            label = object.__getattribute__(self, "label")
            events = object.__getattribute__(self, "events")
            events.append(("module", label, index))
            outcomes = object.__getattribute__(self, "module_outcomes")
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return object.__getattribute__(self, name)


class _ObservedList(list):
    def __init__(self, values, events):
        super().__init__(values)
        self.events = events

    def __iter__(self):
        for index, value in enumerate(super().__iter__()):
            self.events.append(("yield", index))
            yield value
        self.events.append(("finished", len(self)))


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CompilerAllowInGraphReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "compiler.allow_in_graph differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
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
                argument = argument.replace("_actual_", "_shared_")
                argument = argument.replace("_reference_", "_shared_")
            shape.append((opcode.name, argument))
        return shape

    def function_outcome(self, module):
        calls = []
        sentinel = object()

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        calculate.custom_attribute = sentinel
        before = (
            calculate.__name__,
            calculate.__qualname__,
            calculate.__doc__,
            calculate.__annotations__.copy(),
            calculate.__defaults__,
            calculate.__kwdefaults__.copy(),
            dict(calculate.__dict__),
        )
        result = module.compiler.allow_in_graph(calculate)
        after = (
            calculate.__name__,
            calculate.__qualname__,
            calculate.__doc__,
            calculate.__annotations__,
            calculate.__defaults__,
            calculate.__kwdefaults__,
            calculate.__dict__,
        )
        return (
            result is calculate,
            before == after,
            calculate.custom_attribute is sentinel,
            calculate(3, scale=2),
            calculate(3, scale=2),
            calls,
            str(inspect.signature(calculate)),
            hasattr(calculate, "_dynamo_marked_constant"),
            hasattr(calculate, "_torchdynamo_disable"),
        )

    def method_and_callable_outcome(self, module):
        class Accumulator:
            def __init__(self):
                self.total = 0

            @module.compiler.allow_in_graph
            def add(self, value):
                self.total += value
                return self.total

        raw_method = Accumulator.__dict__["add"]
        left = Accumulator()
        right = Accumulator()
        bound = left.add
        allowed_bound = module.compiler.allow_in_graph(bound)
        target = _CallableTarget()
        allowed_target = module.compiler.allow_in_graph(target)
        return (
            Accumulator.add is raw_method,
            allowed_bound is bound,
            bound.__self__ is left,
            bound.__func__ is raw_method,
            left.add(2),
            left.add(3),
            right.add(7),
            allowed_target is target,
            target(6),
            dict(target.__dict__),
        )

    def sequence_outcome(self, module):
        def first():
            return "first"

        target = _CallableTarget()
        inner_tuple = (target, len)
        inner_list = [first]
        events = []
        source = _ObservedList((first, inner_tuple, inner_list), events)
        result = module.compiler.allow_in_graph(source)
        empty_list = []
        empty_tuple = ()
        converted_empty_list = module.compiler.allow_in_graph(empty_list)
        converted_empty_tuple = module.compiler.allow_in_graph(empty_tuple)
        return (
            type(result) is list,
            result is source,
            result[0] is first,
            type(result[1]) is list,
            result[1] is inner_tuple,
            result[1][0] is target,
            result[1][1] is len,
            type(result[2]) is list,
            result[2] is inner_list,
            result[2][0] is first,
            events,
            type(converted_empty_list) is list,
            converted_empty_list == [],
            converted_empty_list is empty_list,
            type(converted_empty_tuple) is list,
            converted_empty_tuple == [],
            converted_empty_tuple is empty_tuple,
        )

    def lookup_outcome(self, module, hash_outcomes, module_outcomes):
        events = []
        target = _LookupProbe("target", hash_outcomes, module_outcomes, events)
        try:
            result = module.compiler.allow_in_graph(target)
        except BaseException as error:
            outcome = (type(error).__name__, str(error), error.args)
        else:
            outcome = ("ok", result is target)
        return events, outcome

    def collection_lookup_outcome(self, module):
        events = []
        accepted = _LookupProbe(
            "accepted", (101, 101), ("sample.module", "sample.module"), events
        )
        failing = _LookupProbe("failing", (RuntimeError("hash failed"),), (), events)
        unreached = _LookupProbe(
            "unreached", (101, 101), ("sample.module", "sample.module"), events
        )
        values = _ObservedList((accepted, failing, unreached), events)
        try:
            module.compiler.allow_in_graph(values)
        except BaseException as error:
            outcome = (type(error).__name__, str(error), error.args)
        else:
            outcome = None
        return events, outcome

    def test_function_method_callable_and_sequence_semantics_match(self):
        self.assertEqual(
            self.function_outcome(torch),
            self.function_outcome(reference_torch),
        )
        self.assertEqual(
            self.method_and_callable_outcome(torch),
            self.method_and_callable_outcome(reference_torch),
        )
        self.assertEqual(
            self.sequence_outcome(torch),
            self.sequence_outcome(reference_torch),
        )

    def test_lookup_validation_errors_and_side_effects_match_pytorch_2_13(self):
        cases = (
            ((101, 101), ("sample.module", "sample.module")),
            ((RuntimeError("first hash"),), ()),
            ((TypeError("unhashable"),), ()),
            ((ValueError("unhashable"),), ()),
            ((101,), (RuntimeError("first module"),)),
            ((101,), ("sample.module", RuntimeError("second module"))),
            (
                (101, RuntimeError("second hash")),
                ("sample.module", "sample.module"),
            ),
            (
                (101, TypeError("second hash")),
                ("sample.module", "sample.module"),
            ),
            ((101,), (42,)),
        )
        for case, (hash_outcomes, module_outcomes) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(
                    self.lookup_outcome(torch, hash_outcomes, module_outcomes),
                    self.lookup_outcome(
                        reference_torch, hash_outcomes, module_outcomes
                    ),
                )

        self.assertEqual(
            self.collection_lookup_outcome(torch),
            self.collection_lookup_outcome(reference_torch),
        )

    def test_invalid_targets_error_order_and_call_shapes_match(self):
        direct_slot = _SlotCallable()
        list_slot = _SlotCallable()
        first_slot = _SlotCallable()
        second_slot = _SlotCallable()
        cases = (
            lambda function: function(),
            lambda function: function(lambda: None, lambda: None),
            lambda function: function(function=lambda: None),
            lambda function: function(lambda: None, fn=lambda: None),
            lambda function: function(None),
            lambda function: function(1),
            lambda function: function("callable"),
            lambda function: function(object()),
            lambda function: function(property()),
            lambda function: function((value for value in (len,))),
            lambda function: function(direct_slot),
            lambda function: function([lambda: None, None, list_slot]),
            lambda function: function((first_slot, None)),
            lambda function: function((None, second_slot)),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(torch.compiler.allow_in_graph),
                    lambda: call(reference_torch.compiler.allow_in_graph),
                )

        for module in (torch, reference_torch):
            events = []
            values = _ObservedList((lambda: None, None, _SlotCallable()), events)
            with self.assertRaises(AssertionError):
                module.compiler.allow_in_graph(values)
            self.assertEqual(events, [("yield", 0), ("yield", 1)])

    def test_signature_documentation_and_metadata_match_pytorch_2_13(self):
        actual_compiler = importlib.import_module("torch_rs.compiler")
        expected_compiler = importlib.import_module("torch.compiler")
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph

        self.assertIs(torch.compiler, actual_compiler)
        self.assertIs(reference_torch.compiler, expected_compiler)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
        self.assertEqual(actual.__annotations__, expected.__annotations__)
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

    def test_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual_compiler = torch.compiler
        expected_compiler = reference_torch.compiler
        actual = actual_compiler.allow_in_graph
        expected = expected_compiler.allow_in_graph

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
        self.assertEqual(torch.__all__.count("allow_in_graph"), 0)
        self.assertEqual(reference_torch.__all__.count("allow_in_graph"), 0)

        for module in (actual_compiler, expected_compiler):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            for name in SUPPORTED_COMPILER_EXPORTS:
                self.assertIs(namespace[name], getattr(module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("compiler", namespace)
            self.assertNotIn("allow_in_graph", namespace)

        for actual_value, expected_value in (
            (actual, expected),
            (_actual_picklable_function, _reference_picklable_function),
        ):
            self.assertIs(copy.copy(actual_value), actual_value)
            self.assertIs(copy.copy(expected_value), expected_value)
            self.assertIs(copy.deepcopy(actual_value), actual_value)
            self.assertIs(copy.deepcopy(expected_value), expected_value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=actual_value.__name__, protocol=protocol):
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_value, protocol)), actual_value
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_value, protocol)),
                        expected_value,
                    )

    def state_outcome(self, module):
        compiler = module.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            before = (
                module.is_grad_enabled(),
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            @compiler.allow_in_graph
            def function(value):
                return value + 1

            result = compiler.allow_in_graph((function, len))
            after = (
                module.is_grad_enabled(),
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            return (
                before,
                result[0] is function,
                result[1] is len,
                function(2),
                after,
                compiler.get_default_backend() is backend,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_eager_calls_preserve_state_like_pytorch_2_13(self):
        self.assertEqual(self.state_outcome(torch), self.state_outcome(reference_torch))

    def reload_outcome(self, package_name):
        script = r"""
import copy
import importlib
import json
import pickle
import sys

module = importlib.import_module(sys.argv[1])
compiler = importlib.import_module(f"{sys.argv[1]}.compiler")
old_function = compiler.allow_in_graph
old_exports = compiler.__all__
reloaded = importlib.reload(compiler)
new_function = reloaded.allow_in_graph

try:
    pickle.dumps(old_function)
except BaseException as error:
    old_pickle_error = [type(error).__name__, "not the same object" in str(error)]
else:
    old_pickle_error = None

new_pickle_results = [
    pickle.loads(pickle.dumps(new_function, protocol)) is new_function
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
]
converted = new_function((len,))
print(json.dumps([
    reloaded is compiler,
    module.compiler is compiler,
    old_function is new_function,
    old_exports is compiler.__all__,
    converted[0] is len,
    copy.copy(old_function) is old_function,
    copy.deepcopy(old_function) is old_function,
    old_pickle_error,
    new_pickle_results,
]))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, package_name],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        return json.loads(completed.stdout)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome("torch_rs"),
            self.reload_outcome("torch"),
        )


if __name__ == "__main__":
    unittest.main()
