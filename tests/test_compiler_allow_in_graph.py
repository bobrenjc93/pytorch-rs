import copy
import gc
import importlib
import inspect
import operator
import pickle
import subprocess
import sys
import types
import unittest
import weakref

import torch_rs as torch


FUNCTION_DOC = """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    If you are using :func:`torch.compile` (with backend="inductor" (the default)), or
    :func:`torch.export.export`, and trying to black-box a Python function throughout
    all tracing, do not use this API.
    Instead, please create a custom operator (see `PyTorch Custom Operators Landing Page
    <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_)

    .. warning::

        If you're a typical torch.compile user (e.g. you're applying torch.compile to
        a model to make it run faster), you probably don't want to use this function.
        :func:`allow_in_graph` is a footgun because it skips the compiler frontend
        (Dynamo) that is responsible for doing safety checks (graph breaks, handling
        closures, etc). Incorrect usage will lead to difficult-to-debug silent
        incorrectness issues.

    Given a Python function with no allow_in_graph decorator, regular execution
    of torch.compile traces through the function. :func:`allow_in_graph` changes
    it so that the frontend does not trace inside the function, but the compiler
    backend still traces through it. Compare this to custom operators, which
    treats a function as a black box throughout the torch.compile stack. The following
    table compares these mechanisms.

    +------------------------+-----------------------+--------------------------------+
    | Mechanism              | Frontend (Dynamo)     | Backend (AOTAutograd+Inductor) |
    +========================+=======================+================================+
    | no decorator           | trace inside          | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | allow_in_graph         | opaque callable       | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | custom op              | opaque callable       | opaque callable                |
    +------------------------+-----------------------+--------------------------------+

    One common use case for :func:`allow_in_graph()` is as an escape hatch for the compiler
    frontend: if you know the function works w.r.t. to the downstream components of the
    compilation stack (AOTAutograd and Inductor) but there is a Dynamo bug that prevents it from
    symbolically introspecting the function properly (or if your code is in C/C++ and
    therefore cannot be introspected with Dynamo), then one can decorate said function
    with :func:`allow_in_graph` to bypass Dynamo.

    We require that ``fn`` adhere to the following restrictions. Failure to adhere
    results in undefined behavior:

    - The inputs to ``fn`` must be Proxy-able types in the FX graph. Valid types include:
      Tensor/int/bool/float/None/List[Tensor?]/List[int?]/List[float?]
      Tuple[Tensor?, ...]/Tuple[int?, ...]/Tuple[float?, ...]/torch.dtype/torch.device
    - The outputs to ``fn`` must be Proxy-able types in the FX graph (see previous bullet)
    - all Tensors used inside of ``fn`` must be passed directly as inputs to ``fn``
      (as opposed to being captured variables).

    Args:
        fn: A callable representing the function to be included in the graph.
            If ``fn`` is a list or tuple of callables it recursively applies
            :func:`allow_in_graph()` to each function and returns a new list or
            tuple containing the modified functions.

    Example::

        torch.compiler.allow_in_graph(my_custom_function)


        @torch.compile(...)
        def fn(x):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will capture a single graph containing ``my_custom_function()``.

    """


COMPILER_EXPORTS = [
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
]


@torch.compiler.allow_in_graph
def _picklable_allowed_function(value, *, increment=1):
    return value + increment


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


class _BuiltinLookupProbe(_LookupProbe):
    __hash__ = _LookupProbe.__hash__

    def __eq__(self, other):
        self.events.append(("eq", self.label, other is dict))
        return other is dict


class _BaseModuleProbe:
    def __init__(self, events, outcome):
        self.events = events
        self.outcome = outcome

    def __hash__(self):
        self.events.append(("base_hash",))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _ModuleNameProbe:
    def __init__(self, events, base_module):
        self.events = events
        self.base_module = base_module

    def split(self, separator):
        self.events.append(("split", separator))
        return [self.base_module]


class _ObservedList(list):
    def __init__(self, values, events):
        super().__init__(values)
        self.events = events

    def __iter__(self):
        for index, value in enumerate(super().__iter__()):
            self.events.append(("yield", index))
            yield value
        self.events.append(("finished", len(self)))


class CompilerAllowInGraphTests(unittest.TestCase):
    def lookup_outcome(self, hash_outcomes, module_outcomes):
        events = []
        target = _LookupProbe("target", hash_outcomes, module_outcomes, events)
        try:
            result = torch.compiler.allow_in_graph(target)
        except BaseException as error:
            outcome = (type(error), str(error), error.args)
        else:
            outcome = ("ok", result is target)
        return events, outcome

    def test_decorator_preserves_function_identity_metadata_and_eager_calls(self):
        calls = []
        sentinel = object()

        def calculate(value: int, *, scale: int = 1) -> int:
            """Calculate eagerly."""
            calls.append((value, scale))
            return value * scale + len(calls)

        calculate.custom_attribute = sentinel
        before = dict(calculate.__dict__)
        result = torch.compiler.allow_in_graph(calculate)

        self.assertIs(result, calculate)
        self.assertEqual(calculate.__dict__, before)
        self.assertIs(calculate.custom_attribute, sentinel)
        self.assertEqual(calculate(3, scale=2), 7)
        self.assertEqual(calculate(3, scale=2), 8)
        self.assertEqual(calls, [(3, 2), (3, 2)])
        self.assertEqual(
            str(inspect.signature(calculate)), "(value: int, *, scale: int = 1) -> int"
        )
        self.assertEqual(calculate.__name__, "calculate")
        self.assertEqual(calculate.__doc__, "Calculate eagerly.")
        self.assertFalse(hasattr(calculate, "_dynamo_marked_constant"))
        self.assertFalse(hasattr(calculate, "_torchdynamo_disable"))

    def test_methods_and_callable_objects_keep_eager_behavior(self):
        class Accumulator:
            def __init__(self):
                self.total = 0

            @torch.compiler.allow_in_graph
            def add(self, value):
                self.total += value
                return self.total

        raw_method = Accumulator.__dict__["add"]
        left = Accumulator()
        right = Accumulator()
        bound = left.add
        self.assertIs(torch.compiler.allow_in_graph(bound), bound)
        self.assertIs(Accumulator.add, raw_method)
        self.assertIs(bound.__self__, left)
        self.assertIs(bound.__func__, raw_method)
        self.assertEqual(left.add(2), 2)
        self.assertEqual(left.add(3), 5)
        self.assertEqual(right.add(7), 7)

        target = _CallableTarget()
        self.assertIs(torch.compiler.allow_in_graph(target), target)
        self.assertEqual(target(6), 12)

    def test_lists_tuples_and_nested_sequences_become_fresh_lists(self):
        def first():
            return "first"

        target = _CallableTarget()
        inner_tuple = (target, len)
        inner_list = [first]
        source = [first, inner_tuple, inner_list]

        result = torch.compiler.allow_in_graph(source)

        self.assertIs(type(result), list)
        self.assertIsNot(result, source)
        self.assertIs(result[0], first)
        self.assertIs(type(result[1]), list)
        self.assertIsNot(result[1], inner_tuple)
        self.assertIs(result[1][0], target)
        self.assertIs(result[1][1], len)
        self.assertIs(type(result[2]), list)
        self.assertIsNot(result[2], inner_list)
        self.assertIs(result[2][0], first)

        for empty in ([], ()):
            converted = torch.compiler.allow_in_graph(empty)
            self.assertIs(type(converted), list)
            self.assertEqual(converted, [])
            self.assertIsNot(converted, empty)

    def test_sequence_validation_is_left_to_right_and_stops_at_failure(self):
        def accepted():
            return None

        events = []
        values = _ObservedList((accepted, None, _SlotCallable()), events)
        with self.assertRaisesRegex(
            AssertionError, "^allow_in_graph expects a callable$"
        ):
            torch.compiler.allow_in_graph(values)
        self.assertEqual(events, [("yield", 0), ("yield", 1)])

        state = importlib.import_module("torch_rs._compiler_state")
        slot_target = _SlotCallable()
        try:
            with self.assertRaisesRegex(
                TypeError, "^cannot create weak reference to '_SlotCallable' object$"
            ):
                torch.compiler.allow_in_graph((slot_target, None))
        finally:
            state.allow_in_graph_callable_ids.discard(id(slot_target))
        with self.assertRaisesRegex(
            AssertionError, "^allow_in_graph expects a callable$"
        ):
            torch.compiler.allow_in_graph((None, _SlotCallable()))

    def test_lookup_validation_preserves_hash_and_module_side_effects(self):
        self.assertEqual(
            self.lookup_outcome((101, 101), ("sample.module", "sample.module")),
            (
                [
                    ("hash", "target", 0),
                    ("module", "target", 0),
                    ("module", "target", 1),
                    ("hash", "target", 1),
                ],
                ("ok", True),
            ),
        )

        for error_type in (TypeError, ValueError):
            with self.subTest(error_type=error_type.__name__):
                self.assertEqual(
                    self.lookup_outcome((error_type("unhashable"),), ()),
                    ([("hash", "target", 0)], ("ok", True)),
                )

    def test_lookup_validation_propagates_hash_and_module_errors(self):
        cases = (
            (
                (RuntimeError("first hash"),),
                (),
                [("hash", "target", 0)],
                RuntimeError,
                "first hash",
            ),
            (
                (101,),
                (RuntimeError("first module"),),
                [("hash", "target", 0), ("module", "target", 0)],
                RuntimeError,
                "first module",
            ),
            (
                (101,),
                ("sample.module", RuntimeError("second module")),
                [
                    ("hash", "target", 0),
                    ("module", "target", 0),
                    ("module", "target", 1),
                ],
                RuntimeError,
                "second module",
            ),
            (
                (101, RuntimeError("second hash")),
                ("sample.module", "sample.module"),
                [
                    ("hash", "target", 0),
                    ("module", "target", 0),
                    ("module", "target", 1),
                    ("hash", "target", 1),
                ],
                RuntimeError,
                "second hash",
            ),
            (
                (101, TypeError("second hash")),
                ("sample.module", "sample.module"),
                [
                    ("hash", "target", 0),
                    ("module", "target", 0),
                    ("module", "target", 1),
                    ("hash", "target", 1),
                ],
                TypeError,
                "second hash",
            ),
            (
                (101,),
                (42,),
                [("hash", "target", 0), ("module", "target", 0)],
                AttributeError,
                "'int' object has no attribute 'split'",
            ),
        )
        for hash_outcomes, module_outcomes, events, error_type, message in cases:
            with self.subTest(message=message):
                actual_events, outcome = self.lookup_outcome(
                    hash_outcomes, module_outcomes
                )
                self.assertEqual(actual_events, events)
                self.assertIs(outcome[0], error_type)
                self.assertEqual(outcome[1], message)
                self.assertEqual(outcome[2], (message,))

    def test_builtin_hit_and_lazy_module_key_lookups_are_observable(self):
        builtin_events = []
        builtin_target = _BuiltinLookupProbe(
            "builtin",
            (hash(dict), hash(dict), RuntimeError("third hash")),
            ("sample.module", "sample.module"),
            builtin_events,
        )
        with self.assertRaisesRegex(RuntimeError, "^third hash$"):
            torch.compiler.allow_in_graph(builtin_target)
        self.assertEqual(
            builtin_events,
            [
                ("hash", "builtin", 0),
                ("module", "builtin", 0),
                ("module", "builtin", 1),
                ("hash", "builtin", 1),
                ("eq", "builtin", True),
                ("hash", "builtin", 2),
            ],
        )

        state = importlib.import_module("torch_rs._compiler_state")
        sentinel = "__torch_rs_allow_in_graph_test__"
        state.allow_in_graph_lazy_modules[sentinel] = None
        try:
            module_events = []
            base_module = _BaseModuleProbe(
                module_events, RuntimeError("base module hash")
            )
            module_name = _ModuleNameProbe(module_events, base_module)
            module_target = _LookupProbe(
                "module", (101,), (module_name,), module_events
            )
            with self.assertRaisesRegex(RuntimeError, "^base module hash$"):
                torch.compiler.allow_in_graph(module_target)
            self.assertEqual(
                module_events,
                [
                    ("hash", "module", 0),
                    ("module", "module", 0),
                    ("split", "."),
                    ("base_hash",),
                ],
            )
        finally:
            state.allow_in_graph_lazy_modules.pop(sentinel, None)

    def test_repeated_and_duplicate_callables_skip_completed_lookup(self):
        for use_collection in (False, True):
            with self.subTest(use_collection=use_collection):
                events = []
                target = _LookupProbe(
                    "target",
                    (101, 101, 101, RuntimeError("fourth hash")),
                    (
                        "sample.module",
                        "sample.module",
                        "sample.module",
                        "sample.module",
                    ),
                    events,
                )
                if use_collection:
                    result = torch.compiler.allow_in_graph([target, target])
                    self.assertEqual(len(result), 2)
                    self.assertIs(result[0], target)
                    self.assertIs(result[1], target)
                else:
                    self.assertIs(torch.compiler.allow_in_graph(target), target)
                    self.assertIs(torch.compiler.allow_in_graph(target), target)

                self.assertEqual(
                    events,
                    [
                        ("hash", "target", 0),
                        ("module", "target", 0),
                        ("module", "target", 1),
                        ("hash", "target", 1),
                        ("hash", "target", 2),
                        ("module", "target", 2),
                        ("module", "target", 3),
                    ],
                )

    def test_nonweakrefable_callable_registration_matches_pytorch_lifecycle(self):
        state = importlib.import_module("torch_rs._compiler_state")
        target = operator.methodcaller("upper")
        target_id = id(target)
        state.allow_in_graph_callable_ids.discard(target_id)
        try:
            with self.assertRaisesRegex(
                TypeError,
                "^cannot create weak reference to 'operator.methodcaller' object$",
            ):
                torch.compiler.allow_in_graph(target)
            self.assertIn(target_id, state.allow_in_graph_callable_ids)
            self.assertIs(torch.compiler.allow_in_graph(target), target)
            self.assertEqual(target("value"), "VALUE")
        finally:
            state.allow_in_graph_callable_ids.discard(target_id)

    def test_weak_identity_registration_is_removed_after_collection(self):
        state = importlib.import_module("torch_rs._compiler_state")
        target = _CallableTarget()
        target_id = id(target)
        target_ref = weakref.ref(target)

        result = torch.compiler.allow_in_graph(target)
        self.assertIs(result, target)
        self.assertIn(target_id, state.allow_in_graph_callable_ids)

        del result
        del target
        gc.collect()

        self.assertIsNone(target_ref())
        self.assertNotIn(target_id, state.allow_in_graph_callable_ids)

    def test_collection_lookup_failures_stop_before_later_entries(self):
        events = []
        accepted = _LookupProbe(
            "accepted", (101, 101), ("sample.module", "sample.module"), events
        )
        failing = _LookupProbe("failing", (RuntimeError("hash failed"),), (), events)
        unreached = _LookupProbe(
            "unreached", (101, 101), ("sample.module", "sample.module"), events
        )
        values = _ObservedList((accepted, failing, unreached), events)

        with self.assertRaisesRegex(RuntimeError, "^hash failed$"):
            torch.compiler.allow_in_graph(values)

        self.assertEqual(
            events,
            [
                ("yield", 0),
                ("hash", "accepted", 0),
                ("module", "accepted", 0),
                ("module", "accepted", 1),
                ("hash", "accepted", 1),
                ("yield", 1),
                ("hash", "failing", 0),
            ],
        )

    def test_invalid_targets_and_call_shapes_match_pytorch_2_13(self):
        function = torch.compiler.allow_in_graph
        invalid_targets = (None, 1, "callable", object(), property())
        for target in invalid_targets:
            with self.subTest(target=type(target).__name__):
                with self.assertRaises(AssertionError) as raised:
                    function(target)
                message = "allow_in_graph expects a callable"
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        generator = (value for value in (len,))
        with self.assertRaisesRegex(
            AssertionError, "^allow_in_graph expects a callable$"
        ):
            function(generator)

        state = importlib.import_module("torch_rs._compiler_state")
        slot_target = _SlotCallable()
        try:
            with self.assertRaises(TypeError) as raised:
                function(slot_target)
            message = "cannot create weak reference to '_SlotCallable' object"
            self.assertEqual(str(raised.exception), message)
            self.assertEqual(raised.exception.args, (message,))
        finally:
            state.allow_in_graph_callable_ids.discard(id(slot_target))

        target = lambda: None
        cases = (
            (
                lambda: function(),
                "allow_in_graph() missing 1 required positional argument: 'fn'",
            ),
            (
                lambda: function(target, target),
                "allow_in_graph() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(target, fn=target),
                "allow_in_graph() got multiple values for argument 'fn'",
            ),
            (
                lambda: function(function=target),
                "allow_in_graph() got an unexpected keyword argument 'function'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_signature_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.allow_in_graph

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "(fn)")
        self.assertEqual(function.__annotations__, {})
        self.assertEqual(function.__name__, "allow_in_graph")
        self.assertEqual(function.__qualname__, "allow_in_graph")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copying_and_pickling_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.allow_in_graph

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("allow_in_graph", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("allow_in_graph", top_level_namespace)
        self.assertFalse(hasattr(torch, "allow_in_graph"))

        for value in (function, _picklable_allowed_function):
            self.assertIs(copy.copy(value), value)
            self.assertIs(copy.deepcopy(value), value)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=value.__name__, protocol=protocol):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIs(pickle.loads(payload), value)

        self.assertEqual(_picklable_allowed_function(4, increment=3), 7)

    def test_reload_replaces_only_the_module_function(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            old_function = compiler.allow_in_graph
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = compiler.allow_in_graph

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertIs(
                new_function(_picklable_allowed_function), _picklable_allowed_function
            )
            self.assertEqual(_picklable_allowed_function(2), 3)
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

    def test_calls_do_not_initialize_or_claim_compiler_execution(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

calls = []

@torch.compiler.allow_in_graph
def function(value):
    calls.append(value)
    return (
        value + 1,
        torch.compiler.is_compiling(),
        torch.compiler.is_dynamo_compiling(),
        torch.compiler.is_exporting(),
    )

source = (function, len)
result = torch.compiler.allow_in_graph(source)
assert result == [function, len]
assert result is not source
assert function(1) == (2, False, False, False)
assert function(2) == (3, False, False, False)
assert calls == [1, 2]
assert function.__dict__ == {}
assert not hasattr(torch, "compile")
assert not hasattr(torch, "export")
assert not hasattr(torch.compiler, "compile")
assert not hasattr(torch.compiler, "_dynamo")
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
