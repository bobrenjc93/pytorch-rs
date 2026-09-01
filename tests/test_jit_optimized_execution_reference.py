import contextlib
import copy
import importlib
import inspect
import pickle
import pickletools
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


WILDCARD_SUPPORTED = {
    "Attribute",
    "annotate",
    "export",
    "ignore",
    "isinstance",
    "onednn_fusion_enabled",
    "script_if_tracing",
    "strict_fusion",
    "unused",
}


@torch.jit.optimized_execution(True)
def _actual_picklable_optimized_execution_function(value):
    return value


if reference_torch is not None:

    @reference_torch.jit.optimized_execution(True)
    def _expected_picklable_optimized_execution_function(value):
        return value

else:
    _expected_picklable_optimized_execution_function = None


class BoolLike:
    def __init__(self, value, events=None, label=None):
        self.value = value
        self.events = events
        self.label = label

    def __bool__(self):
        if self.events is not None:
            self.events.append(self.label)
        return self.value

    def __repr__(self):
        return f"BoolLike({self.value!r})"


class BadBool:
    def __bool__(self):
        raise RuntimeError("bool failed")

    def __repr__(self):
        return "BadBool()"


class KeyboardInterruptBool:
    def __bool__(self):
        raise KeyboardInterrupt("bool interrupted")

    def __repr__(self):
        return "KeyboardInterruptBool()"


class SystemExitBool:
    def __bool__(self):
        raise SystemExit("bool exited")

    def __repr__(self):
        return "SystemExitBool()"


class LenOnly:
    def __len__(self):
        return 1

    def __repr__(self):
        return "LenOnly()"


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitOptimizedExecutionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.optimized_execution differentials require pinned PyTorch 2.13.0"
            )

    def normalize(self, value):
        if isinstance(value, str):
            return value.replace("torch_rs", "torch")
        if isinstance(value, tuple):
            return tuple(self.normalize(item) for item in value)
        if isinstance(value, list):
            return [self.normalize(item) for item in value]
        if isinstance(value, dict):
            return {
                self.normalize(key): self.normalize(item)
                for key, item in value.items()
            }
        return value

    def error_outcome(self, call):
        try:
            call()
        except Exception as error:
            return (
                type(error).__module__,
                type(error).__qualname__,
                self.normalize(str(error)),
                self.normalize(error.args),
            )
        return None

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

    def context_outcome(self, module):
        events = []
        states = [module.is_grad_enabled()]
        valid_values = (
            True,
            False,
            1,
            0,
            None,
            BoolLike(True, events, "truthy object"),
            BoolLike(False, events, "falsey object"),
        )
        entries = []
        exits = []
        for value in valid_values:
            context = module.jit.optimized_execution(value)
            entries.append(context.__enter__())
            states.append(module.is_grad_enabled())
            exits.append(context.__exit__(None, None, None))
            states.append(module.is_grad_enabled())

        with module.jit.optimized_execution(BoolLike(True, events, "outer")) as outer:
            with module.jit.optimized_execution(
                BoolLike(False, events, "inner")
            ) as inner:
                nested = (outer, inner, module.is_grad_enabled())

        error = RuntimeError("forwarded failure")
        try:
            with module.jit.optimized_execution(
                BoolLike(True, events, "exception")
            ):
                raise error
        except RuntimeError as raised:
            propagated = raised is error
        else:
            propagated = False

        with module.no_grad():
            states.append(module.is_grad_enabled())
            with module.jit.optimized_execution(True) as no_grad_entered:
                states.append(module.is_grad_enabled())
            states.append(module.is_grad_enabled())
        states.append(module.is_grad_enabled())

        return (
            entries,
            exits,
            events,
            nested,
            propagated,
            no_grad_entered,
            states,
        )

    def invalid_outcome(self, module):
        results = []
        for value in (
            [],
            [1],
            "",
            "yes",
            BadBool(),
            KeyboardInterruptBool(),
            SystemExitBool(),
            LenOnly(),
        ):
            context = module.jit.optimized_execution(value)
            results.append(
                (
                    type(context).__module__,
                    type(context).__qualname__,
                    self.error_outcome(context.__enter__),
                    context.__exit__(None, None, None),
                )
            )
        return self.normalize(results)

    def decorator_outcome(self, module, should_optimize):
        calls = []
        events = []

        def function(
            value: int,
            /,
            scale: int = 2,
            *,
            label: str = "value",
        ) -> tuple[int, int, str]:
            """Function documentation."""
            calls.append((value, scale, label))
            return value, scale, label

        function.custom_attribute = "custom"
        value = (
            BoolLike(True, events, "decorator")
            if should_optimize == "bool-like"
            else should_optimize
        )
        wrapped = module.jit.optimized_execution(value)(function)
        result = wrapped(3, label="scaled")
        return (
            type(wrapped) is types.FunctionType,
            wrapped is function,
            wrapped.__name__,
            wrapped.__qualname__ == function.__qualname__,
            wrapped.__module__ == function.__module__,
            wrapped.__doc__,
            wrapped.__annotations__,
            typing.get_type_hints(wrapped),
            wrapped.__defaults__,
            wrapped.__kwdefaults__,
            wrapped.custom_attribute,
            wrapped.__wrapped__ is function,
            inspect.unwrap(wrapped) is function,
            inspect.signature(wrapped) == inspect.signature(function),
            set(wrapped.__dict__),
            result,
            calls,
            events,
        )

    def invalid_decorator_outcome(self, module):
        calls = []

        def function():
            calls.append("called")

        wrapped = module.jit.optimized_execution("yes")(function)
        return self.error_outcome(wrapped), calls

    def threaded_grad_outcome(self, module):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        outcomes = [None] * worker_count
        errors = []

        def worker(index):
            try:
                grad_context = (
                    module.no_grad() if index % 2 else contextlib.nullcontext()
                )
                with grad_context:
                    barrier.wait(timeout=10)
                    before = module.is_grad_enabled()
                    with module.jit.optimized_execution(index % 2 == 0) as entered:
                        inside = module.is_grad_enabled()
                    outcomes[index] = (
                        before,
                        inside,
                        module.is_grad_enabled(),
                        entered,
                    )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        return (
            outcomes,
            sorted(errors),
            [thread.is_alive() for thread in threads],
            module.is_grad_enabled(),
        )

    def metadata_outcome(self, module):
        jit = importlib.import_module(f"{module.__name__}.jit")
        fuser = importlib.import_module(f"{module.__name__}.jit._fuser")
        function = jit.optimized_execution
        wrapped = function.__wrapped__
        return self.normalize(
            (
                jit.optimized_execution is fuser.optimized_execution,
                type(function).__name__,
                str(inspect.signature(function)),
                function.__annotations__,
                typing.get_type_hints(function),
                function.__name__,
                function.__qualname__,
                function.__module__,
                inspect.getmodule(function).__name__,
                inspect.cleandoc(function.__doc__),
                function.__defaults__,
                function.__kwdefaults__,
                set(function.__dict__),
                hasattr(function, "__text_signature__"),
                type(wrapped).__name__,
                str(inspect.signature(wrapped)),
                wrapped.__annotations__,
                typing.get_type_hints(wrapped),
                wrapped.__name__,
                wrapped.__qualname__,
                wrapped.__module__,
                inspect.getmodule(wrapped).__name__,
                inspect.cleandoc(wrapped.__doc__),
                wrapped.__defaults__,
                wrapped.__kwdefaults__,
                wrapped.__dict__,
            )
        )

    def exports_outcome(self, module):
        jit = importlib.import_module(f"{module.__name__}.jit")
        fuser = importlib.import_module(f"{module.__name__}.jit._fuser")
        function = jit.optimized_execution
        explicit_namespace = {}
        exec(
            f"from {module.__name__}.jit import optimized_execution",
            explicit_namespace,
        )
        wildcard_namespace = {}
        exec(f"from {module.__name__}.jit import *", wildcard_namespace)
        fuser_namespace = {}
        exec(
            f"from {module.__name__}.jit._fuser import optimized_execution",
            fuser_namespace,
        )
        top_level_namespace = {}
        exec(f"from {module.__name__} import *", top_level_namespace)
        return self.normalize(
            (
                "optimized_execution" in jit.__all__,
                "optimized_execution" in vars(jit),
                "optimized_execution" in wildcard_namespace,
                explicit_namespace["optimized_execution"] is function,
                fuser_namespace["optimized_execution"] is function,
                "optimized_execution" in module.__all__,
                hasattr(module, "optimized_execution"),
                "optimized_execution" in top_level_namespace,
                hasattr(fuser, "__all__"),
                "optimized_execution" in vars(fuser),
            )
        )

    def copy_pickle_outcome(self, module, decorated):
        function = module.jit.optimized_execution
        context = function(True)
        context.value = {"items": [1, 2]}
        shallow = copy.copy(context)
        results = [
            copy.copy(function) is function,
            copy.deepcopy(function) is function,
            copy.copy(decorated) is decorated,
            copy.deepcopy(decorated) is decorated,
            type(context).__module__,
            type(context).__qualname__,
            shallow is context,
            type(shallow) is type(context),
            shallow.__dict__["gen"] is context.__dict__["gen"],
            shallow.__dict__["func"].__module__,
            shallow.__dict__["func"].__name__,
            shallow.__dict__["args"],
            shallow.__dict__["kwds"],
            shallow.value is context.value,
            self.error_outcome(lambda: copy.deepcopy(context)),
        ]
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            results.append(self.pickle_shape(function, protocol))
            results.append(
                pickle.loads(pickle.dumps(function, protocol)) is function
            )
            results.append(
                pickle.loads(pickle.dumps(decorated, protocol)) is decorated
            )
            results.append(
                self.error_outcome(
                    lambda protocol=protocol: pickle.dumps(context, protocol)
                )
            )
        return self.normalize(results)

    def reload_outcome(self, module):
        jit = importlib.import_module(f"{module.__name__}.jit")
        fuser = importlib.import_module(f"{module.__name__}.jit._fuser")
        original = jit.optimized_execution
        reloaded_fuser = importlib.reload(fuser)
        fuser_function = reloaded_fuser.optimized_execution
        jit_still_has_original = jit.optimized_execution is original
        package_fuser_has_reloaded = jit._fuser.optimized_execution is fuser_function
        reloaded_jit = importlib.reload(jit)
        return (
            fuser_function is original,
            jit_still_has_original,
            package_fuser_has_reloaded,
            reloaded_jit.optimized_execution is fuser_function,
            module.jit.optimized_execution is fuser_function,
        )

    def test_context_decorator_validation_and_grad_state_match(self):
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )
        self.assertEqual(
            self.invalid_outcome(torch),
            self.invalid_outcome(reference_torch),
        )
        for value in (True, False, 1, 0, None, "bool-like"):
            with self.subTest(value=value):
                self.assertEqual(
                    self.decorator_outcome(torch, value),
                    self.decorator_outcome(reference_torch, value),
                )
        self.assertEqual(
            self.invalid_decorator_outcome(torch),
            self.invalid_decorator_outcome(reference_torch),
        )
        self.assertEqual(
            self.threaded_grad_outcome(torch),
            self.threaded_grad_outcome(reference_torch),
        )

    def test_signature_metadata_imports_copy_pickle_and_reload_match(self):
        self.assertEqual(
            self.metadata_outcome(torch),
            self.metadata_outcome(reference_torch),
        )
        self.assertEqual(
            self.exports_outcome(torch),
            self.exports_outcome(reference_torch),
        )
        self.assertEqual(
            {name for name in vars(torch.jit) if not name.startswith("_")},
            {
                *WILDCARD_SUPPORTED,
                "is_scripting",
                "is_tracing",
                "optimized_execution",
            },
        )
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            WILDCARD_SUPPORTED,
        )
        self.assertNotIn("optimized_execution", expected_namespace)
        self.assertEqual(
            self.copy_pickle_outcome(
                torch,
                _actual_picklable_optimized_execution_function,
            ),
            self.copy_pickle_outcome(
                reference_torch,
                _expected_picklable_optimized_execution_function,
            ),
        )
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_call_errors_and_unsupported_execution_surface_match_scope(self):
        actual = torch.jit.optimized_execution
        expected = reference_torch.jit.optimized_execution
        cases = (
            lambda function: function(),
            lambda function: function(True, False),
            lambda function: function(value=True),
            lambda function: function(True, should_optimize=False),
            lambda function: function(should_optimize=True, value=False),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assertEqual(
                    self.error_outcome(lambda: call(actual)),
                    self.error_outcome(lambda: call(expected)),
                )

        self.assertTrue(callable(torch.jit.optimized_execution))
        self.assertTrue(callable(reference_torch.jit.optimized_execution))
        self.assertNotIn("optimized_execution", torch.jit.__all__)
        self.assertNotIn("optimized_execution", reference_torch.jit.__all__)
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "enable_onednn_fusion",
            "fork",
            "last_executed_optimized_graph",
            "script",
            "set_fusion_strategy",
            "trace",
            "trace_module",
            "wait",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))
        for name in (
            "_get_graph_executor_optimize",
            "_set_graph_executor_optimize",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch._C, name))
                self.assertFalse(hasattr(torch._C, name))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
