import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _ContextBodyError(Exception):
    pass


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("truth-value error should be replaced")


class _TruthValue:
    def __init__(self, value):
        self.value = value

    def __bool__(self):
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class OptimizedExecutionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.optimized_execution differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.jit._fuser")
        self.expected = importlib.import_module("torch.jit._fuser")
        self.original_actual = torch._C._get_graph_executor_optimize()
        self.original_expected = (
            reference_torch._C._get_graph_executor_optimize()
        )
        torch._C._set_graph_executor_optimize(True)
        reference_torch._C._set_graph_executor_optimize(True)

    def tearDown(self):
        torch._C._set_graph_executor_optimize(self.original_actual)
        reference_torch._C._set_graph_executor_optimize(
            self.original_expected
        )

    def normalize(self, value):
        value = str(value).replace("torch_rs", "torch")
        return re.sub(r"0x[0-9a-fA-F]+", "0x...", value)

    def capture_error(self, call):
        try:
            call()
        except Exception as error:
            return (
                type(error).__name__,
                self.normalize(error),
                tuple(self.normalize(argument) for argument in error.args),
            )
        self.fail("expected the call to fail")

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

    def behavior_contract(self, root, module):
        outcomes = []
        for initial in (False, True):
            for requested in (False, True, None, 0, 1):
                root._C._set_graph_executor_optimize(initial)
                context = module.optimized_execution(requested)
                before_entry = root._C._get_graph_executor_optimize()
                entered = context.__enter__()
                inside = root._C._get_graph_executor_optimize()
                exit_result = context.__exit__(None, None, None)
                outcomes.append(
                    (
                        before_entry,
                        entered,
                        inside,
                        exit_result,
                        root._C._get_graph_executor_optimize(),
                    )
                )

        root._C._set_graph_executor_optimize(True)
        with module.optimized_execution(False) as outer:
            outer_state = root._C._get_graph_executor_optimize()
            with module.optimized_execution(True) as inner:
                inner_state = root._C._get_graph_executor_optimize()
            after_inner = root._C._get_graph_executor_optimize()
        outcomes.append(
            (
                outer,
                outer_state,
                inner,
                inner_state,
                after_inner,
                root._C._get_graph_executor_optimize(),
            )
        )

        marker = _ContextBodyError("body failed")
        try:
            with module.optimized_execution(False) as entered:
                state_before_error = root._C._get_graph_executor_optimize()
                raise marker
        except Exception as error:
            outcomes.append(
                (
                    entered,
                    state_before_error,
                    error is marker,
                    type(error).__name__,
                    error.args,
                    root._C._get_graph_executor_optimize(),
                )
            )
        else:
            self.fail("the context suppressed its body exception")

        observations = []

        @module.optimized_execution(False)
        def decorated(value):
            observations.append(root._C._get_graph_executor_optimize())
            return value

        outcomes.append(
            (
                decorated("first"),
                root._C._get_graph_executor_optimize(),
                decorated("second"),
                root._C._get_graph_executor_optimize(),
                observations,
            )
        )
        return outcomes

    def validation_contract(self, root, module):
        outcomes = []
        invalid_values = ("", "enabled", [], object(), _RejectTruthiness())
        for state in (False, True):
            for value in invalid_values:
                root._C._set_graph_executor_optimize(state)
                context = module.optimized_execution(value)
                outcomes.append(
                    (
                        root._C._get_graph_executor_optimize(),
                        self.capture_error(context.__enter__),
                        root._C._get_graph_executor_optimize(),
                    )
                )

        for value in (
            0.0,
            1.5,
            range(0),
            range(1),
            _TruthValue(False),
            _TruthValue(True),
        ):
            root._C._set_graph_executor_optimize(True)
            with module.optimized_execution(value) as entered:
                outcomes.append(
                    (entered, root._C._get_graph_executor_optimize())
                )
            outcomes.append(root._C._get_graph_executor_optimize())

        root._C._set_graph_executor_optimize(True)
        for call in (
            lambda: module.optimized_execution(),
            lambda: module.optimized_execution(True, False),
            lambda: module.optimized_execution(enabled=True),
            lambda: module.optimized_execution(
                False, should_optimize=True
            ),
        ):
            outcomes.append(
                (
                    self.capture_error(call),
                    root._C._get_graph_executor_optimize(),
                )
            )
        return outcomes

    def thread_contract(self, root, module):
        root._C._set_graph_executor_optimize(False)
        worker_entered = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(
                    ("worker-default", root._C._get_graph_executor_optimize())
                )
                with module.optimized_execution(False) as entered:
                    observations.append(
                        (
                            "worker-enter",
                            entered,
                            root._C._get_graph_executor_optimize(),
                        )
                    )
                    worker_entered.set()
                    if not main_changed.wait(timeout=10):
                        raise RuntimeError("timed out waiting for main thread")
                    observations.append(
                        (
                            "worker-resume",
                            root._C._get_graph_executor_optimize(),
                        )
                    )
                observations.append(
                    ("worker-exit", root._C._get_graph_executor_optimize())
                )
            except BaseException as error:
                errors.append((type(error).__name__, self.normalize(error)))
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        ready = worker_entered.wait(timeout=10)
        main_before = root._C._get_graph_executor_optimize()
        root._C._set_graph_executor_optimize(True)
        try:
            main_after = root._C._get_graph_executor_optimize()
        finally:
            main_changed.set()
            thread.join(timeout=10)

        return (
            ready,
            main_before,
            main_after,
            not thread.is_alive(),
            errors,
            observations,
            root._C._get_graph_executor_optimize(),
        )

    def context_metadata_contract(self, module):
        context = module.optimized_execution(False)
        shallow = copy.copy(context)
        deep_error = self.capture_error(lambda: copy.deepcopy(context))
        pickle_errors = [
            self.capture_error(
                lambda protocol=protocol: pickle.dumps(context, protocol=protocol)
            )
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        ]
        return (
            type(context).__module__,
            type(context).__qualname__,
            context.__doc__,
            context.func is module.optimized_execution.__wrapped__,
            context.args,
            context.kwds,
            shallow is not context,
            shallow.gen is context.gen,
            deep_error,
            pickle_errors,
        )

    def reload_contract(self, root, module):
        jit = root.jit
        root._C._set_graph_executor_optimize(True)
        old_function = module.optimized_execution
        old_wrapped = old_function.__wrapped__
        namespace = module.__dict__
        active_context = old_function(False)
        entry_result = active_context.__enter__()
        state_before_reload = root._C._get_graph_executor_optimize()
        reloaded = importlib.reload(module)
        state_after_reload = root._C._get_graph_executor_optimize()
        package_still_has_old = jit.optimized_execution is old_function
        exit_result = active_context.__exit__(None, None, None)
        state_after_exit = root._C._get_graph_executor_optimize()

        context_results = []
        for function in (old_function, module.optimized_execution):
            with function(False) as entered:
                context_results.append(
                    (entered, root._C._get_graph_executor_optimize())
                )
            context_results.append(root._C._get_graph_executor_optimize())

        old_pickle_error = self.capture_error(lambda: pickle.dumps(old_function))
        new_pickle = pickle.loads(pickle.dumps(module.optimized_execution))
        jit_namespace = jit.__dict__
        reloaded_jit = importlib.reload(jit)
        return (
            entry_result,
            state_before_reload,
            reloaded is module,
            module.__dict__ is namespace,
            sys.modules[module.__name__] is module,
            module.optimized_execution is not old_function,
            module.optimized_execution.__wrapped__ is not old_wrapped,
            package_still_has_old,
            state_after_reload,
            exit_result,
            state_after_exit,
            context_results,
            old_pickle_error,
            new_pickle is module.optimized_execution,
            reloaded_jit is jit,
            jit.__dict__ is jit_namespace,
            jit.optimized_execution is module.optimized_execution,
        )

    def eager_contract(self, root, module):
        outcomes = []
        for should_optimize in (False, True):
            leaf = root.tensor([2.0, 3.0], requires_grad=True)
            with module.optimized_execution(should_optimize) as entered:
                output = leaf * 4.0
                output_values = output.tolist()
                output.sum().backward()
                inside_grad = root.is_grad_enabled()
            outcomes.append(
                (
                    entered,
                    output_values,
                    leaf.grad.tolist(),
                    inside_grad,
                    root.is_grad_enabled(),
                )
            )

        with root.no_grad():
            before = root.is_grad_enabled()
            with module.optimized_execution(False) as entered:
                inside = root.is_grad_enabled()
            after_context = root.is_grad_enabled()
        outcomes.append(
            (before, entered, inside, after_context, root.is_grad_enabled())
        )
        return outcomes

    def test_context_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.behavior_contract(torch, self.actual),
            self.behavior_contract(reference_torch, self.expected),
        )

    def test_deferred_validation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.validation_contract(torch, self.actual),
            self.validation_contract(reference_torch, self.expected),
        )

    def test_thread_local_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(torch, self.actual),
            self.thread_contract(reference_torch, self.expected),
        )

    def test_metadata_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_function = actual.optimized_execution
        expected_function = expected.optimized_execution
        actual_wrapped = actual_function.__wrapped__
        expected_wrapped = expected_function.__wrapped__

        self.assertIs(torch.jit.optimized_execution, actual_function)
        self.assertIs(reference_torch.jit.optimized_execution, expected_function)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(hasattr(actual, "__all__"), hasattr(expected, "__all__"))
        supported_module_names = {
            "contextlib",
            "optimized_execution",
            "torch",
            "warnings",
        }
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name in supported_module_names
            },
        )

        for actual_item, expected_item in (
            (actual_function, expected_function),
            (actual_wrapped, expected_wrapped),
        ):
            self.assertIs(type(actual_item), types.FunctionType)
            self.assertIs(type(expected_item), types.FunctionType)
            self.assertEqual(
                str(inspect.signature(actual_item)),
                str(inspect.signature(expected_item)),
            )
            self.assertEqual(
                inspect.get_annotations(actual_item),
                inspect.get_annotations(expected_item),
            )
            self.assertEqual(actual_item.__name__, expected_item.__name__)
            self.assertEqual(actual_item.__qualname__, expected_item.__qualname__)
            self.assertEqual(
                actual_item.__module__.replace("torch_rs", "torch"),
                expected_item.__module__,
            )
            self.assertEqual(actual_item.__doc__, expected_item.__doc__)
            self.assertEqual(actual_item.__defaults__, expected_item.__defaults__)
            self.assertEqual(
                actual_item.__kwdefaults__, expected_item.__kwdefaults__
            )
            self.assertEqual(set(actual_item.__dict__), set(expected_item.__dict__))
            self.assertEqual(
                hasattr(actual_item, "__text_signature__"),
                hasattr(expected_item, "__text_signature__"),
            )
            self.assertEqual(actual_item.__code__.co_names, expected_item.__code__.co_names)
            self.assertEqual(
                actual_item.__code__.co_freevars,
                expected_item.__code__.co_freevars,
            )
            self.assertEqual(
                actual_item.__code__.co_cellvars,
                expected_item.__code__.co_cellvars,
            )

        self.assertIs(inspect.getmodule(actual_function), actual)
        self.assertIs(inspect.getmodule(expected_function), expected)
        self.assertIs(inspect.getmodule(actual_wrapped), actual)
        self.assertIs(inspect.getmodule(expected_wrapped), expected)

        for package_name, module in (
            ("torch_rs", actual),
            ("torch", expected),
        ):
            package_import = {}
            module_import = {}
            package_wildcard = {}
            module_wildcard = {}
            exec(
                f"from {package_name}.jit import optimized_execution",
                package_import,
            )
            exec(
                f"from {package_name}.jit._fuser import optimized_execution",
                module_import,
            )
            exec(f"from {package_name}.jit import *", package_wildcard)
            exec(f"from {package_name}.jit._fuser import *", module_wildcard)
            self.assertIs(package_import["optimized_execution"], module.optimized_execution)
            self.assertIs(module_import["optimized_execution"], module.optimized_execution)
            self.assertNotIn("optimized_execution", package_wildcard)
            self.assertIs(
                module_wildcard["optimized_execution"],
                module.optimized_execution,
            )

        self.assertNotIn("optimized_execution", torch.jit.__all__)
        self.assertNotIn("optimized_execution", reference_torch.jit.__all__)
        self.assertEqual(
            torch.__all__.count("optimized_execution"),
            reference_torch.__all__.count("optimized_execution"),
        )
        self.assertFalse(hasattr(torch, "optimized_execution"))
        self.assertFalse(hasattr(reference_torch, "optimized_execution"))

        for function in (actual_function, expected_function):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_function, protocol)),
                    actual_function,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_function, protocol)),
                    expected_function,
                )
                self.assertEqual(
                    self.pickle_shape(actual_function, protocol),
                    self.pickle_shape(expected_function, protocol),
                )

        self.assertEqual(
            self.context_metadata_contract(actual),
            self.context_metadata_contract(expected),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_eager_tensor_and_autograd_execution_remain_unchanged(self):
        self.assertEqual(
            self.eager_contract(torch, self.actual),
            self.eager_contract(reference_torch, self.expected),
        )


if __name__ == "__main__":
    unittest.main()
