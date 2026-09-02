import copy
import gc
import importlib
import inspect
import pickle
import threading
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class Truthy:
    def __bool__(self):
        return True


class SetGradEnabledTests(unittest.TestCase):
    def test_state_transitions_decorator_generator_and_thread_isolation(self):
        value = torch.tensor([2.0], requires_grad=True)
        self.assertIs(torch.is_grad_enabled(), True)

        with torch.set_grad_enabled(False) as entered:
            self.assertIsNone(entered)
            self.assertIs(torch.is_grad_enabled(), False)
            self.assertFalse((value * value).requires_grad)
            with torch.set_grad_enabled(True):
                self.assertIs(torch.is_grad_enabled(), True)
                self.assertTrue((value * value).requires_grad)
                with torch.set_grad_enabled(False):
                    self.assertIs(torch.is_grad_enabled(), False)
                    self.assertFalse((value * value).requires_grad)
                self.assertIs(torch.is_grad_enabled(), True)
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

        with torch.no_grad():
            self.assertIs(torch.is_grad_enabled(), False)
            with torch.set_grad_enabled(True):
                self.assertIs(torch.is_grad_enabled(), True)
                self.assertTrue((value * value).requires_grad)
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

        context = torch.set_grad_enabled(False)
        try:
            self.assertIs(torch.is_grad_enabled(), False)
            self.assertFalse((value * value).requires_grad)
        finally:
            context.__exit__(None, None, None)
        self.assertIs(torch.is_grad_enabled(), True)

        with self.assertRaisesRegex(RuntimeError, "restore grad mode"):
            with torch.set_grad_enabled(False):
                self.assertIs(torch.is_grad_enabled(), False)
                raise RuntimeError("restore grad mode")
        self.assertIs(torch.is_grad_enabled(), True)

        @torch.set_grad_enabled(False)
        def disabled(input_value: object, scale: float = 1.0) -> object:
            """A metadata-bearing set-grad callable."""
            return input_value * scale

        self.assertIs(torch.is_grad_enabled(), True)
        self.assertFalse(disabled(value, scale=3.0).requires_grad)
        self.assertEqual(disabled.__name__, "disabled")
        self.assertEqual(disabled.__doc__, "A metadata-bearing set-grad callable.")
        self.assertEqual(
            disabled.__annotations__,
            {"input_value": object, "scale": float, "return": object},
        )
        self.assertEqual(
            inspect.signature(disabled), inspect.signature(disabled.__wrapped__)
        )

        with torch.no_grad():
            @torch.set_grad_enabled(True)
            def enabled(input_value):
                return input_value * input_value

            self.assertIs(torch.is_grad_enabled(), False)
            self.assertTrue(enabled(value).requires_grad)
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

        events = []

        @torch.set_grad_enabled(False)
        def generate():
            events.append(("next", torch.is_grad_enabled()))
            request = yield value * value
            events.append(("send", request, torch.is_grad_enabled()))
            try:
                yield value * value
            except ValueError as error:
                events.append(("throw", str(error), torch.is_grad_enabled()))
                yield value * value
            finally:
                events.append(("close", torch.is_grad_enabled()))

        generator = generate()
        self.assertFalse(next(generator).requires_grad)
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertFalse(generator.send("request").requires_grad)
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertFalse(generator.throw(ValueError("injected")).requires_grad)
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIsNone(generator.close())
        self.assertEqual(
            events,
            [
                ("next", False),
                ("send", "request", False),
                ("throw", "injected", False),
                ("close", False),
            ],
        )

        worker_states = []
        worker_failures = []

        def worker():
            try:
                worker_states.append(torch.is_grad_enabled())
                with torch.set_grad_enabled(False):
                    worker_states.append(torch.is_grad_enabled())
                worker_states.append(torch.is_grad_enabled())
                with torch.no_grad():
                    with torch.set_grad_enabled(True):
                        worker_states.append(torch.is_grad_enabled())
                    worker_states.append(torch.is_grad_enabled())
            except BaseException as error:
                worker_failures.append(error)

        with torch.set_grad_enabled(False):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertIs(torch.is_grad_enabled(), False)
        self.assertEqual(worker_failures, [])
        self.assertEqual(worker_states, [True, False, True, True, False])
        self.assertIs(torch.is_grad_enabled(), True)

    def test_namespace_copy_pickle_reload_and_wildcard_behavior(self):
        autograd = importlib.import_module("torch_rs.autograd")
        grad_mode = importlib.import_module("torch_rs.autograd.grad_mode")
        from torch_rs.autograd import set_grad_enabled as autograd_set_grad_enabled
        from torch_rs.autograd.grad_mode import (
            set_grad_enabled as grad_mode_set_grad_enabled,
        )

        context_type = torch.set_grad_enabled
        self.assertIs(context_type, autograd.set_grad_enabled)
        self.assertIs(context_type, grad_mode.set_grad_enabled)
        self.assertIs(context_type, autograd_set_grad_enabled)
        self.assertIs(context_type, grad_mode_set_grad_enabled)
        self.assertEqual(context_type.__name__, "set_grad_enabled")
        self.assertEqual(context_type.__qualname__, "set_grad_enabled")
        self.assertEqual(context_type.__module__, "torch_rs.autograd.grad_mode")
        self.assertIs(inspect.getmodule(context_type), grad_mode)
        self.assertEqual(str(inspect.signature(context_type)), "(mode: bool) -> None")

        top_wildcard = {}
        autograd_wildcard = {}
        grad_mode_wildcard = {}
        exec("from torch_rs import *", top_wildcard)
        exec("from torch_rs.autograd import *", autograd_wildcard)
        exec("from torch_rs.autograd.grad_mode import *", grad_mode_wildcard)
        self.assertNotIn("set_grad_enabled", top_wildcard)
        self.assertIs(autograd_wildcard["set_grad_enabled"], context_type)
        self.assertIs(grad_mode_wildcard["set_grad_enabled"], context_type)

        self.assertIs(copy.copy(context_type), context_type)
        self.assertIs(copy.deepcopy(context_type), context_type)

        for mode in (False, True):
            with self.subTest(mode=mode):
                instance = context_type(mode)
                try:
                    self.assertEqual(instance.__dict__, {"prev": True, "mode": mode})
                    self.assertEqual(
                        str(instance),
                        "torch_rs.autograd.grad_mode."
                        f"set_grad_enabled(mode={mode})",
                    )
                    self.assertEqual(repr(instance), str(instance))
                    self.assertIs(torch.is_grad_enabled(), mode)
                    restored_contexts = [copy.copy(instance), copy.deepcopy(instance)]
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                        payload = pickle.dumps(instance, protocol=protocol)
                        self.assertIn(b"torch_rs.autograd.grad_mode", payload)
                        restored_contexts.append(pickle.loads(payload))
                finally:
                    instance.__exit__(None, None, None)
                self.assertIs(torch.is_grad_enabled(), True)

                for restored in restored_contexts:
                    self.assertIs(type(restored), context_type)
                    self.assertEqual(restored.__dict__, {"prev": True, "mode": mode})
                    with torch.no_grad():
                        with restored:
                            self.assertIs(torch.is_grad_enabled(), mode)
                        self.assertIs(torch.is_grad_enabled(), True)

        old_autograd_exports = torch.autograd.__all__
        old_grad_mode_exports = grad_mode.__all__
        self.assertIs(importlib.reload(grad_mode), grad_mode)
        self.assertIs(importlib.reload(torch.autograd), torch.autograd)
        self.assertIs(torch.autograd.grad_mode, grad_mode)
        self.assertIs(torch.set_grad_enabled, grad_mode.set_grad_enabled)
        self.assertIs(torch.autograd.set_grad_enabled, grad_mode.set_grad_enabled)
        self.assertEqual(torch.autograd.__all__, old_autograd_exports)
        self.assertEqual(grad_mode.__all__, old_grad_mode_exports)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SetGradEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "set_grad_enabled differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def _reset_grad_enabled(module):
        if not module.is_grad_enabled():
            module.set_grad_enabled(True)

    def tearDown(self):
        self._reset_grad_enabled(torch)
        self._reset_grad_enabled(reference_torch)

    @staticmethod
    def state_contract(module):
        value = module.tensor([2.0], requires_grad=True)
        states = [module.is_grad_enabled()]

        with module.set_grad_enabled(False) as entered:
            states.extend(
                [
                    entered,
                    module.is_grad_enabled(),
                    (value * value).requires_grad,
                ]
            )
            with module.set_grad_enabled(True):
                states.extend(
                    [module.is_grad_enabled(), (value * value).requires_grad]
                )
            states.append(module.is_grad_enabled())
        states.append(module.is_grad_enabled())

        with module.no_grad():
            with module.set_grad_enabled(True):
                states.extend(
                    [module.is_grad_enabled(), (value * value).requires_grad]
                )
            states.append(module.is_grad_enabled())
        states.append(module.is_grad_enabled())

        context = module.set_grad_enabled(False)
        try:
            states.extend(
                [module.is_grad_enabled(), (value * value).requires_grad]
            )
        finally:
            context.__exit__(None, None, None)
        states.append(module.is_grad_enabled())

        @module.set_grad_enabled(False)
        def disabled(input_value: object, scale: float = 1.0) -> object:
            """decorated docs"""
            return input_value * scale

        state_after_decorator_definition = module.is_grad_enabled()
        disabled_result = disabled(value, scale=3.0).requires_grad
        state_after_decorator_call = module.is_grad_enabled()

        with module.no_grad():
            @module.set_grad_enabled(True)
            def enabled(input_value):
                return input_value * input_value

            enabled_definition_state = module.is_grad_enabled()
            enabled_result = enabled(value).requires_grad
            enabled_call_state = module.is_grad_enabled()

        events = []

        @module.set_grad_enabled(False)
        def generate():
            events.append(("next", module.is_grad_enabled()))
            request = yield value * value
            events.append(("send", request, module.is_grad_enabled()))
            try:
                yield value * value
            except ValueError as error:
                events.append(("throw", str(error), module.is_grad_enabled()))
                yield value * value
            finally:
                events.append(("close", module.is_grad_enabled()))

        generator = generate()
        generator_results = [
            next(generator).requires_grad,
            module.is_grad_enabled(),
            generator.send("request").requires_grad,
            module.is_grad_enabled(),
            generator.throw(ValueError("injected")).requires_grad,
            module.is_grad_enabled(),
            generator.close(),
            module.is_grad_enabled(),
        ]

        abandoned_events = []

        @module.set_grad_enabled(False)
        def abandoned():
            try:
                yield value * value
            finally:
                abandoned_events.append(module.is_grad_enabled())

        abandoned_generator = abandoned()
        abandoned_result = next(abandoned_generator).requires_grad
        del abandoned_generator
        gc.collect()

        return (
            states,
            state_after_decorator_definition,
            disabled_result,
            state_after_decorator_call,
            enabled_definition_state,
            enabled_result,
            enabled_call_state,
            disabled.__name__,
            disabled.__doc__,
            disabled.__annotations__,
            str(inspect.signature(disabled)),
            disabled.__wrapped__.__name__,
            inspect.isgeneratorfunction(generate),
            generator_results,
            events,
            abandoned_result,
            abandoned_events,
            module.is_grad_enabled(),
        )

    @staticmethod
    def namespace_contract(module):
        autograd = importlib.import_module(f"{module.__name__}.autograd")
        grad_mode = importlib.import_module(f"{module.__name__}.autograd.grad_mode")
        context_type = module.set_grad_enabled
        top_wildcard = {}
        autograd_wildcard = {}
        grad_mode_wildcard = {}
        exec(f"from {module.__name__} import *", top_wildcard)
        exec(f"from {module.__name__}.autograd import *", autograd_wildcard)
        exec(
            f"from {module.__name__}.autograd.grad_mode import *",
            grad_mode_wildcard,
        )

        return {
            "aliases": (
                autograd.set_grad_enabled is context_type,
                grad_mode.set_grad_enabled is context_type,
            ),
            "metadata": (
                context_type.__name__,
                context_type.__qualname__,
                context_type.__module__.replace(module.__name__, "torch"),
                str(inspect.signature(context_type)).replace(module.__name__, "torch"),
                type(context_type) is type,
            ),
            "wildcard": (
                "set_grad_enabled" in top_wildcard,
                autograd_wildcard["set_grad_enabled"] is context_type,
                grad_mode_wildcard["set_grad_enabled"] is context_type,
            ),
            "exports": (
                module.__all__.count("set_grad_enabled"),
                autograd.__all__.count("set_grad_enabled"),
                grad_mode.__all__.count("set_grad_enabled"),
            ),
            "class_copy": (
                copy.copy(context_type) is context_type,
                copy.deepcopy(context_type) is context_type,
            ),
        }

    @staticmethod
    def copy_pickle_contract(module, mode):
        context_type = module.set_grad_enabled
        instance = context_type(mode)
        try:
            context_state = module.is_grad_enabled()
            copies = (copy.copy(instance), copy.deepcopy(instance))
            copy_states = tuple(
                (type(copied) is context_type, copied.__dict__)
                for copied in copies
            )
            pickle_states = []
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                restored = pickle.loads(pickle.dumps(instance, protocol=protocol))
                pickle_states.append(
                    (type(restored) is context_type, restored.__dict__)
                )
        finally:
            instance.__exit__(None, None, None)

        return (
            context_state,
            instance.__dict__,
            str(instance).replace(module.__name__, "torch"),
            repr(instance).replace(module.__name__, "torch"),
            copy_states,
            tuple(pickle_states),
            module.is_grad_enabled(),
        )

    @staticmethod
    def argument_error_contract(module):
        def function():
            return None

        invalid_values = (
            np.bool_(True),
            np.bool_(False),
            1,
            0,
            [],
            [1],
            None,
            "true",
            Truthy(),
            object(),
            module.tensor([1.0]),
        )
        invalid_errors = []
        for value in invalid_values:
            try:
                module.set_grad_enabled(value)
            except TypeError as error:
                invalid_errors.append(str(error))
            else:
                raise AssertionError(f"{module.__name__} accepted mode={value!r}")
            invalid_errors.append(module.is_grad_enabled())

        signature_calls = (
            lambda: module.set_grad_enabled(),
            lambda: module.set_grad_enabled(True, False),
            lambda: module.set_grad_enabled(True, mode=False),
            lambda: module.set_grad_enabled(function),
        )
        signature_errors = []
        for call in signature_calls:
            try:
                call()
            except TypeError as error:
                signature_errors.append((type(error).__name__, str(error), error.args))
            else:
                raise AssertionError(f"{module.__name__} accepted invalid arguments")
            signature_errors.append(module.is_grad_enabled())

        keyword_context = module.set_grad_enabled(mode=False)
        try:
            keyword_state = module.is_grad_enabled()
        finally:
            keyword_context.__exit__(None, None, None)

        return invalid_errors, signature_errors, keyword_state, module.is_grad_enabled()

    @staticmethod
    def thread_contract(module):
        worker_states = []

        def worker():
            worker_states.append(module.is_grad_enabled())
            with module.set_grad_enabled(False):
                worker_states.append(module.is_grad_enabled())
            worker_states.append(module.is_grad_enabled())
            with module.no_grad():
                with module.set_grad_enabled(True):
                    worker_states.append(module.is_grad_enabled())
                worker_states.append(module.is_grad_enabled())

        with module.set_grad_enabled(False):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            main_inside = module.is_grad_enabled()
        return worker_states, main_inside, module.is_grad_enabled()

    def test_state_transitions_match_pytorch_2_13(self):
        self.assertEqual(
            self.state_contract(torch),
            self.state_contract(reference_torch),
        )

    def test_namespace_copy_pickle_and_wildcard_match_pytorch_2_13(self):
        self.assertEqual(
            self.namespace_contract(torch),
            self.namespace_contract(reference_torch),
        )
        for mode in (False, True):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.copy_pickle_contract(torch, mode),
                    self.copy_pickle_contract(reference_torch, mode),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.argument_error_contract(torch),
            self.argument_error_contract(reference_torch),
        )

    def test_thread_isolation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(torch),
            self.thread_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
