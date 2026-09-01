import copy
import importlib
import inspect
import itertools
import pickle
import pickletools
import re
import sys
import threading
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _ContextBodyError(Exception):
    pass


class _TruthProbe:
    def __init__(self, name, log, result=True, error=None):
        self.name = name
        self.log = log
        self.result = result
        self.error = error

    def __bool__(self):
        self.log.append(self.name)
        if self.error is not None:
            raise self.error
        return self.result


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaSdpKernelReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cuda.sdp_kernel differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.original_actual = self.states(self.actual)
        self.original_expected = self.states(self.expected)
        self.original_actual_reduction = (
            torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        )
        self.original_expected_reduction = (
            reference_torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        )

    def tearDown(self):
        self.set_states(self.actual, self.original_actual)
        self.set_states(self.expected, self.original_expected)
        self.actual.allow_fp16_bf16_reduction_math_sdp(
            self.original_actual_reduction
        )
        self.expected.allow_fp16_bf16_reduction_math_sdp(
            self.original_expected_reduction
        )

    def states(self, module):
        return (
            module.flash_sdp_enabled(),
            module.math_sdp_enabled(),
            module.mem_efficient_sdp_enabled(),
            module.cudnn_sdp_enabled(),
        )

    def set_states(self, module, states):
        flash, math, mem_efficient, cudnn = states
        module.enable_flash_sdp(flash)
        module.enable_math_sdp(math)
        module.enable_mem_efficient_sdp(mem_efficient)
        module.enable_cudnn_sdp(cudnn)

    def make_context(self, module, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            context = module.sdp_kernel(*args, **kwargs)
        warning_details = [
            (warning.category.__name__, str(warning.message))
            for warning in caught
        ]
        return context, warning_details

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

    def context_contract(self, module, initial, arguments, keywords=None):
        keywords = {} if keywords is None else keywords
        self.set_states(module, initial)
        module.allow_fp16_bf16_reduction_math_sdp(True)
        context, warning_details = self.make_context(
            module,
            *arguments,
            **keywords,
        )
        after_create = self.states(module)
        entered = context.__enter__()
        inside = self.states(module)
        reduction_inside = module.fp16_bf16_reduction_math_sdp_allowed()
        exit_result = context.__exit__(None, None, None)
        after_exit = self.states(module)
        reduction_after = module.fp16_bf16_reduction_math_sdp_allowed()
        return (
            warning_details,
            after_create,
            type(entered).__name__,
            entered,
            inside,
            reduction_inside,
            exit_result,
            after_exit,
            reduction_after,
        )

    def test_defaults_and_explicit_boolean_combinations_match_pytorch_2_13(self):
        initial_states = (
            (True, True, True, True),
            (False, True, False, True),
        )
        cases = [((), {})]
        cases.extend((combo, {}) for combo in itertools.product((False, True), repeat=4))
        cases.append(((), {"enable_flash": False, "enable_cudnn": True}))

        for initial in initial_states:
            for arguments, keywords in cases:
                with self.subTest(initial=initial, arguments=arguments, keywords=keywords):
                    self.assertEqual(
                        self.context_contract(
                            self.actual,
                            initial,
                            arguments,
                            keywords,
                        ),
                        self.context_contract(
                            self.expected,
                            initial,
                            arguments,
                            keywords,
                        ),
                    )

    def truthy_contract(self, module):
        self.set_states(module, (True, False, True, False))
        context, warning_details = self.make_context(module, 1, 0, "", object())
        entered = context.__enter__()
        inside = self.states(module)
        exit_result = context.__exit__(None, None, None)
        return warning_details, entered, inside, exit_result, self.states(module)

    def error_contract(self, module):
        self.set_states(module, (True, False, True, False))
        error = _ContextBodyError("truthiness failed")
        log = []
        context, warning_details = self.make_context(
            module,
            _TruthProbe("flash", log),
            _TruthProbe("math", log),
            _TruthProbe("mem_efficient", log, error=error),
            _TruthProbe("cudnn", log),
        )
        try:
            context.__enter__()
        except Exception as raised:
            error_details = (
                type(raised).__name__,
                str(raised),
                raised.args,
                raised is error,
            )
        else:
            self.fail("truthiness exception was not propagated")
        return warning_details, log, error_details, self.states(module)

    def binding_contract(self, module, case):
        self.set_states(module, (True, False, True, False))
        calls = (
            lambda: module.sdp_kernel(True, False, True, False, True),
            lambda: module.sdp_kernel(foo=True),
            lambda: module.sdp_kernel(True, enable_flash=False),
            lambda: module.sdp_kernel(_enabled=False),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                calls[case]()
            except Exception as raised:
                error_details = (
                    type(raised).__name__,
                    str(raised),
                    raised.args,
                )
            else:
                self.fail("binding error did not occur")
        warning_details = [
            (warning.category.__name__, str(warning.message))
            for warning in caught
        ]
        return warning_details, error_details, self.states(module)

    def test_truthy_values_invalid_truthiness_and_binding_errors_match(self):
        self.assertEqual(
            self.truthy_contract(self.actual),
            self.truthy_contract(self.expected),
        )
        self.assertEqual(
            self.error_contract(self.actual),
            self.error_contract(self.expected),
        )
        for case in range(4):
            with self.subTest(case=case):
                self.assertEqual(
                    self.binding_contract(self.actual, case),
                    self.binding_contract(self.expected, case),
                )

    def decorator_contract(self, module):
        self.set_states(module, (True, True, True, True))
        context, warning_details = self.make_context(module, False, True, False, True)
        observations = []

        @context
        def decorated(value):
            observations.append(self.states(module))
            if value == "raise":
                raise _ContextBodyError("decorated body failed")
            return value

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = decorated("first")
            second = decorated("second")
            try:
                decorated("raise")
            except Exception as raised:
                error_details = (
                    type(raised).__name__,
                    str(raised),
                    raised.args,
                )
            else:
                self.fail("decorated body did not raise")
        call_warnings = [
            (warning.category.__name__, str(warning.message))
            for warning in caught
        ]
        return (
            warning_details,
            first,
            second,
            error_details,
            observations,
            self.states(module),
            call_warnings,
        )

    def thread_contract(self, module):
        self.set_states(module, (True, True, True, True))
        worker_entered = threading.Event()
        main_context_exited = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    with module.sdp_kernel(False, False, False, False) as entered:
                        observations.append(("worker-enter", entered, self.states(module)))
                        worker_entered.set()
                        if not main_context_exited.wait(timeout=10):
                            raise RuntimeError("timed out waiting for main context")
                        observations.append(("worker-resume", self.states(module)))
                observations.append(("worker-exit", self.states(module)))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_entered.set()

        thread = threading.Thread(target=worker)
        thread.start()
        try:
            worker_ready = worker_entered.wait(timeout=10)
            after_worker_enter = self.states(module)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                with module.sdp_kernel(True, False, True, False) as entered:
                    main_entered = entered
                    inside_main = self.states(module)
            after_main_exit = self.states(module)
        finally:
            main_context_exited.set()
            thread.join(timeout=10)

        return (
            worker_ready,
            after_worker_enter,
            main_entered,
            inside_main,
            after_main_exit,
            not thread.is_alive(),
            errors,
            observations,
            self.states(module),
        )

    def test_decorator_and_thread_visible_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.decorator_contract(self.actual),
            self.decorator_contract(self.expected),
        )
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def metadata_contract(self, package_name, module, function):
        imported = {}
        wildcard = {}
        exec(f"from {package_name}.backends.cuda import sdp_kernel", imported)
        exec(f"from {package_name}.backends.cuda import *", wildcard)

        context, warning_details = self.make_context(
            module,
            False,
            True,
            False,
            True,
        )
        copied_context = copy.copy(context)
        context_pickle_error = None
        try:
            pickle.dumps(context)
        except Exception as error:
            context_pickle_error = (type(error).__name__, str(error))
        else:
            self.fail("context was unexpectedly pickleable")
        deepcopy_error = None
        try:
            copy.deepcopy(context)
        except Exception as error:
            deepcopy_error = (type(error).__name__, str(error))
        else:
            self.fail("context was unexpectedly deepcopyable")

        wrapped = function.__wrapped__
        inner = wrapped.__wrapped__
        return (
            function is imported["sdp_kernel"],
            function is wildcard["sdp_kernel"],
            "sdp_kernel" in module.__all__,
            str(inspect.signature(function)),
            inspect.get_annotations(function),
            function.__name__,
            function.__qualname__,
            function.__module__.replace("torch_rs", "torch"),
            inspect.getmodule(function) is module,
            function.__doc__,
            function.__defaults__,
            function.__kwdefaults__,
            sorted(function.__dict__),
            getattr(function, "__deprecated__"),
            wrapped.__module__.replace("torch_rs", "torch"),
            wrapped.__doc__,
            wrapped.__defaults__,
            wrapped.__kwdefaults__,
            sorted(wrapped.__dict__),
            getattr(wrapped, "__deprecated__"),
            inner.__defaults__,
            inner.__kwdefaults__,
            sorted(inner.__dict__),
            type(context).__module__,
            type(context).__qualname__,
            context.__doc__,
            context.func is wrapped,
            context.args,
            context.kwds,
            copied_context is context,
            copied_context.gen is context.gen,
            context_pickle_error,
            deepcopy_error,
            warning_details,
        )

    def test_metadata_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual.sdp_kernel
        expected = self.expected.sdp_kernel
        self.assertEqual(
            self.metadata_contract("torch_rs", self.actual, actual),
            self.metadata_contract("torch", self.expected, expected),
        )
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def reload_contract(self, root, module):
        self.set_states(module, (True, True, True, True))
        old_function = module.sdp_kernel
        old_wrapped = old_function.__wrapped__
        namespace = module.__dict__
        active_context, warning_details = self.make_context(
            module,
            False,
            False,
            False,
            False,
        )
        active_context.__enter__()
        reloaded = importlib.reload(module)
        state_after_reload = self.states(module)
        exit_result = active_context.__exit__(None, None, None)
        state_after_exit = self.states(module)

        stale_error = None
        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs",
                    "torch",
                ),
            )
        else:
            self.fail("stale sdp_kernel function remained pickleable")

        return (
            warning_details,
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.cuda is module,
            sys.modules[module.__name__] is module,
            module.sdp_kernel is not old_function,
            module.sdp_kernel.__wrapped__ is not old_wrapped,
            state_after_reload,
            exit_result,
            state_after_exit,
            stale_error,
            pickle.loads(pickle.dumps(module.sdp_kernel)) is module.sdp_kernel,
        )

    def test_reload_preserves_active_contexts_and_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_unsupported_execution_boundary_stays_narrow(self):
        self.assertFalse(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
        self.assertFalse(hasattr(torch.nn, "attention"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertFalse(hasattr(torch, "compile"))
        if reference_torch.cuda.is_available():
            device = reference_torch.device("cuda", 0)
            tensor = reference_torch.tensor([1.0], device=device)
            self.assertEqual(tensor.device.type, "cuda")
            reference_torch.cuda.synchronize(device)


if __name__ == "__main__":
    unittest.main()
