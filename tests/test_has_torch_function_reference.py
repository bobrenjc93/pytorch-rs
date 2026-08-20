import copy
import importlib
import inspect
import pickle
import pickletools
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class HasTorchFunctionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "has_torch_function differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            str(actual_raised.exception).replace("torch_rs", "torch"),
            str(expected_raised.exception),
        )
        self.assertEqual(
            tuple(
                value.replace("torch_rs", "torch")
                if isinstance(value, str)
                else value
                for value in actual_raised.exception.args
            ),
            expected_raised.exception.args,
        )

    def basic_observation(self, module):
        function = module.overrides.has_torch_function
        tensor = module.tensor([1.0])

        class Override:
            __torch_function__ = None

        class Disabled:
            __torch_function__ = reference_torch._C._disabled_torch_function_impl

        class RaisingDescriptor:
            def __init__(self):
                self.lookups = 0

            def __get__(self, instance, owner):
                self.lookups += 1
                raise RuntimeError("descriptor boom")

        descriptor = RaisingDescriptor()

        class Broken:
            __torch_function__ = descriptor

        results = [
            function(()),
            function((tensor,)),
            function((module.Tensor,)),
            function((Override(),)),
            function((Disabled(),)),
            function((Broken(),)),
            function((None,)),
        ]
        return results, descriptor.lookups, [type(result) is bool for result in results]

    def test_exact_custom_disabled_and_broken_overrides_match(self):
        self.assertEqual(
            self.basic_observation(torch),
            self.basic_observation(reference_torch),
        )

    def iterable_observation(self, module):
        function = module.overrides.has_torch_function

        class Override:
            __torch_function__ = None

        class RecordingIterable:
            def __init__(self, values):
                self.values = values
                self.iterations = 0
                self.yielded = 0

            def __iter__(self):
                self.iterations += 1
                for value in self.values:
                    self.yielded += 1
                    yield value

        iterable = RecordingIterable([Override(), object()])
        iterable_result = function(iterable)
        generator = (value for value in [object(), Override()])
        generator_result = function(generator)
        generator_remainder = list(generator)
        mapping_result = function({Override(): "value"})
        nested_result = function([[Override()]])

        class LateFailure:
            def __init__(self):
                self.yielded = 0

            def __iter__(self):
                self.yielded += 1
                yield Override()
                self.yielded += 1
                raise RuntimeError("late iteration failure")

        late = LateFailure()
        try:
            function(late)
        except Exception as error:
            late_error = type(error).__name__, str(error), error.args
        else:
            late_error = None
        return (
            iterable_result,
            iterable.iterations,
            iterable.yielded,
            generator_result,
            generator_remainder,
            mapping_result,
            nested_result,
            late_error,
            late.yielded,
        )

    def test_arbitrary_iterable_materialization_matches_pytorch_2_13(self):
        self.assertEqual(
            self.iterable_observation(torch),
            self.iterable_observation(reference_torch),
        )

    def mode_observation(self, module):
        function = module.overrides.has_torch_function
        tensor = module.tensor([1.0])

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        lower = Mode("lower")
        upper = Mode("upper")
        observations = [function(()), function((tensor,)), function((object(),))]
        with lower:
            before = module.overrides._get_current_function_mode_stack()
            observations.extend(
                [
                    function(()),
                    function((tensor,)),
                    function((object(),)),
                    module.overrides._get_current_function_mode_stack() == before,
                ]
            )
            with upper:
                nested_before = module.overrides._get_current_function_mode_stack()
                observations.extend(
                    [
                        [mode.label for mode in nested_before],
                        function((tensor,)),
                        module.overrides._get_current_function_mode_stack()
                        == nested_before,
                    ]
                )
            observations.append(
                module.overrides._get_current_function_mode_stack() == [lower]
            )
        observations.extend(
            [
                module.overrides._get_current_function_mode_stack() == [],
                function((tensor,)),
            ]
        )
        return observations

    def test_active_mode_results_and_stack_preservation_match(self):
        self.assertEqual(
            self.mode_observation(torch),
            self.mode_observation(reference_torch),
        )

    def threaded_observation(self, module):
        function = module.overrides.has_torch_function
        tensor = module.tensor([1.0])
        worker_entered = threading.Event()
        worker_leave = threading.Event()
        worker_results = []
        errors = []

        class Mode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        main_mode = Mode("main")

        def stack_labels():
            return [
                mode.label
                for mode in module.overrides._get_current_function_mode_stack()
            ]

        def worker():
            worker_mode = Mode("worker")
            try:
                worker_results.append((function((tensor,)), stack_labels()))
                with worker_mode:
                    worker_results.append((function((tensor,)), stack_labels()))
                    worker_entered.set()
                    if not worker_leave.wait(timeout=10):
                        raise RuntimeError("main thread did not release worker")
                worker_results.append((function((tensor,)), stack_labels()))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_entered.set()

        thread = threading.Thread(target=worker)
        with main_mode:
            thread.start()
            entered = worker_entered.wait(timeout=10)
            main_result = function((tensor,)), stack_labels()
            worker_leave.set()
        thread.join(timeout=10)
        return (
            entered,
            thread.is_alive(),
            errors,
            main_result,
            worker_results,
            stack_labels(),
        )

    def test_thread_local_mode_stack_preservation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.threaded_observation(torch),
            self.threaded_observation(reference_torch),
        )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_cuda_exact_tensor_observes_and_preserves_reference_mode_stack(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0], device="cuda")
        self.assertTrue(expected_tensor.is_cuda)

        def observation(module, tensor):
            function = module.overrides.has_torch_function

            class Mode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    return NotImplemented

            mode = Mode()
            before = function((tensor,))
            with mode:
                stack_before = module.overrides._get_current_function_mode_stack()
                active = function((tensor,))
                stack_after = module.overrides._get_current_function_mode_stack()
            return (
                before,
                active,
                function((tensor,)),
                stack_before == [mode],
                stack_after == [mode],
                module.overrides._get_current_function_mode_stack() == [],
            )

        self.assertEqual(
            observation(torch, actual_tensor),
            observation(reference_torch, expected_tensor),
        )

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

    def test_native_metadata_aliases_and_supported_exports_match(self):
        actual_module = importlib.import_module("torch_rs.overrides")
        expected_module = importlib.import_module("torch.overrides")
        actual = actual_module.has_torch_function
        expected = expected_module.has_torch_function

        self.assertIs(actual, actual_module._has_torch_function)
        self.assertIs(expected, expected_module._has_torch_function)
        self.assertIs(actual, torch._C._has_torch_function)
        self.assertIs(expected, reference_torch._C._has_torch_function)
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(inspect.getmodule(actual), torch._C)
        self.assertIs(inspect.getmodule(expected), reference_torch._C)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        for function in (actual, expected):
            with self.assertRaises(ValueError) as raised:
                inspect.signature(function)
            self.assertEqual(
                str(raised.exception),
                "no signature found for builtin <built-in function _has_torch_function>",
            )

        self.assertEqual(actual_module.__all__.count("has_torch_function"), 1)
        self.assertEqual(expected_module.__all__.count("has_torch_function"), 1)
        self.assertNotIn("_has_torch_function", torch._C.__all__)
        self.assertNotIn("has_torch_function", torch.__all__)
        self.assertFalse(hasattr(torch, "has_torch_function"))
        for module, function in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertIs(namespace["has_torch_function"], function)
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

    def test_argument_and_iteration_errors_match_pytorch_2_13(self):
        actual = torch.overrides.has_torch_function
        expected = reference_torch.overrides.has_torch_function
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual((), ()), lambda: expected((), ())),
            (
                lambda: actual(relevant_args=()),
                lambda: expected(relevant_args=()),
            ),
            (lambda: actual(42), lambda: expected(42)),
            (lambda: actual(None), lambda: expected(None)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        class BrokenIterable:
            def __iter__(self):
                raise ValueError("custom iteration failure")

        self.assert_error_matches(
            lambda: actual(BrokenIterable()),
            lambda: expected(BrokenIterable()),
        )


if __name__ == "__main__":
    unittest.main()
