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
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class JitStrictFusionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "jit.strict_fusion differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

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

    def signature_text(self, function):
        try:
            return str(inspect.signature(function))
        except NameError:
            # PyTorch 2.13 deletes torch.jit.Any after defining this class.
            # Python 3.14 resolves annotations lazily, so preserve unresolved
            # forward references when inspecting the reference method.
            import annotationlib

            return str(
                inspect.signature(
                    function,
                    annotation_format=annotationlib.Format.FORWARDREF,
                )
            )

    def annotation_shape(self, function):
        try:
            annotations = function.__annotations__
        except NameError:
            import annotationlib

            annotations = inspect.get_annotations(
                function,
                format=annotationlib.Format.STRING,
            )

        def normalize(annotation):
            if annotation is typing.Any or annotation in ("Any", "_Any"):
                return "Any"
            if annotation is None or annotation == "None":
                return "None"
            return repr(annotation)

        return {
            name: normalize(annotation)
            for name, annotation in annotations.items()
        }

    def warning_outcome(self, module):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            context = module.jit.strict_fusion()
        warning = caught[0]

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            try:
                module.jit.strict_fusion()
            except Exception as error:
                error_outcome = (
                    type(error).__module__,
                    type(error).__qualname__,
                    str(error),
                    error.args,
                )
            else:
                error_outcome = None

        return (
            type(context).__module__.replace("torch_rs", "torch"),
            type(context).__qualname__,
            context.__dict__,
            len(caught),
            type(warning.message) is UserWarning,
            warning.category is UserWarning,
            str(warning.message),
            warning.message.args,
            warning.filename == __file__,
            warning.lineno == warning_line,
            error_outcome,
        )

    def context_outcome(self, module):
        states = [module.is_grad_enabled()]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            context = module.jit.strict_fusion()
            with context as first_entered:
                states.append(module.is_grad_enabled())
                with module.jit.strict_fusion() as nested_entered:
                    states.append(module.is_grad_enabled())
                states.append(module.is_grad_enabled())
            states.append(module.is_grad_enabled())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with module.no_grad():
                states.append(module.is_grad_enabled())
                with context as second_entered:
                    states.append(module.is_grad_enabled())
                    with module.jit.strict_fusion():
                        states.append(module.is_grad_enabled())
                states.append(module.is_grad_enabled())
            states.append(module.is_grad_enabled())

            error = RuntimeError("forwarded failure")
            try:
                with module.jit.strict_fusion():
                    raise error
            except RuntimeError as raised:
                propagated = raised is error
            else:
                propagated = False

        return (
            first_entered,
            nested_entered,
            second_entered,
            states,
            [
                (
                    type(warning.message).__name__,
                    warning.category.__name__,
                    str(warning.message),
                )
                for warning in caught
            ],
            context.__enter__(),
            context.__exit__(object(), object(), object()),
            propagated,
        )

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
                    with module.jit.strict_fusion() as entered:
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
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

    def copy_outcome(self, module):
        strict_fusion = module.jit.strict_fusion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            context = strict_fusion()
        value = {"items": [1, 2]}
        context.value = value
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shallow = copy.copy(context)
            deep = copy.deepcopy(context)
        return (
            copy.copy(strict_fusion) is strict_fusion,
            copy.deepcopy(strict_fusion) is strict_fusion,
            shallow is context,
            type(shallow) is strict_fusion,
            shallow.value is value,
            deep is context,
            type(deep) is strict_fusion,
            deep.value == value,
            deep.value is value,
            len(caught),
        )

    def scripting_predicate_outcome(self, module):
        internal = module._jit_internal
        public = module.jit
        original_internal = internal.is_scripting
        original_public = public.is_scripting
        try:
            internal.is_scripting = lambda: True
            with warnings.catch_warnings(record=True) as internal_caught:
                warnings.simplefilter("always")
                internal_context = public.strict_fusion()

            internal.is_scripting = original_internal
            public.is_scripting = lambda: True
            with warnings.catch_warnings(record=True) as public_caught:
                warnings.simplefilter("always")
                public_context = public.strict_fusion()
        finally:
            internal.is_scripting = original_internal
            public.is_scripting = original_public

        return (
            type(internal_context) is public.strict_fusion,
            [str(warning.message) for warning in internal_caught],
            type(public_context) is public.strict_fusion,
            [str(warning.message) for warning in public_caught],
        )

    def test_warning_context_nesting_exceptions_and_grad_state_match(self):
        self.assertEqual(
            self.warning_outcome(torch),
            self.warning_outcome(reference_torch),
        )
        self.assertEqual(
            self.context_outcome(torch),
            self.context_outcome(reference_torch),
        )
        self.assertEqual(
            self.threaded_grad_outcome(torch),
            self.threaded_grad_outcome(reference_torch),
        )
        self.assertEqual(
            self.scripting_predicate_outcome(torch),
            self.scripting_predicate_outcome(reference_torch),
        )

    def test_signature_documentation_and_module_ownership_match(self):
        actual_jit = importlib.import_module("torch_rs.jit")
        expected_jit = importlib.import_module("torch.jit")
        actual = actual_jit.strict_fusion
        expected = expected_jit.strict_fusion

        self.assertIs(torch.jit, actual_jit)
        self.assertIs(reference_torch.jit, expected_jit)
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual.__bases__, expected.__bases__)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_jit)
        self.assertIs(inspect.getmodule(expected), expected_jit)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        for name in (
            "__module__",
            "__doc__",
            "__init__",
            "__enter__",
            "__exit__",
            "__dict__",
            "__weakref__",
        ):
            with self.subTest(class_field=name):
                self.assertIn(name, actual.__dict__)
                self.assertIn(name, expected.__dict__)
        self.assertEqual(
            {name for name in actual.__dict__ if not name.startswith("__")},
            {name for name in expected.__dict__ if not name.startswith("__")},
        )

        for actual_method, expected_method in (
            (actual.__init__, expected.__init__),
            (actual.__enter__, expected.__enter__),
            (actual.__exit__, expected.__exit__),
        ):
            with self.subTest(method=actual_method.__name__):
                self.assertIs(type(actual_method), types.FunctionType)
                self.assertIs(type(expected_method), types.FunctionType)
                self.assertEqual(
                    self.signature_text(actual_method),
                    self.signature_text(expected_method),
                )
                self.assertEqual(
                    self.annotation_shape(actual_method),
                    self.annotation_shape(expected_method),
                )
                self.assertEqual(actual_method.__name__, expected_method.__name__)
                self.assertEqual(
                    actual_method.__qualname__, expected_method.__qualname__
                )
                self.assertEqual(
                    actual_method.__module__.replace("torch_rs", "torch"),
                    expected_method.__module__,
                )
                self.assertIs(inspect.getmodule(actual_method), actual_jit)
                self.assertIs(inspect.getmodule(expected_method), expected_jit)
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
                self.assertEqual(
                    actual_method.__defaults__, expected_method.__defaults__
                )
                self.assertEqual(
                    actual_method.__kwdefaults__, expected_method.__kwdefaults__
                )
                self.assertEqual(actual_method.__dict__, expected_method.__dict__)

    def test_exports_copying_and_pickling_match_the_supported_scope(self):
        actual_jit = torch.jit
        expected_jit = reference_torch.jit
        actual = actual_jit.strict_fusion
        expected = expected_jit.strict_fusion
        wildcard_supported = {
            "Attribute",
            "annotate",
            "enable_onednn_fusion",
            "export",
            "ignore",
            "isinstance",
            "onednn_fusion_enabled",
            "script_if_tracing",
            "strict_fusion",
            "unused",
        }
        public_supported = {*wildcard_supported, "is_scripting", "is_tracing"}

        self.assertEqual(
            actual_jit.__all__,
            [
                name
                for name in expected_jit.__all__
                if name in wildcard_supported
            ],
        )
        self.assertEqual(
            {name for name in vars(actual_jit) if not name.startswith("_")},
            public_supported,
        )
        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.jit import *", actual_namespace)
        exec("from torch.jit import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            wildcard_supported,
        )
        self.assertIs(actual_namespace["strict_fusion"], actual)
        self.assertIs(expected_namespace["strict_fusion"], expected)

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("strict_fusion", namespace)
            self.assertFalse(hasattr(module, "strict_fusion"))
        self.assertEqual(
            torch.__all__.count("strict_fusion"),
            reference_torch.__all__.count("strict_fusion"),
        )
        self.assertEqual(
            self.copy_outcome(torch), self.copy_outcome(reference_torch)
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            actual_context = actual()
            expected_context = expected()
        actual_context.value = {"items": [1, 2]}
        expected_context.value = {"items": [1, 2]}
        for actual_value, expected_value in (
            (actual, expected),
            (actual_context, expected_context),
        ):
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(value=type(actual_value), protocol=protocol):
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )
                    with warnings.catch_warnings(record=True) as actual_caught:
                        warnings.simplefilter("always")
                        actual_restored = pickle.loads(
                            pickle.dumps(actual_value, protocol)
                        )
                    with warnings.catch_warnings(record=True) as expected_caught:
                        warnings.simplefilter("always")
                        expected_restored = pickle.loads(
                            pickle.dumps(expected_value, protocol)
                        )
                    self.assertEqual(actual_caught, [])
                    self.assertEqual(expected_caught, [])
                    if isinstance(actual_value, type):
                        self.assertIs(actual_restored, actual)
                        self.assertIs(expected_restored, expected)
                    else:
                        self.assertIs(type(actual_restored), actual)
                        self.assertIs(type(expected_restored), expected)
                        self.assertEqual(
                            actual_restored.__dict__, expected_restored.__dict__
                        )

    def test_call_errors_match_pytorch_2_13(self):
        actual = torch.jit.strict_fusion
        expected = reference_torch.jit.strict_fusion
        cases = (
            lambda strict_fusion: strict_fusion(1),
            lambda strict_fusion: strict_fusion(1, 2),
            lambda strict_fusion: strict_fusion(value=1),
            lambda strict_fusion: strict_fusion.__enter__(),
            lambda strict_fusion: strict_fusion.__exit__(object()),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(actual),
                    lambda: call(expected),
                )

    def test_scripting_tracing_and_fusion_execution_stay_outside_scope(self):
        self.assertTrue(callable(torch.jit.strict_fusion))
        self.assertTrue(callable(reference_torch.jit.strict_fusion))
        self.assertIs(torch.jit.is_scripting(), False)
        self.assertIs(torch.jit.is_tracing(), False)
        expected_public = {
            name for name in vars(reference_torch.jit) if not name.startswith("_")
        }
        for name in (
            "CompilationUnit",
            "ScriptFunction",
            "ScriptModule",
            "script",
            "set_fusion_strategy",
            "trace",
            "trace_module",
        ):
            with self.subTest(name=name):
                self.assertIn(name, expected_public)
                self.assertFalse(hasattr(torch.jit, name))
        self.assertFalse(hasattr(torch, "compile"))


if __name__ == "__main__":
    unittest.main()
