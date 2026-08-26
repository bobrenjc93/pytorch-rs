import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _ExplodingTruth:
    def __bool__(self):
        raise AssertionError("the stored fastpath value was coerced to bool")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MhaFastpathReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.mha differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_actual = torch.backends.mha.get_fastpath_enabled()
        self.original_expected = reference_torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(True)
        reference_torch.backends.mha.set_fastpath_enabled(True)

    def tearDown(self):
        torch.backends.mha.set_fastpath_enabled(self.original_actual)
        reference_torch.backends.mha.set_fastpath_enabled(self.original_expected)

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
            shape.append((opcode.name, argument))
        return shape

    def threaded_outcome(self, root):
        mha = root.backends.mha
        initial = object()
        updated = object()
        worker_value = object()
        ready = threading.Event()
        continue_reading = threading.Event()
        observations = []
        errors = []
        mha.set_fastpath_enabled(initial)

        def observer():
            try:
                observations.append(mha.get_fastpath_enabled() is initial)
                ready.set()
                if not continue_reading.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the fastpath update")
                observations.append(mha.get_fastpath_enabled() is updated)
                observations.append(mha.set_fastpath_enabled(worker_value) is None)
                observations.append(mha.get_fastpath_enabled() is worker_value)
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(ready.wait(timeout=10))
        set_result = mha.set_fastpath_enabled(updated)
        continue_reading.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        return (
            set_result is None,
            observations,
            mha.get_fastpath_enabled() is worker_value,
        )

    def test_default_identity_scripting_and_threads_match_pytorch_2_13(self):
        values = (False, None, 0, "", [], {}, object(), _ExplodingTruth())
        for value in values:
            with self.subTest(value=value):
                actual_result = torch.backends.mha.set_fastpath_enabled(value)
                expected_result = (
                    reference_torch.backends.mha.set_fastpath_enabled(value)
                )
                self.assertIs(actual_result, expected_result)
                self.assertIs(actual_result, None)
                self.assertIs(torch.backends.mha.get_fastpath_enabled(), value)
                self.assertIs(
                    reference_torch.backends.mha.get_fastpath_enabled(), value
                )

        actual_marker = object()
        expected_marker = object()
        torch.backends.mha.set_fastpath_enabled(actual_marker)
        reference_torch.backends.mha.set_fastpath_enabled(expected_marker)
        with mock.patch.object(
            torch.jit, "is_scripting", return_value=True
        ), mock.patch.object(
            reference_torch.jit, "is_scripting", return_value=True
        ):
            self.assertIs(torch.backends.mha.get_fastpath_enabled(), True)
            self.assertIs(reference_torch.backends.mha.get_fastpath_enabled(), True)
        self.assertIs(torch.backends.mha.get_fastpath_enabled(), actual_marker)
        self.assertIs(
            reference_torch.backends.mha.get_fastpath_enabled(), expected_marker
        )

        self.assertEqual(
            self.threaded_outcome(torch),
            self.threaded_outcome(reference_torch),
        )

    def test_signature_documentation_and_identity_match_pytorch_2_13(self):
        actual_module = importlib.import_module("torch_rs.backends.mha")
        expected_module = importlib.import_module("torch.backends.mha")

        self.assertIs(torch.backends.mha, actual_module)
        self.assertIs(reference_torch.backends.mha, expected_module)
        self.assertIs(sys.modules[actual_module.__name__], actual_module)
        self.assertIs(sys.modules[expected_module.__name__], expected_module)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)
        self.assertEqual(
            hasattr(actual_module, "__all__"),
            hasattr(expected_module, "__all__"),
        )
        self.assertEqual(
            {name for name in vars(actual_module) if not name.startswith("_")},
            {name for name in vars(expected_module) if not name.startswith("_")},
        )

        for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
            with self.subTest(function=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)), str(inspect.signature(expected))
                )
                self.assertEqual(
                    inspect.get_annotations(actual), inspect.get_annotations(expected)
                )
                self.assertEqual(
                    typing.get_type_hints(actual), typing.get_type_hints(expected)
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_module)
                self.assertIs(inspect.getmodule(expected), expected_module)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )
                self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
                self.assertEqual(
                    actual.__code__.co_freevars, expected.__code__.co_freevars
                )
                self.assertEqual(
                    actual.__code__.co_cellvars, expected.__code__.co_cellvars
                )

    def test_imports_wildcards_copying_and_pickling_match_pytorch_2_13(self):
        actual_backends = importlib.import_module("torch_rs.backends")
        expected_backends = importlib.import_module("torch.backends")
        actual_module = importlib.import_module("torch_rs.backends.mha")
        expected_module = importlib.import_module("torch.backends.mha")

        self.assertIs(torch.backends, actual_backends)
        self.assertIs(reference_torch.backends, expected_backends)
        self.assertIs(actual_backends.mha, actual_module)
        self.assertIs(expected_backends.mha, expected_module)

        for package_name, module in (
            ("torch_rs", actual_module),
            ("torch", expected_module),
        ):
            backend_import = {}
            function_import = {}
            exec(f"from {package_name}.backends import mha", backend_import)
            exec(
                f"from {package_name}.backends.mha import "
                "get_fastpath_enabled, set_fastpath_enabled",
                function_import,
            )
            self.assertIs(backend_import["mha"], module)
            for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
                self.assertIs(function_import[name], getattr(module, name))

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs.backends import *", actual_parent_wildcard)
        exec("from torch.backends import *", expected_parent_wildcard)
        self.assertEqual(
            {
                name
                for name in actual_parent_wildcard
                if not name.startswith("__")
            },
            {
                name
                for name in expected_parent_wildcard
                if name in {"cuda", "cudnn", "mha", "mkl", "nnpack", "openmp"}
            },
        )

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.backends.mha import *", actual_child_wildcard)
        exec("from torch.backends.mha import *", expected_child_wildcard)
        self.assertEqual(
            {name for name in actual_child_wildcard if not name.startswith("__")},
            {name for name in expected_child_wildcard if not name.startswith("__")},
        )

        for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            self.assertIs(copy.copy(actual), actual)
            self.assertIs(copy.copy(expected), expected)
            self.assertIs(copy.deepcopy(actual), actual)
            self.assertIs(copy.deepcopy(expected), expected)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=name, protocol=protocol):
                    self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.mha
        old_getter = module.get_fastpath_enabled
        old_setter = module.set_fastpath_enabled
        namespace = module.__dict__
        marker = object()
        replacement = object()
        old_setter(marker)

        reloaded = importlib.reload(module)
        reset_value = module.get_fastpath_enabled()
        old_value = old_getter()
        old_set_result = old_setter(marker)
        value_after_old_setter = module.get_fastpath_enabled() is marker
        new_set_result = module.set_fastpath_enabled(replacement)
        value_from_old_getter = old_getter() is replacement

        stale_pickle_errors = []
        for function in (old_getter, old_setter):
            try:
                pickle.dumps(function)
            except Exception as error:
                stale_pickle_errors.append(
                    (
                        type(error).__name__,
                        re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                            "torch_rs", "torch"
                        ),
                    )
                )
            else:
                self.fail("a stale MHA fastpath function remained pickleable")

        return (
            parent.mha is module,
            reloaded is module,
            module.__dict__ is namespace,
            sys.modules[module.__name__] is module,
            old_getter is not module.get_fastpath_enabled,
            old_setter is not module.set_fastpath_enabled,
            reset_value is True,
            old_value is True,
            old_set_result is None,
            value_after_old_setter,
            new_set_result is None,
            value_from_old_getter,
            tuple(
                copy.copy(function) is function
                and copy.deepcopy(function) is function
                and pickle.loads(pickle.dumps(function)) is function
                for function in (
                    module.get_fastpath_enabled,
                    module.set_fastpath_enabled,
                )
            ),
            tuple(stale_pickle_errors),
        )

    def test_reload_reset_and_old_callable_behavior_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        for name in ("get_fastpath_enabled", "set_fastpath_enabled"):
            actual = getattr(torch.backends.mha, name)
            expected = getattr(reference_torch.backends.mha, name)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=name, protocol=protocol):
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_call_shape_errors_match_pytorch_2_13(self):
        actual_getter = torch.backends.mha.get_fastpath_enabled
        expected_getter = reference_torch.backends.mha.get_fastpath_enabled
        actual_setter = torch.backends.mha.set_fastpath_enabled
        expected_setter = reference_torch.backends.mha.set_fastpath_enabled
        marker = object()
        actual_setter(marker)
        expected_setter(marker)

        cases = (
            (lambda: actual_getter(None), lambda: expected_getter(None)),
            (
                lambda: actual_getter(enabled=True),
                lambda: expected_getter(enabled=True),
            ),
            (lambda: actual_setter(), lambda: expected_setter()),
            (
                lambda: actual_setter(None, None),
                lambda: expected_setter(None, None),
            ),
            (
                lambda: actual_setter(enabled=True),
                lambda: expected_setter(enabled=True),
            ),
            (
                lambda: actual_setter(None, value=True),
                lambda: expected_setter(None, value=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
                self.assertIs(torch.backends.mha.get_fastpath_enabled(), marker)
                self.assertIs(
                    reference_torch.backends.mha.get_fastpath_enabled(), marker
                )

    def test_attention_and_transformer_surface_stays_explicitly_unsupported(self):
        self.assertTrue(hasattr(reference_torch.nn, "MultiheadAttention"))
        self.assertTrue(hasattr(reference_torch.nn, "Transformer"))
        self.assertTrue(
            hasattr(reference_torch.nn.functional, "multi_head_attention_forward")
        )
        self.assertTrue(hasattr(reference_torch, "_native_multi_head_attention"))

        for name in (
            "MultiheadAttention",
            "Transformer",
            "TransformerDecoder",
            "TransformerDecoderLayer",
            "TransformerEncoder",
            "TransformerEncoderLayer",
        ):
            with self.subTest(module_name=name):
                self.assertFalse(hasattr(torch.nn, name))
                self.assertTrue(hasattr(reference_torch.nn, name))

        for name in (
            "multi_head_attention_forward",
            "scaled_dot_product_attention",
        ):
            with self.subTest(functional_name=name):
                self.assertFalse(hasattr(torch.nn.functional, name))
                self.assertTrue(hasattr(reference_torch.nn.functional, name))

        for name in ("_native_multi_head_attention", "_transformer_encoder_layer_fwd"):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))


if __name__ == "__main__":
    unittest.main()
