import copy
import importlib
import inspect
import pickle
import pickletools
import re
import sys
import types
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_CUDA_NAMES = {
    "allow_fp16_bf16_reduction_math_sdp",
    "cuBLASModule",
    "enable_flash_sdp",
    "enable_math_sdp",
    "enable_mem_efficient_sdp",
    "flash_sdp_enabled",
    "fp16_bf16_reduction_math_sdp_allowed",
    "is_built",
    "is_ck_sdpa_available",
    "is_flash_attention_available",
    "math_sdp_enabled",
    "matmul",
    "mem_efficient_sdp_enabled",
    "sdp_kernel",
}


class _BoolProbe:
    def __init__(self, label, result=True, record=None, error=None):
        self.label = label
        self.result = result
        self.record = record
        self.error = error

    def __bool__(self):
        if self.record is not None:
            self.record.append(self.label)
        if self.error is not None:
            raise self.error
        return self.result


class _TruthinessError(Exception):
    pass


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
        self.actual_original = self.states(self.actual)
        self.expected_original = self.states(self.expected)
        self.expected_cudnn_original = self.expected.cudnn_sdp_enabled()
        self.set_states(self.actual, (True, True, True))
        self.set_states(self.expected, (True, True, True))
        self.expected.enable_cudnn_sdp(True)

    def tearDown(self):
        self.set_states(self.actual, self.actual_original)
        self.set_states(self.expected, self.expected_original)
        self.expected.enable_cudnn_sdp(self.expected_cudnn_original)

    def context(self, function, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return function(*args, **kwargs)

    def context_with_warnings(self, function, *args, **kwargs):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            context = function(*args, **kwargs)
        return context, [
            (type(record.message), str(record.message)) for record in records
        ]

    def states(self, module):
        return (
            module.flash_sdp_enabled(),
            module.math_sdp_enabled(),
            module.mem_efficient_sdp_enabled(),
        )

    def set_states(self, module, states):
        flash, math, mem_efficient = states
        module.enable_flash_sdp(flash)
        module.enable_math_sdp(math)
        module.enable_mem_efficient_sdp(mem_efficient)

    def normalize(self, value):
        if isinstance(value, str):
            return value.replace("torch_rs.backends.cuda", "torch.backends.cuda")
        if isinstance(value, tuple):
            return tuple(self.normalize(item) for item in value)
        return value

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(
            self.normalize(str(actual_raised.exception)),
            str(expected_raised.exception),
        )
        self.assertEqual(
            self.normalize(actual_raised.exception.args),
            expected_raised.exception.args,
        )

    def test_signature_exports_and_imports_match_supported_pytorch_2_13_subset(self):
        actual = self.actual.sdp_kernel
        expected = self.expected.sdp_kernel

        self.assertEqual(
            self.actual.__all__,
            [name for name in self.expected.__all__ if name in SUPPORTED_CUDA_NAMES],
        )
        self.assertEqual(
            {name for name in vars(self.actual) if not name.startswith("_")},
            {
                name
                for name in vars(self.expected)
                if name in SUPPORTED_CUDA_NAMES | {"torch"}
            },
        )
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), self.actual)
        self.assertIs(inspect.getmodule(expected), self.expected)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__.keys(), expected.__dict__.keys())
        self.assertIs(actual.__dict__["__wrapped__"], actual.__wrapped__)
        self.assertIs(expected.__dict__["__wrapped__"], expected.__wrapped__)
        self.assertEqual(actual.__deprecated__, expected.__deprecated__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

        actual_wrapped = actual.__wrapped__
        expected_wrapped = expected.__wrapped__
        actual_original = actual_wrapped.__wrapped__
        expected_original = expected_wrapped.__wrapped__
        self.assertEqual(type(actual_wrapped), type(expected_wrapped))
        self.assertEqual(inspect.signature(actual_wrapped), inspect.signature(expected_wrapped))
        self.assertEqual(
            inspect.get_annotations(actual_wrapped),
            inspect.get_annotations(expected_wrapped),
        )
        self.assertEqual(actual_wrapped.__defaults__, expected_wrapped.__defaults__)
        self.assertEqual(actual_wrapped.__kwdefaults__, expected_wrapped.__kwdefaults__)
        self.assertEqual(actual_wrapped.__dict__.keys(), expected_wrapped.__dict__.keys())
        self.assertIs(actual_wrapped.__dict__["__wrapped__"], actual_original)
        self.assertIs(expected_wrapped.__dict__["__wrapped__"], expected_original)
        self.assertEqual(actual_wrapped.__deprecated__, expected_wrapped.__deprecated__)
        self.assertEqual(type(actual_original), type(expected_original))
        self.assertEqual(inspect.signature(actual_original), inspect.signature(expected_original))
        self.assertEqual(
            inspect.get_annotations(actual_original),
            inspect.get_annotations(expected_original),
        )
        self.assertEqual(actual_original.__defaults__, expected_original.__defaults__)
        self.assertEqual(actual_original.__kwdefaults__, expected_original.__kwdefaults__)

        for package_name, module in (("torch_rs", self.actual), ("torch", self.expected)):
            function_import = {}
            child_wildcard = {}
            exec(f"from {package_name}.backends.cuda import sdp_kernel", function_import)
            exec(f"from {package_name}.backends.cuda import *", child_wildcard)
            self.assertIs(function_import["sdp_kernel"], module.sdp_kernel)
            self.assertIs(child_wildcard["sdp_kernel"], module.sdp_kernel)
            self.assertEqual(
                {name for name in child_wildcard if name in SUPPORTED_CUDA_NAMES},
                SUPPORTED_CUDA_NAMES,
            )

    def test_deprecation_warning_matches_at_context_creation(self):
        self.set_states(self.actual, (False, True, False))
        self.set_states(self.expected, (False, True, False))

        actual_context, actual_warnings = self.context_with_warnings(
            self.actual.sdp_kernel,
        )
        expected_context, expected_warnings = self.context_with_warnings(
            self.expected.sdp_kernel,
        )
        self.assertEqual(actual_warnings, expected_warnings)
        self.assertEqual(self.states(self.actual), self.states(self.expected))
        self.assertEqual(self.states(self.actual), (False, True, False))

        with actual_context:
            with expected_context:
                self.assertEqual(self.states(self.actual), self.states(self.expected))
                self.assertEqual(self.states(self.actual), (True, True, True))
        self.assertEqual(self.states(self.actual), self.states(self.expected))
        self.assertEqual(self.states(self.actual), (False, True, False))

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            with self.assertRaises(FutureWarning) as actual_raised:
                self.actual.sdp_kernel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            with self.assertRaises(FutureWarning) as expected_raised:
                self.expected.sdp_kernel()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_context_transitions_match_supported_preferences(self):
        for initial in (
            (False, False, False),
            (True, False, True),
        ):
            for requested in (
                (True, True, True),
                (False, True, False),
                (True, False, False),
            ):
                with self.subTest(initial=initial, requested=requested):
                    self.set_states(self.actual, initial)
                    self.set_states(self.expected, initial)
                    actual_context = self.context(self.actual.sdp_kernel, *requested)
                    expected_context = self.context(self.expected.sdp_kernel, *requested)
                    self.assertEqual(self.states(self.actual), initial)
                    self.assertEqual(self.states(self.expected), initial)

                    self.assertEqual(actual_context.__enter__(), expected_context.__enter__())
                    self.assertEqual(
                        self.states(self.actual),
                        self.states(self.expected),
                    )
                    self.assertEqual(self.states(self.actual), requested)

                    self.assertIs(
                        actual_context.__exit__(None, None, None),
                        expected_context.__exit__(None, None, None),
                    )
                    self.assertEqual(self.states(self.actual), initial)
                    self.assertEqual(self.states(self.expected), initial)

        self.set_states(self.actual, (True, True, True))
        self.set_states(self.expected, (True, True, True))
        with self.context(self.actual.sdp_kernel, False, False, True):
            with self.context(self.expected.sdp_kernel, False, False, True):
                self.assertEqual(
                    self.states(self.actual),
                    self.states(self.expected),
                )
                self.assertEqual(self.states(self.actual), (False, False, True))
                with self.context(self.actual.sdp_kernel, True, False, False):
                    with self.context(self.expected.sdp_kernel, True, False, False):
                        self.assertEqual(
                            self.states(self.actual),
                            self.states(self.expected),
                        )
                        self.assertEqual(self.states(self.actual), (True, False, False))
                self.assertEqual(
                    self.states(self.actual),
                    self.states(self.expected),
                )
                self.assertEqual(self.states(self.actual), (False, False, True))
        self.assertEqual(self.states(self.actual), (True, True, True))
        self.assertEqual(self.states(self.expected), (True, True, True))

    def test_truthy_arguments_match_supported_preferences(self):
        cases = (
            ({"enable_flash": 1}, (True, True, True)),
            ({"enable_flash": 0}, (False, True, True)),
            ({"enable_flash": None}, (False, True, True)),
            ({"enable_math": 1}, (True, True, True)),
            ({"enable_math": 0}, (True, False, True)),
            ({"enable_math": None}, (True, False, True)),
            ({"enable_mem_efficient": object()}, (True, True, True)),
            ({"enable_mem_efficient": []}, (True, True, False)),
            ({"enable_mem_efficient": None}, (True, True, False)),
            ({"enable_cudnn": object()}, (True, True, True)),
            ({"enable_cudnn": 0}, (True, True, True)),
            (
                {
                    "enable_flash": False,
                    "enable_math": "",
                    "enable_mem_efficient": (1,),
                    "enable_cudnn": None,
                },
                (False, False, True),
            ),
        )

        for kwargs, requested in cases:
            with self.subTest(kwargs=kwargs):
                self.set_states(self.actual, (False, True, False))
                self.set_states(self.expected, (False, True, False))
                actual_context = self.context(self.actual.sdp_kernel, **kwargs)
                expected_context = self.context(self.expected.sdp_kernel, **kwargs)
                self.assertEqual(self.states(self.actual), self.states(self.expected))
                self.assertEqual(self.states(self.actual), (False, True, False))

                self.assertEqual(actual_context.__enter__(), expected_context.__enter__())
                self.assertEqual(self.states(self.actual), self.states(self.expected))
                self.assertEqual(self.states(self.actual), requested)
                self.assertIs(
                    actual_context.__exit__(None, None, None),
                    expected_context.__exit__(None, None, None),
                )
                self.assertEqual(self.states(self.actual), self.states(self.expected))
                self.assertEqual(self.states(self.actual), (False, True, False))

    def test_truthiness_order_and_errors_match_without_state_mutation(self):
        actual_order = []
        expected_order = []
        self.set_states(self.actual, (False, True, False))
        self.set_states(self.expected, (False, True, False))
        actual_context = self.context(
            self.actual.sdp_kernel,
            enable_flash=_BoolProbe("flash", result=False, record=actual_order),
            enable_math=_BoolProbe("math", result=False, record=actual_order),
            enable_mem_efficient=_BoolProbe(
                "mem_efficient",
                result=True,
                record=actual_order,
            ),
            enable_cudnn=_BoolProbe("cudnn", result=False, record=actual_order),
        )
        expected_context = self.context(
            self.expected.sdp_kernel,
            enable_flash=_BoolProbe("flash", result=False, record=expected_order),
            enable_math=_BoolProbe("math", result=False, record=expected_order),
            enable_mem_efficient=_BoolProbe(
                "mem_efficient",
                result=True,
                record=expected_order,
            ),
            enable_cudnn=_BoolProbe("cudnn", result=False, record=expected_order),
        )

        self.assertEqual(actual_context.__enter__(), expected_context.__enter__())
        self.assertEqual(actual_order, expected_order)
        self.assertEqual(actual_order, ["flash", "mem_efficient", "math", "cudnn"])
        self.assertEqual(self.states(self.actual), self.states(self.expected))
        self.assertEqual(self.states(self.actual), (False, False, True))
        self.assertIs(
            actual_context.__exit__(None, None, None),
            expected_context.__exit__(None, None, None),
        )
        self.assertEqual(self.states(self.actual), self.states(self.expected))
        self.assertEqual(self.states(self.actual), (False, True, False))

        for parameter in (
            "enable_flash",
            "enable_math",
            "enable_mem_efficient",
            "enable_cudnn",
        ):
            with self.subTest(parameter=parameter):
                self.set_states(self.actual, (False, True, False))
                self.set_states(self.expected, (False, True, False))
                actual_context = self.context(
                    self.actual.sdp_kernel,
                    **{
                        parameter: _BoolProbe(
                            parameter,
                            error=_TruthinessError(
                                f"{parameter} truthiness failed"
                            ),
                        )
                    },
                )
                expected_context = self.context(
                    self.expected.sdp_kernel,
                    **{
                        parameter: _BoolProbe(
                            parameter,
                            error=_TruthinessError(
                                f"{parameter} truthiness failed"
                            ),
                        )
                    },
                )
                self.assert_error_matches(
                    actual_context.__enter__,
                    expected_context.__enter__,
                )
                self.assertEqual(self.states(self.actual), self.states(self.expected))
                self.assertEqual(self.states(self.actual), (False, True, False))

    def test_errors_copying_pickling_and_reload_match_supported_shape(self):
        actual = self.actual.sdp_kernel
        expected = self.expected.sdp_kernel
        cases = (
            (
                lambda: self.context(actual, True, True, True, True, True),
                lambda: self.context(expected, True, True, True, True, True),
            ),
            (
                lambda: self.context(actual, _enabled=False),
                lambda: self.context(expected, _enabled=False),
            ),
            (
                lambda: self.context(actual, True, enable_flash=False),
                lambda: self.context(expected, True, enable_flash=False),
            ),
        )
        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)

        actual_context = self.context(actual, False, True, False)
        expected_context = self.context(expected, False, True, False)
        self.assertEqual(type(actual_context), type(expected_context))
        self.assertEqual(actual_context.__doc__, expected_context.__doc__)
        self.assertIs(actual_context.func, actual.__wrapped__)
        self.assertIs(expected_context.func, expected.__wrapped__)
        self.assertEqual(actual_context.args, expected_context.args)
        self.assertEqual(actual_context.kwds, expected_context.kwds)

        actual_old = actual
        expected_old = expected
        actual_reloaded = importlib.reload(self.actual)
        expected_reloaded = importlib.reload(self.expected)
        try:
            pickle.dumps(actual_old)
        except Exception as error:
            actual_stale_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale torch_rs sdp_kernel function remained pickleable")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                pickle.dumps(expected_old)
        except Exception as error:
            expected_stale_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)),
            )
        else:
            self.fail("a stale PyTorch sdp_kernel function remained pickleable")

        self.assertIs(actual_reloaded, self.actual)
        self.assertIs(expected_reloaded, self.expected)
        self.assertIsNot(self.actual.sdp_kernel, actual_old)
        self.assertIsNot(self.expected.sdp_kernel, expected_old)
        self.assertEqual(actual_stale_error, expected_stale_error)

    def test_cudnn_sdpa_helpers_and_execution_remain_unsupported(self):
        self.assertTrue(hasattr(self.expected, "sdp_kernel"))
        self.assertTrue(hasattr(self.actual, "sdp_kernel"))
        for name in (
            "SDPAParams",
            "can_use_cudnn_attention",
            "can_use_efficient_attention",
            "can_use_flash_attention",
            "cudnn_sdp_enabled",
            "enable_cudnn_sdp",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(self.actual, name))
                self.assertTrue(hasattr(self.expected, name))

        self.assertFalse(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "scaled_dot_product_attention"))
        self.assertTrue(hasattr(reference_torch, "cuda"))


if __name__ == "__main__":
    unittest.main()
