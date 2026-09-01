import copy
import importlib
import pickle
import pickletools
import re
import sys
import threading
import types
import unittest

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_CUDA_NAMES = {
    "allow_fp16_bf16_reduction_math_sdp",
    "cuBLASModule",
    "cudnn_sdp_enabled",
    "enable_cudnn_sdp",
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
}
MATMUL_ATTRIBUTES = (
    "allow_tf32",
    "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction",
)


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cuda.matmul preferences must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaMatmulPreferenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cuda.matmul differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.actual_precision_original = torch.get_float32_matmul_precision()
        self.expected_precision_original = (
            reference_torch.get_float32_matmul_precision()
        )
        self.actual_original = self.states(self.actual.matmul)
        self.expected_original = self.states(self.expected.matmul)
        torch.set_float32_matmul_precision("highest")
        reference_torch.set_float32_matmul_precision("highest")
        self.set_states(self.actual.matmul, (False, True, True))
        self.set_states(self.expected.matmul, (False, True, True))

    def tearDown(self):
        self.set_states(self.actual.matmul, self.actual_original)
        self.set_states(self.expected.matmul, self.expected_original)
        torch.set_float32_matmul_precision(self.actual_precision_original)
        reference_torch.set_float32_matmul_precision(
            self.expected_precision_original
        )

    def normalize(self, value):
        if isinstance(value, str):
            return (
                value.replace("torch_rs.torch_rs", "torch._C")
                .replace("torch_rs.backends.cuda", "torch.backends.cuda")
                .replace("torch_rs", "torch")
            )
        if isinstance(value, tuple):
            return tuple(self.normalize(item) for item in value)
        return value

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

    def pickle_shape(self, obj, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(obj, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = self.normalize(argument)
            shape.append((opcode.name, argument))
        return shape

    def states(self, matmul):
        return tuple(getattr(matmul, name) for name in MATMUL_ATTRIBUTES)

    def set_states(self, matmul, states):
        for name, state in zip(MATMUL_ATTRIBUTES, states):
            setattr(matmul, name, state)

    def test_defaults_and_transitions_match_pytorch_2_13(self):
        self.assertEqual(
            self.states(self.actual.matmul),
            self.states(self.expected.matmul),
        )
        for states in (
            (True, False, False),
            (False, True, True),
            (False, False, True),
            (True, True, False),
            (False, True, True),
        ):
            with self.subTest(states=states):
                self.set_states(self.actual.matmul, states)
                self.set_states(self.expected.matmul, states)
                self.assertEqual(
                    self.states(self.actual.matmul),
                    self.states(self.expected.matmul),
                )
                self.assertEqual(
                    tuple(type(value) for value in self.states(self.actual.matmul)),
                    tuple(type(value) for value in self.states(self.expected.matmul)),
                )
                self.assertEqual(
                    torch.get_float32_matmul_precision(),
                    reference_torch.get_float32_matmul_precision(),
                )

    def test_invalid_assignments_match_pytorch_2_13(self):
        actual_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
            torch.tensor(True),
            torch.float32,
            torch.device("cpu"),
            torch.strided,
            torch.Size([1]),
            torch.finfo(torch.float32),
        )
        expected_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
            reference_torch.tensor(True),
            reference_torch.float32,
            reference_torch.device("cpu"),
            reference_torch.strided,
            reference_torch.Size([1]),
            reference_torch.finfo(reference_torch.float32),
        )
        for state in (False, True):
            self.actual.matmul.allow_tf32 = state
            self.expected.matmul.allow_tf32 = state
            for case, (actual_value, expected_value) in enumerate(
                zip(actual_values, expected_values)
            ):
                with self.subTest(attribute="allow_tf32", state=state, case=case):
                    self.assert_error_matches(
                        lambda value=actual_value: setattr(
                            self.actual.matmul, "allow_tf32", value
                        ),
                        lambda value=expected_value: setattr(
                            self.expected.matmul, "allow_tf32", value
                        ),
                    )
                    self.assertIs(self.actual.matmul.allow_tf32, state)
                    self.assertIs(self.expected.matmul.allow_tf32, state)

        for attribute in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ):
            for value in (
                None,
                0,
                1,
                0.0,
                np.bool_(True),
                "",
                [],
                object(),
                _RejectTruthiness(),
            ):
                with self.subTest(attribute=attribute, value=type(value).__name__):
                    self.set_states(self.actual.matmul, (False, True, True))
                    self.set_states(self.expected.matmul, (False, True, True))
                    before_actual = self.states(self.actual.matmul)
                    before_expected = self.states(self.expected.matmul)
                    self.assert_error_matches(
                        lambda value=value, attribute=attribute: setattr(
                            self.actual.matmul, attribute, value
                        ),
                        lambda value=value, attribute=attribute: setattr(
                            self.expected.matmul, attribute, value
                        ),
                    )
                    self.assertEqual(self.states(self.actual.matmul), before_actual)
                    self.assertEqual(self.states(self.expected.matmul), before_expected)

            for actual_value, expected_value in (
                (torch.tensor(True), reference_torch.tensor(True)),
                (torch.float32, reference_torch.float32),
                (torch.device("cpu"), reference_torch.device("cpu")),
                (torch.strided, reference_torch.strided),
                (torch.Size([1]), reference_torch.Size([1])),
                (
                    torch.finfo(torch.float32),
                    reference_torch.finfo(reference_torch.float32),
                ),
            ):
                with self.subTest(
                    attribute=attribute,
                    value=type(actual_value).__name__,
                ):
                    self.set_states(self.actual.matmul, (False, True, True))
                    self.set_states(self.expected.matmul, (False, True, True))
                    before_actual = self.states(self.actual.matmul)
                    before_expected = self.states(self.expected.matmul)
                    self.assert_error_matches(
                        lambda value=actual_value, attribute=attribute: setattr(
                            self.actual.matmul, attribute, value
                        ),
                        lambda value=expected_value, attribute=attribute: setattr(
                            self.expected.matmul, attribute, value
                        ),
                    )
                    self.assertEqual(self.states(self.actual.matmul), before_actual)
                    self.assertEqual(self.states(self.expected.matmul), before_expected)

    def test_reduction_tuple_assignments_match_pytorch_2_13(self):
        for attribute in (
            "allow_fp16_reduced_precision_reduction",
            "allow_bf16_reduced_precision_reduction",
        ):
            for value in (
                (False,),
                [False],
                (False, False),
                [False, True],
                (True, True),
            ):
                with self.subTest(attribute=attribute, value=value):
                    setattr(self.actual.matmul, attribute, value)
                    setattr(self.expected.matmul, attribute, value)
                    self.assertEqual(
                        self.states(self.actual.matmul),
                        self.states(self.expected.matmul),
                    )
            self.assert_error_matches(
                lambda attribute=attribute: setattr(
                    self.actual.matmul, attribute, (True, False)
                ),
                lambda attribute=attribute: setattr(
                    self.expected.matmul, attribute, (True, False)
                ),
            )

    def thread_contract(self, module):
        self.set_states(module.matmul, (False, True, True))
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(self.states(module.matmul))
                self.set_states(module.matmul, (True, False, False))
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self.states(module.matmul))
                self.set_states(module.matmul, (False, False, True))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        state_after_worker = self.states(module.matmul)
        self.set_states(module.matmul, (False, True, False))
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            state_after_worker,
            not thread.is_alive(),
            errors,
            observations,
            self.states(module.matmul),
        )

    def test_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def reload_contract(self, root):
        parent = root.backends
        module = parent.cuda
        old_matmul = module.matmul
        old_class = module.cuBLASModule
        namespace = module.__dict__
        self.set_states(module.matmul, (True, False, False))

        reloaded = importlib.reload(module)
        new_matmul = module.matmul
        self.set_states(new_matmul, (False, True, True))
        old_sees_new = self.states(old_matmul)

        try:
            pickle.dumps(old_matmul)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale cuBLASModule instance remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cuda is module,
            sys.modules[module.__name__] is module,
            module.cuBLASModule is not old_class,
            new_matmul is not old_matmul,
            self.states(new_matmul),
            old_sees_new,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_metadata_imports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in SUPPORTED_CUDA_NAMES],
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name in SUPPORTED_CUDA_NAMES | {"torch"}
            },
        )
        self.assertIs(type(actual.matmul), actual.cuBLASModule)
        self.assertIs(type(expected.matmul), expected.cuBLASModule)
        self.assertEqual(actual.cuBLASModule.__name__, expected.cuBLASModule.__name__)
        self.assertEqual(
            actual.cuBLASModule.__qualname__,
            expected.cuBLASModule.__qualname__,
        )
        self.assertEqual(
            actual.cuBLASModule.__module__.replace("torch_rs", "torch"),
            expected.cuBLASModule.__module__,
        )
        self.assertEqual(vars(actual.matmul), vars(expected.matmul))
        for name in MATMUL_ATTRIBUTES:
            self.assertEqual(name in dir(actual.matmul), name in dir(expected.matmul))

        for package_name, module in (("torch_rs", actual), ("torch", expected)):
            object_import = {}
            wildcard = {}
            exec(
                f"from {package_name}.backends.cuda import cuBLASModule, matmul",
                object_import,
            )
            exec(f"from {package_name}.backends.cuda import *", wildcard)
            self.assertIs(object_import["cuBLASModule"], module.cuBLASModule)
            self.assertIs(object_import["matmul"], module.matmul)
            self.assertEqual(
                {name for name in wildcard if name in SUPPORTED_CUDA_NAMES},
                SUPPORTED_CUDA_NAMES,
            )

        for obj_name in ("cuBLASModule", "matmul"):
            actual_obj = getattr(actual, obj_name)
            expected_obj = getattr(expected, obj_name)
            with self.subTest(obj=obj_name):
                self.assertIs(
                    copy.copy(actual_obj) is actual_obj,
                    copy.copy(expected_obj) is expected_obj,
                )
                self.assertIs(
                    copy.deepcopy(actual_obj) is actual_obj,
                    copy.deepcopy(expected_obj) is expected_obj,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertEqual(
                        self.pickle_shape(actual_obj, protocol),
                        self.pickle_shape(expected_obj, protocol),
                    )
                    self.assertIs(
                        type(pickle.loads(pickle.dumps(actual_obj, protocol))),
                        type(actual_obj)
                        if obj_name == "matmul"
                        else type(actual.cuBLASModule),
                    )

        for name in (
            "_get_cublas_allow_tf32",
            "_set_cublas_allow_tf32",
            "_get_cublas_allow_fp16_reduced_precision_reduction",
            "_set_cublas_allow_fp16_reduced_precision_reduction",
            "_get_cublas_allow_bf16_reduced_precision_reduction",
            "_set_cublas_allow_bf16_reduced_precision_reduction",
        ):
            with self.subTest(private_accessor=name):
                actual_function = getattr(torch._C, name)
                expected_function = getattr(reference_torch._C, name)
                self.assertIs(type(actual_function), types.BuiltinFunctionType)
                self.assertIs(type(expected_function), types.BuiltinFunctionType)
                self.assertEqual(actual_function.__name__, expected_function.__name__)
                self.assertEqual(
                    actual_function.__qualname__,
                    expected_function.__qualname__,
                )
                self.assertEqual(
                    self.normalize(actual_function.__module__),
                    expected_function.__module__,
                )
                self.assertEqual(actual_function.__doc__, expected_function.__doc__)
                self.assertEqual(
                    getattr(actual_function, "__text_signature__", None),
                    getattr(expected_function, "__text_signature__", None),
                )
                self.assertIs(copy.copy(actual_function), actual_function)
                self.assertIs(copy.copy(expected_function), expected_function)
                self.assertIs(copy.deepcopy(actual_function), actual_function)
                self.assertIs(copy.deepcopy(expected_function), expected_function)
                self.assertIs(
                    pickle.loads(pickle.dumps(actual_function)),
                    actual_function,
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(expected_function)),
                    expected_function,
                )
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
                self.assertNotIn(name, torch._C.__all__)

    def test_private_accessor_binding_errors_match_pytorch_2_13(self):
        for actual_call, expected_call in (
            (
                lambda: torch._C._get_cublas_allow_tf32(None),
                lambda: reference_torch._C._get_cublas_allow_tf32(None),
            ),
            (
                lambda: torch._C._get_cublas_allow_tf32(value=None),
                lambda: reference_torch._C._get_cublas_allow_tf32(value=None),
            ),
            (
                lambda: torch._C._set_cublas_allow_tf32(),
                lambda: reference_torch._C._set_cublas_allow_tf32(),
            ),
            (
                lambda: torch._C._set_cublas_allow_tf32(True, False),
                lambda: reference_torch._C._set_cublas_allow_tf32(True, False),
            ),
            (
                lambda: torch._C._set_cublas_allow_tf32(object=False),
                lambda: reference_torch._C._set_cublas_allow_tf32(object=False),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

        for name in (
            "_set_cublas_allow_fp16_reduced_precision_reduction",
            "_set_cublas_allow_bf16_reduced_precision_reduction",
        ):
            actual = getattr(torch._C, name)
            expected = getattr(reference_torch._C, name)
            for actual_call, expected_call in (
                (
                    lambda actual=actual: actual(),
                    lambda expected=expected: expected(),
                ),
                (
                    lambda actual=actual: actual(False, False, False),
                    lambda expected=expected: expected(False, False, False),
                ),
                (
                    lambda actual=actual: actual(allow_reduced_precision=False),
                    lambda expected=expected: expected(allow_reduced_precision=False),
                ),
                (
                    lambda actual=actual: actual(False, allow_splitk=False),
                    lambda expected=expected: expected(False, allow_splitk=False),
                ),
            ):
                with self.subTest(name=name):
                    self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
