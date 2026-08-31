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


MATMUL_ATTRS = (
    "allow_tf32",
    "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction",
)


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("matmul preference assignment must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudaMatmulPreferenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "CUDA matmul preference differentials require pinned "
                "PyTorch 2.13.0"
            )

    def setUp(self):
        self.actual = importlib.import_module("torch_rs.backends.cuda")
        self.expected = importlib.import_module("torch.backends.cuda")
        self.original_actual = self.native_state(torch)
        self.original_expected = self.native_state(reference_torch)
        self.set_native_state(torch, False, (True, True), (True, True))
        self.set_native_state(reference_torch, False, (True, True), (True, True))

    def tearDown(self):
        self.set_native_state(torch, *self.original_actual)
        self.set_native_state(reference_torch, *self.original_expected)

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
                str(arg).replace("torch_rs", "torch")
                if isinstance(arg, str)
                else arg
                for arg in actual_raised.exception.args
            ),
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
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def native_state(self, root):
        return (
            root._C._get_cublas_allow_tf32(),
            root._C._get_cublas_allow_fp16_reduced_precision_reduction(),
            root._C._get_cublas_allow_bf16_reduced_precision_reduction(),
        )

    def set_native_state(self, root, allow_tf32, fp16_state, bf16_state):
        root._C._set_cublas_allow_tf32(allow_tf32)
        root._C._set_cublas_allow_fp16_reduced_precision_reduction(*fp16_state)
        root._C._set_cublas_allow_bf16_reduced_precision_reduction(*bf16_state)

    def matmul_state(self, matmul):
        return tuple(getattr(matmul, name) for name in MATMUL_ATTRS)

    def transition_contract(self, root):
        matmul = root.backends.cuda.matmul
        states = []
        for name in MATMUL_ATTRS:
            for enabled in (False, True, True, False):
                setattr(matmul, name, enabled)
                states.append(
                    (
                        name,
                        getattr(matmul, name),
                        type(getattr(matmul, name)) is bool,
                        self.native_state(root),
                    )
                )
        return states

    def tuple_list_contract(self, root):
        matmul = root.backends.cuda.matmul
        outcomes = []
        for name in MATMUL_ATTRS[1:]:
            for value in ((True,), [True], (False,), [False], (False, False), [False, True]):
                setattr(matmul, name, True)
                result = setattr(matmul, name, value)
                outcomes.append((name, tuple(value), result, getattr(matmul, name)))
        return outcomes

    def invalid_contract(self, root, values):
        matmul = root.backends.cuda.matmul
        errors = []
        for state in (False, True):
            for name in MATMUL_ATTRS:
                setattr(matmul, name, state)
            for actual_value in values:
                try:
                    matmul.allow_tf32 = actual_value
                except Exception as error:
                    errors.append(
                        (
                            "allow_tf32",
                            state,
                            type(error).__name__,
                            str(error),
                            error.args,
                            matmul.allow_tf32,
                        )
                    )
                else:
                    self.fail("allow_tf32 unexpectedly accepted an invalid value")

                for name in MATMUL_ATTRS[1:]:
                    try:
                        setattr(matmul, name, actual_value)
                    except Exception as error:
                        errors.append(
                            (
                                name,
                                state,
                                type(error).__name__,
                                str(error),
                                error.args,
                                getattr(matmul, name),
                            )
                        )
                    else:
                        self.fail(f"{name} unexpectedly accepted an invalid value")

        edge_values = (
            (),
            [],
            (1,),
            [1],
            (True, 1),
            [True, 1],
            (False, False, False),
            [False, False, False],
            (True, False),
            [True, False],
        )
        for name in MATMUL_ATTRS[1:]:
            for value in edge_values:
                setattr(matmul, name, True)
                try:
                    setattr(matmul, name, value)
                except Exception as error:
                    errors.append(
                        (
                            name,
                            tuple(value),
                            type(error).__name__,
                            str(error),
                            error.args,
                            getattr(matmul, name),
                        )
                    )
                else:
                    self.fail(f"{name} unexpectedly accepted {value!r}")
        return errors

    def thread_contract(self, module):
        matmul = module.matmul
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        def worker():
            try:
                observations.append(self.matmul_state(matmul))
                matmul.allow_tf32 = True
                matmul.allow_fp16_reduced_precision_reduction = False
                matmul.allow_bf16_reduced_precision_reduction = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(self.matmul_state(matmul))
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        state_after_worker = self.matmul_state(matmul)
        matmul.allow_tf32 = False
        matmul.allow_fp16_reduced_precision_reduction = True
        matmul.allow_bf16_reduced_precision_reduction = True
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            state_after_worker,
            not thread.is_alive(),
            errors,
            observations,
            self.matmul_state(matmul),
        )

    def reload_contract(self, root, module):
        matmul = module.matmul
        old_class = module.cuBLASModule
        namespace = module.__dict__
        matmul.allow_tf32 = True
        matmul.allow_fp16_reduced_precision_reduction = False
        matmul.allow_bf16_reduced_precision_reduction = False
        reloaded = importlib.reload(module)
        new_matmul = module.matmul
        stale_errors = []
        for stale in (matmul, old_class):
            try:
                pickle.dumps(stale)
            except Exception as error:
                stale_errors.append(
                    (
                        type(error).__name__,
                        re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                            "torch_rs", "torch"
                        ),
                    )
                )
            else:
                self.fail("stale cuBLASModule object remained pickleable")
        matmul.allow_tf32 = False
        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.backends.cuda is module,
            sys.modules[module.__name__] is module,
            new_matmul is not matmul,
            module.cuBLASModule is not old_class,
            type(new_matmul) is module.cuBLASModule,
            self.matmul_state(new_matmul),
            self.matmul_state(matmul),
            stale_errors,
        )

    def test_defaults_transitions_threads_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.matmul_state(self.actual.matmul),
            self.matmul_state(self.expected.matmul),
        )
        self.assertEqual(
            self.transition_contract(torch),
            self.transition_contract(reference_torch),
        )
        self.set_native_state(torch, False, (True, True), (True, True))
        self.set_native_state(reference_torch, False, (True, True), (True, True))
        self.assertEqual(
            self.tuple_list_contract(torch),
            self.tuple_list_contract(reference_torch),
        )
        self.set_native_state(torch, False, (True, True), (True, True))
        self.set_native_state(reference_torch, False, (True, True), (True, True))
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )
        self.set_native_state(torch, False, (True, True), (True, True))
        self.set_native_state(reference_torch, False, (True, True), (True, True))
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_invalid_assignments_match_pytorch_2_13(self):
        actual_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            object(),
            _RejectTruthiness(),
            torch.tensor(True),
            torch.float32,
            torch.device("cpu"),
            torch.strided,
            torch.finfo(torch.float32),
        )
        expected_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            object(),
            _RejectTruthiness(),
            reference_torch.tensor(True),
            reference_torch.float32,
            reference_torch.device("cpu"),
            reference_torch.strided,
            reference_torch.finfo(reference_torch.float32),
        )
        actual_errors = self.invalid_contract(torch, actual_values)
        expected_errors = self.invalid_contract(reference_torch, expected_values)
        self.assertEqual(
            [
                (
                    *entry[:3],
                    entry[3].replace("torch_rs", "torch"),
                    tuple(
                        arg.replace("torch_rs", "torch")
                        if isinstance(arg, str)
                        else arg
                        for arg in entry[4]
                    ),
                    entry[5],
                )
                for entry in actual_errors
            ],
            expected_errors,
        )

    def test_metadata_exports_copying_and_pickling_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        supported = {
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
        }

        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in supported],
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name in supported | {"torch"}
            },
        )
        self.assertIs(type(actual.matmul), actual.cuBLASModule)
        self.assertIs(type(expected.matmul), expected.cuBLASModule)
        self.assertEqual(
            type(actual.matmul).__module__.replace("torch_rs", "torch"),
            type(expected.matmul).__module__,
        )
        self.assertEqual(
            type(actual.matmul).__qualname__,
            type(expected.matmul).__qualname__,
        )
        self.assertEqual(vars(actual.matmul), vars(expected.matmul))
        self.assertEqual(actual.matmul.__doc__, expected.matmul.__doc__)
        for name in ("__name__", "__all__"):
            self.assert_error_matches(
                lambda name=name: getattr(actual.matmul, name),
                lambda name=name: getattr(expected.matmul, name),
            )

        for package_name, module in (("torch_rs", actual), ("torch", expected)):
            backend_import = {}
            class_import = {}
            matmul_import = {}
            wildcard = {}
            exec(f"from {package_name}.backends import cuda", backend_import)
            exec(f"from {package_name}.backends.cuda import cuBLASModule", class_import)
            exec(f"from {package_name}.backends.cuda import matmul", matmul_import)
            exec(f"from {package_name}.backends.cuda import *", wildcard)
            self.assertIs(backend_import["cuda"], module)
            self.assertIs(class_import["cuBLASModule"], module.cuBLASModule)
            self.assertIs(matmul_import["matmul"], module.matmul)
            self.assertIs(wildcard["cuBLASModule"], module.cuBLASModule)
            self.assertIs(wildcard["matmul"], module.matmul)
            with self.assertRaises(ImportError):
                exec(f"from {package_name}.backends.cuda import allow_tf32", {})
            with self.assertRaises(ModuleNotFoundError):
                importlib.import_module(f"{package_name}.backends.cuda.matmul")

        for actual_object, expected_object in (
            (actual.cuBLASModule, expected.cuBLASModule),
            (actual.matmul, expected.matmul),
        ):
            with self.subTest(object=type(actual_object).__name__):
                self.assertEqual(
                    copy.copy(actual_object) is actual_object,
                    copy.copy(expected_object) is expected_object,
                )
                self.assertEqual(
                    copy.deepcopy(actual_object) is actual_object,
                    copy.deepcopy(expected_object) is expected_object,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertEqual(
                        pickle.loads(pickle.dumps(actual_object, protocol))
                        is actual_object,
                        pickle.loads(pickle.dumps(expected_object, protocol))
                        is expected_object,
                    )
                    self.assertEqual(
                        self.pickle_shape(actual_object, protocol),
                        self.pickle_shape(expected_object, protocol),
                    )

    def test_private_accessors_and_unsupported_boundaries_match_scope(self):
        for root in (torch, reference_torch):
            for getter_name, setter_name in (
                ("_get_cublas_allow_tf32", "_set_cublas_allow_tf32"),
                (
                    "_get_cublas_allow_fp16_reduced_precision_reduction",
                    "_set_cublas_allow_fp16_reduced_precision_reduction",
                ),
                (
                    "_get_cublas_allow_bf16_reduced_precision_reduction",
                    "_set_cublas_allow_bf16_reduced_precision_reduction",
                ),
            ):
                self.assertTrue(hasattr(root._C, getter_name))
                self.assertTrue(hasattr(root._C, setter_name))
                self.assertFalse(hasattr(root, getter_name))
                self.assertFalse(hasattr(root, setter_name))
                if hasattr(root._C, "__all__"):
                    self.assertNotIn(getter_name, root._C.__all__)
                    self.assertNotIn(setter_name, root._C.__all__)

        actual_matmul = self.actual.matmul
        expected_matmul = self.expected.matmul
        for name in (
            "allow_fp16_accumulation",
            "fp32_precision",
            "allow_fp16_reduced_precision_reduction_split_k",
            "allow_bf16_reduced_precision_reduction_split_k",
        ):
            with self.subTest(unsupported=name):
                self.assertFalse(hasattr(actual_matmul, name))
                self.assertTrue(hasattr(expected_matmul, name))
                with self.assertRaises(AttributeError):
                    setattr(actual_matmul, name, True)

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertFalse(hasattr(self.actual, "enable_cudnn_sdp"))
        self.assertFalse(hasattr(self.actual, "cudnn_sdp_enabled"))

        self.assertTrue(hasattr(reference_torch, "cuda"))
        self.assertTrue(hasattr(reference_torch.Tensor, "cuda"))
        self.assertTrue(hasattr(reference_torch.Tensor, "to"))
        self.assertTrue(hasattr(reference_torch, "compile"))
        self.assertTrue(
            hasattr(reference_torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertTrue(hasattr(self.expected, "enable_cudnn_sdp"))
        self.assertTrue(hasattr(self.expected, "cudnn_sdp_enabled"))


if __name__ == "__main__":
    unittest.main()
