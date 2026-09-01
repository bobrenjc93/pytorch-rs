import copy
import importlib
import inspect
import json
import pickle
import re
import subprocess
import sys
import threading
import types
import typing
import unittest

import numpy as np

import torch_rs as torch


REDUCTION_ALLOWED_DOC = """
    .. warning:: This flag is beta and subject to change.

    Returns whether fp16/bf16 reduction in math scaled dot product attention is enabled or not.
    """

ALLOW_REDUCTION_DOC = """
    .. warning:: This flag is beta and subject to change.

    Enables or disables fp16/bf16 reduction in math scaled dot product attention.
    """

if sys.version_info >= (3, 13):
    REDUCTION_ALLOWED_DOC = "\n" + inspect.cleandoc(REDUCTION_ALLOWED_DOC) + "\n"
    ALLOW_REDUCTION_DOC = "\n" + inspect.cleandoc(ALLOW_REDUCTION_DOC) + "\n"


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("the reduction setter must not request truthiness")


class CudaReducedPrecisionMathSdpTests(unittest.TestCase):
    def setUp(self):
        self.cuda = importlib.import_module("torch_rs.backends.cuda")
        self.original = torch._C._get_math_sdp_allow_fp16_bf16_reduction()
        self.original_flash = torch._C._get_flash_sdp_enabled()
        self.original_math = torch._C._get_math_sdp_enabled()
        self.original_mem_efficient = torch._C._get_mem_efficient_sdp_enabled()
        self.original_cudnn = torch._C._get_cudnn_sdp_enabled()
        self.cuda.allow_fp16_bf16_reduction_math_sdp(False)

    def tearDown(self):
        self.cuda.allow_fp16_bf16_reduction_math_sdp(self.original)
        self.cuda.enable_flash_sdp(self.original_flash)
        self.cuda.enable_math_sdp(self.original_math)
        self.cuda.enable_mem_efficient_sdp(self.original_mem_efficient)
        self.cuda.enable_cudnn_sdp(self.original_cudnn)

    def test_fresh_process_defaults_to_exact_false_without_execution_support(self):
        script = r'''
import json

import torch_rs as torch

cuda = torch.backends.cuda
flash_before = cuda.flash_sdp_enabled()
math_before = cuda.math_sdp_enabled()
mem_efficient_before = cuda.mem_efficient_sdp_enabled()
cudnn_before = cuda.cudnn_sdp_enabled()
initial = cuda.fp16_bf16_reduction_math_sdp_allowed()
first = cuda.allow_fp16_bf16_reduction_math_sdp(True)
enabled = cuda.fp16_bf16_reduction_math_sdp_allowed()
second = cuda.allow_fp16_bf16_reduction_math_sdp(False)
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "first": first,
    "enabled": enabled,
    "second": second,
    "restored": cuda.fp16_bf16_reduction_math_sdp_allowed(),
    "flash_unchanged": cuda.flash_sdp_enabled() is flash_before,
    "math_unchanged": cuda.math_sdp_enabled() is math_before,
    "mem_efficient_unchanged": (
        cuda.mem_efficient_sdp_enabled() is mem_efficient_before
    ),
    "cudnn_unchanged": cuda.cudnn_sdp_enabled() is cudnn_before,
    "built": cuda.is_built(),
    "ck_available": cuda.is_ck_sdpa_available(),
    "flash_available": cuda.is_flash_attention_available(),
    "float16": hasattr(torch, "float16"),
    "bfloat16": hasattr(torch, "bfloat16"),
    "cuda": hasattr(torch, "cuda"),
    "execution": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "initial": False,
                "initial_type": "bool",
                "first": None,
                "enabled": True,
                "second": None,
                "restored": False,
                "flash_unchanged": True,
                "math_unchanged": True,
                "mem_efficient_unchanged": True,
                "cudnn_unchanged": True,
                "built": False,
                "ck_available": False,
                "flash_available": False,
                "float16": False,
                "bfloat16": False,
                "cuda": True,
                "execution": False,
            },
        )

    def test_exact_bool_updates_are_independent_preferences(self):
        cuda = self.cuda
        flash_state = cuda.flash_sdp_enabled()
        math_state = cuda.math_sdp_enabled()
        mem_efficient_state = cuda.mem_efficient_sdp_enabled()
        cudnn_state = cuda.cudnn_sdp_enabled()

        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), False)
        self.assertIs(type(cuda.fp16_bf16_reduction_math_sdp_allowed()), bool)
        for enabled in (True, False, False, True, True, False):
            with self.subTest(enabled=enabled):
                self.assertIs(
                    cuda.allow_fp16_bf16_reduction_math_sdp(enabled),
                    None,
                )
                self.assertIs(
                    cuda.fp16_bf16_reduction_math_sdp_allowed(),
                    enabled,
                )
                self.assertIs(
                    torch._C._get_math_sdp_allow_fp16_bf16_reduction(),
                    enabled,
                )
                self.assertIs(cuda.flash_sdp_enabled(), flash_state)
                self.assertIs(cuda.math_sdp_enabled(), math_state)
                self.assertIs(
                    cuda.mem_efficient_sdp_enabled(),
                    mem_efficient_state,
                )
                self.assertIs(cuda.cudnn_sdp_enabled(), cudnn_state)
                self.assertIs(cuda.is_built(), False)
                self.assertIs(cuda.is_ck_sdpa_available(), False)
                self.assertIs(cuda.is_flash_attention_available(), False)

    def test_non_bool_values_are_rejected_without_state_change(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            (np.bool_(True), "numpy.bool"),
            ("", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for state in (False, True):
            self.cuda.allow_fp16_bf16_reduction_math_sdp(state)
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = f"set_sdp_use_math expects a bool, but got {type_name}"
                    with self.assertRaises(RuntimeError) as raised:
                        self.cuda.allow_fp16_bf16_reduction_math_sdp(value)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(
                        self.cuda.fp16_bf16_reduction_math_sdp_allowed(),
                        state,
                    )
                    self.assertIs(
                        torch._C._get_math_sdp_allow_fp16_bf16_reduction(),
                        state,
                    )

    def test_state_is_process_global_across_threads_and_aliases(self):
        cuda = self.cuda
        imported = importlib.import_module("torch_rs.backends.cuda")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cuda)

        def worker():
            try:
                observations.append(
                    cuda.fp16_bf16_reduction_math_sdp_allowed()
                )
                observations.append(
                    imported.allow_fp16_bf16_reduction_math_sdp(True)
                )
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(
                    imported.fp16_bf16_reduction_math_sdp_allowed()
                )
                observations.append(
                    cuda.allow_fp16_bf16_reduction_math_sdp(True)
                )
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), True)
        self.assertIs(cuda.allow_fp16_bf16_reduction_math_sdp(False), None)
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [False, None, False, None])
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), True)

    def test_reload_preserves_state_and_replaces_public_functions(self):
        cuda = self.cuda
        old_getter = cuda.fp16_bf16_reduction_math_sdp_allowed
        old_setter = cuda.allow_fp16_bf16_reduction_math_sdp
        namespace = cuda.__dict__

        self.assertIs(old_setter(True), None)
        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.fp16_bf16_reduction_math_sdp_allowed, old_getter)
        self.assertIsNot(cuda.allow_fp16_bf16_reduction_math_sdp, old_setter)
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), True)
        self.assertIs(cuda.allow_fp16_bf16_reduction_math_sdp(False), None)
        self.assertIs(old_getter(), False)
        self.assertIs(old_setter(True), None)
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), True)
        self.assertIs(cuda.allow_fp16_bf16_reduction_math_sdp(False), None)

        for name, old_function in (
            ("fp16_bf16_reduction_math_sdp_allowed", old_getter),
            ("allow_fp16_bf16_reduction_math_sdp", old_setter),
        ):
            with self.subTest(stale_function=name):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(old_function)
                message = re.sub(
                    r"0x[0-9a-fA-F]+",
                    "0x...",
                    str(raised.exception),
                )
                self.assertEqual(
                    message,
                    f"Can't pickle <function {name} at 0x...>: "
                    "it's not the same object as "
                    f"torch_rs.backends.cuda.{name}",
                )

    def test_metadata_exports_copying_and_pickling_match_public_contract(self):
        cuda = self.cuda
        getter = cuda.fp16_bf16_reduction_math_sdp_allowed
        setter = cuda.allow_fp16_bf16_reduction_math_sdp

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertIsNone(cuda.__doc__)
        self.assertEqual(
            cuda.__all__,
            [
                "is_built",
                "cuBLASModule",
                "is_ck_sdpa_available",
                "matmul",
                "enable_cudnn_sdp",
                "cudnn_sdp_enabled",
                "enable_flash_sdp",
                "flash_sdp_enabled",
                "enable_mem_efficient_sdp",
                "mem_efficient_sdp_enabled",
                "math_sdp_enabled",
                "enable_math_sdp",
                "allow_fp16_bf16_reduction_math_sdp",
                "fp16_bf16_reduction_math_sdp_allowed",
                "is_flash_attention_available",
                "sdp_kernel",
            ],
        )
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {
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
                "sdp_kernel",
                "math_sdp_enabled",
                "matmul",
                "mem_efficient_sdp_enabled",
                "torch",
            },
        )

        cases = (
            (
                getter,
                "fp16_bf16_reduction_math_sdp_allowed",
                "()",
                {},
                REDUCTION_ALLOWED_DOC,
                ("torch", "_C", "_get_math_sdp_allow_fp16_bf16_reduction"),
            ),
            (
                setter,
                "allow_fp16_bf16_reduction_math_sdp",
                "(enabled: bool)",
                {"enabled": bool},
                ALLOW_REDUCTION_DOC,
                ("torch", "_C", "_set_math_sdp_allow_fp16_bf16_reduction"),
            ),
        )
        for function, name, signature, annotations, doc, code_names in cases:
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(inspect.get_annotations(function), annotations)
                self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.backends.cuda")
                self.assertIs(inspect.getmodule(function), cuda)
                self.assertEqual(function.__doc__, doc)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__code__.co_names, code_names)
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol)),
                        function,
                    )

        getter_import = {}
        setter_import = {}
        wildcard = {}
        exec(
            "from torch_rs.backends.cuda import "
            "fp16_bf16_reduction_math_sdp_allowed",
            getter_import,
        )
        exec(
            "from torch_rs.backends.cuda import "
            "allow_fp16_bf16_reduction_math_sdp",
            setter_import,
        )
        exec("from torch_rs.backends.cuda import *", wildcard)
        self.assertIs(
            getter_import["fp16_bf16_reduction_math_sdp_allowed"],
            getter,
        )
        self.assertIs(
            setter_import["allow_fp16_bf16_reduction_math_sdp"],
            setter,
        )
        self.assertIs(wildcard["fp16_bf16_reduction_math_sdp_allowed"], getter)
        self.assertIs(wildcard["allow_fp16_bf16_reduction_math_sdp"], setter)

    def test_binding_errors_and_private_accessors_are_atomic(self):
        cuda = self.cuda
        cuda.allow_fp16_bf16_reduction_math_sdp(False)
        unexpected_keyword = (
            "allow_fp16_bf16_reduction_math_sdp() got an unexpected "
            "keyword argument '_enabled'"
        )
        if sys.version_info >= (3, 13):
            unexpected_keyword += ". Did you mean 'enabled'?"
        cases = (
            (
                lambda: cuda.fp16_bf16_reduction_math_sdp_allowed(None),
                "fp16_bf16_reduction_math_sdp_allowed() takes 0 positional "
                "arguments but 1 was given",
            ),
            (
                lambda: cuda.fp16_bf16_reduction_math_sdp_allowed(enabled=True),
                "fp16_bf16_reduction_math_sdp_allowed() got an unexpected "
                "keyword argument 'enabled'",
            ),
            (
                lambda: cuda.allow_fp16_bf16_reduction_math_sdp(),
                "allow_fp16_bf16_reduction_math_sdp() missing 1 required "
                "positional argument: 'enabled'",
            ),
            (
                lambda: cuda.allow_fp16_bf16_reduction_math_sdp(True, False),
                "allow_fp16_bf16_reduction_math_sdp() takes 1 positional "
                "argument but 2 were given",
            ),
            (
                lambda: cuda.allow_fp16_bf16_reduction_math_sdp(_enabled=False),
                unexpected_keyword,
            ),
            (
                lambda: cuda.allow_fp16_bf16_reduction_math_sdp(
                    True, enabled=False
                ),
                "allow_fp16_bf16_reduction_math_sdp() got multiple values "
                "for argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(
                    cuda.fp16_bf16_reduction_math_sdp_allowed(),
                    False,
                )

        self.assertIs(
            cuda.allow_fp16_bf16_reduction_math_sdp(enabled=True),
            None,
        )
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), True)

        getter_name = "_get_math_sdp_allow_fp16_bf16_reduction"
        setter_name = "_set_math_sdp_allow_fp16_bf16_reduction"
        self.assertTrue(hasattr(torch._C, getter_name))
        self.assertTrue(hasattr(torch._C, setter_name))
        self.assertFalse(hasattr(torch, getter_name))
        self.assertFalse(hasattr(torch, setter_name))
        self.assertNotIn(getter_name, torch._C.__all__)
        self.assertNotIn(setter_name, torch._C.__all__)

        math_state = cuda.math_sdp_enabled()
        mem_efficient_state = cuda.mem_efficient_sdp_enabled()
        flash_state = cuda.flash_sdp_enabled()
        self.assertIs(getattr(torch._C, setter_name)(False), None)
        self.assertIs(getattr(torch._C, getter_name)(), False)
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), False)
        self.assertIs(cuda.enable_flash_sdp(not flash_state), None)
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), False)
        self.assertIs(cuda.enable_math_sdp(not math_state), None)
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), False)
        self.assertIs(
            cuda.enable_mem_efficient_sdp(not mem_efficient_state),
            None,
        )
        self.assertIs(cuda.fp16_bf16_reduction_math_sdp_allowed(), False)
        self.assertFalse(hasattr(torch, "float16"))
        self.assertFalse(hasattr(torch, "bfloat16"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )


if __name__ == "__main__":
    unittest.main()
