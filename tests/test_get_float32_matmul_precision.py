import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import textwrap
import threading
import types
import typing
import unittest
import warnings

import numpy as np
import torch_rs as torch


GETTER_DOC = """Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """

SETTER_DOC = """Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """

INVALID_WARNING = (
    " is not one of 'highest', 'high', or 'medium'; the current"
    "setFloat32MatmulPrecision call has no effect."
)


class Float32MatmulPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        torch.set_float32_matmul_precision(self.original_precision)

    def test_supported_modes_update_the_process_state_and_return_none(self):
        for precision in ("highest", "high", "medium", "highest"):
            with self.subTest(precision=precision):
                self.assertIsNone(torch.set_float32_matmul_precision(precision))
                result = torch.get_float32_matmul_precision()
                self.assertIs(type(result), str)
                self.assertEqual(result, precision)

        class StringSubclass(str):
            pass

        class BytesSubclass(bytes):
            pass

        for value, expected in (
            (StringSubclass("high"), "high"),
            (b"medium", "medium"),
            (BytesSubclass(b"highest"), "highest"),
        ):
            with self.subTest(value=repr(value)):
                self.assertIsNone(torch.set_float32_matmul_precision(value))
                self.assertEqual(torch.get_float32_matmul_precision(), expected)

    def test_updates_are_visible_across_threads_and_grad_modes(self):
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        torch.set_float32_matmul_precision("high")

        def observer():
            try:
                with torch.no_grad():
                    observations.append(
                        (
                            torch.is_grad_enabled(),
                            torch.get_float32_matmul_precision(),
                        )
                    )
                    worker_ready.set()
                    if not read_updated.wait(timeout=10):
                        raise RuntimeError("timed out waiting for precision update")
                    observations.append(
                        (
                            torch.is_grad_enabled(),
                            torch.get_float32_matmul_precision(),
                        )
                    )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        self.assertIsNone(torch.set_float32_matmul_precision("medium"))
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [(False, "high"), (False, "medium")])
        self.assertTrue(torch.is_grad_enabled())

        worker_result = []

        def writer():
            worker_result.append(torch.set_float32_matmul_precision("highest"))
            worker_result.append(torch.get_float32_matmul_precision())

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_result, [None, "highest"])
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_all_modes_leave_supported_cpu_matmul_and_autograd_unchanged(self):
        def outcome(precision):
            torch.set_float32_matmul_precision(precision)
            left = torch.tensor(
                [[1.25, -2.5, 3.75], [4.5, 5.25, -6.0]],
                requires_grad=True,
            )
            right = torch.tensor(
                [[-7.0, 8.5], [9.25, -10.0], [11.5, 12.75]],
                requires_grad=True,
            )
            before = (
                left.requires_grad,
                left.is_leaf,
                left.grad,
                right.requires_grad,
                right.is_leaf,
                right.grad,
            )
            result = torch.matmul(left, right)
            try:
                result.sum().backward()
            except Exception as error:
                backward = (type(error), str(error), error.args)
            else:
                backward = None
            after = (
                left.requires_grad,
                left.is_leaf,
                left.grad,
                right.requires_grad,
                right.is_leaf,
                right.grad,
            )
            return (
                result.tolist(),
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.requires_grad,
                result.is_leaf,
                before,
                backward,
                after,
            )

        baseline = outcome("highest")
        self.assertIsNotNone(baseline[7])
        for precision in ("high", "medium", "highest"):
            with self.subTest(precision=precision):
                self.assertEqual(outcome(precision), baseline)

    def test_invalid_strings_warn_and_do_not_change_the_state(self):
        torch.set_float32_matmul_precision("high")
        for value, visible in (
            ("Highest", "Highest" + INVALID_WARNING),
            ("low", "low" + INVALID_WARNING),
            ("", INVALID_WARNING),
            ("medium\x00ignored", "medium"),
            (b"highest\x00ignored", "highest"),
        ):
            with self.subTest(value=repr(value)):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.assertIsNone(torch.set_float32_matmul_precision(value))
                self.assertEqual(torch.get_float32_matmul_precision(), "high")
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    str(caught[0].message).split(" (Triggered internally at ", 1)[0],
                    visible,
                )
                self.assertEqual(caught[0].filename, torch.__file__)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                torch.set_float32_matmul_precision("unsupported")
        self.assertEqual(torch.get_float32_matmul_precision(), "high")

    def test_type_validation_errors_preserve_the_state(self):
        class StringConvertible:
            def __str__(self):
                return "medium"

        torch.set_float32_matmul_precision("medium")
        for value, type_name in (
            (None, "NoneType"),
            (True, "bool"),
            (1, "int"),
            (1.5, "float"),
            (bytearray(b"high"), "bytearray"),
            (memoryview(b"high"), "memoryview"),
            ([], "list"),
            ({}, "dict"),
            (StringConvertible(), "StringConvertible"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (np.array([1.0], dtype=np.float32), "numpy.ndarray"),
        ):
            with self.subTest(type=type_name):
                message = (
                    "set_float32_matmul_precision expects a str, but got "
                    f"{type_name}"
                )
                with self.assertRaises(RuntimeError) as raised:
                    torch.set_float32_matmul_precision(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(torch.get_float32_matmul_precision(), "medium")

        with self.assertRaises(RuntimeError) as raised:
            torch.set_float32_matmul_precision("\ud800")
        self.assertEqual(str(raised.exception), "error unpacking string as utf-8")
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")

        with self.assertRaises(UnicodeDecodeError):
            torch.set_float32_matmul_precision(b"\xff")
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_invalid_precision_raises_bad_alloc_instead_of_aborting(self):
        script = textwrap.dedent(
            """\
            import os
            import resource

            import torch_rs as torch

            precision = b"x" * (64 * 1024 * 1024)
            torch.set_float32_matmul_precision("medium")
            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            limit = current_virtual_size + 8 * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            try:
                torch.set_float32_matmul_precision(precision)
            except RuntimeError as error:
                assert str(error) == "std::bad_alloc", repr(error)
            else:
                raise AssertionError("the constrained call unexpectedly succeeded")
            assert torch.get_float32_matmul_precision() == "medium"
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if completed.returncode == 77:
            self.skipTest("process hard address-space limit is too low")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_signature_annotations_documentation_and_module_identity(self):
        package = importlib.import_module("torch_rs")
        getter = package.get_float32_matmul_precision
        setter = package.set_float32_matmul_precision

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        for function in (getter, setter):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs")
            self.assertIs(inspect.getmodule(function), package)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(str(inspect.signature(getter)), "() -> str")
        self.assertEqual(getter.__annotations__, {"return": str})
        self.assertEqual(typing.get_type_hints(getter), {"return": str})
        self.assertEqual(getter.__name__, "get_float32_matmul_precision")
        self.assertEqual(getter.__qualname__, "get_float32_matmul_precision")
        self.assertEqual(inspect.cleandoc(getter.__doc__), inspect.cleandoc(GETTER_DOC))

        self.assertEqual(
            str(inspect.signature(setter)), "(precision: str) -> None"
        )
        self.assertEqual(
            setter.__annotations__, {"precision": str, "return": None}
        )
        self.assertEqual(
            typing.get_type_hints(setter),
            {"precision": str, "return": type(None)},
        )
        self.assertEqual(setter.__name__, "set_float32_matmul_precision")
        self.assertEqual(setter.__qualname__, "set_float32_matmul_precision")
        self.assertEqual(inspect.cleandoc(setter.__doc__), inspect.cleandoc(SETTER_DOC))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        functions = {
            "get_float32_matmul_precision": torch.get_float32_matmul_precision,
            "set_float32_matmul_precision": torch.set_float32_matmul_precision,
        }
        namespace = {}
        exec("from torch_rs import *", namespace)

        for name, function in functions.items():
            with self.subTest(name=name):
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertIs(namespace[name], function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs", payload)
                    self.assertIs(pickle.loads(payload), function)

        self.assertTrue(hasattr(torch._C, "_get_float32_matmul_precision"))
        self.assertTrue(hasattr(torch._C, "_set_float32_matmul_precision"))
        self.assertNotIn("_get_float32_matmul_precision", torch._C.__all__)
        self.assertNotIn("_set_float32_matmul_precision", torch._C.__all__)

    def test_argument_errors_match_pytorch_2_13(self):
        getter = torch.get_float32_matmul_precision
        setter = torch.set_float32_matmul_precision
        cases = (
            (
                lambda: getter(None),
                "get_float32_matmul_precision() takes 0 positional arguments "
                "but 1 was given",
            ),
            (
                lambda: getter(precision=None),
                "get_float32_matmul_precision() got an unexpected keyword "
                "argument 'precision'",
            ),
            (
                lambda: setter(),
                "set_float32_matmul_precision() missing 1 required positional "
                "argument: 'precision'",
            ),
            (
                lambda: setter("high", "medium"),
                "set_float32_matmul_precision() takes 1 positional argument "
                "but 2 were given",
            ),
            (
                lambda: setter(mode="high"),
                "set_float32_matmul_precision() got an unexpected keyword "
                "argument 'mode'",
            ),
            (
                lambda: setter("high", precision="medium"),
                "set_float32_matmul_precision() got multiple values for argument "
                "'precision'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                before = getter()
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(getter(), before)

        self.assertIsNone(setter(precision="high"))
        self.assertEqual(getter(), "high")

    def test_reload_preserves_state_for_old_and_new_functions(self):
        package = importlib.import_module("torch_rs")
        old_getter = package.get_float32_matmul_precision
        old_setter = package.set_float32_matmul_precision
        old_setter("medium")

        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch, package)
        self.assertEqual(package.get_float32_matmul_precision(), "medium")
        self.assertEqual(old_getter(), "medium")

        self.assertIsNone(old_setter("high"))
        self.assertEqual(package.get_float32_matmul_precision(), "high")
        self.assertIsNone(package.set_float32_matmul_precision("highest"))
        self.assertEqual(old_getter(), "highest")

    def test_importing_reloading_and_calling_does_not_import_pytorch(self):
        script = r"""
import importlib
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.get_float32_matmul_precision() == "highest"
assert torch.set_float32_matmul_precision("medium") is None
assert torch.get_float32_matmul_precision() == "medium"
old_getter = torch.get_float32_matmul_precision
old_setter = torch.set_float32_matmul_precision
assert importlib.reload(torch) is torch
assert torch.get_float32_matmul_precision() == "medium"
assert old_getter() == "medium"
assert old_setter("high") is None
assert torch.get_float32_matmul_precision() == "high"
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
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


if __name__ == "__main__":
    unittest.main()
