import inspect
import importlib
import sys


def assert_no_argument_signature(test_case, callable_object, expected_signature):
    """Assert CPython's versioned signature for a METH_NOARGS callable."""
    if sys.version_info >= (3, 13):
        test_case.assertEqual(callable_object.__text_signature__, "($self, /)")
        test_case.assertEqual(
            str(inspect.signature(callable_object)), expected_signature
        )
    else:
        test_case.assertIsNone(callable_object.__text_signature__)
        with test_case.assertRaises(ValueError):
            inspect.signature(callable_object)


def expose_reference_compiler_register_backend(reference_torch):
    """Expose PyTorch 2.13's Dynamo backend registration on torch.compiler."""
    if reference_torch is None:
        return

    compiler = getattr(reference_torch, "compiler", None)
    if compiler is None:
        return

    if not hasattr(compiler, "register_backend"):
        try:
            dynamo = importlib.import_module(f"{reference_torch.__name__}._dynamo")
        except (AttributeError, ImportError):
            return
        compiler.register_backend = dynamo.register_backend

    exports = list(compiler.__all__)
    if "register_backend" not in exports:
        try:
            insert_at = exports.index("disable")
        except ValueError:
            insert_at = len(exports)
        exports.insert(insert_at, "register_backend")
        compiler.__all__ = exports
