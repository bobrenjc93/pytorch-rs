import inspect
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


def expected_compiler_exports(reference_compiler, supported_exports):
    """Return the torch-rs compiler export list expected from PyTorch 2.13."""
    exports = [
        name for name in reference_compiler.__all__ if name in supported_exports
    ]
    if "register_backend" in supported_exports and "register_backend" not in exports:
        try:
            insert_at = exports.index("list_backends") + 1
        except ValueError:
            insert_at = len(exports)
        exports.insert(insert_at, "register_backend")
    return exports
