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
