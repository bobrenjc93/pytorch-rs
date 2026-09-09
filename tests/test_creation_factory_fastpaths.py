"""Factory shape validation must not consult NumPy for Python integers."""
import builtins
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as torch


class IntSubclass(int):
    def __index__(self):
        raise AssertionError("Python integer subclasses must use their integer value")


class CreationFactoryFastPathTests(unittest.TestCase):
    def test_python_integer_and_bool_paths_do_not_import_numpy(self):
        original_import = builtins.__import__
        numpy_imports = []

        def record_import(name, *args, **kwargs):
            if name == "numpy" or name.startswith("numpy."):
                numpy_imports.append(name)
            return original_import(name, *args, **kwargs)

        for name in ("zeros", "ones", "empty"):
            numpy_imports.clear()
            factory = getattr(torch, name)
            calls = (
                (((2, 0, 3),), {}, (2, 0, 3)),
                (([2, 3],), {}, (2, 3)),
                ((2, 3), {}, (2, 3)),
                ((2,), {}, (2,)),
                ((IntSubclass(2),), {}, (2,)),
                (((IntSubclass(2), False),), {}, (2, 0)),
                ((2, True), {}, (2, 1)),
                ((), {"size": (2, True)}, (2, 1)),
                ((), {"shape": [2, False]}, (2, 0)),
            )
            with patch.object(builtins, "__import__", record_import):
                for args, kwargs, shape in calls:
                    with self.subTest(factory=name, args=args, kwargs=kwargs):
                        self.assertEqual(tuple(factory(*args, **kwargs).shape), shape)
                for args in ((True,), ((True, 2),), ([False, 2],)):
                    with self.subTest(factory=name, rejected=args):
                        with self.assertRaises(TypeError):
                            factory(*args)
            with self.subTest(factory=name, numpy_imports=numpy_imports):
                self.assertEqual(numpy_imports, [])

    def test_numpy_bool_rejection_and_integer_support_across_shape_forms(self):
        for name in ("zeros", "ones", "empty"):
            factory = getattr(torch, name)
            for value in (np.bool_(False), np.bool_(True)):
                for args, kwargs in (((value,), {}), (((value, 2),), {}),
                                     (((2, value),), {}), ((2, value), {}),
                                     ((), {"size": [2, value]}),
                                     ((), {"shape": [value, 2]})):
                    with self.subTest(factory=name, args=args, kwargs=kwargs):
                        message = ("must be tuple of ints, not list" if "shape" in kwargs
                                   else r"numpy\.bool")
                        with self.assertRaisesRegex(TypeError, message):
                            factory(*args, **kwargs)
            for args, kwargs in (((np.int64(2),), {}),
                                 (((np.int32(2), 3),), {}),
                                 ((2, np.int64(3)), {}),
                                 ((), {"size": [np.int64(2), 3]})):
                with self.subTest(factory=name, args=args, kwargs=kwargs):
                    result = factory(*args, **kwargs)
                    self.assertEqual(result.shape[0], 2)


if __name__ == "__main__":
    unittest.main()
