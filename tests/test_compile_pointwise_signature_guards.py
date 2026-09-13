"""Signature guards must not execute user-defined container truthiness hooks."""
import unittest

import torch_rs as native
from torch_rs import _compile_pointwise as frontend

from tests.test_compile_pointwise_jit import available, program


def custom_signature_containers(effects):
    class Defaults(tuple):
        def __bool__(self):
            effects.append('defaults truthiness')
            return False

    class KeywordDefaults(dict):
        def __bool__(self):
            effects.append('keyword defaults truthiness')
            return False

    return (('__defaults__', Defaults()), ('__kwdefaults__', KeywordDefaults()))


class SignatureAdmission(unittest.TestCase):
    def test_custom_signature_containers_rejected_without_callbacks(self):
        effects = []
        for name, container in custom_signature_containers(effects):
            fn = program('def f(x):\n return -x')
            setattr(fn, name, container)
            with self.subTest(attribute=name):
                with self.assertRaises(NotImplementedError):
                    frontend.analyze(fn, 1)
                self.assertEqual(effects, [])


@unittest.skipUnless(available(), 'requires native CUDA and reference PyTorch CUDA')
class SignatureRuntime(unittest.TestCase):
    def tearDown(self):
        native.compiler.reset()

    def test_cold_and_warm_signature_changes_never_invoke_callbacks(self):
        x = native.tensor([1., -2.], dtype=native.float32).to('cuda:0')
        for warmed in (False, True):
            effects = []
            for name, container in custom_signature_containers(effects):
                fn = program('def f(x):\n return -x')
                compiled = native.compile(fn)
                if warmed:
                    compiled(x)
                setattr(fn, name, container)
                with self.subTest(warmed=warmed, attribute=name):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
                    self.assertEqual(effects, [])


if __name__ == '__main__':
    unittest.main()
