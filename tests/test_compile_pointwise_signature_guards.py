"""Signature guards must not execute user-defined container truthiness hooks."""
import unittest
from unittest.mock import patch

import torch_rs as native
from torch_rs import _compile_pointwise as frontend

from tests.test_compile_pointwise_jit import available, program
from tests import test_compile_pointwise_helpers as helper_tests


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

    def test_private_resolve_still_validates_replaced_code_and_constants(self):
        effects = []
        class Hostile:
            def __repr__(self): effects.append('repr'); return 'hostile'
        fn = program('def f(x):\n return -x')
        parsed = frontend.analyze(fn, 1)
        fn.__code__ = fn.__code__.replace(co_consts=fn.__code__.co_consts + (Hostile(),))
        with patch.object(frontend.dis, 'get_instructions', side_effect=AssertionError('unsafe disassembly')):
            with self.assertRaises(NotImplementedError): frontend.resolve(fn, parsed)
        self.assertEqual(effects, [])


class SignatureCache(unittest.TestCase):
    setUp = helper_tests.HelperCache.setUp

    def test_warm_calls_check_mutable_signature_once_and_reuse_admitted_code(self):
        fn = program('def f(x):\n for i in range(0):\n  x=-x\n return -x')
        compiled = native.compile(fn)
        compiled(self.x)
        with (patch.object(frontend, 'validate_signature_containers', wraps=frontend.validate_signature_containers) as signature,
              patch.object(frontend, 'validate_code', side_effect=AssertionError('immutable code revalidated'))):
            compiled(self.x)
        self.assertEqual(signature.call_count, 1)
        original = fn.__code__
        effects = []
        class Hostile:
            def __repr__(self): effects.append('repr'); return 'hostile'
        fn.__code__ = original.replace(co_consts=original.co_consts + (Hostile(),))
        with self.assertRaises(NotImplementedError): compiled(self.x)
        self.assertEqual(effects, [])
        fn.__code__ = original
        for name, container in custom_signature_containers(effects):
            previous = getattr(fn, name)
            setattr(fn, name, container)
            with self.assertRaises(NotImplementedError): compiled(self.x)
            setattr(fn, name, previous)
        with patch.object(frontend, 'validate_code', side_effect=AssertionError('recovery revalidated')):
            compiled(self.x)
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
