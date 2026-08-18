import copy
import inspect
import pickle
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FInfoReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("finfo differentials require pinned PyTorch 2.13.0")

    def normalized_error(self, module, call):
        try:
            call()
        except Exception as error:
            message = str(error).replace(module.finfo.__module__, "torch")
            return type(error).__name__, message
        return None

    def metadata(self, value):
        return (
            value.bits,
            value.dtype,
            value.eps.hex(),
            value.max.hex(),
            value.min.hex(),
            value.resolution.hex(),
            value.smallest_normal.hex(),
            value.tiny.hex(),
            repr(value),
            str(value),
        )

    def test_float32_aliases_values_and_freshness_match_pytorch_2_13(self):
        def values(module):
            return (
                module.finfo(),
                module.finfo(module.float32),
                module.finfo(module.float),
                module.finfo(type=module.float32),
                module.finfo(module.tensor(1.0).dtype),
                module.finfo(module.get_default_dtype()),
            )

        actual = values(torch)
        expected = values(reference_torch)
        self.assertEqual(
            tuple(self.metadata(value) for value in actual),
            tuple(self.metadata(value) for value in expected),
        )
        self.assertEqual(
            len({id(value) for value in actual}),
            len({id(value) for value in expected}),
        )
        self.assertEqual(
            tuple(value == actual[0] for value in actual),
            tuple(value == expected[0] for value in expected),
        )
        self.assertIs(torch.float, torch.float32)
        self.assertIs(reference_torch.float, reference_torch.float32)

    def test_type_constructor_and_descriptor_metadata_match_pytorch_2_13(self):
        actual_class = torch.finfo
        expected_class = reference_torch.finfo

        self.assertEqual(type(actual_class), type(expected_class))
        self.assertEqual(actual_class.__name__, expected_class.__name__)
        self.assertEqual(actual_class.__qualname__, expected_class.__qualname__)
        self.assertEqual(actual_class.__doc__, expected_class.__doc__)
        self.assertEqual(actual_class.__basicsize__, expected_class.__basicsize__)
        self.assertEqual(actual_class.__itemsize__, expected_class.__itemsize__)
        actual_surface = set(vars(actual_class))
        expected_surface = set(vars(expected_class))
        actual_surface.discard("__module__")
        expected_surface.discard("__module__")
        self.assertEqual(actual_surface, expected_surface)
        self.assertEqual(
            hasattr(torch.finfo(), "__getnewargs__"),
            hasattr(reference_torch.finfo(), "__getnewargs__"),
        )
        self.assertEqual(
            repr(actual_class).replace(actual_class.__module__, "torch"),
            repr(expected_class),
        )
        self.assertEqual("finfo" in torch.__all__, "finfo" in reference_torch.__all__)
        for value in (actual_class, expected_class):
            with self.assertRaises(ValueError):
                inspect.signature(value)

        actual_new = inspect.getattr_static(actual_class, "__new__")
        expected_new = inspect.getattr_static(expected_class, "__new__")
        self.assertEqual(type(actual_new), type(expected_new))
        self.assertEqual(actual_new.__name__, expected_new.__name__)
        self.assertEqual(actual_new.__qualname__, expected_new.__qualname__)
        self.assertEqual(actual_new.__doc__, expected_new.__doc__)
        self.assertEqual(
            actual_new.__text_signature__, expected_new.__text_signature__
        )

        for name in (
            "bits",
            "dtype",
            "eps",
            "max",
            "min",
            "resolution",
            "smallest_normal",
            "tiny",
        ):
            actual = inspect.getattr_static(actual_class, name)
            expected = inspect.getattr_static(expected_class, name)
            with self.subTest(name=name):
                self.assertIs(type(actual), types.GetSetDescriptorType)
                self.assertEqual(type(actual), type(expected))
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertIs(actual.__objclass__, actual_class)
                self.assertIs(expected.__objclass__, expected_class)
                self.assertIs(actual.__get__(None, actual_class), actual)
                self.assertIs(expected.__get__(None, expected_class), expected)

    def test_type_immutability_matches_pytorch_2_13(self):
        actual_actions = (
            lambda: setattr(torch.finfo, "extra", 1),
            lambda: setattr(torch.finfo, "__repr__", lambda self: "changed"),
            lambda: delattr(torch.finfo, "bits"),
        )
        expected_actions = (
            lambda: setattr(reference_torch.finfo, "extra", 1),
            lambda: setattr(
                reference_torch.finfo, "__repr__", lambda self: "changed"
            ),
            lambda: delattr(reference_torch.finfo, "bits"),
        )
        for actual_action, expected_action in zip(
            actual_actions, expected_actions, strict=True
        ):
            self.assertEqual(
                self.normalized_error(torch, actual_action),
                self.normalized_error(reference_torch, expected_action),
            )

    def test_constructor_errors_match_pytorch_2_13(self):
        cases = (
            lambda module: module.finfo(dtype=module.float32),
            lambda module: module.finfo(unexpected=module.float32),
            lambda module: module.finfo(None),
            lambda module: module.finfo(type=None),
            lambda module: module.finfo("float32"),
            lambda module: module.finfo(object()),
            lambda module: module.finfo(module.tensor(1.0)),
            lambda module: module.finfo(module.float32, module.float32),
            lambda module: module.finfo(module.float32, type=module.float32),
        )
        for call in cases:
            with self.subTest(call=call):
                self.assertEqual(
                    self.normalized_error(torch, lambda: call(torch)),
                    self.normalized_error(
                        reference_torch, lambda: call(reference_torch)
                    ),
                )

    def test_read_only_equality_hash_and_ordering_match_pytorch_2_13(self):
        actual = torch.finfo()
        expected = reference_torch.finfo()

        for name in (
            "bits",
            "dtype",
            "eps",
            "max",
            "min",
            "resolution",
            "smallest_normal",
            "tiny",
        ):
            actual_descriptor = inspect.getattr_static(torch.finfo, name)
            expected_descriptor = inspect.getattr_static(reference_torch.finfo, name)
            actual_actions = (
                lambda name=name: setattr(actual, name, None),
                lambda name=name: delattr(actual, name),
                lambda descriptor=actual_descriptor: descriptor.__set__(
                    actual, None
                ),
                lambda descriptor=actual_descriptor: descriptor.__delete__(actual),
            )
            expected_actions = (
                lambda name=name: setattr(expected, name, None),
                lambda name=name: delattr(expected, name),
                lambda descriptor=expected_descriptor: descriptor.__set__(
                    expected, None
                ),
                lambda descriptor=expected_descriptor: descriptor.__delete__(
                    expected
                ),
            )
            for actual_action, expected_action in zip(
                actual_actions, expected_actions, strict=True
            ):
                with self.subTest(name=name, action=actual_action):
                    self.assertEqual(
                        self.normalized_error(torch, actual_action),
                        self.normalized_error(reference_torch, expected_action),
                    )

        self.assertEqual(
            self.normalized_error(
                torch, lambda: setattr(actual, "extra", None)
            ),
            self.normalized_error(
                reference_torch, lambda: setattr(expected, "extra", None)
            ),
        )

        actual_other = torch.finfo(torch.float)
        expected_other = reference_torch.finfo(reference_torch.float)
        self.assertEqual(
            (actual == actual_other, actual != actual_other),
            (expected == expected_other, expected != expected_other),
        )
        self.assertEqual(
            self.normalized_error(torch, lambda: hash(actual)),
            self.normalized_error(reference_torch, lambda: hash(expected)),
        )
        for actual_action, expected_action in (
            (lambda: actual < actual_other, lambda: expected < expected_other),
            (lambda: actual <= actual_other, lambda: expected <= expected_other),
            (lambda: actual > actual_other, lambda: expected > expected_other),
            (lambda: actual >= actual_other, lambda: expected >= expected_other),
        ):
            self.assertEqual(
                self.normalized_error(torch, actual_action),
                self.normalized_error(reference_torch, expected_action),
            )

        def foreign_comparison(module):
            class Foreign:
                def __init__(self):
                    self.events = []

                def __eq__(self, other):
                    self.events.append(("eq", other))
                    return "reflected equality sentinel"

                def __ne__(self, other):
                    self.events.append(("ne", other))
                    raise RuntimeError("reflected inequality ran")

            value = module.finfo()
            foreign = Foreign()
            equality = value == foreign
            inequality = value != foreign
            return equality, inequality, tuple(foreign.events)

        self.assertEqual(
            foreign_comparison(torch),
            foreign_comparison(reference_torch),
        )

    def test_unpicklability_and_copy_errors_match_pytorch_2_13(self):
        actual = torch.finfo()
        expected = reference_torch.finfo()

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertEqual(
                    self.normalized_error(
                        torch,
                        lambda protocol=protocol: pickle.dumps(
                            actual, protocol=protocol
                        ),
                    ),
                    self.normalized_error(
                        reference_torch,
                        lambda protocol=protocol: pickle.dumps(
                            expected, protocol=protocol
                        ),
                    ),
                )

        for actual_action, expected_action in (
            (lambda: copy.copy(actual), lambda: copy.copy(expected)),
            (lambda: copy.deepcopy(actual), lambda: copy.deepcopy(expected)),
            (actual.__reduce__, expected.__reduce__),
            (
                lambda: actual.__reduce_ex__(pickle.HIGHEST_PROTOCOL),
                lambda: expected.__reduce_ex__(pickle.HIGHEST_PROTOCOL),
            ),
        ):
            self.assertEqual(
                self.normalized_error(torch, actual_action),
                self.normalized_error(reference_torch, expected_action),
            )


if __name__ == "__main__":
    unittest.main()
