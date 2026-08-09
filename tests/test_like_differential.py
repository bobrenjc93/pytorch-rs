import unittest

import torch as pytorch
import torch_rs


class LikeFactoryDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert pytorch.__version__.split("+")[0] == "2.13.0"

    @staticmethod
    def sources(module):
        ordinary = module.tensor(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ]
        )
        view = ordinary[1].reshape(3, 2)[1].reshape(1, 2)
        return (
            ordinary,
            view,
            module.tensor(3.0),
            module.zeros((2, 0, 3)),
            module.zeros((0, 1)) + 1.0,
        )

    def assert_same_tensor(self, actual, expected):
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(repr(actual.dtype), repr(expected.dtype))
        self.assertEqual(actual.device.type, expected.device.type)
        self.assertEqual(repr(actual.layout), repr(expected.layout))
        self.assertEqual(actual.requires_grad, expected.requires_grad)

    def test_values_shapes_metadata_views_scalars_and_empty_tensors(self):
        for local_input, reference_input in zip(
            self.sources(torch_rs), self.sources(pytorch), strict=True
        ):
            for factory_name in ("zeros_like", "ones_like"):
                local_factory = getattr(torch_rs, factory_name)
                reference_factory = getattr(pytorch, factory_name)
                with self.subTest(factory=factory_name, shape=local_input.shape):
                    self.assert_same_tensor(
                        local_factory(local_input),
                        reference_factory(reference_input),
                    )

    def test_supported_keywords_and_memory_formats(self):
        local_input = torch_rs.zeros((0, 1)) + 1.0
        reference_input = pytorch.zeros((0, 1)) + 1.0
        for factory_name in ("zeros_like", "ones_like"):
            local_factory = getattr(torch_rs, factory_name)
            reference_factory = getattr(pytorch, factory_name)
            for memory_format_name in ("preserve_format", "contiguous_format"):
                local_memory_format = getattr(torch_rs, memory_format_name)
                reference_memory_format = getattr(pytorch, memory_format_name)
                local = local_factory(
                    input=local_input,
                    dtype=torch_rs.float32,
                    layout=torch_rs.strided,
                    device=torch_rs.device("cpu"),
                    requires_grad=False,
                    memory_format=local_memory_format,
                )
                reference = reference_factory(
                    input=reference_input,
                    dtype=pytorch.float32,
                    layout=pytorch.strided,
                    device=pytorch.device("cpu"),
                    requires_grad=False,
                    memory_format=reference_memory_format,
                )
                with self.subTest(
                    factory=factory_name, memory_format=memory_format_name
                ):
                    self.assert_same_tensor(local, reference)

            self.assert_same_tensor(
                local_factory(
                    local_input,
                    dtype=None,
                    layout=None,
                    device=None,
                    memory_format=None,
                ),
                reference_factory(
                    reference_input,
                    dtype=None,
                    layout=None,
                    device=None,
                    memory_format=None,
                ),
            )

    def test_common_keyword_and_call_errors_match_pytorch_categories(self):
        local_input = torch_rs.tensor([1.0])
        reference_input = pytorch.tensor([1.0])
        for factory_name in ("zeros_like", "ones_like"):
            local_factory = getattr(torch_rs, factory_name)
            reference_factory = getattr(pytorch, factory_name)
            call_pairs = (
                (
                    lambda: local_factory(local_input, dtype=object()),
                    lambda: reference_factory(reference_input, dtype=object()),
                    TypeError,
                ),
                (
                    lambda: local_factory(local_input, layout=object()),
                    lambda: reference_factory(reference_input, layout=object()),
                    TypeError,
                ),
                (
                    lambda: local_factory(local_input, device=object()),
                    lambda: reference_factory(reference_input, device=object()),
                    TypeError,
                ),
                (
                    lambda: local_factory(local_input, requires_grad=1),
                    lambda: reference_factory(reference_input, requires_grad=1),
                    TypeError,
                ),
                (
                    lambda: local_factory(local_input, memory_format=object()),
                    lambda: reference_factory(reference_input, memory_format=object()),
                    TypeError,
                ),
                (
                    lambda: local_factory(
                        local_input, memory_format=torch_rs.channels_last
                    ),
                    lambda: reference_factory(
                        reference_input, memory_format=pytorch.channels_last
                    ),
                    RuntimeError,
                ),
                (lambda: local_factory(), lambda: reference_factory(), TypeError),
                (lambda: local_factory([1.0]), lambda: reference_factory([1.0]), TypeError),
                (
                    lambda: local_factory(local_input, torch_rs.float32),
                    lambda: reference_factory(reference_input, pytorch.float32),
                    TypeError,
                ),
                (
                    lambda: local_factory(local_input, out=None),
                    lambda: reference_factory(reference_input, out=None),
                    TypeError,
                ),
            )
            for local_call, reference_call, error_type in call_pairs:
                with self.subTest(factory=factory_name, call=local_call):
                    with self.assertRaises(error_type):
                        local_call()
                    with self.assertRaises(error_type):
                        reference_call()

    def test_docstrings_expose_the_pytorch_signature(self):
        expected = "input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format"
        for factory_name in ("zeros_like", "ones_like"):
            local_doc = " ".join(getattr(torch_rs, factory_name).__doc__.split())
            reference_doc = " ".join(getattr(pytorch, factory_name).__doc__.split())
            with self.subTest(factory=factory_name):
                self.assertIn(expected, local_doc)
                self.assertIn(expected, reference_doc)


if __name__ == "__main__":
    unittest.main()
