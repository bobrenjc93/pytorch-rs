import itertools
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class RankTwoMeanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("rank-2 mean differentials require PyTorch 2.13.0")

    def assert_matches(self, actual, expected):
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertIs(actual.dtype, torch.float32)
        self.assertIs(expected.dtype, reference_torch.float32)
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        actual_values = np.asarray(actual)
        expected_values = expected.detach().numpy()
        np.testing.assert_allclose(
            actual_values, expected_values, rtol=1e-6, atol=1e-7, equal_nan=True
        )
        zeros = expected_values == 0
        np.testing.assert_array_equal(
            np.signbit(actual_values[zeros]), np.signbit(expected_values[zeros])
        )
        if expected.grad_fn is not None:
            self.assertEqual(type(expected.grad_fn).__name__, "MeanBackward1")
            self.assertEqual(
                torch._C._nn_functional_dropout_tensor_autograd_suffix(actual),
                ", grad_fn=<MeanBackward1>",
            )

    @staticmethod
    def cases(module):
        def tensor(values):
            return module.tensor(values, dtype=module.float32)

        dense = tensor(np.arange(30, dtype=np.float32).reshape(5, 6).tolist())
        cube = tensor(np.arange(60, dtype=np.float32).reshape(3, 4, 5).tolist())
        rng = np.random.default_rng(217)
        return (
            ("dense", dense),
            ("offset", dense[1:4]),
            ("transposed", dense.transpose(0, 1)),
            ("transposed offset", dense.transpose(0, 1)[1:5]),
            ("both strides nonunit", cube.transpose(0, 2)[2]),
            ("empty rows", module.zeros((0, 3))),
            ("empty columns", module.zeros((2, 0))),
            ("both empty", module.zeros((0, 0))),
            ("empty offset", module.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("singleton row", tensor([[1.0, 2.0, 4.0]])),
            ("singleton column", tensor([[1.0], [2.0], [4.0]])),
            ("signed zeros", tensor([[-0.0, 0.0, -0.0], [-0.0, -0.0, -0.0]])),
            ("nonfinite", tensor([
                [float("inf"), float("nan"), float("-inf"), 3.0],
                [float("-inf"), 2.0, float("-inf"), 6.0],
            ])),
            ("vectorized", tensor(rng.normal(size=(33, 37)).astype(np.float32).tolist())),
        )

    @staticmethod
    def call(module, source, dim, keepdim, form):
        if form == "method positional":
            return source.mean(dim, keepdim)
        if form == "method defaults":
            return source.mean(dim, keepdim=True) if keepdim else source.mean(dim)
        if form == "method tuple none":
            return source.mean(dim=(dim,), keepdim=keepdim, dtype=None)
        if form == "method list float":
            return source.mean(dim=[dim], keepdim=keepdim, dtype=module.float)
        if form == "top positional":
            return module.mean(source, dim, keepdim)
        if form == "top defaults":
            if keepdim:
                return module.mean(source, dim, keepdim=True)
            return module.mean(source, dim)
        if form == "top tuple none":
            return module.mean(input=source, dim=(dim,), keepdim=keepdim, dtype=None, out=None)
        if form == "top list float32":
            return module.mean(source, dim=[dim], keepdim=keepdim, dtype=module.float32)
        raise AssertionError(form)

    def test_values_layouts_and_bindings(self):
        forms = (
            "method positional", "method defaults", "method tuple none",
            "method list float", "top positional", "top defaults",
            "top tuple none", "top list float32",
        )
        for (name, actual), (_, expected) in zip(
            self.cases(torch), self.cases(reference_torch), strict=True
        ):
            before = np.asarray(actual).copy()
            for dim, keepdim, form in itertools.product((0, 1, -1, -2), (False, True), forms):
                with self.subTest(case=name, dim=dim, keepdim=keepdim, form=form):
                    result = self.call(torch, actual, dim, keepdim, form)
                    self.assert_matches(
                        result, self.call(reference_torch, expected, dim, keepdim, form)
                    )
                    self.assertFalse(result.is_set_to(actual))
                    if result.numel():
                        self.assertNotEqual(result.data_ptr(), actual.data_ptr())
                    np.testing.assert_array_equal(np.asarray(actual), before)

    def test_integer_protocols_input_aliases_and_empty_dimension_sequences(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return -2

        for dim in (IntSubclass(-1), np.int64(-2), (IndexOnly(),), [np.int64(1)]):
            for alias in ("input", "x", "a", "x1"):
                with self.subTest(dim=dim, alias=alias):
                    outputs = []
                    for module in (torch, reference_torch):
                        source = module.tensor([[1.0, 2.0, 4.0], [7.0, 8.0, 10.0]])
                        outputs.append(module.mean(**{alias: source}, dim=dim, keepdim=True))
                    self.assert_matches(*outputs)
        # Empty sequences keep their existing full-reduction meaning.
        for dim, keepdim, top in itertools.product(((), []), (False, True), (False, True)):
            outputs = []
            for module in (torch, reference_torch):
                source = module.ones((2, 3))
                outputs.append(
                    module.mean(source, dim, keepdim) if top else source.mean(dim, keepdim)
                )
            self.assert_matches(*outputs)

    def test_weighted_backward_views_accumulation_and_no_grad(self):
        for dim, keepdim, top_level, transposed in itertools.product(
            (0, 1, -1, -2), (False, True), (False, True), (False, True)
        ):
            with self.subTest(dim=dim, keepdim=keepdim, top=top_level, transposed=transposed):
                leaves = []
                for module in (torch, reference_torch):
                    leaf = module.tensor(
                        np.arange(20, dtype=np.float32).reshape(4, 5).tolist(),
                        dtype=module.float32, requires_grad=True,
                    )
                    leaves.append(leaf)
                for iteration in range(2):
                    outputs = []
                    for module, leaf in zip((torch, reference_torch), leaves, strict=True):
                        view = leaf.transpose(0, 1)[1:4] if transposed else leaf[1:]
                        call = module.mean if top_level else module.Tensor.mean
                        output = call(view, dim, keepdim, dtype=module.float32)
                        weights = module.tensor(
                            (7 - 3 * np.arange(output.numel(), dtype=np.float32))
                            .reshape(tuple(output.shape)).tolist(),
                            dtype=module.float32,
                        )
                        outputs.append(output)
                        # Build a fresh graph each time to check leaf accumulation.
                        (output * weights).sum().backward()
                        with module.no_grad():
                            untracked = call(view, dim, keepdim)
                        self.assertFalse(untracked.requires_grad)
                        self.assertTrue(untracked.is_leaf)
                    self.assert_matches(*outputs)
                    np.testing.assert_array_equal(
                        np.asarray(leaves[0].grad), leaves[1].grad.numpy(),
                        err_msg=f"accumulation iteration {iteration}",
                    )

    def test_empty_and_nonfinite_backward(self):
        for shape, dim, keepdim, top in itertools.product(
            ((0, 3), (2, 0), (0, 0)), (0, 1, -1, -2), (False, True), (False, True)
        ):
            with self.subTest(shape=shape, dim=dim, keepdim=keepdim, top=top):
                results, gradients = [], []
                for module in (torch, reference_torch):
                    leaf = module.zeros(shape, requires_grad=True)
                    view = leaf.transpose(0, 1)
                    output = module.mean(view, dim, keepdim) if top else view.mean(dim, keepdim)
                    results.append(output)
                    output.sum().backward()
                    gradients.append(leaf.grad)
                self.assert_matches(*results)
                self.assert_matches(*gradients)
        for dim in (0, 1):
            gradients = []
            for module in (torch, reference_torch):
                leaf = module.tensor(
                    [[float("nan"), float("inf")], [float("-inf"), 2.0]],
                    requires_grad=True,
                )
                (leaf.mean(dim) * 7.0).sum().backward()
                gradients.append(leaf.grad)
            self.assert_matches(*gradients)

    def test_invalid_dimensions_and_binding_errors(self):
        for dim, top in itertools.product((2, -3, True, "bad", 2**100, (2,), [-3]), (False, True)):
            errors = []
            for module in (torch, reference_torch):
                source = module.ones((2, 3))
                call = module.mean if top else module.Tensor.mean
                with self.assertRaises(Exception) as raised:
                    call(source, dim)
                errors.append(raised.exception)
            with self.subTest(dim=dim, top=top):
                self.assertIs(type(errors[0]), type(errors[1]))
                self.assertEqual(str(errors[0]), str(errors[1]))

    def test_unsupported_forms_remain_explicitly_rejected(self):
        def calls(module):
            source = module.ones((2, 3))
            return (
                lambda: source.mean((0, 1)),
                lambda: module.mean(source, dim=[0, 1]),
                lambda: module.ones((2, 3, 4)).mean(1),
                lambda: module.mean(module.ones((2, 3, 4)), -1, True),
                lambda: module.tensor(2.0).mean(0),
                lambda: module.mean(source, 0, out=module.zeros((3,))),
                lambda: source.mean(1, dtype=reference_torch.float64),
                lambda: module.mean(source, 0, dtype=reference_torch.float64),
            )
        for index, (actual, expected) in enumerate(zip(calls(torch), calls(reference_torch), strict=True)):
            with self.subTest(case=index):
                with self.assertRaises((NotImplementedError, TypeError)):
                    actual()
                expected()


if __name__ == "__main__":
    unittest.main()
