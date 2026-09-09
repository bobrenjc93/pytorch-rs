import itertools
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference
except ImportError:
    reference = None


@unittest.skipIf(reference is None, "install the reference dependency group")
class TopLevelHstackReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("hstack differentials require pinned PyTorch 2.13.0")

    def assert_matches(self, actual, expected):
        self.assertIs(type(actual), torch.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            expected.detach().numpy().reshape(-1).view(np.uint32),
        )

    @staticmethod
    def error(call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("call unexpectedly succeeded")

    def check_forward(self, build):
        actual_inputs, expected_inputs = build(torch), build(reference)
        actual = torch.hstack(tensors=actual_inputs, out=None)
        expected = reference.hstack(tensors=expected_inputs, out=None)
        self.assert_matches(actual, expected)
        for source in actual_inputs:
            self.assertIsNot(actual, source)
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())
        return actual, expected

    def test_scalar_vector_matrix_empty_shape_combinations(self):
        shapes = ((), (0,), (1,), (3,), (0, 0), (0, 2), (2, 0), (1, 1), (1, 3), (2, 1), (2, 3))
        for pair in itertools.product(shapes, repeat=2):
            with self.subTest(shapes=pair):
                def build(module):
                    return [
                        module.tensor(
                            np.arange(np.prod(s, dtype=int), dtype=np.float32)
                            .reshape(s).tolist(),
                            dtype=module.float32,
                        )
                        for s in pair
                    ]
                expected_inputs = build(reference)
                try:
                    reference.hstack(expected_inputs)
                except RuntimeError:
                    self.assertEqual(
                        self.error(lambda: torch.hstack(build(torch))),
                        self.error(lambda: reference.hstack(expected_inputs)),
                    )
                else:
                    self.check_forward(build)

    def test_strided_offset_views_special_values_and_first_operand_axis(self):
        def strided_vectors(module):
            base = module.tensor([[1., -0., 3.], [4., 5., 6.]], dtype=module.float32)
            return (
                base.transpose(0, 1)[1],
                module.tensor(float("inf")),
                module.tensor([float("nan"), -float("inf")]),
            )

        def strided_matrices(module):
            base = module.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
            return (base[1].transpose(0, 1), module.zeros((4, 0)), base[0].transpose(0, 1))

        for build in (
            strided_vectors,
            strided_matrices,
            lambda m: [m.tensor([]), m.ones((2, 3)), m.ones((4, 3))],
            lambda m: [m.ones((2, 3)), m.tensor([]), m.ones((2, 4))],
            lambda m: [m.tensor(2.5)],
            lambda m: [m.tensor([])],
            lambda m: [m.ones((2, 3)).transpose(0, 1)],
        ):
            with self.subTest(build=build):
                self.check_forward(build)

    def test_gradients_through_views_repeated_inputs_and_empties(self):
        def scalar_vector(module):
            scalar = module.tensor(1.5, requires_grad=True)
            base = module.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
            empty = module.tensor([], requires_grad=True)
            return [scalar, base, empty], [scalar, base.transpose(0, 1)[1], empty, scalar]

        def matrix_columns(module):
            base = module.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(), requires_grad=True)
            empty = module.tensor([], requires_grad=True)
            view = base[1].transpose(0, 1)
            return [base, empty], [view, empty, view]

        def leading_empty(module):
            empty = module.tensor([], requires_grad=True)
            matrix = module.ones((2, 3), requires_grad=True)
            return [empty, matrix], [empty, matrix, matrix]

        def all_empty(module):
            empty = module.zeros((2, 0), requires_grad=True)
            return [empty], [empty, empty]

        def scalar_view(module):
            base = module.tensor([1., 2., 3.], requires_grad=True)
            constant = module.tensor([4., 5.])
            return [base, constant], [base[1], constant, base[1]]

        def single_scalar(module):
            scalar = module.tensor(2., requires_grad=True)
            return [scalar], [scalar]

        for build in (scalar_vector, matrix_columns, leading_empty, all_empty, scalar_view, single_scalar):
            with self.subTest(build=build):
                actual_leaves, actual_inputs = build(torch)
                expected_leaves, expected_inputs = build(reference)
                actual, expected = torch.hstack(actual_inputs), reference.hstack(expected_inputs)
                self.assert_matches(actual, expected)
                weights = np.arange(1, actual.numel() + 1, dtype=np.float32).reshape(actual.shape).tolist()
                (actual * torch.tensor(weights).reshape(actual.shape)).sum().backward()
                (expected * reference.tensor(weights).reshape(expected.shape)).sum().backward()
                for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
                    if expected_leaf.grad is None:
                        self.assertIsNone(actual_leaf.grad)
                    else:
                        self.assert_matches(actual_leaf.grad, expected_leaf.grad)
                with torch.no_grad():
                    untracked = torch.hstack(actual_inputs)
                with reference.no_grad():
                    expected_untracked = reference.hstack(expected_inputs)
                self.assert_matches(untracked, expected_untracked)

    def test_argument_errors_and_precedence(self):
        cases = (
            lambda m, x: m.hstack(),
            lambda m, x: m.hstack(out=x),
            lambda m, x: m.hstack([]),
            lambda m, x: m.hstack(()),
            lambda m, x: m.hstack(x),
            lambda m, x: m.hstack(tensors=x),
            lambda m, x: m.hstack(iter([x])),
            lambda m, x: m.hstack([x, None]),
            lambda m, x: m.hstack([x], None),
            lambda m, x: m.hstack([x], out=1),
            lambda m, x: m.hstack([x], tensors=[x]),
            lambda m, x: m.hstack([x], dim=0),
            lambda m, x: m.hstack([x], axis=0),
            lambda m, x: m.hstack([x], dtype=m.float32),
            lambda m, x: m.hstack([x, 1], dim=0),
            lambda m, x: m.hstack([x], out=1, dim=0),
            lambda m, x: m.hstack([m.zeros((2, 3)), m.zeros((3, 2))]),
            lambda m, x: m.hstack([m.tensor([]), m.zeros((2, 3)), m.zeros((2, 4))]),
        )
        for i, case in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(
                    self.error(lambda: case(torch, torch.tensor([1.0]))),
                    self.error(lambda: case(reference, reference.tensor([1.0]))),
                )

    def test_sequence_out_and_mode_override_dispatch(self):
        def observe(module, kind):
            events = []
            marker = object()
            function = module.hstack

            class Parent:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((
                        cls.__name__, func is function,
                        tuple(t.__name__ for t in types),
                        args[0] is original_args[0], kwargs == original_kwargs,
                    ))
                    return NotImplemented

            class Child(Parent):
                pass

            class Accept:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((
                        cls.__name__, func is function,
                        tuple(t.__name__ for t in types),
                        args[0] is original_args[0], kwargs == original_kwargs,
                    ))
                    return marker

            class Mode(module.overrides.TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    events.append(("mode", func is function, tuple(t.__name__ for t in types)))
                    return func(*args, **(kwargs or {}))

            if kind == "sequence":
                original_args, original_kwargs = (Accept(),), {}
            elif kind == "out":
                original_args, original_kwargs = ([module.tensor([1.])],), {"out": Accept()}
            else:
                original_args, original_kwargs = ([Parent(), Child(), Parent(), Accept()],), {"out": None}
            with Mode():
                result = function(*original_args, **original_kwargs)
            return result is marker, events, len(module.overrides._get_current_function_mode_stack())

        for kind in ("sequence", "elements", "out"):
            with self.subTest(kind=kind):
                self.assertEqual(observe(torch, kind), observe(reference, kind))

        for module in (torch, reference):
            class Decline:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    return NotImplemented

            with self.assertRaisesRegex(TypeError, "Multiple dispatch failed for 'torch.hstack'"):
                module.hstack([Decline()])

    def test_real_cuda_inputs_are_explicitly_rejected(self):
        if not reference.cuda.is_available():
            self.skipTest("requires an NVIDIA GPU and the CUDA reference runtime")
        for data in (1.0, [1.0, 2.0], [[1.0], [2.0]], []):
            with self.subTest(data=data):
                actual = torch.tensor(data).to("cuda:0")
                expected = reference.tensor(data, device="cuda:0")
                self.assertEqual(str(actual.device), "cuda:0")
                reference.hstack([expected])
                reference.cuda.synchronize(0)
                for inputs in ([actual], [torch.tensor([]), actual]):
                    with self.assertRaisesRegex(NotImplementedError, "hstack.*only exact native CPU float32"):
                        torch.hstack(inputs)
                self.assertEqual(actual.cpu().tolist(), data)

    def test_public_callable_metadata(self):
        for attribute in ("__name__", "__qualname__", "__module__"):
            self.assertEqual(getattr(torch.hstack, attribute), getattr(reference.hstack, attribute))
        self.assertEqual(torch.__all__.count("hstack"), reference.__all__.count("hstack"))
        self.assertEqual(hasattr(torch.functional, "hstack"), hasattr(reference.functional, "hstack"))


if __name__ == "__main__":
    unittest.main()
