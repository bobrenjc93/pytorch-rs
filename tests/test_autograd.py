import gc
import inspect
import statistics
import threading
import time
import unittest
import weakref

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class AutogradApiTests(unittest.TestCase):
    def test_requires_grad_leaf_grad_accumulation_and_nonleaf_grad(self):
        x = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad)

        square = x * x
        loss = square.sum()
        self.assertTrue(square.requires_grad)
        self.assertTrue(loss.requires_grad)
        self.assertIsNone(square.grad)
        loss.backward()
        retained_grad = x.grad
        self.assertIs(retained_grad, x.grad)
        np.testing.assert_array_equal(np.asarray(retained_grad), [-4.0, 1.0, 6.0])

        (x * x).sum().backward()
        self.assertIs(retained_grad, x.grad)
        np.testing.assert_array_equal(np.asarray(retained_grad), [-8.0, 2.0, 12.0])

    def test_requires_grad_requires_a_builtin_bool(self):
        class Truthy:
            def __bool__(self):
                return True

        for invalid in (np.bool_(True), np.bool_(False), 1, 0, None, "true", Truthy()):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(TypeError, "requires_grad.*must be bool"):
                    torch.tensor([1.0], requires_grad=invalid)

        self.assertTrue(torch.tensor([1.0], requires_grad=True).requires_grad)
        self.assertFalse(torch.tensor([1.0], requires_grad=False).requires_grad)

    def test_detach_and_no_grad_context_decorator_are_boundaries(self):
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        detached = x.detach()
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.tolist(), x.tolist())
        self.assertFalse((detached * detached).sum().requires_grad)

        with torch.no_grad():
            self.assertFalse((x * x).requires_grad)
            with torch.no_grad():
                self.assertFalse(x.sum().requires_grad)
            self.assertFalse((x * 4.0).requires_grad)
        self.assertTrue((x * x).requires_grad)

        @torch.no_grad()
        def square(value):
            return value * value

        self.assertFalse(square(x).requires_grad)
        self.assertTrue((x * x).requires_grad)

        with self.assertRaisesRegex(ValueError, "restore recording"):
            with torch.no_grad():
                raise ValueError("restore recording")
        self.assertTrue((x * x).requires_grad)

    def test_backward_errors_repeated_policy_and_graph_lifetime(self):
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.tensor(1.0).backward()
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            torch.tensor([1.0, 2.0]).backward()
        with self.assertRaisesRegex(RuntimeError, "implicitly created only for scalar"):
            torch.tensor([1.0, 2.0], requires_grad=True).backward()

        x = torch.tensor([2.0, 3.0], requires_grad=True)
        intermediate = x * x
        output = intermediate.sum()
        del intermediate
        gc.collect()
        output.backward()
        np.testing.assert_array_equal(np.asarray(x.grad), [4.0, 6.0])
        with self.assertRaisesRegex(RuntimeError, "backward through the graph a second time"):
            output.backward()

        scalar_leaf = torch.tensor([7.0], requires_grad=True)
        scalar_leaf.backward()
        scalar_leaf.backward()
        self.assertEqual(scalar_leaf.grad.tolist(), [2.0])

        summed_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        summed = summed_leaf.sum()
        summed.backward()
        summed.backward()
        self.assertEqual(summed_leaf.grad.tolist(), [2.0, 2.0])

        transformed_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        transformed = transformed_leaf.T.clone().sum()
        transformed.backward()
        transformed.backward()
        np.testing.assert_array_equal(
            np.asarray(transformed_leaf.grad), [[2.0, 2.0], [2.0, 2.0]]
        )

    def test_deep_graph_transformations_and_negative_zero(self):
        deep_leaf = torch.tensor(3.0, requires_grad=True)
        deep_output = deep_leaf
        for _ in range(20_000):
            deep_output = deep_output * 1.0
        deep_output.backward()
        self.assertEqual(deep_leaf.grad.item(), 1.0)

        leaf = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True)
        transformed = leaf.transpose(1, 2)
        self.assertTrue(transformed.requires_grad)
        transformed = transformed.contiguous().squeeze(0)[1].reshape(2, 1).clone()
        self.assertTrue(transformed.requires_grad)
        (transformed * transformed).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.array([[[0.0, 4.0, 0.0], [0.0, 10.0, 0.0]]], dtype=np.float32),
        )

        matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.assertTrue(matrix.T.requires_grad)
        (matrix.T * matrix.T).sum().backward()
        np.testing.assert_array_equal(np.asarray(matrix.grad), [[2.0, 4.0], [6.0, 8.0]])

        signed = torch.tensor([2.0], requires_grad=True)
        (signed * torch.tensor([-0.0])).sum().backward()
        self.assertEqual(np.asarray(signed.grad).view(np.uint32)[0], 0x8000_0000)

        broadcast_signed = torch.tensor(2.0, requires_grad=True)
        (broadcast_signed * torch.tensor([-0.0, -0.0])).sum().backward()
        self.assertEqual(np.asarray(broadcast_signed.grad).view(np.uint32).item(), 0)

    def test_no_grad_decorator_binds_methods_and_is_cross_thread_callable(self):
        class Model:
            @torch.no_grad()
            def forward(self, value, scale=1.0):
                return value * scale

        value = torch.tensor([2.0], requires_grad=True)
        model = Model()
        self.assertFalse(model.forward(value, scale=3.0).requires_grad)
        self.assertFalse(Model.forward(model, value).requires_grad)

        results = []
        failures = []

        @torch.no_grad()
        def worker(input_value):
            return input_value * input_value

        def run_worker():
            try:
                results.append(worker(value).requires_grad)
                results.append((value * value).requires_grad)
            except BaseException as error:  # PanicException inherits BaseException.
                failures.append(error)

        thread = threading.Thread(target=run_worker)
        thread.start()
        thread.join()
        self.assertEqual(failures, [])
        self.assertEqual(results, [False, True])
        self.assertTrue((value * value).requires_grad)

        context = torch.no_grad()
        context_results = []
        context_failures = []

        def run_context():
            try:
                with context:
                    context_results.append((value * value).requires_grad)
                context_results.append((value * value).requires_grad)
            except BaseException as error:
                context_failures.append(error)

        context_thread = threading.Thread(target=run_context)
        context_thread.start()
        context_thread.join()
        self.assertEqual(context_failures, [])
        self.assertEqual(context_results, [False, True])

    def test_no_grad_public_contract_and_callable_cycles(self):
        value = torch.tensor([2.0], requires_grad=True)
        with torch.no_grad() as entered:
            self.assertIsNone(entered)
            self.assertFalse((value * value).requires_grad)

        @torch.no_grad
        def direct(input_value: object, scale: float = 1.0) -> object:
            """A metadata-bearing no-grad callable."""
            return input_value * scale

        self.assertFalse(direct(value, scale=3.0).requires_grad)
        self.assertEqual(direct.__name__, "direct")
        self.assertEqual(direct.__doc__, "A metadata-bearing no-grad callable.")
        self.assertEqual(
            direct.__annotations__,
            {"input_value": object, "scale": float, "return": object},
        )
        self.assertEqual(inspect.signature(direct), inspect.signature(direct.__wrapped__))
        self.assertTrue(gc.is_tracked(direct))

        def make_callable_cycle():
            wrapped = None

            def function():
                return wrapped

            wrapped = torch.no_grad()(function)
            return weakref.ref(wrapped)

        callable_reference = make_callable_cycle()
        gc.collect()
        self.assertIsNone(callable_reference())

    def test_no_grad_decorator_guards_every_generator_resume(self):
        value = torch.tensor([2.0], requires_grad=True)
        events = []

        @torch.no_grad()
        def generate():
            events.append(("next", (value * value).requires_grad))
            request = yield value * value
            events.append(("send", request, (value * value).requires_grad))
            try:
                yield value * value
            except ValueError as error:
                events.append(("throw", str(error), (value * value).requires_grad))
                yield value * value
            finally:
                events.append(("close", (value * value).requires_grad))

        self.assertTrue(inspect.isgeneratorfunction(generate))
        generator = generate()
        self.assertTrue(inspect.isgenerator(generator))
        self.assertTrue(gc.is_tracked(generator))
        self.assertIs(iter(generator), generator)
        self.assertFalse(next(generator).requires_grad)
        self.assertTrue((value * value).requires_grad)
        self.assertFalse(generator.send("request").requires_grad)
        self.assertTrue((value * value).requires_grad)
        self.assertFalse(generator.throw(ValueError("injected")).requires_grad)
        self.assertTrue((value * value).requires_grad)
        self.assertIsNone(generator.close())
        self.assertEqual(
            events,
            [
                ("next", False),
                ("send", "request", False),
                ("throw", "injected", False),
                ("close", False),
            ],
        )
        self.assertTrue((value * value).requires_grad)

        abandoned_events = []

        @torch.no_grad()
        def abandoned():
            try:
                yield value * value
            finally:
                abandoned_events.append((value * value).requires_grad)

        generator = abandoned()
        self.assertFalse(next(generator).requires_grad)
        del generator
        gc.collect()
        self.assertEqual(abandoned_events, [False])
        self.assertTrue((value * value).requires_grad)

        cyclic_events = []

        @torch.no_grad()
        def cyclic():
            proxy = yield None
            try:
                yield proxy
            finally:
                cyclic_events.append((value * value).requires_grad)

        generator = cyclic()
        next(generator)
        generator.send(generator)
        generator_reference = weakref.ref(generator)
        del generator
        gc.collect()
        self.assertIsNone(generator_reference())
        self.assertEqual(cyclic_events, [False])
        self.assertTrue((value * value).requires_grad)

    def test_unconsumed_deep_graph_drop_and_detach_are_stack_safe(self):
        leaf = torch.tensor(3.0, requires_grad=True)
        output = leaf
        for _ in range(100_000):
            output = output * 1.0
        del output
        gc.collect()

        output = leaf
        for _ in range(100_000):
            output = output * 1.0
        detached = output.detach()
        del output
        gc.collect()
        self.assertFalse(detached.requires_grad)
        self.assertEqual(detached.item(), 3.0)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AutogradReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("autograd differentials require pinned PyTorch 2.13.0")

    def assert_gradient_pair(self, native, expected):
        self.assertEqual(native.shape, tuple(expected.shape))
        self.assertEqual(native.grad.shape, tuple(expected.grad.shape))
        np.testing.assert_allclose(
            np.asarray(native.grad),
            expected.grad.detach().cpu().numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    def test_requires_grad_argument_types_match_pytorch_2_13(self):
        class Truthy:
            def __bool__(self):
                return True

        values = [np.bool_(True), np.bool_(False), 1, 0, None, "true", Truthy()]
        outcomes = []
        for module in (torch, reference_torch):
            errors = []
            for value in values:
                try:
                    module.tensor([1.0], requires_grad=value)
                except TypeError as error:
                    errors.append(str(error))
                else:
                    self.fail(f"{module.__name__} accepted requires_grad={value!r}")
            outcomes.append(errors)

        self.assertEqual(outcomes[0], outcomes[1])

    def test_leaf_grad_identity_and_live_accumulation_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor([2.0, 3.0], requires_grad=True)
            loss = leaf.sum()
            loss.backward()
            retained = leaf.grad
            first_identity = retained is leaf.grad
            first_values = np.asarray(retained).copy()

            loss.backward()
            outcomes.append(
                (
                    first_identity,
                    retained is leaf.grad,
                    first_values,
                    np.asarray(retained).copy(),
                )
            )

        self.assertEqual(outcomes[0][:2], outcomes[1][:2])
        np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])
        np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])

    def test_seeded_square_sum_and_broadcast_gradients_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xA670_213)
        shape_pairs = [
            ((17,), (17,)),
            ((2, 1, 3), (1, 5, 1)),
            ((3, 1), (1, 7)),
            ((), (4, 3)),
            ((0,), (1,)),
        ]
        for left_shape, right_shape in shape_pairs:
            left_values = rng.normal(size=left_shape or ()).astype(np.float32)
            right_values = rng.normal(size=right_shape or ()).astype(np.float32)
            native_left_data = left_values.item() if left_shape == () else left_values.tolist()
            native_right_data = right_values.item() if right_shape == () else right_values.tolist()
            native_left = torch.tensor(native_left_data, requires_grad=True)
            native_right = torch.tensor(native_right_data, requires_grad=True)
            expected_left = reference_torch.tensor(left_values, requires_grad=True)
            expected_right = reference_torch.tensor(right_values, requires_grad=True)

            native_output = (native_left * native_right).sum()
            expected_output = (expected_left * expected_right).sum()
            native_output.backward()
            expected_output.backward()

            with self.subTest(left=left_shape, right=right_shape):
                np.testing.assert_allclose(
                    native_output.item(), expected_output.item(), rtol=1.0e-5, atol=1.0e-6
                )
                self.assert_gradient_pair(native_left, expected_left)
                self.assert_gradient_pair(native_right, expected_right)

        square_values = rng.normal(size=(11, 7)).astype(np.float32)
        native_square = torch.tensor(square_values.tolist(), requires_grad=True)
        expected_square = reference_torch.tensor(square_values, requires_grad=True)
        (native_square * native_square).sum().backward()
        (expected_square * expected_square).sum().backward()
        self.assert_gradient_pair(native_square, expected_square)

    def test_boundaries_scalar_empty_and_errors_match_pytorch_2_13(self):
        accumulated_gradients = []
        for module in (torch, reference_torch):
            scalar = module.tensor(3.0, requires_grad=True)
            (scalar * 5.0).backward()
            self.assertEqual(scalar.grad.item(), 5.0)

            empty = module.tensor(np.empty((0,), dtype=np.float32), requires_grad=True)
            empty.sum().backward()
            self.assertEqual(tuple(empty.grad.shape), (0,))
            self.assertEqual(empty.grad.numel(), 0)

            boundary = module.tensor([2.0], requires_grad=True)
            self.assertFalse((boundary.detach() * boundary.detach()).requires_grad)
            with module.no_grad():
                self.assertFalse((boundary * boundary).requires_grad)
            self.assertTrue((boundary * boundary).requires_grad)

            intermediate = boundary * boundary
            output = intermediate.sum()
            del intermediate
            gc.collect()
            output.backward()
            with self.assertRaises(RuntimeError):
                output.backward()
            (boundary * boundary).sum().backward()
            accumulated_gradients.append(np.asarray(boundary.grad).copy())
            with self.assertRaises(RuntimeError):
                module.tensor([1.0, 2.0], requires_grad=True).backward()
            with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                module.tensor([1.0, 2.0]).backward()
        np.testing.assert_array_equal(accumulated_gradients[0], accumulated_gradients[1])

    def test_transform_and_signed_zero_gradients_match_pytorch_2_13(self):
        gradients = []
        reshape_gradients = []
        signed_zero_bits = []
        broadcast_zero_bits = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], requires_grad=True
            )
            transformed = leaf.transpose(1, 2).contiguous().squeeze(0)[1].reshape(2, 1).clone()
            (transformed * transformed).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())

            reshape_leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            reshaped_copy = reshape_leaf.transpose(0, 1).reshape(6)
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (reshaped_copy * weights).sum().backward()
            reshape_gradients.append(np.asarray(reshape_leaf.grad).copy())

            signed = module.tensor([2.0], requires_grad=True)
            (signed * module.tensor([-0.0])).sum().backward()
            signed_zero_bits.append(np.asarray(signed.grad).view(np.uint32)[0])

            broadcast_signed = module.tensor(2.0, requires_grad=True)
            (broadcast_signed * module.tensor([-0.0, -0.0])).sum().backward()
            broadcast_zero_bits.append(
                np.asarray(broadcast_signed.grad).view(np.uint32).item()
            )

        np.testing.assert_array_equal(gradients[0], gradients[1])
        np.testing.assert_array_equal(reshape_gradients[0], reshape_gradients[1])
        self.assertEqual(signed_zero_bits, [0x8000_0000, 0x8000_0000])
        self.assertEqual(broadcast_zero_bits, [0, 0])

    def test_no_grad_generator_protocol_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            value = module.tensor([2.0], requires_grad=True)
            events = []

            @module.no_grad()
            def generate():
                events.append(("next", (value * value).requires_grad))
                request = yield value * value
                events.append(("send", request, (value * value).requires_grad))
                try:
                    yield value * value
                except ValueError as error:
                    events.append(("throw", str(error), (value * value).requires_grad))
                    yield value * value
                finally:
                    events.append(("close", (value * value).requires_grad))

            generator = generate()
            requires_grad = [next(generator).requires_grad]
            requires_grad.append(generator.send("request").requires_grad)
            requires_grad.append(generator.throw(ValueError("injected")).requires_grad)
            generator.close()
            abandoned_events = []

            @module.no_grad()
            def abandoned():
                try:
                    yield value * value
                finally:
                    abandoned_events.append((value * value).requires_grad)

            abandoned_generator = abandoned()
            requires_grad.append(next(abandoned_generator).requires_grad)
            del abandoned_generator
            gc.collect()
            outcomes.append(
                (
                    requires_grad,
                    events,
                    abandoned_events,
                    (value * value).requires_grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_no_grad_views_and_public_contract_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            source = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            with module.no_grad() as entered:
                views = [
                    source.transpose(0, 1),
                    source.T,
                    source.reshape(4),
                    source.squeeze(),
                    source[0],
                ]
                materialized = [source.clone(), source.T.contiguous()]

            with self.assertRaisesRegex(RuntimeError, "implicitly created only for scalar"):
                views[2].backward()

            (views[0] * views[0]).sum().backward()

            @module.no_grad
            def decorated(value: object, scale: float = 1.0) -> object:
                """decorated docs"""
                return value * scale

            @module.no_grad()
            def generator():
                yield source * source

            outcomes.append(
                (
                    entered,
                    [view.requires_grad for view in views],
                    [tensor.requires_grad for tensor in materialized],
                    source.grad is None,
                    decorated.__name__,
                    decorated.__doc__,
                    decorated.__annotations__,
                    str(inspect.signature(decorated)),
                    decorated.__wrapped__.__name__,
                    inspect.isgeneratorfunction(generator),
                    next(generator()).requires_grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_reusable_metadata_only_graphs_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            summed_leaf = module.tensor([1.0, 2.0], requires_grad=True)
            summed = summed_leaf.sum()
            summed.backward()
            summed.backward()

            transformed_leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            transformed = transformed_leaf.T.clone().sum()
            transformed.backward()
            transformed.backward()

            multiplied_leaf = module.tensor([1.0, 2.0], requires_grad=True)
            multiplied = (multiplied_leaf * multiplied_leaf).sum()
            multiplied.backward()
            with self.assertRaisesRegex(
                RuntimeError, "backward through the graph a second time"
            ):
                multiplied.backward()

            outcomes.append(
                (
                    np.asarray(summed_leaf.grad).copy(),
                    np.asarray(transformed_leaf.grad).copy(),
                    np.asarray(multiplied_leaf.grad).copy(),
                )
            )

        for native, expected in zip(outcomes[0], outcomes[1]):
            np.testing.assert_array_equal(native, expected)

    def test_square_sum_backward_performance_smoke_is_equivalent_work(self):
        values = np.linspace(-2.0, 2.0, 16_384, dtype=np.float32)

        def measure(module):
            inputs = [module.tensor(values, requires_grad=True) for _ in range(10)]
            samples = []
            checksum = 0.0
            for index, value in enumerate(inputs):
                started = time.perf_counter()
                (value * value).sum().backward()
                elapsed = time.perf_counter() - started
                checksum += float((value.grad * value.grad).sum().item())
                if index >= 3:
                    samples.append(elapsed)
            return statistics.median(samples), checksum

        gc.collect()
        native_seconds, native_checksum = measure(torch)
        gc.collect()
        reference_seconds, reference_checksum = measure(reference_torch)

        self.assertGreater(native_seconds, 0.0)
        self.assertGreater(reference_seconds, 0.0)
        np.testing.assert_allclose(native_checksum, reference_checksum, rtol=1.0e-5)
        self.assertLess(native_seconds / reference_seconds, 250.0)


if __name__ == "__main__":
    unittest.main()
