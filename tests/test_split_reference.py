import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, 'install the reference dependency group')
class TensorSplitReferenceTests(unittest.TestCase):
    @staticmethod
    def split(module, source, size, dim=0):
        return source.split(split_size=size, dim=dim)

    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split('+')[0] != '2.13.0':
            raise AssertionError('split differentials require pinned PyTorch 2.13.0')

    def layout_contract(self, module):
        base = module.tensor([float(i) for i in range(30)], dtype=module.float32).reshape(2, 3, 5)
        cases = [(base, 2, -1), (base, 2, 1), (base, 100, 0),
                 (base, 2**63 - 1, 0), (base.transpose(0, 2), 2, 0),
                 (base[1].t(), 2, 0), (base[1].t().narrow(0, 3, 0), 0, 0)]
        for shape, dim in (((0,), 0), ((2, 0, 3), 1), ((2, 3, 0), -1)):
            cases.extend((module.zeros(shape), size, dim) for size in (0, 1, 10))
        cases.append((module.zeros((0, 5)), 2, 1))
        result = []
        for source, size, dim in cases:
            outputs = self.split(module, source, size, dim)
            self.assertIs(type(outputs), tuple)
            start = 0
            details = []
            for output in outputs:
                direct = source.narrow(dim, start, output.shape[dim])
                details.append((output.tolist(), tuple(output.shape), output.stride(),
                                output.storage_offset(), output.output_nr,
                                output.requires_grad, output.is_leaf,
                                str(output.dtype), str(output.device),
                                output.data_ptr() == direct.data_ptr(), output.is_set_to(direct)))
                start += output.shape[dim]
            result.append(details)
        return result

    def test_values_layout_aliasing_empty_and_oversized_match(self):
        self.assertEqual(self.layout_contract(torch), self.layout_contract(reference_torch))

    def gradient_contract(self, module):
        gradients = []
        for selected in ((2,), (0, 1, 2), (1, 1, 2)):
            leaf = module.tensor([float(i) for i in range(30)], dtype=module.float32, requires_grad=True)
            source = (leaf * 2.).reshape(2, 3, 5)[1].t()
            outputs = self.split(module, source, 2)
            loss = outputs[selected[0]].sum()
            for i in selected[1:]:
                loss = loss + outputs[i].sum()
            loss.backward()
            gradients.append(([(o.output_nr, o.requires_grad, o.is_leaf) for o in outputs], leaf.grad.tolist()))
        leaf = module.tensor([1., 2.], requires_grad=True)
        first, second = self.split(module, leaf, 1)
        ((first * -0.).sum() + second.sum()).backward()
        gradients.append(np.asarray(leaf.grad).view(np.uint32).tolist())
        leaf = module.tensor([float(i) for i in range(7)], requires_grad=True)
        outputs = self.split(module, leaf, 3)
        nested = self.split(module, outputs[1], 2)
        (nested[1].sum() + outputs[0].sum() + leaf.sum()).backward()
        gradients.append(leaf.grad.tolist())
        return gradients

    def test_selected_combined_repeated_nested_and_signed_zero_gradients_match(self):
        self.assertEqual(self.gradient_contract(torch), self.gradient_contract(reference_torch))

    def empty_and_no_grad_contract(self, module):
        result = []
        for size in (0, 2):
            leaf = module.zeros((2, 0, 3), requires_grad=True)
            outputs = self.split(module, leaf, size, 1)
            result.append([(o.output_nr, o.requires_grad, o.is_leaf) for o in outputs])
            outputs[0].sum().backward()
            result.append((tuple(leaf.grad.shape), leaf.grad.tolist()))
        for requires_grad in (False, True):
            leaf = module.ones((5,), requires_grad=requires_grad)
            with module.no_grad():
                outputs = self.split(module, leaf, 2)
            result.append([(o.output_nr, o.requires_grad, o.is_leaf) for o in outputs])
        return result

    def test_empty_backward_and_no_grad_match(self):
        self.assertEqual(self.empty_and_no_grad_contract(torch), self.empty_and_no_grad_contract(reference_torch))

    def error_contract(self, module):
        source = module.zeros((2, 3))
        scalar = module.tensor(1.)
        calls = [lambda: source.split(), lambda: source.split(2, 0, 0),
                 lambda: source.split(2, split_size=2), lambda: source.split(2, extra=0),
                 lambda: source.split(-1), lambda: source.split(0),
                 lambda: source.split(-1, 10), lambda: source.split(0, 10),
                 lambda: source.split(1, -3), lambda: source.split(1, 2),
                 lambda: source.split(2**100), lambda: source.split(1, 2**100),
                 lambda: scalar.split(1), lambda: scalar.split(-1)]
        errors = []
        for call in calls:
            with self.assertRaises(Exception) as raised:
                call()
            errors.append((type(raised.exception).__name__, str(raised.exception)))
        return errors

    def test_binding_bounds_and_overflow_errors_match(self):
        self.assertEqual(self.error_contract(torch), self.error_contract(reference_torch))

    def test_integer_subclasses_and_invalid_argument_types_match(self):
        class IntegerSubclass(int):
            pass
        for module in (torch, reference_torch):
            source = module.zeros((2, 3))
            self.assertEqual(len(self.split(module, source, IntegerSubclass(1), np.int64(1))), 3)
            for value in (True, None, 1.5, '2', np.int64(2)):
                with self.subTest(module=module.__name__, value=value):
                    with self.assertRaises(TypeError):
                        self.split(module, source, value)
            for dimension in (True, None, 1.5, '0'):
                with self.subTest(module=module.__name__, dimension=dimension):
                    with self.assertRaises(TypeError):
                        self.split(module, source, 1, dimension)


if __name__ == '__main__':
    unittest.main()
