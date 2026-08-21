import copy
import importlib
import inspect
import pickle
import threading
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class StatefulEnableGrad(torch.enable_grad):
    init_calls = 0

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, required):
        type(self).init_calls += 1
        super().__init__()
        self.required = required


class NewArgsEnableGrad(torch.enable_grad):
    __slots__ = ("constructed",)
    init_calls = 0
    new_calls = 0

    def __new__(cls, constructed):
        cls.new_calls += 1
        instance = super().__new__(cls)
        instance.constructed = constructed
        return instance

    def __init__(self, constructed):
        type(self).init_calls += 1
        super().__init__()
        self.mutated = {"state": [7, 8]}

    def __getnewargs__(self):
        return (self.constructed,)

    def __getstate__(self):
        return self.__dict__


if reference_torch is not None:

    class ReferenceStatefulEnableGrad(reference_torch.enable_grad):
        init_calls = 0

        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __init__(self, required):
            type(self).init_calls += 1
            super().__init__()
            self.required = required

    class ReferenceNewArgsEnableGrad(reference_torch.enable_grad):
        __slots__ = ("constructed",)
        init_calls = 0
        new_calls = 0

        def __new__(cls, constructed):
            cls.new_calls += 1
            instance = super().__new__(cls)
            instance.constructed = constructed
            return instance

        def __init__(self, constructed):
            type(self).init_calls += 1
            super().__init__()
            self.mutated = {"state": [7, 8]}

        def __getnewargs__(self):
            return (self.constructed,)

        def __getstate__(self):
            return self.__dict__

else:
    ReferenceStatefulEnableGrad = None
    ReferenceNewArgsEnableGrad = None


class EnableGradTests(unittest.TestCase):
    def test_canonical_aliases_and_unsupported_siblings(self):
        autograd = importlib.import_module("torch_rs.autograd")
        grad_mode = importlib.import_module("torch_rs.autograd.grad_mode")
        from torch_rs.autograd import enable_grad as autograd_enable_grad
        from torch_rs.autograd.grad_mode import enable_grad as grad_mode_enable_grad

        self.assertIs(torch.autograd, autograd)
        self.assertIs(autograd.grad_mode, grad_mode)
        self.assertIs(torch.enable_grad, autograd.enable_grad)
        self.assertIs(torch.enable_grad, grad_mode.enable_grad)
        self.assertIs(torch.enable_grad, autograd_enable_grad)
        self.assertIs(torch.enable_grad, grad_mode_enable_grad)
        self.assertEqual(torch.__all__.count("enable_grad"), 1)
        self.assertEqual(autograd.__all__, ["grad_mode", "enable_grad", "no_grad"])
        self.assertEqual(grad_mode.__all__, ["no_grad", "enable_grad"])

        for module in (torch, autograd, grad_mode):
            for unsupported in ("set_grad_enabled", "inference_mode"):
                with self.subTest(module=module.__name__, unsupported=unsupported):
                    self.assertFalse(hasattr(module, unsupported))

    def test_metadata_signatures_copy_and_pickle(self):
        context_type = torch.enable_grad
        grad_mode = torch.autograd.grad_mode

        self.assertEqual(context_type.__name__, "enable_grad")
        self.assertEqual(context_type.__qualname__, "enable_grad")
        self.assertEqual(context_type.__module__, "torch_rs.autograd.grad_mode")
        self.assertIs(inspect.getmodule(context_type), grad_mode)
        self.assertEqual(
            str(inspect.signature(context_type)),
            "(orig_func: Optional[~F] = None) -> Union[Self, ~F]",
        )
        self.assertEqual(
            str(inspect.signature(context_type.__new__)),
            "(cls, orig_func: Optional[~F] = None) -> Union[Self, ~F]",
        )
        self.assertEqual(
            str(inspect.signature(context_type.__call__)),
            "(self, orig_func: ~F) -> ~F",
        )
        self.assertEqual(str(inspect.signature(context_type.__enter__)), "(self) -> None")
        self.assertEqual(
            str(inspect.signature(context_type.__exit__)),
            "(self, exc_type: Any, exc_value: Any, traceback: Any) -> None",
        )
        self.assertEqual(
            str(inspect.signature(context_type())), "(orig_func: ~F) -> ~F"
        )
        self.assertIs(copy.copy(context_type), context_type)
        self.assertIs(copy.deepcopy(context_type), context_type)

        instance = context_type()
        self.assertEqual(instance.__dict__, {})
        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__):
                restored = operation(instance)
                self.assertIsNot(restored, instance)
                self.assertIs(type(restored), context_type)
                self.assertEqual(restored.__dict__, {})

        canonical_path = b"torch_rs.autograd.grad_mode"
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, value="class"):
                payload = pickle.dumps(context_type, protocol=protocol)
                self.assertIn(canonical_path, payload)
                self.assertIs(pickle.loads(payload), context_type)

            with self.subTest(protocol=protocol, value="instance"):
                payload = pickle.dumps(instance, protocol=protocol)
                self.assertIn(canonical_path, payload)
                restored = pickle.loads(payload)
                self.assertIsNot(restored, instance)
                self.assertIs(type(restored), context_type)
                self.assertEqual(restored.__dict__, {})

        with torch.no_grad():
            with instance:
                self.assertEqual(instance.__dict__, {"prev": False})
            self.assertFalse(torch.is_grad_enabled())
        for operation in (copy.copy, copy.deepcopy):
            with self.subTest(operation=operation.__name__, state="entered"):
                restored = operation(instance)
                self.assertEqual(restored.__dict__, {"prev": False})
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, state="entered"):
                restored = pickle.loads(pickle.dumps(instance, protocol=protocol))
                self.assertEqual(restored.__dict__, {"prev": False})

    def test_context_nesting_exceptions_and_recording(self):
        value = torch.tensor([2.0], requires_grad=True)
        self.assertTrue(torch.is_grad_enabled())

        with torch.enable_grad() as entered:
            self.assertIsNone(entered)
            self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse((value * value).requires_grad)
            with torch.enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                enabled = value * value
                self.assertTrue(enabled.requires_grad)
                with torch.no_grad():
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertFalse((value * value).requires_grad)
                self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(torch.is_grad_enabled())

        enabled.sum().backward()
        self.assertEqual(value.grad.tolist(), [4.0])

        with self.assertRaisesRegex(RuntimeError, "restore enable-grad state"):
            with torch.no_grad():
                with torch.enable_grad():
                    self.assertTrue(torch.is_grad_enabled())
                    raise RuntimeError("restore enable-grad state")
        self.assertTrue(torch.is_grad_enabled())

    def test_decorated_functions_generators_and_threads(self):
        value = torch.tensor([2.0], requires_grad=True)

        @torch.enable_grad()
        def decorated(scale=1.0):
            return (value * scale).requires_grad

        @torch.enable_grad
        def direct():
            return (value * value).requires_grad

        @torch.enable_grad()
        def failing():
            self.assertTrue(torch.is_grad_enabled())
            raise RuntimeError("decorated failure")

        events = []

        @torch.autograd.grad_mode.enable_grad()
        def generate():
            events.append(("next", torch.is_grad_enabled()))
            request = yield (value * value).requires_grad
            events.append(("send", request, torch.is_grad_enabled()))
            try:
                yield (value * value).requires_grad
            except ValueError as error:
                events.append(("throw", str(error), torch.is_grad_enabled()))
                yield (value * value).requires_grad
            finally:
                events.append(("close", torch.is_grad_enabled()))

        with torch.no_grad():
            self.assertTrue(decorated(scale=3.0))
            self.assertTrue(direct())
            with self.assertRaisesRegex(RuntimeError, "decorated failure"):
                failing()
            self.assertFalse(torch.is_grad_enabled())
            generator = generate()
            self.assertTrue(inspect.isgenerator(generator))
            self.assertTrue(next(generator))
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(generator.send("request"))
            self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(generator.throw(ValueError("injected")))
            self.assertFalse(torch.is_grad_enabled())
            self.assertIsNone(generator.close())
            self.assertFalse(torch.is_grad_enabled())

        self.assertEqual(
            events,
            [
                ("next", True),
                ("send", "request", True),
                ("throw", "injected", True),
                ("close", True),
            ],
        )
        self.assertTrue(torch.is_grad_enabled())

        worker_states = []

        def worker():
            worker_states.append(torch.is_grad_enabled())
            with torch.no_grad():
                worker_states.append(torch.is_grad_enabled())
                worker_states.append(decorated())
                worker_states.append(torch.is_grad_enabled())
            worker_states.append(torch.is_grad_enabled())

        with torch.no_grad():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertFalse(torch.is_grad_enabled())

        self.assertEqual(worker_states, [True, False, True, False, True])
        self.assertTrue(torch.is_grad_enabled())

    def test_argument_errors_match_the_public_protocol(self):
        context_type = torch.enable_grad
        instance = context_type()
        cases = (
            (
                lambda: context_type(1, 2),
                "_NoParamDecoratorContextManager.__new__() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: context_type(original_function=None),
                "_NoParamDecoratorContextManager.__new__() got an unexpected keyword argument 'original_function'",
            ),
            (
                lambda: context_type(foo=1),
                "_NoParamDecoratorContextManager.__new__() got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: instance(),
                "_DecoratorContextManager.__call__() missing 1 required positional argument: 'orig_func'",
            ),
            (
                lambda: instance(1, 2),
                "_DecoratorContextManager.__call__() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: instance(function=lambda: None),
                "_DecoratorContextManager.__call__() got an unexpected keyword argument 'function'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        self.assertIs(type(context_type(None)), context_type)
        self.assertIs(type(context_type(orig_func=None)), context_type)

    def test_copy_and_pickle_preserve_instance_and_subclass_state(self):
        instance = torch.enable_grad()
        instance.payload = {"values": [1, 2]}

        shallow = copy.copy(instance)
        self.assertIs(shallow.payload, instance.payload)
        deep = copy.deepcopy(instance)
        self.assertEqual(deep.payload, instance.payload)
        self.assertIsNot(deep.payload, instance.payload)

        StatefulEnableGrad.init_calls = 0
        required = {"required": [3, 4]}
        subclass_instance = StatefulEnableGrad(required)
        subclass_instance.mutated = {"mutated": [5, 6]}
        shallow_subclass = copy.copy(subclass_instance)
        deep_subclass = copy.deepcopy(subclass_instance)
        self.assertIs(type(shallow_subclass), StatefulEnableGrad)
        self.assertIs(shallow_subclass.required, required)
        self.assertIs(shallow_subclass.mutated, subclass_instance.mutated)
        self.assertIs(type(deep_subclass), StatefulEnableGrad)
        self.assertEqual(deep_subclass.required, required)
        self.assertEqual(deep_subclass.mutated, subclass_instance.mutated)
        self.assertIsNot(deep_subclass.required, required)
        self.assertIsNot(deep_subclass.mutated, subclass_instance.mutated)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(
                    pickle.dumps(subclass_instance, protocol=protocol)
                )
                self.assertIs(type(restored), StatefulEnableGrad)
                self.assertEqual(restored.required, required)
                self.assertEqual(restored.mutated, subclass_instance.mutated)
                self.assertIsNot(restored.required, required)
                self.assertIsNot(restored.mutated, subclass_instance.mutated)

        self.assertEqual(StatefulEnableGrad.init_calls, 1)

    def test_copy_and_pickle_honor_subclass_getnewargs(self):
        NewArgsEnableGrad.init_calls = 0
        NewArgsEnableGrad.new_calls = 0
        constructed = {"constructed": [1, 2]}
        instance = NewArgsEnableGrad(constructed)

        shallow = copy.copy(instance)
        self.assertIs(type(shallow), NewArgsEnableGrad)
        self.assertIs(shallow.constructed, constructed)
        deep = copy.deepcopy(instance)
        self.assertIs(type(deep), NewArgsEnableGrad)
        self.assertEqual(deep.constructed, constructed)
        self.assertIsNot(deep.constructed, constructed)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(instance, protocol=protocol))
                self.assertIs(type(restored), NewArgsEnableGrad)
                self.assertEqual(restored.mutated, instance.mutated)
                if protocol < 2:
                    self.assertFalse(hasattr(restored, "constructed"))
                else:
                    self.assertEqual(restored.constructed, constructed)
                    self.assertIsNot(restored.constructed, constructed)

        self.assertEqual(NewArgsEnableGrad.init_calls, 1)
        self.assertEqual(NewArgsEnableGrad.new_calls, pickle.HIGHEST_PROTOCOL + 2)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EnableGradReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "enable-grad differentials require pinned PyTorch 2.13.0"
            )

    def metadata_contract(self, module):
        context_type = module.enable_grad
        instance = context_type()
        return {
            "aliases": (
                module.autograd.enable_grad is context_type,
                module.autograd.grad_mode.enable_grad is context_type,
            ),
            "metadata": (
                context_type.__name__,
                context_type.__qualname__,
                context_type.__module__.removeprefix(module.__name__),
                context_type.__doc__,
                context_type.__new__.__module__.removeprefix(module.__name__),
                context_type.__new__.__qualname__,
                context_type.__call__.__module__.removeprefix(module.__name__),
                context_type.__call__.__qualname__,
            ),
            "signatures": tuple(
                str(inspect.signature(value))
                for value in (
                    context_type,
                    context_type.__new__,
                    context_type.__call__,
                    context_type.__enter__,
                    context_type.__exit__,
                    instance,
                )
            ),
            "copy": (
                copy.copy(context_type) is context_type,
                copy.deepcopy(context_type) is context_type,
                type(copy.copy(instance)) is context_type,
                type(copy.deepcopy(instance)) is context_type,
            ),
            "pickle": tuple(
                (
                    pickle.loads(pickle.dumps(context_type, protocol=protocol))
                    is context_type,
                    type(
                        pickle.loads(pickle.dumps(instance, protocol=protocol))
                    )
                    is context_type,
                )
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def error_contract(self, module):
        context_type = module.enable_grad
        instance = context_type()
        calls = (
            lambda: context_type(1, 2),
            lambda: context_type(original_function=None),
            lambda: context_type(foo=1),
            lambda: instance(),
            lambda: instance(1, 2),
            lambda: instance(function=lambda: None),
        )
        errors = []
        for call in calls:
            try:
                call()
            except TypeError as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail(f"{module.__name__}.enable_grad accepted invalid arguments")
        return errors

    def state_contract(self, module):
        states = [module.is_grad_enabled()]
        value = module.tensor([2.0], requires_grad=True)
        with module.no_grad():
            states.append(module.is_grad_enabled())
            with module.enable_grad():
                states.append(module.is_grad_enabled())
                output = value * value
                states.append(output.requires_grad)
                with module.no_grad():
                    states.append(module.is_grad_enabled())
                states.append(module.is_grad_enabled())
            states.append(module.is_grad_enabled())
        states.append(module.is_grad_enabled())
        output.sum().backward()
        return states, value.grad.tolist()

    def test_metadata_signatures_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_contract(torch), self.metadata_contract(reference_torch)
        )

    def test_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(self.error_contract(torch), self.error_contract(reference_torch))

    def test_nested_recording_matches_pytorch_2_13(self):
        self.assertEqual(self.state_contract(torch), self.state_contract(reference_torch))


if __name__ == "__main__":
    unittest.main()
