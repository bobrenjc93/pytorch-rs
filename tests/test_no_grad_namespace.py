import copy
import importlib
import inspect
import pickle
import threading
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class StatefulNoGrad(torch.no_grad):
    init_calls = 0

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, required):
        type(self).init_calls += 1
        super().__init__()
        self.required = required


class NewArgsNoGrad(torch.no_grad):
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
    class ReferenceStatefulNoGrad(reference_torch.no_grad):
        init_calls = 0

        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __init__(self, required):
            type(self).init_calls += 1
            super().__init__()
            self.required = required

    class ReferenceNewArgsNoGrad(reference_torch.no_grad):
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
    ReferenceStatefulNoGrad = None
    ReferenceNewArgsNoGrad = None
    ReferenceStatefulEnableGrad = None
    ReferenceNewArgsEnableGrad = None


class NoGradNamespaceTests(unittest.TestCase):
    def assert_context_restores(self, context, *, expected_enabled):
        if not expected_enabled:
            with context:
                self.assertFalse(torch.is_grad_enabled())
            self.assertTrue(torch.is_grad_enabled())
            return

        self.assertTrue(torch.is_grad_enabled())
        with context:
            self.assertTrue(torch.is_grad_enabled())
        self.assertTrue(torch.is_grad_enabled())

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            with context:
                self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(torch.is_grad_enabled())

    def test_canonical_imports_are_identical_and_minimal(self):
        autograd = importlib.import_module("torch_rs.autograd")
        grad_mode = importlib.import_module("torch_rs.autograd.grad_mode")
        from torch_rs import enable_grad as top_level_enable_grad
        from torch_rs.autograd import no_grad as autograd_no_grad
        from torch_rs.autograd import enable_grad as autograd_enable_grad
        from torch_rs.autograd.grad_mode import no_grad as grad_mode_no_grad
        from torch_rs.autograd.grad_mode import enable_grad as grad_mode_enable_grad

        self.assertIs(torch.autograd, autograd)
        self.assertIs(autograd.grad_mode, grad_mode)
        self.assertIs(torch.enable_grad, top_level_enable_grad)
        self.assertIs(torch.no_grad, autograd.no_grad)
        self.assertIs(torch.enable_grad, autograd.enable_grad)
        self.assertIs(torch.no_grad, grad_mode.no_grad)
        self.assertIs(torch.enable_grad, grad_mode.enable_grad)
        self.assertIs(torch.no_grad, autograd_no_grad)
        self.assertIs(torch.enable_grad, autograd_enable_grad)
        self.assertIs(torch.no_grad, grad_mode_no_grad)
        self.assertIs(torch.enable_grad, grad_mode_enable_grad)
        self.assertEqual(
            autograd.__all__, ["backward", "grad_mode", "enable_grad", "no_grad"]
        )
        self.assertEqual(grad_mode.__all__, ["no_grad", "enable_grad"])
        self.assertNotIn("autograd", torch.__all__)
        self.assertIn("enable_grad", torch.__all__)

        for module in (autograd, grad_mode):
            for unsupported in ("grad",):
                with self.subTest(module=module.__name__, unsupported=unsupported):
                    self.assertFalse(hasattr(module, unsupported))
        for unsupported in ("set_grad_enabled", "inference_mode"):
            with self.subTest(module=torch.__name__, unsupported=unsupported):
                self.assertFalse(hasattr(torch, unsupported))
        self.assertIs(autograd.backward, torch.autograd.backward)
        self.assertFalse(hasattr(grad_mode, "backward"))

    def test_grad_mode_reloads_keep_canonical_context_classes(self):
        autograd = importlib.import_module("torch_rs.autograd")
        grad_mode = importlib.import_module("torch_rs.autograd.grad_mode")
        reloaded_grad_mode = importlib.reload(grad_mode)
        reloaded_autograd = importlib.reload(autograd)

        self.assertIs(reloaded_grad_mode.no_grad, torch.no_grad)
        self.assertIs(reloaded_grad_mode.enable_grad, torch.enable_grad)
        self.assertIs(reloaded_autograd.no_grad, torch.no_grad)
        self.assertIs(reloaded_autograd.enable_grad, torch.enable_grad)
        self.assertIs(reloaded_autograd.grad_mode, reloaded_grad_mode)
        self.assertEqual(reloaded_grad_mode.__all__, ["no_grad", "enable_grad"])
        self.assertEqual(
            reloaded_autograd.__all__,
            ["backward", "grad_mode", "enable_grad", "no_grad"],
        )

    def test_metadata_copy_and_pickle_resolve_through_grad_mode(self):
        grad_mode = torch.autograd.grad_mode

        canonical_path = b"torch_rs.autograd.grad_mode"
        for context_name in ("no_grad", "enable_grad"):
            context_type = getattr(torch, context_name)
            expected_enabled = context_name == "enable_grad"

            with self.subTest(context_name=context_name, metadata=True):
                self.assertEqual(context_type.__name__, context_name)
                self.assertEqual(context_type.__qualname__, context_name)
                self.assertEqual(
                    context_type.__module__, "torch_rs.autograd.grad_mode"
                )
                self.assertIs(inspect.getmodule(context_type), grad_mode)
                self.assertIs(copy.copy(context_type), context_type)
                self.assertIs(copy.deepcopy(context_type), context_type)

            instance = context_type()
            self.assert_context_restores(
                instance, expected_enabled=expected_enabled
            )
            for operation in (copy.copy, copy.deepcopy):
                with self.subTest(
                    context_name=context_name, operation=operation.__name__
                ):
                    restored = operation(instance)
                    self.assertIsNot(restored, instance)
                    self.assertIs(type(restored), context_type)
                    self.assert_context_restores(
                        restored, expected_enabled=expected_enabled
                    )

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    context_name=context_name, protocol=protocol, value="class"
                ):
                    payload = pickle.dumps(context_type, protocol=protocol)
                    self.assertIn(canonical_path, payload)
                    self.assertIs(pickle.loads(payload), context_type)

                with self.subTest(
                    context_name=context_name, protocol=protocol, value="instance"
                ):
                    payload = pickle.dumps(instance, protocol=protocol)
                    self.assertIn(canonical_path, payload)
                    restored = pickle.loads(payload)
                    self.assertIsNot(restored, instance)
                    self.assertIs(type(restored), context_type)
                    self.assert_context_restores(
                        restored, expected_enabled=expected_enabled
                    )

    def test_copy_and_pickle_preserve_instance_and_subclass_state(self):
        for context_name, stateful_type in (
            ("no_grad", StatefulNoGrad),
            ("enable_grad", StatefulEnableGrad),
        ):
            with self.subTest(context_name=context_name, setup=True):
                context_type = getattr(torch, context_name)
                expected_enabled = context_name == "enable_grad"
                instance = context_type()
                instance.payload = {"values": [1, 2]}

                shallow = copy.copy(instance)
                self.assertEqual(shallow.payload, instance.payload)
                self.assertIs(shallow.payload, instance.payload)

                deep = copy.deepcopy(instance)
                self.assertEqual(deep.payload, instance.payload)
                self.assertIsNot(deep.payload, instance.payload)

                stateful_type.init_calls = 0
                required = {"required": [3, 4]}
                subclass_instance = stateful_type(required)
                subclass_instance.mutated = {"mutated": [5, 6]}

                shallow_subclass = copy.copy(subclass_instance)
                self.assertIs(type(shallow_subclass), stateful_type)
                self.assertIs(shallow_subclass.required, required)
                self.assertIs(shallow_subclass.mutated, subclass_instance.mutated)

                deep_subclass = copy.deepcopy(subclass_instance)
                self.assertIs(type(deep_subclass), stateful_type)
                self.assertEqual(deep_subclass.required, required)
                self.assertEqual(deep_subclass.mutated, subclass_instance.mutated)
                self.assertIsNot(deep_subclass.required, required)
                self.assertIsNot(deep_subclass.mutated, subclass_instance.mutated)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(context_name=context_name, protocol=protocol):
                    restored_instance = pickle.loads(
                        pickle.dumps(instance, protocol=protocol)
                    )
                    self.assertIs(type(restored_instance), context_type)
                    self.assertEqual(restored_instance.payload, instance.payload)
                    self.assertIsNot(restored_instance.payload, instance.payload)

                    restored = pickle.loads(
                        pickle.dumps(subclass_instance, protocol=protocol)
                    )
                    self.assertIs(type(restored), stateful_type)
                    self.assertEqual(restored.required, required)
                    self.assertEqual(restored.mutated, subclass_instance.mutated)
                    self.assertIsNot(restored.required, required)
                    self.assertIsNot(restored.mutated, subclass_instance.mutated)
                    self.assert_context_restores(
                        restored, expected_enabled=expected_enabled
                    )

            self.assertEqual(stateful_type.init_calls, 1)

    def test_copy_and_pickle_honor_subclass_getnewargs(self):
        for context_name, newargs_type in (
            ("no_grad", NewArgsNoGrad),
            ("enable_grad", NewArgsEnableGrad),
        ):
            newargs_type.init_calls = 0
            newargs_type.new_calls = 0
            constructed = {"constructed": [1, 2]}
            instance = newargs_type(constructed)

            shallow = copy.copy(instance)
            self.assertIs(type(shallow), newargs_type)
            self.assertIs(shallow.constructed, constructed)
            self.assertEqual(shallow.mutated, instance.mutated)

            deep = copy.deepcopy(instance)
            self.assertIs(type(deep), newargs_type)
            self.assertEqual(deep.constructed, constructed)
            self.assertIsNot(deep.constructed, constructed)
            self.assertEqual(deep.mutated, instance.mutated)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(context_name=context_name, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(instance, protocol=protocol))
                    self.assertIs(type(restored), newargs_type)
                    self.assertEqual(restored.mutated, instance.mutated)
                    if protocol < 2:
                        self.assertFalse(hasattr(restored, "constructed"))
                    else:
                        self.assertEqual(restored.constructed, constructed)
                        self.assertIsNot(restored.constructed, constructed)

            self.assertEqual(newargs_type.init_calls, 1)
            self.assertEqual(newargs_type.new_calls, pickle.HIGHEST_PROTOCOL + 2)

    def test_aliases_preserve_context_decorator_generator_and_thread_behavior(self):
        autograd_no_grad = torch.autograd.no_grad
        grad_mode_no_grad = torch.autograd.grad_mode.no_grad
        value = torch.tensor([2.0], requires_grad=True)

        with autograd_no_grad():
            self.assertFalse((value * value).requires_grad)
            with grad_mode_no_grad():
                self.assertFalse((value * value).requires_grad)
            self.assertFalse((value * value).requires_grad)
        self.assertTrue((value * value).requires_grad)

        @autograd_no_grad()
        def decorated():
            return (value * value).requires_grad

        @grad_mode_no_grad()
        def generate():
            request = yield (value * value).requires_grad
            yield request, (value * value).requires_grad

        self.assertFalse(decorated())
        generator = generate()
        self.assertFalse(next(generator))
        self.assertTrue((value * value).requires_grad)
        self.assertEqual(generator.send("resume"), ("resume", False))
        self.assertTrue((value * value).requires_grad)

        worker_states = []

        def worker():
            worker_states.append(torch.is_grad_enabled())
            with grad_mode_no_grad():
                worker_states.append(torch.is_grad_enabled())
            worker_states.append(torch.is_grad_enabled())

        with autograd_no_grad():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertFalse(torch.is_grad_enabled())

        self.assertEqual(worker_states, [True, False, True])
        self.assertTrue(torch.is_grad_enabled())

    def test_enable_grad_reenables_recording_inside_no_grad_scopes(self):
        autograd_enable_grad = torch.autograd.enable_grad
        grad_mode_enable_grad = torch.autograd.grad_mode.enable_grad
        value = torch.tensor([2.0], requires_grad=True)

        with torch.no_grad():
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse((value * value).requires_grad)
            with autograd_enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                tracked = value * value
                self.assertTrue(tracked.requires_grad)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse((value * value).requires_grad)

            with grad_mode_enable_grad():
                self.assertTrue(torch.is_grad_enabled())
                tracked.sum().backward()
            self.assertFalse(torch.is_grad_enabled())

        self.assertTrue(torch.is_grad_enabled())
        np.testing.assert_array_equal(np.asarray(value.grad), [4.0])

    def test_enable_grad_restores_state_after_exception(self):
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "restore grad mode"):
                with torch.enable_grad():
                    self.assertTrue(torch.is_grad_enabled())
                    raise RuntimeError("restore grad mode")
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(torch.is_grad_enabled())

    def test_enable_grad_decorator_generator_and_thread_behavior(self):
        value = torch.tensor([3.0], requires_grad=True)

        @torch.enable_grad()
        def decorated():
            return torch.is_grad_enabled(), (value * value).requires_grad

        @torch.enable_grad
        def bare_decorated():
            return torch.is_grad_enabled(), (value * value).requires_grad

        @torch.autograd.grad_mode.enable_grad()
        def generate():
            first_request = yield (
                torch.is_grad_enabled(),
                (value * value).requires_grad,
            )
            yield first_request, torch.is_grad_enabled(), (value * value).requires_grad

        with torch.no_grad():
            self.assertEqual(decorated(), (True, True))
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(bare_decorated(), (True, True))
            self.assertFalse(torch.is_grad_enabled())

            generator = generate()
            self.assertEqual(next(generator), (True, True))
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(generator.send("resume"), ("resume", True, True))
            self.assertFalse(torch.is_grad_enabled())

        worker_states = []

        def worker():
            worker_states.append(torch.is_grad_enabled())
            with torch.enable_grad():
                worker_states.append(torch.is_grad_enabled())
            worker_states.append(torch.is_grad_enabled())

        with torch.no_grad():
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertFalse(torch.is_grad_enabled())

        self.assertEqual(worker_states, [True, True, True])
        self.assertTrue(torch.is_grad_enabled())


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NoGradNamespaceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "grad-mode namespace differentials require pinned PyTorch 2.13.0"
            )

    def contract(self, module, context_name):
        context_type = getattr(module, context_name)
        instance = context_type()
        module_prefix = module.__name__
        return {
            "aliases": (
                getattr(module.autograd, context_name) is context_type,
                getattr(module.autograd.grad_mode, context_name) is context_type,
            ),
            "metadata": (
                context_type.__name__,
                context_type.__qualname__,
                context_type.__module__.removeprefix(module_prefix),
                inspect.getmodule(context_type) is module.autograd.grad_mode,
            ),
            "class_copy": (
                copy.copy(context_type) is context_type,
                copy.deepcopy(context_type) is context_type,
            ),
            "instance_copy": tuple(
                (copied is instance, type(copied) is context_type)
                for copied in (copy.copy(instance), copy.deepcopy(instance))
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

    def state_contract(self, module, context_name, subclass_type):
        context_type = getattr(module, context_name)
        instance = context_type()
        instance.payload = {"values": [1, 2]}
        shallow = copy.copy(instance)
        deep = copy.deepcopy(instance)

        subclass_type.init_calls = 0
        required = {"required": [3, 4]}
        subclass_instance = subclass_type(required)
        subclass_instance.mutated = {"mutated": [5, 6]}
        shallow_subclass = copy.copy(subclass_instance)
        deep_subclass = copy.deepcopy(subclass_instance)
        base_pickle_results = []
        pickle_results = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored_instance = pickle.loads(
                pickle.dumps(instance, protocol=protocol)
            )
            base_pickle_results.append(
                (
                    type(restored_instance) is context_type,
                    restored_instance.payload == instance.payload,
                    restored_instance.payload is instance.payload,
                )
            )
            restored = pickle.loads(
                pickle.dumps(subclass_instance, protocol=protocol)
            )
            pickle_results.append(
                (
                    type(restored) is subclass_type,
                    restored.required == required,
                    restored.mutated == subclass_instance.mutated,
                    restored.required is required,
                    restored.mutated is subclass_instance.mutated,
                )
            )

        return {
            "base_copy": (
                shallow.payload == instance.payload,
                shallow.payload is instance.payload,
                deep.payload == instance.payload,
                deep.payload is instance.payload,
            ),
            "base_pickle": tuple(base_pickle_results),
            "subclass_copy": (
                type(shallow_subclass) is subclass_type,
                shallow_subclass.required is required,
                shallow_subclass.mutated is subclass_instance.mutated,
                type(deep_subclass) is subclass_type,
                deep_subclass.required == required,
                deep_subclass.mutated == subclass_instance.mutated,
                deep_subclass.required is required,
                deep_subclass.mutated is subclass_instance.mutated,
            ),
            "pickle": tuple(pickle_results),
            "init_calls": subclass_type.init_calls,
        }

    def newargs_contract(self, subclass_type):
        subclass_type.init_calls = 0
        subclass_type.new_calls = 0
        constructed = {"constructed": [1, 2]}
        instance = subclass_type(constructed)
        shallow = copy.copy(instance)
        deep = copy.deepcopy(instance)
        pickle_results = []
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            restored = pickle.loads(pickle.dumps(instance, protocol=protocol))
            pickle_results.append(
                (
                    type(restored) is subclass_type,
                    hasattr(restored, "constructed"),
                    getattr(restored, "constructed", None) == constructed,
                    getattr(restored, "constructed", None) is constructed,
                    restored.mutated == instance.mutated,
                )
            )

        return {
            "copy": (
                type(shallow) is subclass_type,
                shallow.constructed == constructed,
                shallow.constructed is constructed,
                shallow.mutated == instance.mutated,
                type(deep) is subclass_type,
                deep.constructed == constructed,
                deep.constructed is constructed,
                deep.mutated == instance.mutated,
            ),
            "pickle": tuple(pickle_results),
            "init_calls": subclass_type.init_calls,
            "new_calls": subclass_type.new_calls,
        }

    def grad_state_contract(self, module):
        value = module.tensor([2.0], requires_grad=True)
        states = [module.is_grad_enabled(), (value * value).requires_grad]
        with module.no_grad():
            states.extend(
                (module.is_grad_enabled(), (value * value).requires_grad)
            )
            with module.enable_grad():
                states.extend(
                    (module.is_grad_enabled(), (value * value).requires_grad)
                )
            states.extend(
                (module.is_grad_enabled(), (value * value).requires_grad)
            )
        states.extend((module.is_grad_enabled(), (value * value).requires_grad))
        return tuple(states)

    def test_namespace_metadata_copy_and_pickle_match_pytorch_2_13(self):
        for context_name in ("no_grad", "enable_grad"):
            with self.subTest(context_name=context_name):
                self.assertEqual(
                    self.contract(torch, context_name),
                    self.contract(reference_torch, context_name),
                )

    def test_instance_and_subclass_state_match_pytorch_2_13(self):
        for context_name, actual_subclass, expected_subclass in (
            ("no_grad", StatefulNoGrad, ReferenceStatefulNoGrad),
            ("enable_grad", StatefulEnableGrad, ReferenceStatefulEnableGrad),
        ):
            with self.subTest(context_name=context_name):
                self.assertEqual(
                    self.state_contract(torch, context_name, actual_subclass),
                    self.state_contract(
                        reference_torch, context_name, expected_subclass
                    ),
                )

    def test_subclass_getnewargs_matches_pytorch_2_13(self):
        for context_name, actual_subclass, expected_subclass in (
            ("no_grad", NewArgsNoGrad, ReferenceNewArgsNoGrad),
            ("enable_grad", NewArgsEnableGrad, ReferenceNewArgsEnableGrad),
        ):
            with self.subTest(context_name=context_name):
                self.assertEqual(
                    self.newargs_contract(actual_subclass),
                    self.newargs_contract(expected_subclass),
                )

    def test_enable_grad_state_restoration_matches_pytorch_2_13(self):
        self.assertEqual(
            self.grad_state_contract(torch),
            self.grad_state_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
