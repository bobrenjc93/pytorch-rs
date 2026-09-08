import importlib
import inspect
import types
import unittest
import warnings
from collections import OrderedDict

import torch_rs as torch
import torch_rs.nn as nn


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, *args, **kwargs):
        self.seen.append((args, kwargs, self.training))
        return self.seen[-1]


class _MissingForward(nn.Module):
    pass


class ModuleTests(unittest.TestCase):
    def test_canonical_imports_exports_and_deliberate_surface(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_modules = importlib.import_module("torch_rs.nn.modules")
        imported_module_file = importlib.import_module("torch_rs.nn.modules.module")
        from torch_rs.nn import Module
        from torch_rs.nn.modules import Module as ModulesModule

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.modules, imported_modules)
        self.assertIs(Module, nn.Module)
        self.assertIs(ModulesModule, nn.Module)
        self.assertIs(imported_modules.Module, nn.Module)
        self.assertIs(imported_module_file.Module, nn.Module)
        self.assertEqual(imported_modules.__all__, ["Module"])
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(nn, "Parameter"))
        self.assertFalse(hasattr(imported_modules, "Parameter"))
        self.assertNotIn("nn", torch.__all__)

        nn_wildcard = {}
        exec("from torch_rs.nn import *", nn_wildcard)
        self.assertIs(nn_wildcard["Module"], nn.Module)
        self.assertIs(nn_wildcard["factory_kwargs"], nn.factory_kwargs)
        self.assertEqual(
            {name for name in nn_wildcard if not name.startswith("_")},
            {"Module", "factory_kwargs", "functional", "init", "modules"},
        )

        modules_wildcard = {}
        exec("from torch_rs.nn.modules import *", modules_wildcard)
        self.assertEqual(
            {name for name in modules_wildcard if not name.startswith("__")},
            {"Module"},
        )
        self.assertIs(modules_wildcard["Module"], nn.Module)

    def test_class_metadata_and_constructor_errors(self):
        module_type = nn.Module

        self.assertIs(type(module_type), type)
        self.assertEqual(module_type.__name__, "Module")
        self.assertEqual(module_type.__qualname__, "Module")
        self.assertEqual(module_type.__module__, "torch_rs.nn.modules.module")
        self.assertEqual(module_type._version, 1)
        self.assertIs(inspect.getmodule(module_type), nn.modules.module)
        self.assertIs(type(module_type.__call__), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(module_type)),
            "(*args: Any, **kwargs: Any) -> None",
        )
        self.assertEqual(
            str(inspect.signature(module_type.train)),
            "(self, mode: bool = True) -> typing_extensions.Self",
        )
        self.assertEqual(
            str(inspect.signature(module_type.eval)),
            "(self) -> typing_extensions.Self",
        )

        with self.assertRaisesRegex(
            TypeError,
            r"^Module\.__init__\(\) takes 1 positional argument but 2 were given$",
        ):
            nn.Module(1)
        with self.assertRaisesRegex(
            TypeError,
            r"^Module\.__init__\(\) got an unexpected keyword argument 'name'$",
        ):
            nn.Module(name="value")

        with self.assertRaisesRegex(
            TypeError,
            r"^_Recorder\.__init__\(\) takes 1 positional argument but 2 were given$",
        ):
            _Recorder(1)

    def test_training_mode_and_call_forward_dispatch(self):
        module = _Recorder()

        self.assertIs(module.training, True)
        self.assertEqual(module.__dict__["_parameters"], {})
        self.assertEqual(module.__dict__["_buffers"], {})
        self.assertEqual(module.__dict__["_modules"], {})

        self.assertEqual(module(1, 2, label="value"), ((1, 2), {"label": "value"}, True))
        self.assertEqual(module.seen, [((1, 2), {"label": "value"}, True)])

        self.assertIs(module.train(False), module)
        self.assertIs(module.training, False)
        self.assertEqual(module("x"), (("x",), {}, False))

        self.assertIs(module.train(), module)
        self.assertIs(module.training, True)

        self.assertIs(module.eval(), module)
        self.assertIs(module.training, False)

        for mode in (0, 1, None, "false"):
            with self.subTest(mode=mode):
                before = module.training
                with self.assertRaisesRegex(
                    ValueError,
                    "^training mode is expected to be boolean$",
                ):
                    module.train(mode)
                self.assertIs(module.training, before)

    def test_missing_forward_matches_module_error_shape(self):
        for module, name in ((nn.Module(), "Module"), (_MissingForward(), "_MissingForward")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf'^Module \[{name}\] is missing the required "forward" function$',
                ):
                    module(1)

    def test_empty_parameters_buffers_and_state_dict(self):
        module = _Recorder()

        for iterator in (
            module.parameters(),
            module.parameters(False),
            module.named_parameters(),
            module.named_parameters("prefix.", False, False),
            module.buffers(),
            module.buffers(False),
            module.named_buffers(),
            module.named_buffers("prefix.", False, False),
        ):
            with self.subTest(iterator=iterator):
                self.assertIs(type(iterator), types.GeneratorType)
                self.assertEqual(list(iterator), [])

        state = module.state_dict()
        self.assertIs(type(state), OrderedDict)
        self.assertEqual(list(state.items()), [])
        self.assertEqual(state._metadata, OrderedDict([("", {"version": 1})]))

        destination = OrderedDict([("existing", object())])
        self.assertIs(
            module.state_dict(destination=destination, prefix="child."),
            destination,
        )
        self.assertEqual(list(destination), ["existing"])
        self.assertFalse(hasattr(destination, "_metadata"))

        destination_with_metadata = OrderedDict()
        destination_with_metadata._metadata = OrderedDict(
            [("existing", {"version": 0})]
        )
        self.assertIs(
            module.state_dict(destination=destination_with_metadata, prefix="child."),
            destination_with_metadata,
        )
        self.assertEqual(
            destination_with_metadata._metadata,
            OrderedDict([("existing", {"version": 0}), ("child", {"version": 1})]),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            positional = module.state_dict(None, "root.", True)
        self.assertIs(type(positional), OrderedDict)
        self.assertEqual(positional._metadata, OrderedDict([("root", {"version": 1})]))
        self.assertEqual(len(caught), 1)
        self.assertIs(type(caught[0].message), FutureWarning)
        self.assertIn("Positional args are being deprecated", str(caught[0].message))

    def test_plain_attributes_are_not_registered(self):
        module = _Recorder()
        tensor = torch.tensor([1.0, 2.0])

        module.answer = 42
        module.tensor = tensor

        self.assertEqual(module.answer, 42)
        self.assertIs(module.tensor, tensor)
        self.assertEqual(list(module.named_parameters()), [])
        self.assertEqual(list(module.named_buffers()), [])
        self.assertEqual(list(module.state_dict().items()), [])

    def test_unsupported_registration_hooks_conversion_and_loading_raise(self):
        module = _Recorder()
        unsupported = (
            lambda: module.register_parameter("weight", None),
            lambda: module.register_buffer("running", torch.tensor(1.0)),
            lambda: module.add_module("child", _Recorder()),
            lambda: module.register_module("child", _Recorder()),
            lambda: setattr(module, "child", _Recorder()),
            lambda: module.register_forward_pre_hook(lambda *args: None),
            lambda: module.register_forward_hook(lambda *args: None),
            lambda: module.register_full_backward_pre_hook(lambda *args: None),
            lambda: module.register_full_backward_hook(lambda *args: None),
            lambda: module.register_backward_hook(lambda *args: None),
            lambda: module.register_state_dict_pre_hook(lambda *args: None),
            lambda: module.register_state_dict_post_hook(lambda *args: None),
            lambda: module.register_load_state_dict_pre_hook(lambda *args: None),
            lambda: module.register_load_state_dict_post_hook(lambda *args: None),
            lambda: module.to("cpu"),
            lambda: module.to_empty(device="cpu"),
            lambda: module.cpu(),
            lambda: module.cuda(),
            lambda: module.type("torch.FloatTensor"),
            lambda: module.float(),
            lambda: module.double(),
            lambda: module.half(),
            lambda: module.bfloat16(),
            lambda: module.load_state_dict({}),
        )

        for index, call in enumerate(unsupported):
            with self.subTest(index=index):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^torch_rs\.nn\.Module does not support .+; only "
                    r"parameterless eager modules are implemented$",
                ):
                    call()
