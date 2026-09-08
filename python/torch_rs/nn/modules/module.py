import warnings
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any

from typing_extensions import Self


def _unsupported(feature: str) -> NotImplementedError:
    return NotImplementedError(
        f"torch_rs.nn.Module does not support {feature}; only parameterless "
        "eager modules are implemented"
    )


def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )


class Module:
    _version = 1

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs:
            name = next(iter(kwargs))
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected keyword "
                f"argument '{name}'"
            )
        if args:
            count = len(args) + 1
            noun = "argument" if count == 1 else "arguments"
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional "
                f"argument but {count} were given"
            )

        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            raise _unsupported("child module registration")
        object.__setattr__(self, name, value)

    forward = _forward_unimplemented

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True) -> Self:
        if type(mode) is not bool:
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        return self

    def eval(self) -> Self:
        return self.train(False)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        if self._parameters or self._modules:
            raise _unsupported("parameters or child module traversal")
        if False:
            yield prefix, recurse, remove_duplicate

    def parameters(self, recurse: bool = True) -> Iterator[Any]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def named_buffers(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, Any]]:
        if self._buffers or self._modules:
            raise _unsupported("buffers or child module traversal")
        if False:
            yield prefix, recurse, remove_duplicate

    def buffers(self, recurse: bool = True) -> Iterator[Any]:
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def state_dict(
        self,
        *args: Any,
        destination: Any = None,
        prefix: str = "",
        keep_vars: bool = False,
    ):
        if args:
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. "
                "Refer to https://pytorch.org/docs/main/generated/"
                "torch.nn.Module.html#torch.nn.Module.state_dict for details.",
                FutureWarning,
                stacklevel=2,
            )
            destination = args[0]
            if len(args) > 1:
                prefix = args[1]
            if len(args) > 2:
                keep_vars = args[2]

        if self._parameters or self._buffers or self._modules:
            raise _unsupported("non-empty module state dictionaries")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = {"version": self._version}

        return destination

    def register_parameter(self, name: str, param: Any) -> None:
        raise _unsupported("parameter registration")

    def register_buffer(
        self,
        name: str,
        tensor: Any,
        persistent: bool = True,
    ) -> None:
        raise _unsupported("buffer registration")

    def add_module(self, name: str, module: "Module | None") -> None:
        raise _unsupported("child module registration")

    def register_module(self, name: str, module: "Module | None") -> None:
        raise _unsupported("child module registration")

    def register_forward_pre_hook(
        self,
        hook: Any,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ):
        raise _unsupported("module hooks")

    def register_forward_hook(
        self,
        hook: Any,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ):
        raise _unsupported("module hooks")

    def register_full_backward_pre_hook(
        self,
        hook: Any,
        prepend: bool = False,
    ):
        raise _unsupported("module hooks")

    def register_full_backward_hook(
        self,
        hook: Any,
        prepend: bool = False,
    ):
        raise _unsupported("module hooks")

    def register_backward_hook(self, hook: Any):
        raise _unsupported("module hooks")

    def register_state_dict_pre_hook(self, hook: Any):
        raise _unsupported("module hooks")

    def register_state_dict_post_hook(self, hook: Any):
        raise _unsupported("module hooks")

    def register_load_state_dict_pre_hook(self, hook: Any):
        raise _unsupported("module hooks")

    def register_load_state_dict_post_hook(self, hook: Any):
        raise _unsupported("module hooks")

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        raise _unsupported("state-dict loading")

    def to(self, *args: Any, **kwargs: Any) -> Self:
        raise _unsupported("module device or dtype conversion")

    def to_empty(self, *, device: Any, recurse: bool = True) -> Self:
        raise _unsupported("module device or dtype conversion")

    def cpu(self) -> Self:
        raise _unsupported("module device or dtype conversion")

    def cuda(self, device: Any = None) -> Self:
        raise _unsupported("module device or dtype conversion")

    def type(self, dst_type: Any) -> Self:
        raise _unsupported("module device or dtype conversion")

    def float(self) -> Self:
        raise _unsupported("module device or dtype conversion")

    def double(self) -> Self:
        raise _unsupported("module device or dtype conversion")

    def half(self) -> Self:
        raise _unsupported("module device or dtype conversion")

    def bfloat16(self) -> Self:
        raise _unsupported("module device or dtype conversion")
