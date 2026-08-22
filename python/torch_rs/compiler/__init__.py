from collections.abc import Callable
from typing import Any


__all__ = [
    "assume_constant_result",
    "list_backends",
    "get_default_backend",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
]


def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function.
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """
    fn._dynamo_marked_constant = True
    return fn


def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """
    return []


def get_default_backend() -> str | Callable[..., Any]:
    """Return the current default backend for ``torch.compile``.

    Returns:
        The current default backend (string or callable). Initially ``"inductor"``.
    """
    return "inductor"


def is_compiling() -> bool:
    """
    Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().

    Note that there are 2 other related flags that should deprecated eventually:
      * torch._dynamo.external_utils.is_compiling()
      * torch._utils.is_compiling()

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_compiling():
        >>>        pass # ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    return False


def is_dynamo_compiling() -> bool:
    """
    Indicates whether a graph is traced via TorchDynamo.

    It's stricter than is_compiling() flag, as it would only be set to True when
    TorchDynamo is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_dynamo_compiling():
        >>>        pass # ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    return False


def is_exporting() -> bool:
    """
    Indicated whether we're under exporting.

    It's stricter than is_compiling() flag, as it would only be set to True when
    torch.export is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_exporting():
        >>>        pass # ...logic that is not needed in export...
        >>>
        >>>     # ...rest of the function...
    """
    return False
