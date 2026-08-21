"""Context-manager decorator helpers shared by public compatibility APIs."""

import functools
import inspect
import sys
import warnings
from collections.abc import Callable
from typing import Any, Optional, TypeVar, Union, cast, overload

from typing_extensions import Self


F = TypeVar("F", bound=Callable[..., Any])


def _wrap_generator(ctx_factory, func):
    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        generator = func(*args, **kwargs)
        try:
            with ctx_factory():
                response = generator.send(None)

            while True:
                try:
                    request = yield response
                except GeneratorExit:
                    with ctx_factory():
                        generator.close()
                    raise
                except BaseException:
                    with ctx_factory():
                        response = generator.throw(*sys.exc_info())
                else:
                    with ctx_factory():
                        response = generator.send(request)
        except StopIteration as error:
            return error.value

    return generator_context


def context_decorator(ctx, func):
    if callable(ctx) and hasattr(ctx, "__enter__"):
        raise AssertionError(
            f"Passed in {ctx} is both callable and also a valid context manager "
            "(has __enter__), making it ambiguous which interface to use.  If you "
            "intended to pass a context manager factory, rewrite your call as "
            "context_decorator(lambda: ctx()); if you intended to pass a context "
            "manager directly, rewrite your call as context_decorator(lambda: ctx)"
        )

    if not callable(ctx):

        def ctx_factory():
            return ctx

    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        if inspect.isclass(orig_func):
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else.",
                FutureWarning,
                stacklevel=2,
            )
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        return self.__class__()


class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""

    @overload
    def __new__(cls, orig_func: F) -> F: ...

    @overload
    def __new__(cls, orig_func: None = None) -> Self: ...

    def __new__(
        cls, orig_func: Optional[F] = None
    ) -> Union[Self, F]:  # type: ignore[misc]
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)
