import sys as _sys
import types as _types
from contextlib import contextmanager as _contextmanager


# Freezing catches unscoped backend preference changes that can leak between
# tests or otherwise affect unrelated work in the same process.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags():
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen():
    return not __allow_nonbracketed_mutation_flag


@_contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context "
                "manager instead"
            )


class PropModule(_types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)


class GenericModule(PropModule):
    pass


# Keep forwarded configuration helpers directly importable while matching
# PyTorch's module namespace, wildcard imports, and replacement-on-reload.
_sys.modules[__name__] = GenericModule(_sys.modules[__name__], __name__)


from . import cpu as cpu
from . import cuda as cuda
from . import cusparselt as cusparselt
from . import cudnn as cudnn
from . import kleidiai as kleidiai
from . import mha as mha
from . import mkl as mkl
from . import nnpack as nnpack
from . import openmp as openmp
