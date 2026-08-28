from contextlib import contextmanager as _contextmanager


__allow_nonbracketed_mutation_flag = True


@_contextmanager
def __allow_nonbracketed_mutation():
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


from . import cpu as cpu
from . import cuda as cuda
from . import cudnn as cudnn
from . import kleidiai as kleidiai
from . import mha as mha
from . import mkl as mkl
from . import nnpack as nnpack
from . import openmp as openmp
