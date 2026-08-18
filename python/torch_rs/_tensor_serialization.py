"""Private, reload-stable serialization state for native Tensor descriptors."""

import copyreg
import types
import weakref
from multiprocessing import reduction

from .torch_rs import Tensor as _NativeTensor


_CANONICAL_TENSOR_TO = _NativeTensor.__base__.__dict__["to"]
_OWNED_REDUCERS = []


def restore_tensor_to():
    return _CANONICAL_TENSOR_TO


def owned_reducer_previous(reducer):
    for reference, previous in tuple(_OWNED_REDUCERS):
        if reference() is reducer:
            return True, previous
    return False, None


def remember_owned_reducer(reducer, previous):
    def forget(reference):
        _OWNED_REDUCERS[:] = [
            entry for entry in _OWNED_REDUCERS if entry[0] is not reference
        ]

    _OWNED_REDUCERS.append((weakref.ref(reducer, forget), previous))


def unwrapped_reducer(reducer):
    owned, previous = owned_reducer_previous(reducer)
    return previous if owned else reducer


def make_reducer(previous, fallback=None):
    canonical_descriptor = _CANONICAL_TENSOR_TO
    restore_descriptor = restore_tensor_to

    def reduce_method_descriptor(descriptor):
        if descriptor is canonical_descriptor:
            return restore_descriptor, ()
        if previous is not None:
            return previous(descriptor)
        if fallback is not None:
            return fallback(descriptor)
        return descriptor.__reduce__()

    remember_owned_reducer(reduce_method_descriptor, previous)
    return reduce_method_descriptor


def install():
    descriptor_type = types.MethodDescriptorType
    previous = unwrapped_reducer(copyreg.dispatch_table.get(descriptor_type))
    copyreg_reducer = make_reducer(previous)
    copyreg.pickle(descriptor_type, copyreg_reducer)

    forking_pickler = reduction.ForkingPickler
    previous = unwrapped_reducer(
        forking_pickler._extra_reducers.get(descriptor_type)
    )
    forking_reducer = make_reducer(previous, fallback=copyreg_reducer)
    forking_pickler.register(descriptor_type, forking_reducer)
