import abc
import functools
import numpy as np
import attr


# Abstract base class compatible with both Python 2 and 3.
ABC = abc.ABCMeta('ABC', (object,), {})


def array_copy_immutable(data):
    """Return a copy of data as an immutable numpy array.

    Useful as an attr.ib converter for frozen classes.
    """
    a = np.copy(data)
    a.flags.writeable = False
    return a


def attrib(doc=None, **kwargs):
    """Wrapper for attr.ib with docstring support via the 'doc' argument."""
    if doc is not None:
        metadata = kwargs.setdefault('metadata', {})
        metadata['doc'] = doc
    return attr.ib(**kwargs)


def cached_property(fget):
    """Decorator for a read-only property whose value is only evaluated once."""
    cache_attr = f"_{fget.__name__}"
    def wrapper(self):
        if not hasattr(self, cache_attr):
            object.__setattr__(self, cache_attr, fget(self))
        return getattr(self, cache_attr)
    return property(functools.update_wrapper(wrapper, fget))
