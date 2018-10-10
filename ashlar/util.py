import abc
import numpy as np


# Abstract base class compatible with both Python 2 and 3.
ABC = abc.ABCMeta('ABC', (object,), {})


def array_copy_immutable(data):
    """Return a copy of data as an immutable numpy array.

    Useful as an attr.ib converter for frozen classes.
    """
    a = np.copy(data)
    a.flags.writeable = False
    return a
