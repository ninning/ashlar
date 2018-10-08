import numbers
import attr
import numpy as np


@attr.s(frozen=True)
class Vector(object):
    y = attr.ib(converter=float)
    x = attr.ib(converter=float)

    @classmethod
    def from_ndarray(cls, a):
        if a.shape != (2,):
            raise ValueError("array shape must be (2,)")
        return cls(*a)

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise NotImplementedError
        return Vector(self.y + other.y, self.x + other.x)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            raise NotImplementedError
        return Vector(self.y - other.y, self.x - other.x)

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            raise NotImplementedError
        return Vector(self.y * other, self.x * other)

    def __div__(self, other):
        if not isinstance(other, numbers.Number):
            raise NotImplementedError
        return Vector(self.y / other, self.x / other)

    def __array__(self):
        return np.array([self.y, self.x])


@attr.s(frozen=True)
class Rectangle(object):
    vector1 = attr.ib()
    vector2 = attr.ib()

    @classmethod
    def from_shape(cls, vector, shape):
        return cls(vector, vector + shape)

    @classmethod
    def rmin(cls, v1, v2):
        """Return Vector at "rectangle minimum" of v1 and v2."""
        return Vector(x=min(v1.x, v2.x), y=min(v1.y, v2.y))

    @classmethod
    def rmax(cls, v1, v2):
        """Return Vector at "rectangle maximum" of v1 and v2."""
        return Vector(x=max(v1.x, v2.x), y=max(v1.y, v2.y))

    def __attrs_post_init__(self):
        # Normalize to make vector1 the lower corner and vector2 the upper.
        v1 = self.rmin(self.vector1, self.vector2)
        v2 = self.rmax(self.vector1, self.vector2)
        object.__setattr__(self, 'vector1', v1)
        object.__setattr__(self, 'vector2', v2)

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise NotImplementedError
        return attr.evolve(self, vector2=self.vector2 + other)

    @property
    def shape(self):
        return self.vector2 - self.vector1

    @property
    def area(self):
        s = self.shape
        return s.x * s.y

    def intersection(self, other):
        p1 = self.rmax(self.vector1, other.vector1)
        p2 = self.rmin(self.vector2, other.vector2)
        # Check for degenerate rectangle.
        if p1.x >= p2.x or p1.y >= p2.y:
            # FIXME is this the right solution?
            p1 = p2 = self.vector1
        return Rectangle(p1, p2)
